#!/usr/bin/env python
"""End-to-end throughput measurement for MMLU multiple-choice training."""

import argparse
import logging
import math
import os
import random
import sys
from datetime import timedelta
from pathlib import Path
import time

import datasets
import torch
from accelerate import Accelerator, DistributedDataParallelKwargsCustom, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datetime import datetime

from new_comm_hooks.comm_hooks import (
    P2P_THC_compress_hook,
    P2P_dynamiQ_hook,
    P2P_dynamiQ_dynamic_bitrate_hook,
    P2P_MXfp8_compress_hook,
    P2P_bf16_compress_hook,
    P2P_omnireduce_topk_hook,
    P2P_slicing_compress_hook,
)
from new_comm_hooks.comm_hooks import CHUNK_SIZE_THRESHOLD
import torch.distributed as dist



import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)
from transformers.utils import send_example_telemetry
import numpy as np

from torch.utils.data import Sampler
import pickle
from dataset_llm_factories import multiple_choice_dataset


MEASURE_START_STEP = 4
MEASURE_END_STEP = 150
LR_DECAY_EPOCHS = 2
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"


class CustomSelectiveDataCollator(DataCollatorForTokenClassification):
    def __init__(self, tokenizer, pad_fields=None, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.pad_fields = tuple(pad_fields or ("input_ids", "attention_mask", "labels"))

    def __call__(self, features):
        # Separate out the fields to pad and those to leave untouched
        pad_features = [
            {k: f[k] for k in f if k in self.pad_fields}
            for f in features
        ]
        non_pad_features = [
            {k: f[k] for k in f if k not in self.pad_fields}
            for f in features
        ]

        batch = super().__call__(pad_features)

        # Merge back non-padded fields (assumes they are same across samples or don't need padding)
        for key in non_pad_features[0]:
            batch[key] = [f[key] for f in non_pad_features]

        return batch

class PrecomputedSampler(Sampler):
    def __init__(self, rank, model_name_or_path=None):
        total_rank = dist.get_world_size()

        if "gemma" in model_name_or_path or "GPT2" in model_name_or_path:
            index_path = MODELS_DIR / f"indices_mmlu_new_gemma_{total_rank}.pkl"
        else:
            index_path = MODELS_DIR / f"indices_mmlu_new_{total_rank}.pkl"
        with index_path.open("rb") as fin:
            self.indices_list_for_all = pickle.load(fin)
        self.indices_list = self.indices_list_for_all[rank]
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.epoch >= len(self.indices_list):
            print("Warning: self.epoch >= len(self.indices_list)")
            self.epoch = 0

    def __iter__(self):
        return iter(self.indices_list[self.epoch])

    def __len__(self):
        return len(self.indices_list[self.epoch])



logger = get_logger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cais/mmlu",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="all",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
        default="meta-llama/Llama-3.2-1B"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )

    parser.add_argument(
        "--data_cache_dir",
        type=str,
        default="/cluster/project2/gcreduce_data/data"
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default="/cluster/project2/gcreduce_data/pretrained_models/language_model"
    )

    parser.add_argument("--normalized", default=1, type=int)
    parser.add_argument("--normalized_chunk_size", default=1024, type=int)
    parser.add_argument("--aggregation_method", type=str, default="None")
    parser.add_argument('--nclients', default=8, type=int, help='Number of clients')

    parser.add_argument('--ef', default=False, type=bool, help='use error feedback')
    ### INCA tables' parameters (not all exist) 
    parser.add_argument('--overflow_frequency', default=1024, type=int, help='one over the expected number of overflowed coordinates')
    ### rotate the gradients before quantization
    parser.add_argument('--rotation', default="True", type=str, help='use rotation')
    ### quantization levels to use
    parser.add_argument('--quantization_levels', default=64, type=int, help='# quantization levels to use')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_norm_div_factor", type=float, default=1.0)
    parser.add_argument("--nbits_communication", type=int, default=8)
    parser.add_argument("--resume_dir", type=str, default="None")
    parser.add_argument("--fine_tuning", type=str, default="False")
    parser.add_argument("--last_step", type=int, default=0)
    parser.add_argument("--to_perm", type=str, default="False")
    parser.add_argument("--max_chunk_size", type=int, default=32)
    parser.add_argument("--smaller_max_chunk_size", type=int, default=32)
    parser.add_argument("--agg_chunk_size", type=int, default=16)
    parser.add_argument("--compression", type=str, default="None")
    parser.add_argument("--sparsity", type=str, default="None")

    parser.add_argument("--to_shrimp", type=str, default="False")

    args = parser.parse_args()

    # User can set dist._DEFAULT_FIRST_BUCKET_BYTES to tune DDP first
    dist._DEFAULT_FIRST_BUCKET_BYTES = 4000 * 1024 * 1024 if "dynamiq" in args.aggregation_method.lower() else 256 * 1024 * 1024


    if args.model_name_or_path == "gemma":
        args.model_name_or_path = "google/gemma-3-1b-it"
    
    elif args.model_name_or_path == "DistilGPT2":
        args.model_name_or_path = "distilbert/distilgpt2"

    return args


def get_suffix():
    return datetime.now().strftime("%m_%d_%Y__%H_%M_%S") + "," + str(hash(str(sys.argv)))


def cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_comm_state(args):
    params = {
        "args": args,
        "nclients": args.nclients,
        "seed": args.seed,
        "table_dir": "../simulation/compression/new_tables",
        "d": {},
        "size": {},
        "keys": [],
        "measure_points": {},
        "max_norm": {},
        "smaller_max_norm": {},
        "norm": {},
        "agg_chunk_size": args.agg_chunk_size,
        "overflow_prob": 8,
        "max_chunk_size": args.max_chunk_size,
        "smaller_max_chunk_size": args.smaller_max_chunk_size,
        "heuristic": "chunk_size" if args.aggregation_method.lower() != "dynamiq_mixed" else "bitrate",
        "lr_adjust_param": 1,
        "MAX_BUCKET_SIZE": CHUNK_SIZE_THRESHOLD,
        "ef": args.ef,
        "rotation": args.rotation == "True",
        "quantization_levels": args.quantization_levels,
        "overflow_frequency": args.overflow_frequency,
        "normalized": args.normalized,
        "chunk_size": args.normalized_chunk_size,
        "supergroup": 16,
        "device": "cuda",
        "is_correlated": "correlated" in args.aggregation_method,
    }
    if args.sparsity != "None":
        params["target_topk"] = float(args.sparsity)
    if args.to_shrimp == "True":
        params["to_shrimp"] = True

    return {
        "batch_idx": -1,
        "params": params,
        "start_idx": {},
        "partition_len": {},
        "ret_tensor": {},
        "start_interm_idx": {},
        "interm_reduce_tensor": {},
        "args": args,
    }


def normalize_comm_args(args):
    if "thc" in args.aggregation_method.lower():
        args.quantization_levels = 31


def select_comm_hook(aggregation_method):
    method = aggregation_method.lower()
    if "thc" in method:
        return P2P_THC_compress_hook
    if "omnireduce" in method or "omni" in method:
        return P2P_omnireduce_topk_hook
    if "bf16" in method:
        return P2P_bf16_compress_hook
    if "dynamiq" in method and "bitrate" in method:
        return P2P_dynamiQ_dynamic_bitrate_hook
    if "dynamiq" in method:
        return P2P_dynamiQ_hook
    if "fp8" in method:
        return P2P_MXfp8_compress_hook
    if "fp4" in method or "fp6" in method or "zero" in method:
        return P2P_slicing_compress_hook
    raise ValueError(
        "Unsupported aggregation_method for train_llm_mmlu.py. "
        "Use bf16, fp8/MXfp8, fp4/fp6/zero, thc, omnireduce, or dynamiQ_aee/dynamiQ_mee."
    )


def build_lr_scheduler(optimizer, num_update_steps_per_epoch):
    decay_steps = max(1, LR_DECAY_EPOCHS * num_update_steps_per_epoch)
    return torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=1.0 / 8.0,
        total_iters=decay_steps,
    )


def maybe_step_lr_scheduler(lr_scheduler, epoch):
    if epoch < LR_DECAY_EPOCHS:
        lr_scheduler.step()


def train(comm_state, sampler):
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    timing_started = False
    start_time_secs = None
    measured_steps = 0

    if accelerator.is_main_process:
        txt_log_file = params["txt_result_file"] # logging the results

    for epoch in range(args.num_train_epochs):
        model.train()
        sampler.set_epoch(epoch)

        comm_state["batch_idx"] = -1

        measurement_data = []
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            measurement_data.append([input_ids, attention_mask, labels])
            if step >= MEASURE_END_STEP + 2:
                break
        

        for step, batch in enumerate(measurement_data):
            comm_state["batch_idx"] += 1
            step_number = completed_steps + 1

            if step_number == MEASURE_START_STEP:
                cuda_synchronize()
                start_time_secs = time.perf_counter()
                timing_started = True
            elif timing_started and step_number % 10 == MEASURE_START_STEP:
                # cuda_synchronize()
                logger.info("batch_idx=%s elapsed=%.6f", comm_state["batch_idx"], time.perf_counter() - start_time_secs)

            input_ids, attention_mask, labels = batch[0], batch[1], batch[2]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            accelerator.backward(loss)
            optimizer.step()
            maybe_step_lr_scheduler(lr_scheduler, epoch)
            optimizer.zero_grad()

            completed_steps += 1

            if timing_started and step_number <= MEASURE_END_STEP:
                measured_steps += 1

            if step_number >= MEASURE_END_STEP or completed_steps >= args.max_train_steps:
                cuda_synchronize()
                end_time_secs = time.perf_counter()
                elapsed = end_time_secs - start_time_secs if start_time_secs is not None else 0.0
                avg_step_time = elapsed / measured_steps if measured_steps else float("nan")
                if accelerator.is_main_process:
                    line = (
                        f"measured_steps {measured_steps} "
                        f"elapsed_seconds {elapsed:.6f} "
                        f"avg_seconds_per_step {avg_step_time:.6f}\n"
                    )
                    txt_log_file.write(line)
                    txt_log_file.flush()
                    print(line, end="")
                return
        



if __name__ == "__main__":
    args = parse_args()

    set_seed(args.seed)

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm_no_trainer", args)

    suffix = get_suffix()
    if args.output_dir == None:
        args.output_dir = os.path.join("./results", suffix)

    normalize_comm_args(args)
    comm_state = build_comm_state(args)
    params = comm_state["params"]
    comm_hook = select_comm_hook(args.aggregation_method)

    ddp_kwargs = DistributedDataParallelKwargsCustom(bucket_cap_mb=4000 if "dynamiq" in args.aggregation_method.lower() else 256, comm_hook=comm_hook, comm_state_option=[comm_state]) # avoid rdma buffer overflow...

    init_process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=40000))

    

    proj_config = ProjectConfiguration(total_limit=1)


    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, init_process_group_kwargs], gradient_accumulation_steps=args.gradient_accumulation_steps, project_config=proj_config)

    accelerator.wait_for_everyone()

    def process_group_warm_up():
        reduce_vec = torch.tensor([1.], device="cuda")
        dist.all_reduce(reduce_vec)

        print(reduce_vec)
    
    process_group_warm_up()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info("model_name_or_path=%s", args.model_name_or_path)
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Get the datasets.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        print(args.dataset_name, args.dataset_config_name, args.data_cache_dir)
        raw_datasets = load_dataset(
            args.dataset_name, args.dataset_config_name, trust_remote_code=args.trust_remote_code, cache_dir=args.data_cache_dir, split="auxiliary_train"
        )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    config_kwargs = {
        "cache_dir": args.model_cache_dir,
        "revision": "main",
        "token": None,
        "trust_remote_code": args.trust_remote_code,
    }

    config = AutoConfig.from_pretrained(args.config_name or args.model_name_or_path, **config_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer,
        **config_kwargs,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        torch_dtype=torch.bfloat16,
        **config_kwargs
    )
    accelerator.wait_for_everyone()

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        print("Warning::gemma's tokenizer contains a <image_soft_token> which is unused.")

    
    
    

    with accelerator.main_process_first():
        split_dataset = raw_datasets.train_test_split(test_size=0.1, seed=args.seed)
        lm_datasets = split_dataset.map(
            lambda examples: multiple_choice_dataset(examples, tokenizer),
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
        )

    MAX_LENGTH = 1024

    def filter_long_examples(example):
        return len(example["input_ids"]) < MAX_LENGTH

    with accelerator.main_process_first():
        lm_dataset_new = lm_datasets.filter(filter_long_examples, num_proc=args.preprocessing_num_workers)
        lm_datasets = lm_dataset_new

    

    del raw_datasets # free up memory

    train_dataset = lm_datasets["train"].remove_columns(["question", "subject", "choices", "answer"])

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    # train_dataloader = DataLoader(
    #     train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    # )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sampler = PrecomputedSampler(rank=dist.get_rank(), model_name_or_path=args.model_name_or_path)
    train_dataloader = DataLoader( 
        train_dataset, sampler=sampler, collate_fn=CustomSelectiveDataCollator(tokenizer), batch_size=args.per_device_train_batch_size, num_workers=16
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=0.0)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    model, optimizer = accelerator.prepare(model, optimizer)
    

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    lr_scheduler = build_lr_scheduler(optimizer, num_update_steps_per_epoch)

    if accelerator.is_main_process:
        params["txt_log_file"] = open(os.path.join(args.output_dir, "log.txt"), "a")
        params["txt_result_file"] = open(os.path.join(args.output_dir, "results.txt"), "a")

        txt_config_file = open(os.path.join(args.output_dir, 'config.txt'), "a")    
        txt_config_file.write(str(vars(args)) + "\n")
        txt_config_file.write(str(params))
        txt_config_file.close()


    accelerator.wait_for_everyone()
    train(comm_state, sampler)
