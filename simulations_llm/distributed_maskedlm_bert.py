#!/usr/bin/env python
"""End-to-end throughput measurement for BERT masked language modeling."""

import argparse
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta
from itertools import chain

import datasets
import numpy as np
import torch
import torch.distributed as dist
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargsCustom, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    default_data_collator,
)
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from new_comm_hooks.comm_hooks import CHUNK_SIZE_THRESHOLD, wrapper_hook


MEASURE_START_STEP = 4
MEASURE_END_STEP = 150
DEFAULT_LR_DECAY_EPOCHS = 15
DEFAULT_FIXED_LR_EPOCHS = 6
LR_END_FACTOR = 1.0 / 16.0

logger = get_logger(__name__)

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Measure BERT MaskedLM throughput on Wikitext103")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="The name of the dataset to use via the datasets library.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="wikitext-103-v1",
        help="The configuration name of the dataset to use via the datasets library.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv, txt or json file containing the training data.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-large-cased",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, use a slow tokenizer instead of a fast tokenizer.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device for the training dataloader.",
    )
    parser.add_argument(
        "--init_lr",
        type=float,
        default=5e-5,
        help="Initial learning rate for the fixed BERT MLM schedule.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Compatibility alias for --init_lr. If set, it overrides --init_lr.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=DEFAULT_LR_DECAY_EPOCHS,
        help="Number of epochs over which to linearly decay the learning rate.",
    )
    parser.add_argument(
        "--fixed_lr_epochs",
        type=int,
        default=DEFAULT_FIXED_LR_EPOCHS,
        help="Number of extra epochs to run after LR decay with fixed final LR.",
    )
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
        help="Number of update steps to accumulate before a backward/update pass.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store logs/results.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--block_size",
        type=int,
        default=512,
        help="Input sequence length including BERT special tokens.",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=69632,
        help="Maximum number of chunked training examples to keep. Use <=0 to disable.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for preprocessing.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="The number of worker processes for the training dataloader.",
    )
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite cached datasets.")
    parser.add_argument("--no_keep_linebreaks", action="store_true", help="Do not keep line breaks for TXT files.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote dataset/model code.",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help="Create the model as an empty shell before loading pretrained weights.",
    )
    parser.add_argument(
        "--data_cache_dir",
        type=str,
        default=os.environ.get("DYNAMIQ_DATA_CACHE"),
        help="Dataset cache directory. Defaults to Hugging Face's cache unless DYNAMIQ_DATA_CACHE is set.",
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=os.environ.get("DYNAMIQ_MODEL_CACHE"),
        help="Model cache directory. Defaults to Hugging Face's cache unless DYNAMIQ_MODEL_CACHE is set.",
    )

    parser.add_argument("--normalized", default=1, type=int)
    parser.add_argument("--normalized_chunk_size", default=1024, type=int)
    parser.add_argument("--aggregation_method", type=str, default="None")
    parser.add_argument("--nclients", default=8, type=int, help="Number of clients")
    parser.add_argument("--ef", default=False, type=bool, help="Use error feedback")
    parser.add_argument("--overflow_frequency", default=1024, type=int)
    parser.add_argument("--rotation", default="True", type=str)
    parser.add_argument("--quantization_levels", default=64, type=int)
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
    parser.add_argument("--sparsity", type=str, default="None")
    parser.add_argument("--to_shrimp", type=str, default="False")
    parser.add_argument("--measure_comm_error", action="store_true", help="Enable per-bucket L2 communication-error logging.")

    args = parser.parse_args()

    if args.dataset_name in {"", "None", "none"}:
        args.dataset_name = None
    if args.dataset_config_name in {"", "None", "none"}:
        args.dataset_config_name = None

    if args.learning_rate is not None:
        args.init_lr = args.learning_rate
    args.learning_rate = args.init_lr

    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    if args.train_file is not None:
        extension = args.train_file.split(".")[-1]
        if extension not in ["csv", "json", "txt"]:
            raise ValueError("Dataset files should be csv, json or txt files.")

    if args.model_name_or_path == "bert":
        args.model_name_or_path = "bert-large-cased"

    logger.info("model_name_or_path=%s", args.model_name_or_path)

    return args


def get_suffix():
    return datetime.now().strftime("%m_%d_%Y__%H_%M_%S") + "," + str(hash(str(sys.argv)))


def cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_comm_state(args):
    aggregation_method_lower = args.aggregation_method.lower()
    params = {
        "args": args,
        "nclients": args.nclients,
        "seed": args.seed,
        "table_dir": "compression/new_tables",
        "measure_comm_error": args.measure_comm_error,
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
        "heuristic": "bitrate" if (
            aggregation_method_lower == "dynamiq_mixed"
            or ("dynamiq" in aggregation_method_lower and "bitrate" in aggregation_method_lower)
        ) else "chunk_size",
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


def build_lr_scheduler(optimizer, num_update_steps_per_epoch, decay_epochs):
    decay_steps = max(1, decay_epochs * num_update_steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=LR_END_FACTOR,
        total_iters=decay_steps,
    )
    return scheduler, decay_steps


def maybe_step_lr_scheduler(lr_scheduler, completed_steps, lr_decay_steps):
    if completed_steps < lr_decay_steps:
        lr_scheduler.step()


def mask_tokens(inputs, tokenizer, args, special_tokens_mask=None):
    """Prepare masked token inputs/labels: 80% MASK, 10% random, 10% original."""
    if tokenizer.mask_token is None:
        raise ValueError("This tokenizer does not define a mask token, so it cannot be used for MLM.")

    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, args.mlm_probability, device=labels.device)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=labels.device)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=labels.device)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5, device=labels.device)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=labels.device)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels


def resolve_block_sizes(args, tokenizer):
    model_max_length = tokenizer.model_max_length
    if model_max_length is None or model_max_length > 100000:
        model_max_length = 512

    block_size = args.block_size if args.block_size and args.block_size > 0 else model_max_length
    if block_size > model_max_length:
        logger.warning(
            "The block_size passed (%s) is larger than the maximum length for the model (%s). "
            "Using block_size=%s.",
            block_size,
            model_max_length,
            model_max_length,
        )
        block_size = model_max_length

    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    payload_block_size = block_size - special_tokens
    if payload_block_size <= 0:
        raise ValueError(f"block_size={block_size} leaves no room after {special_tokens} special tokens.")

    return block_size, payload_block_size


def load_raw_datasets(args):
    if args.dataset_name is not None:
        return load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.data_cache_dir,
        )

    dataset_args = {}
    data_files = {"train": args.train_file}
    extension = args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
        dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks

    return load_dataset(extension, data_files=data_files, cache_dir=args.data_cache_dir, **dataset_args)


def build_mlm_datasets(args, tokenizer, accelerator):
    raw_datasets = load_raw_datasets(args)
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    block_size, payload_block_size = resolve_block_sizes(args, tokenizer)
    logger.info("block_size=%s payload_block_size=%s", block_size, payload_block_size)

    def tokenize_function(examples):
        texts = [text for text in examples[text_column_name] if text and not text.isspace()]
        if not texts:
            return {"input_ids": []}
        return tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    def group_texts(examples):
        input_ids = list(chain(*examples["input_ids"]))
        total_length = (len(input_ids) // payload_block_size) * payload_block_size
        if total_length == 0:
            return {"input_ids": [], "attention_mask": [], "special_tokens_mask": []}

        result_input_ids = []
        result_attention_mask = []
        result_special_tokens_mask = []
        for i in range(0, total_length, payload_block_size):
            chunk = input_ids[i : i + payload_block_size]
            chunk = tokenizer.build_inputs_with_special_tokens(chunk)
            result_input_ids.append(chunk)
            result_attention_mask.append([1] * len(chunk))
            result_special_tokens_mask.append(
                tokenizer.get_special_tokens_mask(chunk, already_has_special_tokens=True)
            )

        return {
            "input_ids": result_input_ids,
            "attention_mask": result_attention_mask,
            "special_tokens_mask": result_special_tokens_mask,
        }

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=tokenized_datasets["train"].column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"]

    if args.max_train_samples is not None and args.max_train_samples > 0:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))

    return train_dataset


def process_group_warm_up():
    if not (dist.is_available() and dist.is_initialized()):
        return
    reduce_vec = torch.tensor([1.0], device="cuda")
    dist.all_reduce(reduce_vec)


def train(comm_state):
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info("  Num examples = %s", len(train_dataset))
    logger.info("  LR decay epochs = %s", args.num_train_epochs)
    logger.info("  Fixed LR epochs = %s", args.fixed_lr_epochs)
    logger.info("  Num Epochs = %s", args.total_train_epochs)
    logger.info("  Instantaneous batch size per device = %s", args.per_device_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %s", total_batch_size)
    logger.info("  Gradient Accumulation steps = %s", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %s", args.max_train_steps)

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    timing_started = False
    start_time_secs = None
    measured_steps = 0
    wrote_timing_summary = False

    if accelerator.is_main_process:
        txt_log_file = params["txt_result_file"]

    for epoch in range(args.total_train_epochs):
        model.train()

        for step, batch in enumerate(train_dataloader):
            comm_state["batch_idx"] += 1
            step_number = completed_steps + 1

            if step_number == MEASURE_START_STEP:
                cuda_synchronize()
                start_time_secs = time.perf_counter()
                timing_started = True
            elif timing_started and step_number % 10 == MEASURE_START_STEP:
                cuda_synchronize()
                logger.info("batch_idx=%s elapsed=%.6f", comm_state["batch_idx"], time.perf_counter() - start_time_secs)

            input_ids = batch["input_ids"].clone()
            special_tokens_mask = batch.get("special_tokens_mask")
            input_ids, labels = mask_tokens(input_ids, tokenizer, args, special_tokens_mask=special_tokens_mask)

            outputs = model(input_ids=input_ids, attention_mask=batch["attention_mask"], labels=labels)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            maybe_step_lr_scheduler(lr_scheduler, completed_steps, lr_decay_steps)
            optimizer.zero_grad()

            progress_bar.update(1)
            completed_steps += 1

            if timing_started and step_number <= MEASURE_END_STEP:
                measured_steps += 1

            if timing_started and step_number == MEASURE_END_STEP and not wrote_timing_summary:
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
                wrote_timing_summary = True

            if completed_steps >= args.max_train_steps:
                return


if __name__ == "__main__":
    args = parse_args()

    set_seed(args.seed)
    send_example_telemetry("run_mlm_no_trainer", args)

    suffix = get_suffix()
    if args.output_dir is None:
        args.output_dir = os.path.join("./results", suffix)

    normalize_comm_args(args)
    comm_state = build_comm_state(args)
    params = comm_state["params"]

    ddp_kwargs = DistributedDataParallelKwargsCustom(
        bucket_cap_mb=500,
        comm_hook=wrapper_hook,
        comm_state_option=comm_state,
    )
    init_process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=40000))
    proj_config = ProjectConfiguration(total_limit=1)

    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=proj_config,
    )

    accelerator.wait_for_everyone()
    process_group_warm_up()

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

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

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

    if tokenizer.mask_token is None:
        raise ValueError("Masked language modeling requires a tokenizer with a mask token.")

    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        torch_dtype=torch.bfloat16,
        **config_kwargs,
    )
    accelerator.wait_for_everyone()

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    train_dataset = build_mlm_datasets(args, tokenizer, accelerator)

    for index in random.sample(range(len(train_dataset)), min(3, len(train_dataset))):
        logger.info("Sample %s of the training set: %s.", index, train_dataset[index])

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=default_data_collator,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
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
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.init_lr, weight_decay=0.0)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.total_train_epochs = args.num_train_epochs + args.fixed_lr_epochs
        args.max_train_steps = args.total_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    model, optimizer, train_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.total_train_epochs = args.num_train_epochs + args.fixed_lr_epochs
        args.max_train_steps = args.total_train_epochs * num_update_steps_per_epoch
    else:
        args.total_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    lr_scheduler, lr_decay_steps = build_lr_scheduler(optimizer, num_update_steps_per_epoch, args.num_train_epochs)

    if accelerator.is_main_process:
        params["txt_log_file"] = open(os.path.join(args.output_dir, "log.txt"), "a")
        params["txt_result_file"] = open(os.path.join(args.output_dir, "results.txt"), "a")

        with open(os.path.join(args.output_dir, "config.txt"), "a") as txt_config_file:
            txt_config_file.write(str(vars(args)) + "\n")
            txt_config_file.write(str(params))

    accelerator.wait_for_everyone()
    train(comm_state)
