#!/usr/bin/env python
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""

import argparse
import logging
import math
import os
import random
import sys
from datetime import datetime, timedelta

import datasets
import torch
import torch.distributed as dist
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargsCustom, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from tqdm.auto import tqdm

from dataset_llm_factories import multiple_choice_dataset
from new_comm_hooks.comm_hooks import CHUNK_SIZE_THRESHOLD, wrapper_hook
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    DataCollatorForTokenClassification,
)
from transformers.utils import send_example_telemetry
import numpy as np
import pickle


class CustomSelectiveDataCollator(DataCollatorForTokenClassification):
    def __init__(self, tokenizer, pad_fields=["input_ids", "attention_mask", "labels"], **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.pad_fields = pad_fields

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
    def __init__(self, rank,  num_epochs, model_name_or_path=None):
        total_rank = dist.get_world_size()

        if "gemma" in model_name_or_path or "GPT2" in model_name_or_path:
            self.indices_list_for_all = pickle.load(open("models/indices_mmlu_new_gemma_{}.pkl".format(total_rank), "rb"))
        else:
            self.indices_list_for_all = pickle.load(open("models/indices_mmlu_new_{}.pkl".format(total_rank), "rb"))
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

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        "--train_file", type=str, default=None, help="A csv, txt or a json file containing the training data."
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
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
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
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=10, help="Number of steps for the warmup in the lr scheduler."
    ) # get a bit of warmup to avoid NAN
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
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
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
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
        "--checkpointing_steps",
        type=str,
        default="100",
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
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
    parser.add_argument('--nclients', default=8, type=int, help='Number of clients')

    parser.add_argument('--ef', default=False, type=bool, help='use error feedback')
    parser.add_argument('--overflow_frequency', default=1024, type=int, help='one over the expected number of overflowed coordinates')
    parser.add_argument('--rotation', default="True", type=str, help='use rotation')
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
    parser.add_argument("--sparsity", type=str, default="None")
    parser.add_argument("--measure_comm_error", action="store_true", help="Enable per-bucket ground-truth all-reduce and L2 communication-error logging.")

    args = parser.parse_args()


    if args.model_name_or_path == "gemma":
        args.model_name_or_path = "google/gemma-3-1b-it"
    
    elif args.model_name_or_path == "DistilGPT2":
        args.model_name_or_path = "distilbert/distilgpt2"

    print(args.model_name_or_path)

    return args


def get_suffix(args):
    return datetime.now().strftime("%m_%d_%Y__%H_%M_%S") + "," + str(hash(str(sys.argv)))



def get_correctness_multiple_choice(logits, inputs, tokenized_choices):
    total_correct = 0
    total_probability_abcd = 0
    for i in range(len(logits)):
        answer = inputs["res_choice"][i]
        location = inputs["location"][i]
        
        logits_prob = [logits[i][location - 1][choice].item() for choice in tokenized_choices]
        
        predicted_index = logits_prob.index(max(logits_prob))

        if predicted_index == answer:
            total_correct += 1

        logits_distribution = logits[i][location - 1].detach()
        logits_distribution = torch.softmax(logits_distribution, dim=-1)
        probability_abcd = logits_distribution[tokenized_choices].sum().item()
        total_probability_abcd += probability_abcd
    
    return total_correct, total_probability_abcd



def evaluate(accelerator, model, eval_dataloader, tokenized_choices, args, completed_steps):
    model.eval()
    losses = []
    correctness = 0
    total_probability_abcd = 0
    nitems = 0

    for step, batch in enumerate(eval_dataloader):
        if step == 3000:
            break
        if step % 100 == 0:
            print(step)
        with torch.no_grad():
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])

        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
        correct, probability_abcd = get_correctness_multiple_choice(outputs.logits, batch, tokenized_choices)
        correctness += correct
        total_probability_abcd += probability_abcd

        nitems += len(outputs.logits)
        if step % 100 == 0:
            print(correctness, nitems)

    losses = torch.cat(losses)
    eval_loss = torch.mean(losses)
    correctness = torch.tensor([correctness], dtype=torch.float32, device="cuda")
    torch.distributed.all_reduce(correctness, op=torch.distributed.ReduceOp.SUM)

    total_probability_abcd = torch.tensor([total_probability_abcd], dtype=torch.float32, device="cuda")
    torch.distributed.all_reduce(total_probability_abcd, op=torch.distributed.ReduceOp.SUM)

    eval_loss = eval_loss.item()
    correctness = correctness[0].item() / nitems / accelerator.num_processes
    total_probability_abcd = total_probability_abcd[0].item() / nitems / accelerator.num_processes

    model.train()
    return eval_loss, correctness, total_probability_abcd

def train(comm_state, sampler, tokenizer):
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    if accelerator.is_main_process:
        txt_log_file = params["txt_result_file"]

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps

    progress_bar.update(completed_steps)

    tokenized_choices = []
    for choice in [" A", " B", " C", " D"]:
        tokenized_choices.append(tokenizer(choice, add_special_tokens=False).input_ids[0])

    if dist.get_world_size() == 2:
        eval_loss, acc, prob_abcd = 1, 1, 1
    else:
        eval_loss, acc, prob_abcd = evaluate(accelerator, model, eval_dataloader, tokenized_choices, args, 0)

    logger.info(f"step {completed_steps}: acc: {acc} prob_abcd: {prob_abcd} eval_loss: {eval_loss} train_loss: NAN")

    if accelerator.is_main_process:
        txt_log_file.write(f"step {completed_steps}: acc: {acc} prob_abcd: {prob_abcd} eval_loss: {eval_loss} train_loss: NAN\n")
        txt_log_file.flush()

    real_num_train_epochs = args.num_train_epochs + 1
    for epoch in range(starting_epoch, real_num_train_epochs):
        if epoch >= 1 and "7bit" in args.aggregation_method:
            break
        
        model.train()
        sampler.set_epoch(epoch)

        total_loss = 0
        total_acc = 0
        total_prob_abcd = 0
        total_steps = 0
        total_losses_per_step = [0]
        total_acc_per_step = [0]
        total_prob_abcd_per_step = [0]

        for step, batch in enumerate(train_dataloader):
            comm_state["batch_idx"] += 1

            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            loss = outputs.loss

            total_loss += loss.detach().float()
            total_losses_per_step[-1] += loss.detach().float()
            acc, prob_abcd = get_correctness_multiple_choice(outputs.logits, batch, tokenized_choices)
            total_acc += acc / len(batch["input_ids"])
            total_acc_per_step[-1] += acc / len(batch["input_ids"])
            total_prob_abcd += prob_abcd / len(batch["input_ids"])
            total_prob_abcd_per_step[-1] += prob_abcd / len(batch["input_ids"])

            accelerator.backward(loss)
            optimizer.step()
            if epoch < args.num_train_epochs:
                lr_scheduler.step()
            optimizer.zero_grad()
            total_steps += 1

            torch.cuda.empty_cache()

            progress_bar.update(1)
            completed_steps += 1

            if completed_steps % 10 == 0:
                logger.info(f"step {completed_steps}: loss: {total_loss.item() / total_steps} acc: {total_acc / total_steps} prob_abcd: {total_prob_abcd / total_steps} train_loss_per_10_steps: {total_losses_per_step[-1] / 10} train_acc_per_10_steps: {total_acc_per_step[-1] / 10} train_prob_abcd_per_10_steps: {total_prob_abcd_per_step[-1] / 10}")
                if accelerator.is_main_process:
                    txt_log_file.write(f"Step updates: step {completed_steps}: acc: / eval_loss: / train_loss: {total_loss.item() / total_steps} train_loss_per_10_steps: {total_losses_per_step[-1] / 10}  acc: {total_acc / total_steps} train_acc_per_10_steps: {total_acc_per_step[-1] / 10} prob_abcd: {total_prob_abcd / total_steps} train_prob_abcd_per_10_steps: {total_prob_abcd_per_step[-1] / 10}\n")
                    txt_log_file.flush()
                total_losses_per_step.append(0)
                total_acc_per_step.append(0)
                total_prob_abcd_per_step.append(0)

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and completed_steps > 0:
                    eval_loss, acc, prob_abcd = evaluate(accelerator, model, eval_dataloader, tokenized_choices, args, 0)
                    model.train()

                    logger.info(f"step {completed_steps}: acc: {acc} prob_abcd: {prob_abcd} eval_loss: {eval_loss} train_loss: {total_loss.item() / total_steps} train_acc: {total_acc / total_steps} train_prob_abcd: {total_prob_abcd / total_steps}")

                    if accelerator.is_main_process:
                        txt_log_file.write(f"Epoch updates: step {completed_steps}: acc: {acc} prob_abcd: {prob_abcd} eval_loss: {eval_loss} train_loss: {total_loss.item() / total_steps} train_acc: {total_acc / total_steps} train_prob_abcd: {total_prob_abcd / total_steps}\n")
                        txt_log_file.flush()
        eval_loss, acc, prob_abcd = evaluate(accelerator, model, eval_dataloader, tokenized_choices, args, 0)
        model.train()

        logger.info(f"step {completed_steps}: acc: {acc} prob_abcd: {prob_abcd} eval_loss: {eval_loss} train_loss: {total_loss.item() / total_steps} train_acc: {total_acc / total_steps} train_prob_abcd: {total_prob_abcd / total_steps}")

        if accelerator.is_main_process:
            txt_log_file.write(f"Epoch updates: step {completed_steps}: acc: {acc} prob_abcd: {prob_abcd} eval_loss: {eval_loss} train_loss: {total_loss.item() / total_steps} train_acc: {total_acc / total_steps} train_prob_abcd: {total_prob_abcd / total_steps}\n")
            txt_log_file.flush()
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and "bf16" in args.aggregation_method:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            tokenizer.save_pretrained(args.output_dir)



if __name__ == "__main__":
    args = parse_args()

    set_seed(args.seed)
    send_example_telemetry("run_clm_no_trainer", args)

    suffix = get_suffix(args)
    if args.output_dir == None:
        args.output_dir = os.path.join(os.environ.get("DYNAMIQ_OUTPUT_DIR", "simulation_outputs"), suffix)

    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["project_dir"] = args.output_dir

    
    params = dict()
    
    params["args"] = args

    params["nclients"] = args.nclients
    params['seed'] = args.seed
    params["table_dir"] = "compression/new_tables"
    params["measure_comm_error"] = args.measure_comm_error

    params['d'] = dict()
    params["size"] = dict()
    params["keys"] = []
    params["measure_points"] = dict()
    params["max_norm"] = dict()
    params["smaller_max_norm"] = dict()
    params["norm"] = dict()
    params["agg_chunk_size"] = args.agg_chunk_size
    params["overflow_prob"] = 8
    params["max_chunk_size"] = args.max_chunk_size
    params["smaller_max_chunk_size"] = args.smaller_max_chunk_size
    aggregation_method_lower = args.aggregation_method.lower()
    params["heuristic"] = "bitrate" if (
        aggregation_method_lower == "dynamiq_mixed"
        or ("dynamiq" in aggregation_method_lower and "bitrate" in aggregation_method_lower)
    ) else "chunk_size"
    params["lr_adjust_param"] = 1

    params["MAX_BUCKET_SIZE"] = CHUNK_SIZE_THRESHOLD

    params['ef'] = args.ef
    params['rotation'] = True if args.rotation == "True" else False
    params["quantization_levels"] = args.quantization_levels
    params['overflow_frequency'] = args.overflow_frequency
    params["normalized"] = args.normalized
    
    params["chunk_size"] = args.normalized_chunk_size
    params["supergroup"] = 16
    params["device"] = "cuda"
    params["is_correlated"] = True if "correlated" in args.aggregation_method else False
    if args.sparsity != "None":
        params["target_topk"] = float(args.sparsity)

    comm_state = {"batch_idx": -1, "params": params, "start_idx": {}, "partition_len": {}, "ret_tensor": {}, "start_interm_idx": {}, "interm_reduce_tensor": {}, "args": args}
    
    ddp_kwargs = DistributedDataParallelKwargsCustom(bucket_cap_mb=500, comm_hook=wrapper_hook, comm_state_option=comm_state)

    init_process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=40000))
    

    proj_config = ProjectConfiguration(total_limit=1)

    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, init_process_group_kwargs], gradient_accumulation_steps=args.gradient_accumulation_steps, project_config=proj_config, **accelerator_log_kwargs)

    accelerator.wait_for_everyone()

    def process_group_warm_up():
        reduce_vec = torch.tensor([1.], device="cuda")
        dist.all_reduce(reduce_vec)

    process_group_warm_up()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
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

    if args.dataset_name is not None:
        raw_datasets = load_dataset(
            args.dataset_name, args.dataset_config_name, trust_remote_code=args.trust_remote_code, cache_dir=args.data_cache_dir, split="auxiliary_train"
        )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": args.model_cache_dir,
        "revision": "main",
        "token": None,
        "trust_remote_code": args.trust_remote_code,
    }

    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            **config_kwargs,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            **config_kwargs,
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, **config_kwargs
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, **config_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            torch_dtype=torch.bfloat16,
            **config_kwargs
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config, **config_kwargs)
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

    

    max_data_length = 0
    for data in lm_datasets["test"]:
        max_data_length = max(max_data_length, len(data["input_ids"]))
    print("max_data_length: ", max_data_length)
    del raw_datasets

    train_dataset = lm_datasets["train"]

    train_dataset.remove_columns(["question", "subject", "choices", "answer"])

    eval_dataset = lm_datasets["test"]
    eval_dataset.remove_columns(["question", "subject", "choices", "answer"])

    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sampler = PrecomputedSampler(rank=dist.get_rank(), num_epochs=args.num_train_epochs, model_name_or_path=args.model_name_or_path)
    train_dataloader = DataLoader(
        train_dataset, sampler=sampler, collate_fn=CustomSelectiveDataCollator(tokenizer), batch_size=args.per_device_train_batch_size
    )

    eval_dataloader = DataLoader(
        eval_dataset, shuffle=True, collate_fn=CustomSelectiveDataCollator(tokenizer), batch_size=args.per_device_eval_batch_size
    )

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

    overrode_max_train_steps = False
    num_update_steps_per_epoch = len(train_dataloader)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=1. / 8., total_iters=max(1, args.max_train_steps))
    
    model, optimizer, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer,  eval_dataloader, lr_scheduler
    )
    

    num_update_steps_per_epoch = len(train_dataloader)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    if accelerator.is_main_process:
        params["txt_log_file"] = open(os.path.join(args.output_dir, "log.txt"), "a")
        params["txt_result_file"] = open(os.path.join(args.output_dir, "results.txt"), "a")

        txt_config_file = open(os.path.join(args.output_dir, 'config.txt'), "a")    
        txt_config_file.write(str(vars(args)) + "\n")
        txt_config_file.write(str(params))
        txt_config_file.close()


    accelerator.wait_for_everyone()
    train(comm_state, sampler, tokenizer)
