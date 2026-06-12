#!/usr/bin/env python3
"""DDP smoke test for testbed_evaluation/new_comm_hooks.

Example single-node launch:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    testbed_evaluation/smoke_test_new_comm_hooks.py --aggregation_method bf16

Example two-node launch:
  CUDA_VISIBLE_DEVICES=0,1 RING_RDMA_RAILS=1 RING_RDMA_IFACE0=eth0 \
    torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 \
    --master_addr=<node0> --master_port=29500 \
    testbed_evaluation/smoke_test_new_comm_hooks.py \
    --aggregation_method dynamiQ_aee_4bit

Example two-node butterfly RDMA launch:
  CUDA_VISIBLE_DEVICES=0,1 RING_RDMA_RAILS=1 RING_RDMA_IFACE0=eth0 \
    torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 \
    --master_addr=<node0> --master_port=29500 \
    testbed_evaluation/smoke_test_new_comm_hooks.py \
    --aggregation_method dynamiQ_mee_5bit_butterfly \
    --expect_butterfly_rdma

The script intentionally performs no optimizer step. It only runs synthetic
forward/backward passes so PyTorch DDP calls the selected communication hook.
"""

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import time


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from new_comm_hooks.comm_hooks import (  # noqa: E402
    P2P_MXfp8_compress_hook,
    P2P_dynamiQ_hook,
    P2P_dynamiQ_dynamic_bitrate_hook,
    P2P_bf16_compress_hook,
    P2P_omnireduce_topk_hook,
    P2P_slicing_compress_hook,
    P2P_THC_compress_hook,
)


class SingleBucketModel(nn.Module):
    def __init__(self, numel: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        weight = torch.empty(numel, dtype=dtype, device=device)
        weight.normal_(mean=0.0, std=0.02)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.weight * x).sum()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic torchrun smoke test for the refactored DDP communication hooks."
    )
    parser.add_argument("--aggregation_method", "--aggregation-method", default="bf16")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--numel", type=int, default=2 ** 30)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--bucket_cap_mb", "--bucket-cap-mb", type=float, default=512.0)
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--timeout_seconds", "--timeout-seconds", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checksum_atol", "--checksum-atol", type=float, default=1e-2)
    parser.add_argument("--skip_grad_check", "--skip-grad-check", action="store_true")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None)

    parser.add_argument("--normalized", default=1, type=int)
    parser.add_argument("--normalized_chunk_size", "--normalized-chunk-size", default=1024, type=int)
    parser.add_argument("--nclients", default=0, type=int)
    parser.add_argument("--ef", action="store_true")
    parser.add_argument("--overflow_frequency", "--overflow-frequency", default=1024, type=int)
    parser.add_argument("--rotation", default="True")
    parser.add_argument("--quantization_levels", "--quantization-levels", default=64, type=int)
    parser.add_argument("--max_chunk_size", "--max-chunk-size", type=int, default=32)
    parser.add_argument("--smaller_max_chunk_size", "--smaller-max-chunk-size", type=int, default=32)
    parser.add_argument("--agg_chunk_size", "--agg-chunk-size", type=int, default=16)
    parser.add_argument("--compression", default="None")
    parser.add_argument("--sparsity", default="None")
    parser.add_argument("--to_shrimp", "--to-shrimp", default="False")
    parser.add_argument("--rdma_max_bucket_elems", "--rdma-max-bucket-elems", type=int, default=0)
    parser.add_argument(
        "--nsys_profile_step",
        "--nsys-profile-step",
        type=int,
        default=-1,
        help="Call cudaProfilerStart/Stop around this training step. Use with nsys --capture-range=cudaProfilerApi.",
    )
    parser.add_argument(
        "--expect_butterfly_rdma",
        "--expect-butterfly-rdma",
        action="store_true",
        help="Fail unless the butterfly RDMA manager is initialized by the custom callback.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def init_distributed(args: argparse.Namespace) -> Tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("new_comm_hooks smoke test requires CUDA")

    local_rank = args.local_rank
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank is None:
        local_rank = 0

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2 or world_size % 2 != 0:
        raise RuntimeError("launch with an even WORLD_SIZE >= 2 and two ranks per node")
    if local_rank not in (0, 1):
        raise RuntimeError(f"LOCAL_RANK must be 0 or 1 for the ring layout, got {local_rank}")
    if rank % 2 != local_rank:
        raise RuntimeError(
            "expected two-ranks-per-node ordering: rank0/rank1 on node0, rank2/rank3 on node1, ..."
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=args.backend,
        timeout=timedelta(seconds=args.timeout_seconds),
    )
    device = torch.device("cuda", local_rank)
    return rank, world_size, local_rank, device


def build_comm_state(args: argparse.Namespace, world_size: int) -> Dict:
    params = {
        "args": args,
        "nclients": args.nclients or world_size,
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
        "heuristic": "chunk_size"
        if args.aggregation_method.lower() != "dynamiq_mixed"
        else "bitrate",
        "lr_adjust_param": 1,
        "MAX_BUCKET_SIZE": args.rdma_max_bucket_elems or args.numel,
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


def select_hook(aggregation_method: str) -> Callable:
    method = aggregation_method.lower()
    if "thc" in method:
        return P2P_THC_compress_hook
    if "bf16" in method:
        return P2P_bf16_compress_hook
    if "dynamiq" in method and "bitrate" in method:
        return P2P_dynamiQ_dynamic_bitrate_hook
    if "dynamiq" in method:
        return P2P_dynamiQ_hook
    if "omnireduce" in method or "omni" in method:
        return P2P_omnireduce_topk_hook
    if "fp8" in method:
        return P2P_MXfp8_compress_hook
    if "fp4" in method or "fp6" in method or "zero" in method:
        return P2P_slicing_compress_hook
    raise ValueError(
        "unsupported aggregation_method for this smoke test; use bf16, thc, fp8, fp4/fp6/zero, "
        "omnireduce, or dynamiQ_aee_*/dynamiQ_mee_*"
    )


def grad_checksum(model: nn.Module, device: torch.device) -> torch.Tensor:
    checksum = torch.zeros(2, device=device, dtype=torch.float64)
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().to(torch.float64)
        checksum[0].add_(grad.sum())
        checksum[1].add_(grad.norm())
    return checksum


def check_grad_sync(
    model: nn.Module,
    device: torch.device,
    world_size: int,
    atol: float,
) -> float:
    local = grad_checksum(model, device)
    gathered: List[torch.Tensor] = [torch.zeros_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    stacked = torch.stack(gathered)
    max_delta = (stacked - stacked[:1]).abs().max().item()
    if max_delta > atol:
        checksums = stacked.detach().cpu().tolist()
        raise RuntimeError(
            f"gradient checksum mismatch across ranks: max_delta={max_delta:.6g} "
            f"checksums={checksums}"
        )
    return max_delta


def _cuda_profiler_call(name: str, rank: int) -> None:
    torch.cuda.synchronize()
    try:
        ret = getattr(torch.cuda.cudart(), name)()
    except Exception as exc:
        raise RuntimeError(f"{name} failed on rank {rank}: {exc}") from exc
    if isinstance(ret, tuple):
        code = int(ret[0])
    elif ret is None:
        code = 0
    else:
        code = int(ret)
    if code != 0:
        raise RuntimeError(f"{name} failed on rank {rank} with CUDA error code {code}")


def main() -> None:
    args = parse_args()
    if args.steps < 3:
        raise ValueError("--steps must be at least 3 to exercise the custom ring callback")
    if args.numel <= 0:
        raise ValueError("--numel must be positive")
    if args.nsys_profile_step >= args.steps:
        raise ValueError("--nsys_profile_step must be smaller than --steps")
    if args.rdma_max_bucket_elems and args.rdma_max_bucket_elems < args.numel:
        raise ValueError("--rdma_max_bucket_elems must be at least --numel for this smoke test")

    rank, world_size, local_rank, device = init_distributed(args)
    print(rank, world_size, local_rank, device)
    if args.nclients and args.nclients != world_size:
        raise ValueError("--nclients must match WORLD_SIZE for this smoke test")

    dtype = dtype_from_name(args.dtype)
    if "dynamiq" in args.aggregation_method.lower() and dtype == torch.float32:
        raise ValueError("dynamiQ dynamic-range hooks should be tested with --dtype bf16 or --dtype fp16")
    if args.expect_butterfly_rdma:
        if "butterfly" not in args.aggregation_method:
            raise ValueError("--expect_butterfly_rdma requires an aggregation_method containing 'butterfly'")
        if world_size not in (4, 8):
            raise ValueError("butterfly RDMA smoke currently supports WORLD_SIZE 4 or 8")
        if os.environ.get("BUTTERFLY_RDMA_DISABLE", "").lower() in ("1", "true", "yes", "on"):
            raise ValueError("--expect_butterfly_rdma is incompatible with BUTTERFLY_RDMA_DISABLE=1")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    comm_state = build_comm_state(args, world_size)
    hook = select_hook(args.aggregation_method)

    model = SingleBucketModel(args.numel, dtype, device)
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        bucket_cap_mb=args.bucket_cap_mb,
    )
    ddp_model.register_comm_hook([comm_state], hook)

    input_gen = torch.Generator(device=device)
    input_gen.manual_seed(args.seed + 1009 * rank)

    if rank == 0:
        print(
            "new_comm_hooks_smoke "
            f"hook={hook.__name__} aggregation_method={args.aggregation_method} "
            f"world={world_size} dtype={args.dtype} numel={args.numel} steps={args.steps} "
            f"nsys_profile_step={args.nsys_profile_step}",
            flush=True,
        )

    start_ts = None

    for step in range(args.steps):
        comm_state["batch_idx"] = step
        # ddp_model.zero_grad(set_to_none=True)

        if step == 3:
            start_ts = time.perf_counter()
        x = torch.randn(1, device=device, dtype=dtype, generator=input_gen).to(dtype)
        loss = None
        profile_this_step = step == args.nsys_profile_step
        if profile_this_step:
            dist.barrier()
            if rank == 0:
                print(f"starting nsys cudaProfilerApi capture for step {step}", flush=True)
            _cuda_profiler_call("cudaProfilerStart", rank)
            torch.cuda.nvtx.range_push(f"smoke_step_{step}")
        try:
            loss = ddp_model(x).float()
            loss.backward()
            torch.cuda.synchronize(device)
        finally:
            if profile_this_step:
                torch.cuda.nvtx.range_pop()
                dist.barrier()
                _cuda_profiler_call("cudaProfilerStop", rank)
                if rank == 0:
                    print(f"stopped nsys cudaProfilerApi capture for step {step}", flush=True)

        step_elapsed = 0.0 if start_ts is None else time.perf_counter() - start_ts
        print("step {} elapsed {}".format(step, step_elapsed), flush=True)

        max_delta = 0.0
        if not args.skip_grad_check:
            max_delta = check_grad_sync(ddp_model, device, world_size, args.checksum_atol)
        dist.barrier()

        if rank == 0:
            phase = "custom" if step >= 2 else "warmup"
            print(
                f"step={step} phase={phase} loss={loss.item():.6g} "
                f"grad_checksum_max_delta={max_delta:.6g}",
                flush=True,
            )
    measured_steps = max(args.steps - 3, 0)
    if start_ts is not None and measured_steps > 0:
        end_ts = time.perf_counter()
        elapsed = end_ts - start_ts
        print(f"elapsed={elapsed:.2f} seconds for {measured_steps} steps with the custom ring callback. Average time per step = {elapsed / measured_steps}", flush=True)

    manager_ready = any(
        key.startswith("ring_rdma_manager")
        or key.startswith("ring_rdma_pipeline_manager")
        or key.startswith("butterfly_rdma_manager")
        for key in comm_state["params"]
    )
    butterfly_ready = any(
        key.startswith("butterfly_rdma_manager")
        for key in comm_state["params"]
    )
    if args.expect_butterfly_rdma and not butterfly_ready:
        raise RuntimeError("expected butterfly RDMA manager to be initialized, but it was not")

    if rank == 0:
        print(
            "smoke test passed "
            f"rdma_manager_initialized={manager_ready} "
            f"butterfly_rdma_manager_initialized={butterfly_ready}",
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
