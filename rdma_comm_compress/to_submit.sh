./submit_qsub_ring_allreduce.zsh \
  --nodes chip-207-3,chip-207-6 \
  --iters 20 \
  --warmup-iters 3 \
  --numel 536870912 \
  --mode bf16


# ./submit_qsub_single_node_allreduce.zsh \
#   --node chip-207-6 \
#   --iters 20 \
#   --warmup-iters 3 \
#   --numel 268435456