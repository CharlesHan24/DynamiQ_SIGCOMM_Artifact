#!/usr/bin/env bash
set -euo pipefail

NODES="${NODES:-chip-207-3,chip-207-4,chip-207-5,chip-207-6}"
METHODS="${METHODS:-dynamiQ_mee_5bit_dynamic_bitrate,fp4}"

zsh submit_qsub_llm_hook_matrix.zsh \
  --nodes "$NODES" \
  --aggregation_methods "$METHODS" \
  --combos "causal:llama:meta-llama/Llama-3.2-1B,causal:gemma:gemma,mmlu:llama:meta-llama/Llama-3.2-1B,wikitext:bert-large:bert-large-cased" \
  --dynamic-pipeline-rdma 1 \
  --pipeline-chunk-mb 8 \
  --pipeline-inflight 2 \
  --per_device_train_batch_size 1 \
  --rails 1
