# LLM Simulation Jobs

This directory contains the simulation training scripts and Grid Engine launch scripts used for the dynamiQ LLM experiments.

Most users should launch jobs with:

```zsh
simulations_llm/submit_qsub_simulation_matrix.zsh
```

The submitter creates one qsub job per node and uses `simulations_llm/qsub_simulation_matrix_node.zsh` internally to run the right distributed training script with `accelerate launch`.

The same submitter also supports:

- `--launch-mode qsub` (default): submit one SGE job per node.
- `--launch-mode direct`: `ssh` into each selected host and invoke `simulations_llm/qsub_simulation_matrix_node.zsh` directly.
- `--topology butterfly`: automatically run butterfly variants of the selected aggregation methods.

For `--launch-mode direct`, ensure that:

- the repository is visible at the same path on every selected node,
- the launching machine can `ssh` to each selected node without interactive prompts, and
- the requested conda environment exists on every selected node.

## Required Setup

Before launching simulation jobs, generate the fixed sampling orders and correlated random data:

```zsh
python simulations_llm/gen_correlated_rand.py
```

This writes deterministic data under:

```text
simulations_llm/models/
simulations_llm/models/correlated_rand/
```

The `indices_*.pkl`, `indices_gemma_*.pkl`, `indices_mmlu_new_*.pkl`, and `indices_mmlu_new_gemma_*.pkl` files fix the data order so training behavior is reproducible across runs. The `models/correlated_rand/obj_*.pt` files are used by correlated-random communication methods and THC.

The correlated-random tensors are large. If you only need to refresh the fixed data-order files, run:

```zsh
python simulations_llm/gen_correlated_rand.py --skip-correlated-rand --skip-hadamard
```

The LLM datasets and pretrained models are loaded with Hugging Face APIs. If
they are not already cached, the scripts download them automatically. By default
they use Hugging Face's normal cache locations, usually under
`~/.cache/huggingface`.

To use explicit cache directories, either export:

```zsh
export DYNAMIQ_DATA_CACHE="$PWD/hf_data"
export DYNAMIQ_MODEL_CACHE="$PWD/hf_models"
```

or pass cache paths to the simulation launcher:

```zsh
simulations_llm/submit_qsub_simulation_matrix.zsh \
  --data-cache-dir "$PWD/hf_data" \
  --model-cache-dir "$PWD/hf_models"
```

The Llama and Gemma runs use gated Hugging Face model repositories. Before
running those jobs on a new machine, accept the model licenses on Hugging Face
and authenticate once:

```zsh
huggingface-cli login
```

## Quick Start

From the repository root, first inspect the generated qsub commands:

```zsh
simulations_llm/submit_qsub_simulation_matrix.zsh --dry-run
```

Then submit the default simulation matrix:

```zsh
simulations_llm/submit_qsub_simulation_matrix.zsh
```

To inspect the corresponding direct-launch commands instead, run:

```zsh
simulations_llm/submit_qsub_simulation_matrix.zsh \
  --launch-mode direct \
  --dry-run
```

To launch the default simulation matrix directly over `ssh`, run:

```zsh
simulations_llm/submit_qsub_simulation_matrix.zsh \
  --launch-mode direct
```

By default this runs:

- worker counts: `2,4,8`
- one rank per node
- tasks: causal LM with Llama, causal LM with Gemma, MMLU with Llama, and Wikitext-103 masked LM with BERT-large-cased
- aggregation method: `dynamiQ_mee_5bit_dynamic_bitrate`
- output root: `simulations_llm/simulation_matrix_results`

## Common Launches

Run only the 2-worker causal Llama experiment:

```zsh
simulations_llm/submit_qsub_simulation_matrix.zsh \
  --workers 2 \
  --combos causal:llama:meta-llama/Llama-3.2-1B
```

The same experiment in direct mode:

```zsh
simulations_llm/submit_qsub_simulation_matrix.zsh \
  --launch-mode direct \
  --workers 2 \
  --combos causal:llama:meta-llama/Llama-3.2-1B
```

Run all default tasks on a custom host list:

```zsh
simulations_llm/submit_qsub_simulation_matrix.zsh \
  --nodes chip-207-3,chip-207-4,chip-207-5,chip-207-6,chip-207-7,chip-207-8,chip-207-9,chip-207-10
```

The same launch over direct `ssh`:

```zsh
simulations_llm/submit_qsub_simulation_matrix.zsh \
  --launch-mode direct \
  --nodes chip-207-3,chip-207-4,chip-207-5,chip-207-6,chip-207-7,chip-207-8,chip-207-9,chip-207-10
```

Run only BERT masked LM:

```zsh
simulations_llm/submit_qsub_simulation_matrix.zsh \
  --workers 2,4,8 \
  --combos wikitext:bert-large:bert-large-cased
```

Override the aggregation methods:

```zsh
simulations_llm/submit_qsub_simulation_matrix.zsh \
  --aggregation-methods dynamiQ_mee_5bit_dynamic_bitrate,bf16
```

Run an 8-worker direct launch with two local ranks per node:

```zsh
simulations_llm/submit_qsub_simulation_matrix.zsh \
  --launch-mode direct \
  --workers 8 \
  --procs-per-node 2 \
  --nodes chip-207-3,chip-207-4,chip-207-5,chip-207-6
```

Run the same experiment on the butterfly all-reduce topology:

```zsh
simulations_llm/submit_qsub_simulation_matrix.zsh \
  --topology butterfly \
  --workers 8 \
  --procs-per-node 2 \
  --nodes chip-207-3,chip-207-4,chip-207-5,chip-207-6
```

Pass extra arguments directly to the training script:

```zsh
simulations_llm/submit_qsub_simulation_matrix.zsh \
  --extra-train-args "--measure_comm_error"
```

Enable communication-error logging, including vNMSE, for a simulation run:

```zsh
simulations_llm/submit_qsub_simulation_matrix.zsh \
  --workers 2 \
  --combos causal:llama:meta-llama/Llama-3.2-1B \
  --aggregation-methods dynamiQ_mee_5bit_dynamic_bitrate \
  --extra-train-args "--measure_comm_error"
```

With `--measure_comm_error`, rank 0 computes the exact all-reduce result for
each gradient bucket and writes cumulative per-step error data under each
method's `output/` directory. The legacy raw L2 sums are appended to `log.txt`.
The easier-to-parse file is `comm_error_vnmse.tsv`, whose `vnmse` column is
`l2_error / l2_norm`.

## Task Format

Use `--combos` to choose experiments. It is a comma-separated list of triples:

```text
task:label:model
```

Supported tasks are:

- `causal`: runs `distributed_llm_clm.py`
- `mmlu`: runs `distributed_llm_mmlu.py`
- `wikitext`, `maskedlm`, or `mlm`: runs `distributed_maskedlm_bert.py`

The `label` is only used in job names and result paths. The `model` is passed to `--model_name_or_path`. The causal LM script maps `gemma` to `google/gemma-3-1b-it`.

Default combos:

```text
causal:llama:meta-llama/Llama-3.2-1B,
causal:gemma:gemma,
mmlu:llama:meta-llama/Llama-3.2-1B,
wikitext:bert-large:bert-large-cased
```

## Worker And Node Counts

`--workers` accepts `2`, `4`, and `8`, either singly or as a comma-separated list.

The number of qsub node jobs is:

```text
workers / procs-per-node
```

The default is `--procs-per-node 1`, so an 8-worker job uses 8 nodes. If you set `--procs-per-node 2`, the launcher requests two GPUs per qsub job and runs two local ranks per node.

The submitter checks that:

- `workers` is divisible by `procs-per-node`
- enough hosts were provided through `--nodes`

## Learning Rate And Epoch Defaults

The per-node runner computes learning rate and decay epochs automatically unless `--learning-rate` is supplied.

For causal LM:

```text
lr = 1e-5 * sqrt(workers / 2) * (8 / workers)^0.4
```

Gemma causal LM multiplies that value by `0.7`.

For MMLU:

```text
lr = 3e-6 * sqrt(workers / 2) * (8 / workers)^0.4
```

For BERT masked LM:

```text
lr = 5e-5
```

For causal LM and MMLU, the launch script passes `--num_train_epochs 1` when `workers <= 4`, and `--num_train_epochs 2` when `workers == 8`. The training scripts then run one additional epoch at the fixed final learning rate.

For BERT masked LM, the launch script passes `--num_train_epochs 8` when `workers <= 4`, and `--num_train_epochs 15` when `workers == 8`, plus `--fixed_lr_epochs 6`.

## Butterfly All-Reduce

Set:

```text
--topology butterfly
```

to map the selected methods onto their butterfly variants automatically. For example:

- `bf16` becomes `bf16_butterfly`
- `omnireduce` becomes `omnireduce_butterfly`
- `dynamiQ_mee_5bit_dynamic_bitrate` becomes `dynamiQ_mee_5bit_dynamic_bitrate_butterfly`

To reproduce the butterfly-allreduce simulation runs corresponding to Figure 9, run:

```zsh
simulations_llm/submit_qsub_simulation_matrix.zsh \
  --topology butterfly \
  --workers 2,4,8 \
  --combos causal:llama:meta-llama/Llama-3.2-1B,causal:gemma:gemma,mmlu:llama:meta-llama/Llama-3.2-1B,wikitext:bert-large:bert-large-cased \
  --aggregation-methods bf16,omnireduce,thc,dynamiQ_mee_5bit_dynamic_bitrate
```

If you are not using SGE, add `--launch-mode direct` to the same command.

## Useful Options

```text
--nodes <csv>                 Hostnames available to qsub.
--workers <csv>               Worker counts to run. Default: 2,4,8.
--combos <csv>                task:label:model triples.
--aggregation-methods <csv>   Aggregation methods. Default: dynamiQ_mee_5bit_dynamic_bitrate.
--topology <ring|butterfly>   Select the communication topology.
--result-root <path>          Where run directories are created.
--launch-mode <qsub|direct>   Launch through SGE or directly over ssh.
--procs-per-node <n>          Local ranks per qsub job. Default: 1.
--learning-rate <x>           Override the automatic learning rate.
--per-device-train-batch-size <n>
                              Override task defaults.
--data-cache-dir <path>       Hugging Face dataset cache directory.
--model-cache-dir <path>      Hugging Face model cache directory.
--extra-train-args <string>   Extra arguments appended to each training command.
--dry-run                     Print launcher commands without submitting or running.
```

Resource and environment options:

```text
--project <name>              qsub project. Default: cmic_hpc.
--h-rt <seconds>              qsub runtime limit. Default: 60000.
--tmem <value>                qsub memory resource. Default: 1K.
--conda-env <name>            Conda environment activated on each node. Default: llm.
--gpu-max-used-mb <mb>        Maximum used GPU memory for automatic GPU selection. Default: 512.
--numa-node <id|none>         Optionally wrap training with numactl.
```

## Output Layout

Each submitter invocation creates a run directory:

```text
<result-root>/<timestamp>_<job-id-or-manual>_<pid>/
```

Important files:

```text
submit_metadata.log
qsub_commands.log
direct_commands.log           Present for --launch-mode direct.
experiment_<index>_<task>_<label>_<workers>w/
  experiment_metadata.log
  node_<rank>_<host>.qsub.out
  node_<rank>_<host>.qsub.err
  node_<rank>_<host>.direct.out
  node_<rank>_<host>.direct.err
  node_<rank>_metadata.log
  method_<index>_<aggregation_method>/
    command.log
    node_<rank>.out
    node_<rank>.err
    node_<rank>.status
    output/
      results.txt
      log.txt
      comm_error_vnmse.tsv     Present when --measure_comm_error is enabled.
```

`command.log` records the exact `accelerate launch` command used for each aggregation method.

`results.txt` contains the task metrics such as loss, perplexity, or accuracy.
When communication-error logging is enabled, `log.txt` keeps the legacy raw
`l2_error l2_norm pred_l2_norm` rows, and `comm_error_vnmse.tsv` records the same
values with headers plus `vnmse`. These vNMSE results can be used to reproduce Table 3.

## Troubleshooting

Use `--dry-run` first whenever you change nodes, workers, models, or extra arguments.

With `--launch-mode direct`, `--dry-run` prints the exact `ssh` commands that would be executed.

If a node cannot find enough idle GPUs, check:

```text
node_<rank>_gpu_selection_failed.log
```

If a training command fails, check:

```text
method_<index>_<aggregation_method>/node_<rank>.err
method_<index>_<aggregation_method>/command.log
```

To continue running later methods even when one method fails, pass:

```zsh
--continue-on-failure
```

To submit jobs without waiting for the last node of each experiment to finish, pass:

```zsh
--no-sync
```
