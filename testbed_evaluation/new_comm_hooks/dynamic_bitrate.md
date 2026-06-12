# dynamiQ Dynamic Bitrate Hook

This note documents `P2P_dynamiQ_dynamic_bitrate_hook`, the dynamic bitrate variant
of the dynamic range AEE/MEE dynamiQ hook.

## High-Level Algorithm

The dynamic range hook assigns fixed proportions of superchunks to 8-bit,
4-bit, and 2-bit compression.  The dynamic bitrate hook instead assigns each
superchunk a bitrate from the current round's reduced per-superchunk norm
statistics, while adapting a scalar offset `u` so the average assigned bitrate
tracks a target.

For each 256-element superchunk `i`, let `F_i` be the aggregated squared L2 norm
of that superchunk.  The hook computes:

```text
z_i = floor(log2((4 / (512 / 17)) * log2(F_i) + u))
z_i = clamp(z_i, 1, 3)
q_i = 2 ^ z_i
```

After clamping, `q_i` is one of:

```text
z_i = 1 -> q_i = 2 bits
z_i = 2 -> q_i = 4 bits
z_i = 3 -> q_i = 8 bits
```

The target is not the full aggregation-method bitrate `b`; it reserves a fixed
headroom:

```text
target_avg_bits = b - b_headroom
b_headroom = 0.625
```

The adaptation objective is:

```text
sum_i(q_i) / m ~= target_avg_bits
```

where `m` is the total number of superchunks accumulated across all buckets in
the current round.

### Initial Update

The first compressed round uses:

```text
u = 60
```

At `batch_idx == 2`, after all buckets have appended their reduced norm
statistics, the hook chooses the next round's `u` by binary search over:

```text
[-10, 60]
```

For any candidate `u`, increasing `u` can only increase or preserve the assigned
`q_i` values, so the average bitrate is monotone.  The binary search stops when:

```text
abs(delta) <= 0.1
delta = target_avg_bits - average(q_i)
```

or after the fixed maximum iteration budget.

### Later Updates

After the initial binary search, later rounds use an incremental dynamic
adjustment:

```text
old_delta = target_avg_bits - average_qbits(u_prev)
candidate_u = u_prev + old_delta
new_delta = target_avg_bits - average_qbits(candidate_u)
```

If the candidate remains on the same side of the target, it is accepted:

```text
old_delta * new_delta >= 0
```

It is also accepted if it is already close enough:

```text
abs(new_delta) <= 0.1
```

Otherwise, the step is halved and retried:

```text
step = step / 2
candidate_u = u_prev + step
```

The accepted candidate becomes the `u` used by the next round.

## Implementation Details

The implementation lives in:

```text
testbed_evaluation/new_comm_hooks/comm_hooks.py
```

The main hook is:

```text
P2P_dynamiQ_dynamic_bitrate_hook
```

It reuses the same communication and compression structure as
`P2P_dynamiQ_hook`:

1. Mean-center each 256-element superchunk with `superchunk_mean_center`.
2. All-reduce the per-superchunk stats tensor with `ReduceOp.SUM`.
3. Use `stats[:, 1]` as the reduced `F_i` values.
4. Select 2-bit, 4-bit, and 8-bit superchunk groups.
5. Communicate each non-empty group with the selected topology callback.
6. Restore the reduced mean with `superchunk_add_mean_copy`.

The important difference is step 4.  Dynamic bitrate selection computes `q_i`
directly:

```text
qbits = _dynamic_bitrate_qbits(norms, current_u)
indice_8 = nonzero(qbits == 8)
indice_4 = nonzero(qbits == 4)
indice_2 = nonzero(qbits == 2)
```

No sort, top-k proportion, bottom-k proportion, or `torch.cat` is used for the
bitrate decision.

### Accumulating Norms

Each partition's reduced norm view is appended to a Python list:

```text
state["dynamic_bitrate_norms"].append(stats[:, 1])
```

This keeps per-bucket accumulation cheap.  The hook only walks the list when
`bucket.is_last()` is true and it needs to compute the next round's `u`.

The list is reset when a new compressed round starts:

```text
state["dynamic_bitrate_stats_batch_idx"]
state["dynamic_bitrate_norms"]
```

### State Keys

The dynamic bitrate hook stores the following values in `state["params"]`:

```text
dynamic_bitrate_target_bitrate
dynamic_bitrate_target_avg_bits
dynamic_bitrate_u
dynamic_bitrate_last_avg_bits
dynamic_bitrate_last_delta
```

`dynamic_bitrate_u` always means the `u` to use for the current round.  At the
last bucket of that round, it is overwritten with the `u` chosen for the next
round.

### Constants

The constants are defined near the top of `comm_hooks.py`:

```text
DYNAMIC_BITRATE_HEADROOM = 0.625
DYNAMIC_BITRATE_U_MIN = -10.0
DYNAMIC_BITRATE_U_MAX = 60.0
DYNAMIC_BITRATE_TOL = 0.1
DYNAMIC_BITRATE_MAX_ITERS = 32
DYNAMIC_BITRATE_NORM_SCALE = 4.0 / (512.0 / 17.0)
```

If the aggregation method contains an explicit bitrate token such as `3bit`,
`4bit`, or `5bit`, that value is parsed as `b`.  If no such token is present,
the implementation defaults to `b = 5`.

### Hook Selection

The training and smoke entrypoints route aggregation methods containing both
`dynamiQ` and `bitrate` to the dynamic bitrate hook:

```text
dynamiQ_aee_5bit_bitrate
dynamiQ_mee_5bit_bitrate
```

Methods that contain `dynamiQ` but not `bitrate` continue to use
`P2P_dynamiQ_hook`.

The selector updates are in:

```text
testbed_evaluation/train_llm_causal.py
testbed_evaluation/train_llm_mmlu.py
testbed_evaluation/smoke_test_new_comm_hooks.py
```

### Communication Path

The hook preserves the dynamic range hook's existing topology choices:

```text
butterfly method -> butterfly callback
DYNAMIC_AEE_PIPELINE_RDMA=1 -> pipelined ring RDMA dynamiQ callback
otherwise -> regular composable ring RDMA callback
```

For each non-empty bitrate group, the selected rows are copied into a reusable
selection buffer, compressed with the matching 2-bit, 4-bit, or 8-bit compressor
object, all-reduced through the topology callback, and copied back into their
original superchunk positions.
