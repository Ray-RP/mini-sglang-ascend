# Gate 4.12 Verdict — Qwen3-1.7B TP=2 dynamic admission B: 1 → 2 → 1

**Gate ID:** 4.12 (Qwen3-1.7B TP=2 dynamic admission B: 1 → 2 → 1 on Ascend 910B1)
**Verdict:** PASS
**Branch:** `gate4.12-qwen1.7b-tp2-dynamic-admission-b1-b2-b1`
**Base commit:** `643f50c` (tip of `ascend-port`, Gate 4.11 merge)
**Freeze commit:** `149f9c6`
**Date:** 2026-07-11
**Kind:** Real-hardware Ascend 910B1 TP=2 dynamic-admission proof
— two ranks × Qwen3-1.7B × two prompts of unequal tokenized length
(2 vs 12) × greedy × `max_new_tokens=8` per request × staggered
arrival. Request A starts alone, decodes at least one step, then
request B is admitted while A is still active. Both jointly decode
several steps; A completes first; B continues alone until it too
completes. The observed per-step `batch_size` timeline is
`[1, 1, 1, 2, 2, 2, 2, 2, 2, 1]`, containing the required ordered
subsequence `1 → 2 → 1`, on both ranks byte-identically. The
allocator returns to baseline on both ranks with no deferred
aborts and no radix-cache corruption.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5 /
3.1 / 3.2 / 3.3 / 3.4 / 4.1 / 4.2 / 4.3 / 4.4 / 4.5 / 4.6 / 4.7 /
4.8 / 4.9 / 4.10 / 4.11, does not mutate release tag `v0.1.0a1`,
does not touch the GitHub Release, CHANGELOG, or release notes,
and does not extend the Ascend port to TP > 2, B > 2, dynamic
grow-shrink (`B: 1→2→3→2→1`), TP=2 timing, non-Qwen3 architectures,
or Qwen3-4B / 14B / 32B / quantized / MoE variants. The only new
artefacts at this gate are one bring-up script and this verdict;
no runtime source under `python/minisgl/` is modified; no test
file is modified.

---

## 1. Verdict summary

**PASS on all three cases across both ranks.**

| Case | Description | rank 0 | rank 1 |
|---|---|---|---|
| A | TP=2 init — `_init_communication` returns, both ranks report `init_status=PASS` | **PASS** | **PASS** |
| B | Qwen3-1.7B TP=2 model-load — per-rank weight sharding + post-load `check_integrity()` | **PASS** | **PASS** |
| C | Dynamic admission — batch timeline contains `1 → 2 → 1` ordered subsequence; B admitted strictly after A; ≥1 joint decode step; sibling continues after peer leaves | **PASS** | **PASS** |

Cases A / B collapse into the same `LLM` boot (`set_tp_info` is
one-shot); reaching a successful `_snapshot(cache_manager)` after
`StaggeredLLM.__init__` returns proves init + load simultaneously.

Case C proven invariants on both ranks:

* `prompt_token_lengths == [2, 12]` (`Paris is` = 2 tokens on
  Qwen3-1.7B tokenizer; `The largest planet in our solar system by
  mass and volume is` = 12 tokens)
* `max_new_tokens_per_request == [8, 8]`
* `actual_output_tokens_per_request == [8, 8]`
* `request_uids == [0, 1]` — the staggered subclass assigns uid 0
  to request A (revealed first) and uid 1 to request B (revealed
  after the hook signals A alone has decoded once).
* `batch_timeline == [1, 1, 1, 2, 2, 2, 2, 2, 2, 1]`
* Ordered subsequence check `1 → 2 → 1` passes on both ranks
  (`saw_one → saw_two_after_one → saw_one_after_two`).
* `admission_events == {"0": 0, "1": 2}` — A first appears in a
  batch at step 0; B first appears at step 2. `a_first < b_first`
  holds → `admission_status == "PASS"`.
* `completion_events == {"0": 8, "1": 9}` — A last appears in a
  batch at step 8 (its 8th decode token); B last appears at step 9,
  one step after A. B continues alone at step 9.
* `joint_decode_step_count == 6` — six pure-decode steps
  (`query_lengths == [1, 1]`) with both uids active
  (`batch_size == 2`).
* rank 0 `output_texts[i]` byte-identical to rank 1 `output_texts[i]`
  for `i ∈ {0, 1}`
* rank 0 `output_token_ids[i]` byte-identical to rank 1 `output_token_ids[i]`
  for `i ∈ {0, 1}`
* `output_texts[0] == " the capital of France. Paris is the"`
  (matches Gate 4.10 / 4.11 uid-0 output on this prompt)
* `output_texts[1] == " the planet Jupiter. It is the second"`
  (matches Gate 4.10 / 4.11 uid-1 output on this prompt)
* `output_token_ids[0] == [279, 6722, 315, 9625, 13, 12095, 374, 279]`
* `output_token_ids[1] == [279, 11580, 49689, 13, 1084, 374, 279, 2086]`

Post-case allocator invariants held on both ranks:

* `available_tokens_after_case == baseline_available_tokens` (`930656` on both ranks)
* `deferred_abort_uids == 0`
* `cache_integrity_ok == true`
* `free_pages_after_case == 58165` vs `baseline_free_pages == 58166`
  — a **1-page radix-cache retention** absorbed by the
  `available_size` invariant. Same benign pattern documented at
  Gate 4.5 / 4.10 / 4.11: `available_size = free_slots +
  evictable_prefix_pages`, so a 1-page evictable prefix left in
  the radix cache reduces `free_pages` by 1 but keeps
  `available_size` bit-exact against baseline. Not a leak.

Structured log (rank 0, single JSON object; rank 1 identical except
for `device: "npu:1"` and per-rank `generate_ms`):

```
GATE4.12_JSONL rank=0 {
  "rank": 0, "world_size": 2, "tp_size": 2,
  "model_path": "/mnt/nvme/models/Qwen3-1.7B",
  "device": "npu:0",
  "request_uids": [0, 1],
  "prompts": ["Paris is", "The largest planet in our solar system by mass and volume is"],
  "prompt_token_lengths": [2, 12],
  "max_new_tokens_per_request": [8, 8],
  "baseline_available_tokens": 930656, "baseline_free_pages": 58166, "total_pages": 58166,
  "init_status": "PASS", "load_status": "PASS",
  "prefill_status": "PASS", "decode_status": "PASS",
  "admission_status": "PASS", "timeline_status": "PASS",
  "batch_timeline": [1, 1, 1, 2, 2, 2, 2, 2, 2, 1],
  "admission_events": {"0": 0, "1": 2},
  "completion_events": {"0": 8, "1": 9},
  "joint_decode_step_count": 6,
  "all_step_snapshots": [
    {"step_id": 0, "batch_size": 1, "active_uids": [0],    "query_lengths": [2],    "kv_lengths": [2]},
    {"step_id": 1, "batch_size": 1, "active_uids": [0],    "query_lengths": [1],    "kv_lengths": [3]},
    {"step_id": 2, "batch_size": 1, "active_uids": [1],    "query_lengths": [12],   "kv_lengths": [12]},
    {"step_id": 3, "batch_size": 2, "active_uids": [0, 1], "query_lengths": [1, 1], "kv_lengths": [4, 13]},
    {"step_id": 4, "batch_size": 2, "active_uids": [0, 1], "query_lengths": [1, 1], "kv_lengths": [5, 14]},
    {"step_id": 5, "batch_size": 2, "active_uids": [0, 1], "query_lengths": [1, 1], "kv_lengths": [6, 15]},
    {"step_id": 6, "batch_size": 2, "active_uids": [0, 1], "query_lengths": [1, 1], "kv_lengths": [7, 16]},
    {"step_id": 7, "batch_size": 2, "active_uids": [0, 1], "query_lengths": [1, 1], "kv_lengths": [8, 17]},
    {"step_id": 8, "batch_size": 2, "active_uids": [0, 1], "query_lengths": [1, 1], "kv_lengths": [9, 18]},
    {"step_id": 9, "batch_size": 1, "active_uids": [1],    "query_lengths": [1],    "kv_lengths": [19]}
  ],
  "actual_output_tokens_per_request": [8, 8],
  "output_texts": [
    " the capital of France. Paris is the",
    " the planet Jupiter. It is the second"
  ],
  "output_token_ids": [
    [279, 6722, 315, 9625, 13, 12095, 374, 279],
    [279, 11580, 49689, 13, 1084, 374, 279, 2086]
  ],
  "available_tokens_after_case": 930656,
  "free_pages_after_case": 58165,
  "free_pages_before_after": [58166, 58165],
  "deferred_abort_uids": 0,
  "cache_integrity_ok": true,
  "generate_ms": 2859.7381...,
  "cold_start_attempt_id": 1,
  "memory_sync_retry_note": "",
  "status": "PASS",
  "failure_stage": null, "failure_trace_summary": null
}
GATE4.12_JSONL rank=1 { ...device: "npu:1", generate_ms: 3177.90..., otherwise identical... }
```

Rank 0 exit code: 0.

## 2. Envelope (locked)

| Knob | Value |
|---|---|
| Hardware | 2 × Ascend 910B1 (64 GiB HBM each) |
| Container | `<CONTAINER>` on `<HOST>:<PORT>` |
| torch | 2.4.0 |
| torch_npu | 2.9.0.post1 |
| CANN | 8.5.1 |
| Python | 3.11.14 |
| Model | Qwen3-1.7B (`/mnt/nvme/models/Qwen3-1.7B`) |
| dtype | bf16 |
| TP | 2 (torchrun `--nproc_per_node=2`) |
| Attention backend | `npu_fia` |
| Rendezvous | `MINISGL_DISTRIBUTED_ADDR=env://` (reuses torchrun's TCPStore) |
| Distributed collectives | HCCL primary, gloo sidecar; `use_pynccl=False` |
| CUDAGraph batch sizes | `cuda_graph_bs=[]` (torch_npu has no CUDAGraph) |
| Execution mode | eager |
| Sampling | greedy (temperature=0.0, top_k=1, top_p=1.0, `ignore_eos=True`) |
| `memory_ratio` | 0.85 |
| `page_size` | 16 |
| `max_running_req` | 4 |
| `max_new_tokens` (per request) | 8 / 8 |
| Batch envelope | 1 → 2 → 1 |
| Staggered admission | script-local `StaggeredLLM` subclass override of `offline_receive_msg` |
| Metadata capture | script-local monkey-patch of `AscendFIABackend.prepare_metadata`; runtime unchanged |

## 3. Launch command

Same outer-shell retry pattern as Gate 4.10 / 4.11: sleep 8 s
between attempts, bump `--master_port`, thread
`--cold-start-attempt-id` and `--memory-sync-retry-note` into the
driver. Attempt 1 (port 29510) succeeded cleanly — no retry needed.

```bash
ssh -p <PORT> <USER>@<HOST> \
  "docker exec <CONTAINER> bash -c '
    set +e
    cd /mnt/nvme/LR-606/mini-sglang-ascend-gate4.12 &&
    mkdir -p logs &&
    PORT_BASE=29500 &&
    NOTE=\"\" &&
    for ATTEMPT in 1 2 3; do
      PORT=\$((PORT_BASE + ATTEMPT * 10))
      LOG=logs/gate4.12_qwen1.7b_tp2_dynamic_admission_b1_b2_b1_attempt\${ATTEMPT}.log
      PYTHONPATH=./python:\$PYTHONPATH torchrun --nproc_per_node=2 --master_port=\$PORT \\
        scripts/gate4_12_qwen1_7b_tp2_dynamic_admission_b1_b2_b1.py \\
        --model-path /mnt/nvme/models/Qwen3-1.7B \\
        --cold-start-attempt-id \$ATTEMPT \\
        --memory-sync-retry-note \"\$NOTE\" > \$LOG 2>&1
      if grep -q GATE4.12_JSONL \$LOG && ! grep -q \"Memory across TP ranks are imbalanced\" \$LOG; then
        cp \$LOG logs/gate4.12_qwen1.7b_tp2_dynamic_admission_b1_b2_b1.log
        break
      fi
      IMB=\$(grep \"Memory across TP ranks are imbalanced\" \$LOG | head -1)
      NOTE=\"\$NOTE | attempt \$ATTEMPT port \$PORT: \$IMB\"
      sleep 8
    done
  '"
```

## 4. Batch timeline evidence

The metadata hook captured 10 forward-pass snapshots per rank
inside the `run_forever()` window (byte-identical across ranks;
step-by-step breakdown):

| step_id | phase | batch_size | active_uids | query_lengths | kv_lengths |
|---:|---|---:|---|---|---|
| 0 | A prefill (`cached_len==0`) | 1 | `[0]` | `[2]` | `[2]` |
| 1 | A decode 1 (alone) — triggers `ready_to_admit_b` | 1 | `[0]` | `[1]` | `[3]` |
| 2 | B prefill (admitted; `cached_len==0`) | 1 | `[1]` | `[12]` | `[12]` |
| 3 | joint decode 1 | 2 | `[0, 1]` | `[1, 1]` | `[4, 13]` |
| 4 | joint decode 2 | 2 | `[0, 1]` | `[1, 1]` | `[5, 14]` |
| 5 | joint decode 3 | 2 | `[0, 1]` | `[1, 1]` | `[6, 15]` |
| 6 | joint decode 4 | 2 | `[0, 1]` | `[1, 1]` | `[7, 16]` |
| 7 | joint decode 5 | 2 | `[0, 1]` | `[1, 1]` | `[8, 17]` |
| 8 | joint decode 6 (A hits `max_tokens=8`; A retires next tick) | 2 | `[0, 1]` | `[1, 1]` | `[9, 18]` |
| 9 | B decode alone (A gone) | 1 | `[1]` | `[1]` | `[19]` |

**`batch_timeline = [1, 1, 1, 2, 2, 2, 2, 2, 2, 1]`** — contains the
required ordered subsequence `1 → 2 → 1`:

* `saw_one` at step 0 (bs=1)
* `saw_two_after_one` at step 3 (bs=2, after a bs=1 was seen)
* `saw_one_after_two` at step 9 (bs=1, after bs=2 was seen)

`timeline_status == "PASS"` and `joint_decode_step_count == 6 ≥ 1`
on both ranks. Rank 0 and rank 1 metadata traces are byte-identical
— HCCL lockstep guarantees both ranks tick identically, so the
staggered admission trigger fires on the same step on both ranks
and uids get assigned identically.

## 5. Admission / completion event evidence

Derived from the per-step `active_uids` scan:

| uid | prompt | first step in a batch | last step in a batch |
|---:|---|---:|---:|
| 0 (A, `"Paris is"`, 2 tokens) | 0 | 8 |
| 1 (B, `"The largest planet ... is"`, 12 tokens) | 2 | 9 |

* `admission_events == {"0": 0, "1": 2}` on both ranks — request B
  first entered a batch at step 2 (its prefill), strictly after
  request A first entered a batch at step 0. `a_first < b_first`
  → `admission_status == "PASS"`.
* `completion_events == {"0": 8, "1": 9}` on both ranks — request
  A left the batch after step 8 (having produced 8 decode tokens
  from steps 1..8, hitting `max_tokens=8`), and request B
  continued alone at step 9 to produce its 8th token (steps
  2..9 = 1 prefill + 7 decode + 1 final = 8 decode tokens for
  uid 1). Sibling B continued after peer A left → the required
  `B: 1 → 2 → 1` collapse invariant.
* Uid 0 output tokens are exactly 8 (steps 1, 3, 4, 5, 6, 7, 8 are
  its 7 decode ticks in a batch, plus the last decode token was
  sampled after step 8's forward completed); uid 1 output tokens
  are exactly 8 (steps 3, 4, 5, 6, 7, 8, 9 are its 7 batched
  decode ticks, plus the token sampled after step 2's prefill —
  first B decode result — makes 8 total). Both match
  `max_new_tokens=8` per request.

## 6. Per-rank output evidence

| Field | rank 0 | rank 1 |
|---|---|---|
| `device` | `npu:0` | `npu:1` |
| `request_uids` | `[0, 1]` | `[0, 1]` |
| `prompt_token_lengths` | `[2, 12]` | `[2, 12]` |
| `max_new_tokens_per_request` | `[8, 8]` | `[8, 8]` |
| `actual_output_tokens_per_request` | `[8, 8]` | `[8, 8]` |
| `output_texts[0]` | `" the capital of France. Paris is the"` | `" the capital of France. Paris is the"` |
| `output_texts[1]` | `" the planet Jupiter. It is the second"` | `" the planet Jupiter. It is the second"` |
| `output_token_ids[0]` | `[279, 6722, 315, 9625, 13, 12095, 374, 279]` | `[279, 6722, 315, 9625, 13, 12095, 374, 279]` |
| `output_token_ids[1]` | `[279, 11580, 49689, 13, 1084, 374, 279, 2086]` | `[279, 11580, 49689, 13, 1084, 374, 279, 2086]` |
| `batch_timeline` | `[1,1,1,2,2,2,2,2,2,1]` | `[1,1,1,2,2,2,2,2,2,1]` |
| `admission_events` | `{"0": 0, "1": 2}` | `{"0": 0, "1": 2}` |
| `completion_events` | `{"0": 8, "1": 9}` | `{"0": 8, "1": 9}` |
| `joint_decode_step_count` | 6 | 6 |
| `generate_ms` | 2859.74 | 3177.90 |
| `cold_start_attempt_id` | 1 | 1 |
| `admission_status` | `PASS` | `PASS` |
| `timeline_status` | `PASS` | `PASS` |
| `status` | `PASS` | `PASS` |

Per-rank output equality proven by exact byte match on
`output_texts[i]` and on every element of `output_token_ids[i]`
for both `i ∈ {0, 1}`. The uid-0 output matches Gate 4.10's ragged
prefill uid-0 output byte-for-byte on the same short prompt, and
uid-1 output matches Gate 4.10 / 4.11 uid-1 output byte-for-byte
— greedy sampling on all-gathered logits under TP=2 lockstep is
deterministic, and dynamic admission does not perturb the KV
values that go into either request's next-token sampling.

## 7. Per-rank allocator evidence

| Field | rank 0 | rank 1 |
|---|---|---|
| `baseline_available_tokens` | 930656 | 930656 |
| `baseline_free_pages` | 58166 | 58166 |
| `total_pages` | 58166 | 58166 |
| `available_tokens_after_case` | 930656 | 930656 |
| `free_pages_after_case` | 58165 | 58165 |
| `free_pages_before_after` | `[58166, 58165]` | `[58166, 58165]` |
| `deferred_abort_uids` | 0 | 0 |
| `cache_integrity_ok` | true | true |

The **`available_size` invariant is exact** on both ranks
(`930656 → 930656`). The one-page reduction in `free_pages`
(`58166 → 58165`) is a retained evictable radix-cache prefix —
same benign pattern documented at Gate 4.5 on Qwen3-0.6B and at
Gate 4.10 / 4.11 on Qwen3-1.7B: `available_size = free_slots +
evictable_prefix_pages` normalises back to baseline. Not a page
leak; `check_integrity()` confirms radix-cache linkage is intact
on both ranks.

Two requests spanning 10 forward passes, one staggered admission,
and one out-of-order completion left no page leak, no deferred
abort, no radix corruption. Baseline (`930656` / `58166`)
matches the natural Qwen3-1.7B TP=2 budget with the metadata hook
installed — a hair different from Gate 4.10 / 4.11's baselines
because the boot-time footprint varies slightly per cold-start
(same phenomenon Gate 4.9 §7.1 documented at the
`_sync_get_memory()` layer). Both ranks agree bit-for-bit on the
Gate 4.12 baseline (`930656` per rank), so the invariant proof is
not affected.

## 8. Cold-start retry note

**Attempt 1 (port 29510) succeeded cleanly.** Both ranks passed
the pre-load and post-load `_sync_get_memory()` imbalance check on
the first attempt. JSONL reports `cold_start_attempt_id == 1` and
`memory_sync_retry_note == ""` on both ranks.

The outer shell loop was authorised (per Gate 4.12 spec §2) to
sleep 8 s and re-launch with a fresh `--master_port` up to 2 more
times on cold-start imbalance. It was not exercised on this run.
Attempt-1 log lives at
`logs/gate4.12_qwen1.7b_tp2_dynamic_admission_b1_b2_b1_attempt1.log`
and is also copied to
`logs/gate4.12_qwen1.7b_tp2_dynamic_admission_b1_b2_b1.log` as the
canonical trace.

The pre-existing 2 GiB tolerance in `Engine._sync_get_memory()`
at `python/minisgl/engine/engine.py:246` is unchanged. Gate 4.12
did not modify `python/minisgl/`.

## 9. First failing stage

None — the driver reached `status=PASS` on both ranks without
recording a `failure_stage`. `failure_stage` is `null` on both
ranks; `failure_trace_summary` is `null` on both ranks.

## 10. Regression evidence

Per-file pytest on `tests/misc/` (headers only, hermetic per-file
mode) in the same working tree used for the smoke run:

```
tests/misc/test_scheduler_abort_ack.py             → 8/8   PASS  (15.14s)
tests/misc/test_scheduler_overlap_abort_fence.py   → 7/7   PASS  (15.60s)
tests/misc/test_scheduler_prepare_batch_txn.py     → 5/5   PASS  (15.52s)
tests/misc/test_engine_forward_sampler_atomic.py   → 5/5   PASS  (13.10s)
tests/misc/test_scheduler_shutdown_drain.py        → 8/8   PASS  (15.24s)
tests/misc/test_exposed_path_abort_ack.py          → 2/2   PASS  (14.46s)
tests/misc/test_shell_cancel_cleanup.py            → 2/2   PASS  (13.85s)
tests/misc/test_pyproject_config.py                → 14/14 PASS  ( 0.04s)
```

Total: **51 / 51 PASS** in per-file (hermetic) mode. Every count
matches the last measurement at Gate 4.11. No test file was
modified by this gate.

## 11. Known limitations

* **`B: 1 → 2 → 1` only.** `B: 1 → 2 → 3 → 2 → 1` and any wider
  grow-shrink shape are out of scope. Gate 4.6 explored a similar
  boundary on Qwen3-0.6B; Qwen3-1.7B is *not* proven for wider
  shapes by this gate.
* **B ≤ 2.** No B=3 at any point in the timeline.
* **`max_new_tokens=8`** for both requests. Longer decodes are not
  exercised. The demonstration relies on both requests reaching
  the joint decode phase together; 8/8 is comfortably enough to
  produce 6 joint decode steps here.
* **Staggered admission is script-local.** The `StaggeredLLM`
  subclass and the `AscendFIABackend.prepare_metadata` monkey-patch
  live in the driver only, are torn down when the process exits,
  and do not modify runtime source. They do not deep-copy
  metadata; they snapshot the list forms of `query_seq_lens` /
  `kv_seq_lens` at wrapper-return time and the `active_uids` list
  by re-reading `batch.reqs`, both of which are safe against
  runtime mutation after `prepare_metadata` returns.
* **Only the `cached_len == 0` prefill branch is proven** for both
  requests. The Gate 2.2f-documented "ragged + non-zero
  `cached_len` + `extend_len > 1`" branch is deliberately not
  exercised — B's prefill happens on its own tick (step 2), so it
  is never mixed with a decode step of a different uid on the
  ragged code path.
* **Qwen3-1.7B only.** Qwen3-4B / 14B / 32B, quantized weights,
  and MoE variants are out of scope.
* **TP=2 only.** TP=4 / TP=8 not proven for Qwen3-1.7B.
* **Not a benchmark.** `generate_ms` is included in the JSONL for
  auditability only. It reflects the eager + npu_fia + bf16 +
  greedy path with a single fresh boot, the staggered-admission
  wrapper, AND the metadata-capture wrapper on every forward —
  not a stabilised throughput number, no warmup, no repeats. Do
  not quote it as performance.
* **`use_pynccl=False` is mandatory on NPU.** All numbers reflect
  the HCCL + gloo sidecar collective path.
* **1-page radix-cache retention** (`free_pages_after_case ==
  baseline_free_pages - 1`) is benign; `available_size`
  normalises back to baseline through `evictable_prefix_pages`.
  Same pattern as Gate 4.5 / 4.10 / 4.11.
* **Cold-start `_sync_get_memory()` variability** documented at
  Gate 4.9 remains outside `python/minisgl/`. The Gate 4.12 outer
  shell loop authorises up to 2 retries with fresh `master_port`
  and 8 s sleep. On this run only attempt 1 was required.

## 12. Decision matrix

| Question | Answer |
|---|---|
| Does Qwen3-1.7B `LLM.__init__` (via `StaggeredLLM`) return on both ranks under TP=2? | Yes (attempt 1) |
| Does the two-shard weight load complete on both ranks? | Yes |
| Does request A start alone before request B? | Yes (`admission_events == {"0": 0, "1": 2}`) |
| Is `a_first < b_first`? | Yes (0 < 2) |
| Does at least one joint decode step occur with `batch_size==2` and `query_lengths==[1,1]`? | Yes (6 joint decode steps) |
| Does the batch timeline contain the ordered subsequence `1 → 2 → 1`? | Yes (`[1,1,1,2,2,2,2,2,2,1]`) |
| Does a sibling continue alone after its peer leaves the batch? | Yes (uid 1 alone at step 9 after uid 0 retires) |
| Do rank 0 and rank 1 produce byte-identical `output_texts` per uid? | Yes for both uids |
| Do rank 0 and rank 1 produce byte-identical `output_token_ids` per uid? | Yes for both uids |
| Does the allocator `available_size` return to baseline on both ranks? | Yes (`930656 → 930656`) |
| Is the 1-page `free_pages` drift a leak? | No (retained evictable prefix) |
| Are `deferred_abort_uids == 0` after the case? | Yes on both ranks |
| Does `check_integrity()` pass after the case? | Yes on both ranks |
| Is the driver `B: 1 → 2 → 1` only (no B=3, no grow-shrink)? | Yes |
| Does the driver touch timing / benchmarking? | No |
| Does the driver modify `python/minisgl/`? | No |
| Does the driver modify tests? | No |
| Is `use_pynccl=False`? | Yes |
| Is the model Qwen3-1.7B only? | Yes |
| Is TP fixed at 2? | Yes |

**Verdict: PASS.**

## 13. Freeze boundary

The following files are the frozen artefacts for Gate 4.12:

* `scripts/gate4_12_qwen1_7b_tp2_dynamic_admission_b1_b2_b1.py`
* `docs/ascend_port/gate4.12_qwen1.7b_tp2_dynamic_admission_b1_b2_b1_verdict.md`

No files under `python/minisgl/` were modified at this gate. No
tests were modified at this gate. The freeze commit SHA is
recorded in this document header once the driver + verdict pair
is committed, and it is recorded on the `ascend-port` tip once the
branch is merged with `--no-ff`.
