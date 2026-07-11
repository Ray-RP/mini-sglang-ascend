# Gate 4.11 Verdict — Qwen3-1.7B TP=2 B=2 mixed-KV decode explicit evidence

**Gate ID:** 4.11 (Qwen3-1.7B TP=2 B=2 mixed-KV decode explicit evidence on Ascend 910B1)
**Verdict:** PASS
**Branch:** `gate4.11-qwen1.7b-tp2-b2-mixed-kv-decode`
**Base commit:** `fc51292` (tip of `ascend-port`, Gate 4.10 merge)
**Freeze commit:** `6089cfd`
**Date:** 2026-07-11
**Kind:** Real-hardware Ascend 910B1 TP=2 B=2 mixed-KV decode
explicit-evidence proof — two ranks × Qwen3-1.7B × two prompts
with *unequal* tokenized lengths (2 vs 12) × greedy ×
`max_new_tokens=8`. Both ranks return the exact same 8 output
tokens per uid, uid-by-uid outputs match across ranks
byte-for-byte, the FIA metadata is captured per forward pass, all
seven decode steps show `query_lengths == [1, 1]` with unequal
`kv_lengths`, and the allocator returns to baseline on both ranks
with no deferred aborts and no radix-cache corruption.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5 /
3.1 / 3.2 / 3.3 / 3.4 / 4.1 / 4.2 / 4.3 / 4.4 / 4.5 / 4.6 / 4.7 /
4.8 / 4.9 / 4.10, does not mutate release tag `v0.1.0a1`, does not
touch the GitHub Release, CHANGELOG, or release notes, and does not
extend the Ascend port to TP > 2, B > 2, dynamic admission, TP=2
timing, non-Qwen3 architectures, or Qwen3-4B / 14B / 32B /
quantized / MoE variants. The only new artefacts at this gate are
one bring-up script and this verdict; no runtime source under
`python/minisgl/` is modified; no test file is modified.

---

## 1. Verdict summary

**PASS on all four cases across both ranks.**

| Case | Description | rank 0 | rank 1 |
|---|---|---|---|
| A | TP=2 init — `_init_communication` returns, both ranks report `init_status=PASS` | **PASS** | **PASS** |
| B | Qwen3-1.7B TP=2 model-load — per-rank weight sharding + post-load `check_integrity()` | **PASS** | **PASS** |
| C | B=2 unequal-length prefill + decode — `generate([short, long], max_tokens=8)` | **PASS** | **PASS** |
| D | mixed-KV decode metadata evidence — `≥1` decode snapshot with `query_lengths==[1,1]` and `kv_lengths[0] != kv_lengths[1]` | **PASS** (7/7) | **PASS** (7/7) |

Cases A / B collapse into the same `LLM` boot (`set_tp_info` is
one-shot); reaching a successful `_snapshot(cache_manager)` after
`LLM.__init__` returns proves init + load simultaneously.

Case C proven invariants on both ranks:

* `prompt_token_lengths == [2, 12]` (unequal — driver asserts
  `tokenized_lengths[0] != tokenized_lengths[1]` before dispatching
  `generate()`)
* `actual_output_tokens_per_request == [8, 8]`
* `output_texts[0] == " the capital of France. Paris is the"`
* `output_texts[1] == " the planet Jupiter. It is the second"`
* rank 0 `output_texts[i]` byte-identical to rank 1 `output_texts[i]`
  for `i ∈ {0, 1}`
* rank 0 `output_token_ids[i]` byte-identical to rank 1 `output_token_ids[i]`
  for `i ∈ {0, 1}`
* `output_token_ids[0] == [279, 6722, 315, 9625, 13, 12095, 374, 279]`
* `output_token_ids[1] == [279, 11580, 49689, 13, 1084, 374, 279, 2086]`

Case D proven invariants on both ranks (`AscendFIABackend.prepare_metadata`
captured 8 snapshots per rank: 1 prefill + 7 decode):

* `all_step_snapshots[0]` = prefill: `batch_size=2, query_lengths=[2,12], kv_lengths=[2,12]`
* `decode_step_snapshots` = 7 pure-decode steps, each with
  `batch_size=2, query_lengths=[1,1]` and `kv_lengths` strictly
  increasing per uid.
* `mixed_kv_decode_step_count == 7` (every decode step exhibits
  `kv_lengths[0] != kv_lengths[1]`).
* The KV delta per step is exactly `long_prompt_len - short_prompt_len == 10`
  on every decode step, matching the mixed-KV invariant this gate
  exists to prove.

Post-case allocator invariants held on both ranks:

* `available_tokens_after_case == baseline_available_tokens` (`925328` on both ranks)
* `deferred_abort_uids == 0`
* `cache_integrity_ok == true`
* `free_pages_after_case == 57832` vs `baseline_free_pages == 57833`
  — a **1-page radix-cache retention** absorbed by the
  `available_size` invariant. Same benign pattern documented at
  Gate 4.5 / 4.10: `available_size = free_slots +
  evictable_prefix_pages`, so a 1-page evictable prefix left in the
  radix cache reduces `free_pages` by 1 but keeps `available_size`
  bit-exact against baseline. Not a leak.

Structured log (rank 0, single JSON object; rank 1 identical
except for `device: "npu:1"` and per-rank `generate_ms`):

```
GATE4.11_JSONL rank=0 {
  "rank": 0, "world_size": 2, "tp_size": 2,
  "model_path": "/mnt/nvme/models/Qwen3-1.7B",
  "device": "npu:0",
  "prompts": ["Paris is", "The largest planet in our solar system by mass and volume is"],
  "prompt_token_lengths": [2, 12], "batch_size": 2,
  "baseline_available_tokens": 925328, "baseline_free_pages": 57833, "total_pages": 57833,
  "init_status": "PASS", "load_status": "PASS",
  "prefill_status": "PASS", "decode_status": "PASS", "mixed_kv_status": "PASS",
  "actual_output_tokens_per_request": [8, 8],
  "output_texts": [
    " the capital of France. Paris is the",
    " the planet Jupiter. It is the second"
  ],
  "output_token_ids": [
    [279, 6722, 315, 9625, 13, 12095, 374, 279],
    [279, 11580, 49689, 13, 1084, 374, 279, 2086]
  ],
  "all_step_snapshots": [
    {"step_id": 0, "batch_size": 2, "query_lengths": [2, 12], "kv_lengths": [2, 12]},
    {"step_id": 1, "batch_size": 2, "query_lengths": [1, 1],  "kv_lengths": [3, 13]},
    {"step_id": 2, "batch_size": 2, "query_lengths": [1, 1],  "kv_lengths": [4, 14]},
    {"step_id": 3, "batch_size": 2, "query_lengths": [1, 1],  "kv_lengths": [5, 15]},
    {"step_id": 4, "batch_size": 2, "query_lengths": [1, 1],  "kv_lengths": [6, 16]},
    {"step_id": 5, "batch_size": 2, "query_lengths": [1, 1],  "kv_lengths": [7, 17]},
    {"step_id": 6, "batch_size": 2, "query_lengths": [1, 1],  "kv_lengths": [8, 18]},
    {"step_id": 7, "batch_size": 2, "query_lengths": [1, 1],  "kv_lengths": [9, 19]}
  ],
  "decode_step_snapshots": [steps 1..7 above],
  "mixed_kv_decode_step_count": 7,
  "available_tokens_after_case": 925328,
  "free_pages_after_case": 57832,
  "deferred_abort_uids": 0,
  "cache_integrity_ok": true,
  "generate_ms": 5445.965310093015,
  "cold_start_attempt_id": 1,
  "memory_sync_retry_note": "",
  "status": "PASS",
  "failure_stage": null, "failure_trace_summary": null
}
GATE4.11_JSONL rank=1 { ...device: "npu:1", generate_ms: 5606.63..., otherwise identical... }
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
| `max_new_tokens` | 8 |
| Batch size | 2 |
| Prompt inequality | `prompt_token_lengths[0] == 2 != prompt_token_lengths[1] == 12` |
| Metadata capture | script-local monkey-patch of `AscendFIABackend.prepare_metadata`; runtime unchanged |

## 3. Launch command

Same outer-shell retry pattern as Gate 4.10: sleep 8 s between
attempts, bump `--master_port`, thread `--cold-start-attempt-id`
and `--memory-sync-retry-note` into the driver. Attempt 1 (port
29480) succeeded cleanly — no retry needed.

```bash
ssh -p <PORT> <USER>@<HOST> \
  "docker exec <CONTAINER> bash -c '
    set +e
    cd /mnt/nvme/LR-606/mini-sglang-ascend-gate4.11 &&
    mkdir -p logs &&
    PORT_BASE=29470 &&
    NOTE=\"\" &&
    for ATTEMPT in 1 2 3; do
      PORT=\$((PORT_BASE + ATTEMPT * 10))
      LOG=logs/gate4.11_qwen1.7b_tp2_b2_mixed_kv_decode_attempt\${ATTEMPT}.log
      PYTHONPATH=./python:\$PYTHONPATH torchrun --nproc_per_node=2 --master_port=\$PORT \\
        scripts/gate4_11_qwen1_7b_tp2_b2_mixed_kv_decode.py \\
        --model-path /mnt/nvme/models/Qwen3-1.7B \\
        --cold-start-attempt-id \$ATTEMPT \\
        --memory-sync-retry-note \"\$NOTE\" > \$LOG 2>&1
      if grep -q GATE4.11_JSONL \$LOG && ! grep -q \"Memory across TP ranks are imbalanced\" \$LOG; then
        cp \$LOG logs/gate4.11_qwen1.7b_tp2_b2_mixed_kv_decode.log
        break
      fi
      IMB=\$(grep \"Memory across TP ranks are imbalanced\" \$LOG | head -1)
      NOTE=\"\$NOTE | attempt \$ATTEMPT port \$PORT: \$IMB\"
      sleep 8
    done
  '"
```

## 4. Prompt-length evidence

The two prompts were chosen so their tokenized lengths on
Qwen3-1.7B differ by an order of magnitude — the same short/long
pair used at Gate 4.10 for ragged prefill, now reused for
mixed-KV decode evidence. The driver asserts
`tokenized_lengths[0] != tokenized_lengths[1]` before dispatching
`generate()`; the assertion passed cleanly:

| uid | prompt | tokenized length (Qwen3-1.7B tokenizer) |
|---|---|---:|
| 0 (short) | `"Paris is"` | 2 |
| 1 (long)  | `"The largest planet in our solar system by mass and volume is"` | 12 |

Both ranks reported `prompt_token_lengths == [2, 12]` in the JSONL.
The 10-token gap propagates through every decode step: at step
`k` (0-indexed decode) both uids have grown by `k+1` tokens, so
`kv_lengths[1] - kv_lengths[0] == 10` on every decode snapshot.
That constant per-step delta is the mixed-KV signature this gate
proves.

## 5. Decode metadata evidence

`AscendFIABackend.prepare_metadata` is invoked once per forward
pass. The driver installs a script-local wrapper BEFORE `LLM()`
and records `batch_size`, `query_seq_lens`, and `kv_seq_lens`
from `batch.attn_metadata` immediately after the original method
sets it. Nothing under `python/minisgl/` is modified — the wrapper
is torn down when the process exits.

Both ranks captured exactly 8 snapshots inside the `generate()`
window (`all_step_snapshots` slice from `snapshots_pre_generate`):

| step_id | phase | batch_size | query_lengths | kv_lengths | kv_delta (long − short) |
|---:|---|---:|---|---|---:|
| 0 | prefill (ragged, `cached_len == 0`) | 2 | `[2, 12]` | `[2, 12]` | 10 |
| 1 | decode 1 | 2 | `[1, 1]` | `[3, 13]` | 10 |
| 2 | decode 2 | 2 | `[1, 1]` | `[4, 14]` | 10 |
| 3 | decode 3 | 2 | `[1, 1]` | `[5, 15]` | 10 |
| 4 | decode 4 | 2 | `[1, 1]` | `[6, 16]` | 10 |
| 5 | decode 5 | 2 | `[1, 1]` | `[7, 17]` | 10 |
| 6 | decode 6 | 2 | `[1, 1]` | `[8, 18]` | 10 |
| 7 | decode 7 | 2 | `[1, 1]` | `[9, 19]` | 10 |

Gate 4.11 acceptance requires **≥1** decode snapshot with
`query_lengths == [1, 1]` and `kv_lengths[0] != kv_lengths[1]`.
We captured **7 / 7** decode snapshots satisfying that condition
on both ranks (`mixed_kv_decode_step_count == 7`). Rank 0 and
rank 1 metadata traces are byte-identical — the FIA scheduler
plans identical shapes across TP ranks, as expected.

## 6. Per-rank output evidence

| Field | rank 0 | rank 1 |
|---|---|---|
| `device` | `npu:0` | `npu:1` |
| `prompt_token_lengths` | `[2, 12]` | `[2, 12]` |
| `batch_size` | 2 | 2 |
| `actual_output_tokens_per_request` | `[8, 8]` | `[8, 8]` |
| `output_texts[0]` | `" the capital of France. Paris is the"` | `" the capital of France. Paris is the"` |
| `output_texts[1]` | `" the planet Jupiter. It is the second"` | `" the planet Jupiter. It is the second"` |
| `output_token_ids[0]` | `[279, 6722, 315, 9625, 13, 12095, 374, 279]` | `[279, 6722, 315, 9625, 13, 12095, 374, 279]` |
| `output_token_ids[1]` | `[279, 11580, 49689, 13, 1084, 374, 279, 2086]` | `[279, 11580, 49689, 13, 1084, 374, 279, 2086]` |
| `generate_ms` | 5445.97 | 5606.63 |
| `cold_start_attempt_id` | 1 | 1 |
| `mixed_kv_status` | `PASS` | `PASS` |
| `status` | `PASS` | `PASS` |

Per-rank output equality proven by exact byte match on
`output_texts[i]` and on every element of `output_token_ids[i]`
for both `i ∈ {0, 1}`. Output tokens match Gate 4.10's ragged
prefill run byte-for-byte on the same prompt pair — greedy
sampling on all-gathered logits is deterministic under lockstep
TP=2 scheduling, and the metadata-capture wrapper does not
perturb the forward pass.

## 7. Per-rank allocator evidence

| Field | rank 0 | rank 1 |
|---|---|---|
| `baseline_available_tokens` | 925328 | 925328 |
| `baseline_free_pages` | 57833 | 57833 |
| `total_pages` | 57833 | 57833 |
| `available_tokens_after_case` | 925328 | 925328 |
| `free_pages_after_case` | 57832 | 57832 |
| `deferred_abort_uids` | 0 | 0 |
| `cache_integrity_ok` | true | true |

The **`available_size` invariant is exact** on both ranks
(`925328 → 925328`). The one-page reduction in `free_pages`
(`57833 → 57832`) is a retained evictable radix-cache prefix —
same benign pattern documented at Gate 4.5 on Qwen3-0.6B and
Gate 4.10 on Qwen3-1.7B: `available_size = free_slots +
evictable_prefix_pages` normalises back to baseline. Not a page
leak; `check_integrity()` confirms radix-cache linkage is intact
on both ranks.

Note that the Gate 4.11 baseline (`925328` tokens, `57833`
pages) is a few pages lower than Gate 4.10's (`930640` /
`58165`). Both ranks agree on the same baseline (925328 per rank),
so the invariant proof is not affected. The delta reflects
cold-start torch_npu memory-footprint variance between fresh
process boots — the same phenomenon that Gate 4.9 documented at
the `_sync_get_memory()` layer. It is not a regression: Gate 4.11
budget on both ranks is bit-identical, and `available_size`
returns exactly to it after `generate()`.

## 8. Cold-start retry note

**Attempt 1 (port 29480) succeeded cleanly.** Both ranks passed
the pre-load and post-load `_sync_get_memory()` imbalance check
on the first attempt. JSONL reports `cold_start_attempt_id == 1`
and `memory_sync_retry_note == ""` on both ranks.

The outer shell loop was authorised (per Gate 4.11 spec §2) to
sleep 8 s and re-launch with a fresh `--master_port` up to 2 more
times on cold-start imbalance. It was not exercised on this run.
Attempt-1 log lives at
`logs/gate4.11_qwen1.7b_tp2_b2_mixed_kv_decode_attempt1.log` and
is also copied to
`logs/gate4.11_qwen1.7b_tp2_b2_mixed_kv_decode.log` as the
canonical trace.

The pre-existing 2 GiB tolerance in `Engine._sync_get_memory()`
at `python/minisgl/engine/engine.py:246` is unchanged. Gate 4.11
did not modify `python/minisgl/`.

## 9. First failing stage

None — the driver reached `status=PASS` on both ranks without
recording a `failure_stage`. `failure_stage` is `null` on both
ranks; `failure_trace_summary` is `null` on both ranks.

## 10. Regression evidence

Per-file pytest on `tests/misc/` (headers only, hermetic per-file
mode) in the same working tree used for the smoke run:

```
tests/misc/test_scheduler_abort_ack.py             → 8/8   PASS  (16.82s)
tests/misc/test_scheduler_overlap_abort_fence.py   → 7/7   PASS  (15.95s)
tests/misc/test_scheduler_prepare_batch_txn.py     → 5/5   PASS  (15.20s)
tests/misc/test_engine_forward_sampler_atomic.py   → 5/5   PASS  (13.29s)
tests/misc/test_scheduler_shutdown_drain.py        → 8/8   PASS  (15.45s)
tests/misc/test_exposed_path_abort_ack.py          → 2/2   PASS  (15.52s)
tests/misc/test_shell_cancel_cleanup.py            → 2/2   PASS  (16.05s)
tests/misc/test_pyproject_config.py                → 14/14 PASS  ( 0.04s)
```

Total: **51 / 51 PASS** in per-file (hermetic) mode. Every count
matches the last measurement at Gate 4.10. No test file was
modified by this gate.

## 11. Known limitations

* **B=2 mixed-KV decode only.** Dynamic admission and B > 2 on
  Qwen3-1.7B are out of scope. Gate 4.6 proved dynamic admission
  on Qwen3-0.6B; Qwen3-1.7B is *not* proven for those shapes by
  this gate.
* **`max_new_tokens=8`.** Longer decodes are not exercised; the
  mixed-KV pattern proved here scales trivially with token count
  (each step just adds one more KV per uid), so this is a
  sufficiency window, not a limit test.
* **Only the `cached_len == 0` prefill branch is proven.** The
  Gate 2.2f-documented "ragged + non-zero `cached_len` +
  `extend_len > 1`" branch is deliberately not exercised (single
  `generate()` per process).
* **Qwen3-1.7B only.** Qwen3-4B / 14B / 32B, quantized weights,
  and MoE variants are out of scope.
* **TP=2 only.** TP=4 / TP=8 not proven for Qwen3-1.7B.
* **Metadata capture is script-local.** The
  `AscendFIABackend.prepare_metadata` wrapper is torn down when
  the process exits and does not modify runtime source. It also
  does not deep-copy the metadata; it snapshots the list forms of
  `query_seq_lens` and `kv_seq_lens` at wrapper-return time, which
  is safe because the runtime does not mutate those in-place after
  `prepare_metadata` returns.
* **Not a benchmark.** `generate_ms` is included in the JSONL for
  auditability only. It reflects the eager + npu_fia + bf16 +
  greedy path with a single fresh boot AND the metadata-capture
  wrapper on every forward — not a stabilised throughput number,
  no warmup, no repeats. Do not quote it as performance.
* **`use_pynccl=False` is mandatory on NPU.** All numbers reflect
  the HCCL + gloo sidecar collective path.
* **1-page radix-cache retention** (`free_pages_after_case ==
  baseline_free_pages - 1`) is benign; `available_size`
  normalises back to baseline through `evictable_prefix_pages`.
  Same pattern as Gate 4.5 and Gate 4.10.
* **Cold-start `_sync_get_memory()` variability** documented at
  Gate 4.9 remains outside `python/minisgl/`. The Gate 4.11 outer
  shell loop authorises up to 2 retries with fresh `master_port`
  and 8 s sleep. On this run only attempt 1 was required.

## 12. Decision matrix

| Question | Answer |
|---|---|
| Does Qwen3-1.7B `LLM.__init__` return on both ranks under TP=2? | Yes (attempt 1) |
| Does the two-shard weight load complete on both ranks? | Yes |
| Are `prompt_token_lengths[0] != prompt_token_lengths[1]`? | Yes (`2 != 12`) |
| Does `generate([short, long], max_tokens=8)` return `[8, 8]` tokens? | Yes on both ranks |
| Do rank 0 and rank 1 produce byte-identical `output_texts` per uid? | Yes for both uids |
| Do rank 0 and rank 1 produce byte-identical `output_token_ids` per uid? | Yes for both uids |
| Did the driver capture ≥1 decode step with `query_lengths==[1,1]` and unequal `kv_lengths`? | Yes — 7 / 7 decode steps, on both ranks |
| Are `all_step_snapshots` byte-identical across ranks? | Yes |
| Does the allocator `available_size` return to baseline on both ranks? | Yes (`925328 → 925328`) |
| Is the 1-page `free_pages` drift a leak? | No (retained evictable prefix) |
| Are `deferred_abort_uids == 0` after the case? | Yes on both ranks |
| Does `check_integrity()` pass after the case? | Yes on both ranks |
| Is the driver B=2 mixed-KV only (`cached_len == 0` prefill)? | Yes |
| Does the driver touch dynamic admission / timing / B > 2? | No |
| Does the driver modify `python/minisgl/`? | No |
| Does the driver modify tests? | No |
| Is `use_pynccl=False`? | Yes |
| Is the model Qwen3-1.7B only? | Yes |
| Is TP fixed at 2? | Yes |

**Verdict: PASS.**

## 13. Freeze boundary

The following files are the frozen artefacts for Gate 4.11:

* `scripts/gate4_11_qwen1_7b_tp2_b2_mixed_kv_decode.py`
* `docs/ascend_port/gate4.11_qwen1.7b_tp2_b2_mixed_kv_decode_verdict.md`

No files under `python/minisgl/` were modified at this gate. No
tests were modified at this gate. The freeze commit SHA is
recorded in this document header once the driver + verdict pair
is committed, and it is recorded on the `ascend-port` tip once the
branch is merged with `--no-ff`.
