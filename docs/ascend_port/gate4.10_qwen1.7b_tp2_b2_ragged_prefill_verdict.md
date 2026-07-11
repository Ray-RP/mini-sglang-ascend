# Gate 4.10 Verdict — Qwen3-1.7B TP=2 B=2 ragged prefill

**Gate ID:** 4.10 (Qwen3-1.7B TP=2 B=2 ragged prefill on Ascend 910B1)
**Verdict:** PASS
**Branch:** `gate4.10-qwen1.7b-tp2-b2-ragged-prefill`
**Base commit:** `8577594` (tip of `ascend-port`, Gate 4.9 merge)
**Freeze commit:** `ffa6e8f`
**Date:** 2026-07-11
**Kind:** Real-hardware Ascend 910B1 TP=2 B=2 ragged-prefill proof
— two ranks × Qwen3-1.7B × two prompts with *unequal* tokenized
lengths (2 vs 12) × greedy × `max_new_tokens=8`. Both ranks return
the exact same 8 output tokens per uid, uid-by-uid outputs match
across ranks byte-for-byte, and the allocator returns to baseline
on both ranks with no deferred aborts and no radix-cache corruption.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5 /
3.1 / 3.2 / 3.3 / 3.4 / 4.1 / 4.2 / 4.3 / 4.4 / 4.5 / 4.6 / 4.7 /
4.8 / 4.9, does not mutate release tag `v0.1.0a1`, does not touch
the GitHub Release, CHANGELOG, or release notes, and does not
extend the Ascend port to TP > 2, B > 2, mixed-KV decode explicit
evidence, dynamic admission, TP=2 timing, non-Qwen3 architectures,
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
| C | B=2 ragged prefill + decode — `generate([short, long], max_tokens=8)` with unequal tokenized lengths on the FIA `cached_len == 0` branch | **PASS** | **PASS** |

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

Post-case allocator invariants held on both ranks:

* `available_tokens_after_case == baseline_available_tokens` (`930640` on both ranks)
* `deferred_abort_uids == 0`
* `cache_integrity_ok == true`
* `free_pages_after_case == 58164` vs `baseline_free_pages == 58165`
  — a **1-page radix-cache retention** absorbed by the
  `available_size` invariant. This is the same benign pattern seen
  at Gate 4.5 on Qwen3-0.6B: `available_size = free_slots +
  evictable_prefix_pages`, so a 1-page evictable prefix left in the
  radix cache reduces `free_pages` by 1 but keeps `available_size`
  bit-exact against baseline. Not a leak.

Structured log (both ranks, single JSON object each; rank 1 identical
except for `device: "npu:1"` and per-rank `generate_ms`):

```
GATE4.10_JSONL rank=0 {
  "rank": 0, "world_size": 2, "tp_size": 2,
  "model_path": "/mnt/nvme/models/Qwen3-1.7B",
  "device": "npu:0",
  "prompts": ["Paris is", "The largest planet in our solar system by mass and volume is"],
  "prompt_token_lengths": [2, 12], "batch_size": 2,
  "baseline_available_tokens": 930640, "baseline_free_pages": 58165, "total_pages": 58165,
  "init_status": "PASS", "load_status": "PASS",
  "prefill_status": "PASS", "decode_status": "PASS",
  "actual_output_tokens_per_request": [8, 8],
  "output_texts": [
    " the capital of France. Paris is the",
    " the planet Jupiter. It is the second"
  ],
  "output_token_ids": [
    [279, 6722, 315, 9625, 13, 12095, 374, 279],
    [279, 11580, 49689, 13, 1084, 374, 279, 2086]
  ],
  "available_tokens_after_case": 930640,
  "free_pages_after_case": 58164,
  "deferred_abort_uids": 0,
  "cache_integrity_ok": true,
  "generate_ms": 2662.0128797367215,
  "cold_start_attempt_id": 1,
  "memory_sync_retry_note": "",
  "status": "PASS",
  "failure_stage": null, "failure_trace_summary": null
}
GATE4.10_JSONL rank=1 { ...device: "npu:1", generate_ms: 2684.24..., otherwise identical... }
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
| FIA branch | ragged prefill with `cached_len == 0` (fresh boot, single generate() call) |

## 3. Launch command

The Gate 4.10 launch runs the driver inside an outer shell loop
that retries on the transient `_sync_get_memory()` imbalance
documented at Gate 4.9. Each retry uses a fresh `--master_port`,
sleeps 8 s between attempts, and threads
`--cold-start-attempt-id` and `--memory-sync-retry-note` into the
driver so the JSONL always reports which attempt produced the
successful trace.

```bash
ssh -p <PORT> <USER>@<HOST> \
  "docker exec <CONTAINER> bash -c '
    set +e
    cd /mnt/nvme/LR-606/mini-sglang-ascend-gate50 &&
    mkdir -p logs &&
    PORT_BASE=29450 &&
    NOTE=\"\" &&
    for ATTEMPT in 1 2 3; do
      PORT=\$((PORT_BASE + ATTEMPT * 10))
      LOG=logs/gate4.10_qwen1.7b_tp2_b2_ragged_prefill_attempt\${ATTEMPT}.log
      PYTHONPATH=./python:\$PYTHONPATH torchrun --nproc_per_node=2 --master_port=\$PORT \\
        scripts/gate4_10_qwen1_7b_tp2_b2_ragged_prefill.py \\
        --model-path /mnt/nvme/models/Qwen3-1.7B \\
        --cold-start-attempt-id \$ATTEMPT \\
        --memory-sync-retry-note \"\$NOTE\" > \$LOG 2>&1
      if grep -q GATE4.10_JSONL \$LOG && ! grep -q \"Memory across TP ranks are imbalanced\" \$LOG; then
        cp \$LOG logs/gate4.10_qwen1.7b_tp2_b2_ragged_prefill.log
        break
      fi
      IMB=\$(grep \"Memory across TP ranks are imbalanced\" \$LOG | head -1)
      NOTE=\"\$NOTE | attempt \$ATTEMPT port \$PORT: \$IMB\"
      sleep 8
    done
  '"
```

Attempt 1 (port 29460) succeeded cleanly — no retry needed.

## 4. Prompt-length evidence

The two prompts were chosen so their tokenized lengths on
Qwen3-1.7B differ by an order of magnitude — a robust ragged pair
that cannot silently regress into an equal-length batch. The driver
asserts `tokenized_lengths[0] != tokenized_lengths[1]` before
dispatching `generate()`; the assertion passed cleanly:

| uid | prompt | tokenized length (Qwen3-1.7B tokenizer) |
|---|---|---:|
| 0 (short) | `"Paris is"` | 2 |
| 1 (long)  | `"The largest planet in our solar system by mass and volume is"` | 12 |

Both ranks reported `prompt_token_lengths == [2, 12]` in the JSONL,
confirming inequality. This drives the FIA "ragged prefill with
`cached_len == 0`" branch identically to Gate 4.3 on Qwen3-0.6B,
now proven on Qwen3-1.7B. The Gate 2.2f unsupported combination
("ragged + non-zero `cached_len` + `extend_len > 1`") is *not*
touched: only one `generate()` call is made per process, so no
scheduler step ever sees a prefix-hit uid on a ragged batch.

## 5. Per-rank output evidence

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
| `generate_ms` | 2662.01 | 2684.24 |
| `cold_start_attempt_id` | 1 | 1 |
| `status` | `PASS` | `PASS` |

Per-rank output equality proven by exact byte match on
`output_texts[i]` and on every element of `output_token_ids[i]` for
both `i ∈ {0, 1}`. Greedy sampling on all-gathered logits under
lockstep TP=2 scheduling gives bit-identical per-uid outputs across
ranks — same invariant Gate 4.3 established on Qwen3-0.6B, now
extended to Qwen3-1.7B on a ragged prefill batch.

## 6. Per-rank allocator evidence

| Field | rank 0 | rank 1 |
|---|---|---|
| `baseline_available_tokens` | 930640 | 930640 |
| `baseline_free_pages` | 58165 | 58165 |
| `total_pages` | 58165 | 58165 |
| `available_tokens_after_case` | 930640 | 930640 |
| `free_pages_after_case` | 58164 | 58164 |
| `deferred_abort_uids` | 0 | 0 |
| `cache_integrity_ok` | true | true |

The **`available_size` invariant is exact** on both ranks
(`930640 → 930640`). The one-page reduction in `free_pages`
(`58165 → 58164`) is a retained evictable radix-cache prefix —
after `generate()` returns, the scheduler leaves one prefix page
in the radix cache marked evictable; `available_size = free_slots +
evictable_prefix_pages` therefore normalises back to baseline. This
is the same benign pattern documented at Gate 4.5 on Qwen3-0.6B
(`952880 → 952880` with a similar 1-page `free_pages` drift). It
is not a page leak: subsequent requests can re-use that page
freely, and `check_integrity()` confirms the radix cache linkage
is intact on both ranks.

No requests left a deferred-abort uid pending, no requests
corrupted the radix cache linkage. Baseline matches Gate 4.9's
`930640` per rank — confirming the KV cache budget computation is
unaffected by B=2 batch shape variance.

## 7. Cold-start retry note

**Attempt 1 (port 29460) succeeded cleanly.** No retry was needed
this run — both ranks passed the pre-load and post-load
`_sync_get_memory()` imbalance check on the first attempt, unlike
Gate 4.9 which required two retries. The JSONL therefore reports
`cold_start_attempt_id == 1` and `memory_sync_retry_note == ""` on
both ranks.

The outer shell loop was still authorised (per the Gate 4.10 spec
§2) to sleep 8 s and re-launch with a fresh `--master_port` up to
2 more times on cold-start imbalance. It was not exercised on this
run. Attempt-1 log lives at `logs/gate4.10_qwen1.7b_tp2_b2_ragged_prefill_attempt1.log`
and is also copied to `logs/gate4.10_qwen1.7b_tp2_b2_ragged_prefill.log`
as the canonical trace.

The pre-existing 2 GiB tolerance in
`Engine._sync_get_memory()` at
`python/minisgl/engine/engine.py:246` is unchanged. Gate 4.10
did not modify `python/minisgl/`.

## 8. First failing stage

None — the driver reached `status=PASS` on both ranks without
recording a `failure_stage`. `failure_stage` is `null` on both
ranks; `failure_trace_summary` is `null` on both ranks.

## 9. Regression evidence

Per-file pytest on `tests/misc/` (headers only, hermetic per-file mode)
in the same working tree used for the smoke run:

```
tests/misc/test_scheduler_abort_ack.py             → 8/8  PASS  (15.38s)
tests/misc/test_scheduler_overlap_abort_fence.py   → 7/7  PASS  (15.17s)
tests/misc/test_scheduler_prepare_batch_txn.py     → 5/5  PASS  (14.70s)
tests/misc/test_engine_forward_sampler_atomic.py   → 5/5  PASS  (12.27s)
tests/misc/test_scheduler_shutdown_drain.py        → 8/8  PASS  (15.48s)
tests/misc/test_exposed_path_abort_ack.py          → 2/2  PASS  (15.11s)
tests/misc/test_shell_cancel_cleanup.py            → 2/2  PASS  (14.78s)
tests/misc/test_pyproject_config.py                → 14/14 PASS ( 0.04s)
```

Total: **51 / 51 PASS** in per-file (hermetic) mode. Every count
matches the last measurement at Gate 4.9. No test file was modified
by this gate.

## 10. Known limitations

* **B=2 ragged only.** Mixed-KV decode explicit evidence and
  dynamic admission on Qwen3-1.7B are out of scope. Gates 4.4 /
  4.5 / 4.6 proved those shapes on Qwen3-0.6B; Qwen3-1.7B is *not*
  proven for those shapes by this gate.
* **B > 2 not exercised.** Only B=2 tested.
* **`max_new_tokens=8`.** Longer decodes are not exercised.
* **Only the `cached_len == 0` ragged branch is proven.** The
  Gate 2.2f-documented "ragged + non-zero `cached_len` +
  `extend_len > 1`" branch is deliberately not exercised (single
  `generate()` per process).
* **Qwen3-1.7B only.** Qwen3-4B / 14B / 32B, quantized weights, and
  MoE variants are out of scope.
* **TP=2 only.** TP=4 / TP=8 not proven for Qwen3-1.7B by this gate.
* **Not a benchmark.** `generate_ms` is included in the JSONL for
  auditability only. It reflects the eager + npu_fia + bf16 +
  greedy path with a single fresh boot — not a stabilised
  throughput number, no warmup, no repeats. Do not quote it as
  performance.
* **`use_pynccl=False` is mandatory on NPU.** All numbers reflect
  the HCCL + gloo sidecar collective path.
* **1-page radix-cache retention** (`free_pages_after_case ==
  baseline_free_pages - 1`) is benign; `available_size` normalises
  back to baseline through `evictable_prefix_pages`. Same pattern
  as Gate 4.5.
* **Cold-start `_sync_get_memory()` variability** documented at
  Gate 4.9 remains outside `python/minisgl/`. The Gate 4.10 outer
  shell loop authorises up to 2 retries with fresh `master_port`
  and 8 s sleep. On this run only attempt 1 was required.

## 11. Decision matrix

| Question | Answer |
|---|---|
| Does Qwen3-1.7B `LLM.__init__` return on both ranks under TP=2? | Yes (attempt 1) |
| Does the two-shard weight load complete on both ranks? | Yes |
| Are `prompt_token_lengths[0] != prompt_token_lengths[1]`? | Yes (`2 != 12`) |
| Does `generate([short, long], max_tokens=8)` return `[8, 8]` tokens? | Yes on both ranks |
| Do rank 0 and rank 1 produce byte-identical `output_texts` per uid? | Yes for both uids |
| Do rank 0 and rank 1 produce byte-identical `output_token_ids` per uid? | Yes for both uids |
| Does the allocator `available_size` return to baseline on both ranks? | Yes (`930640 → 930640`) |
| Is the 1-page `free_pages` drift a leak? | No (retained evictable prefix; `available_size` absorbs it) |
| Are `deferred_abort_uids == 0` after the case? | Yes on both ranks |
| Does `check_integrity()` pass after the case? | Yes on both ranks |
| Is the driver B=2 ragged only (cached_len == 0 branch)? | Yes |
| Does the driver touch mixed-KV / dynamic admission / timing? | No |
| Does the driver modify `python/minisgl/`? | No |
| Does the driver modify tests? | No |
| Is `use_pynccl=False`? | Yes |
| Is the model Qwen3-1.7B only? | Yes |
| Is TP fixed at 2? | Yes |

**Verdict: PASS.**

## 12. Freeze boundary

The following files are the frozen artefacts for Gate 4.10:

* `scripts/gate4_10_qwen1_7b_tp2_b2_ragged_prefill.py`
* `docs/ascend_port/gate4.10_qwen1.7b_tp2_b2_ragged_prefill_verdict.md`

No files under `python/minisgl/` were modified at this gate. No
tests were modified at this gate. The freeze commit SHA is
recorded in this document header once the driver + verdict pair
is committed, and it is recorded on the `ascend-port` tip once the
branch is merged with `--no-ff`.
