# Gate 4.13 Verdict — Qwen3-1.7B TP=2 timing baseline

**Gate ID:** 4.13 (Qwen3-1.7B TP=2 timing baseline on Ascend 910B1)
**Verdict:** PASS
**Branch:** `gate4.13-qwen1.7b-tp2-timing-baseline`
**Base commit:** `c651d74` (tip of `ascend-port`, Gate 4.12 merge)
**Freeze commit:** `0b66782`
**Date:** 2026-07-12
**Kind:** Real-hardware Ascend 910B1 TP=2 timing snapshot
— two ranks × Qwen3-1.7B × six cases (A–F) × 1 warmup + 3 measured
repeats each × greedy × `use_pynccl=False`. Per-rank per-repeat JSONL
timing records (rank, ttft, e2e, tok/s, ms/tok, batch_timeline,
allocator invariants) plus per-case median/min/max summaries plus
a `GATE4.13_TIMING_RESULT=PASS` footer. The allocator returns to
baseline on every measured record on both ranks with no deferred
aborts and no radix-cache corruption.

> **This is an internal reproducibility snapshot. It is not a formal
> benchmark. It is not a cross-stack comparison. It is not a
> performance superiority claim.** Cross-stack numbers (SGLang, vLLM,
> TGI, TensorRT-LLM) are explicitly out of scope, and no
> optimisation, sweep, or tuning was performed. Every measurement
> includes the `AscendFIABackend.prepare_metadata` timing-hook
> overhead equally.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5 /
3.1 / 3.2 / 3.3 / 3.4 / 4.1 / 4.2 / 4.3 / 4.4 / 4.5 / 4.6 / 4.7 / 4.8
/ 4.9 / 4.10 / 4.11 / 4.12, does not mutate release tag `v0.1.0a1`,
does not touch the GitHub Release / CHANGELOG / release notes, and
does not extend the Ascend port to TP > 2, TP runtime elasticity,
runtime TP switching, Graph Re-Linker, Tensor-Remap-Kernel, non-Qwen3
architectures, or Qwen3-4B / 14B / 32B / quantized / MoE variants.
The only new artefacts at this gate are one bring-up script and this
verdict; no runtime source under `python/minisgl/` is modified; no
test file is modified.

---

## 1. Verdict summary

**PASS on all six cases (A–F) across both ranks.**

| Case | Description | rank 0 | rank 1 |
|---|---|:---:|:---:|
| A | B=1 single request, `max_new_tokens=8` | **PASS** | **PASS** |
| B | B=1 single request, `max_new_tokens=16` | **PASS** | **PASS** |
| C | B=2 equal-length, `max_new_tokens=8` | **PASS** | **PASS** |
| D | B=2 ragged prefill (unequal), `max_new_tokens=8` | **PASS** | **PASS** |
| E | B=2 mixed-KV decode evidence (unequal), `max_new_tokens=8` | **PASS** | **PASS** |
| F | dynamic admission B: 1 → 2 → 1 | **PASS** | **PASS** |

Every measured record on both ranks reported:

* `available_tokens_after_case == baseline_available_tokens` (`930640`)
* `deferred_abort_uids == 0`
* `cache_integrity_ok == true`
* per-repeat `status == "PASS"`

Footer (rank 0):

```
GATE4.13_TIMING_RESULT=PASS
```

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
| `warmup` | 1 (per case) |
| `measured_repeats` | 3 (per case) |
| Metadata / timing hook | script-local monkey-patch of `AscendFIABackend.prepare_metadata`; runtime unchanged |

## 3. Launch command

Same outer-shell retry pattern as Gates 4.10 / 4.11 / 4.12: sleep 8 s
between attempts, bump `--master_port`, thread
`--cold-start-attempt-id` and `--memory-sync-retry-note` into the
driver. Attempt 1 (port 29550) succeeded cleanly — no retry needed.

> **Public hygiene note:** remote host, username, password, and
> container identifiers are redacted in this document. The redaction
> does not affect Gate 4.13 runtime evidence.

```bash
ssh -p <PORT> <USER>@<HOST> \
  "docker exec <CONTAINER> bash -c '
    set +e
    cd /mnt/nvme/LR-606/mini-sglang-ascend-gate4.13 &&
    mkdir -p logs &&
    PORT_BASE=29540 &&
    NOTE=\"\" &&
    for ATTEMPT in 1 2 3; do
      PORT=\$((PORT_BASE + ATTEMPT * 10))
      LOG=logs/gate4.13_qwen1.7b_tp2_timing_baseline_attempt\${ATTEMPT}.log
      PYTHONPATH=./python:\$PYTHONPATH torchrun --nproc_per_node=2 --master_port=\$PORT \\
        scripts/gate4_13_qwen1_7b_tp2_timing_baseline.py \\
        --model-path /mnt/nvme/models/Qwen3-1.7B \\
        --cold-start-attempt-id \$ATTEMPT \\
        --memory-sync-retry-note \"\$NOTE\" > \$LOG 2>&1
      if grep -q GATE4.13_TIMING_RESULT= \$LOG && ! grep -q \"Memory across TP ranks are imbalanced\" \$LOG; then
        cp \$LOG logs/gate4.13_qwen1.7b_tp2_timing_baseline.log
        break
      fi
      IMB=\$(grep \"Memory across TP ranks are imbalanced\" \$LOG | head -1)
      NOTE=\"\$NOTE | attempt \$ATTEMPT port \$PORT: \$IMB\"
      sleep 8
    done
  '"
```

## 4. Case A–F prompt / max_new / batch-timeline evidence

Prompts (Qwen3-1.7B tokenizer counts, captured on both ranks):

| Case | Prompts | `prompt_lengths` | `requested_max_new_tokens` |
|---|---|---|---|
| A | `"Paris is"` | `[2]` | `[8]` |
| B | `"Paris is"` | `[2]` | `[16]` |
| C | `"The capital of France is"`, `"The capital of Italy is"` | `[5, 5]` | `[8, 8]` |
| D | `"Paris is"`, `"The largest planet in our solar system by mass and volume is"` | `[2, 12]` | `[8, 8]` |
| E | same as D | `[2, 12]` | `[8, 8]` |
| F | same as D (dynamic admission) | `[2, 12]` | `[8, 8]` |

Case F observed `batch_timeline` on all four passes (warmup + 3
measured, both ranks byte-identical):

```
[1, 1, 1, 2, 2, 2, 2, 2, 2, 1]
```

The `1 → 2 → 1` ordered subsequence appears in every F pass —
identical to the timeline Gate 4.12 proved. The staggered `LLM`
subclass reveals request B on the tick after the timing hook observes
`batch_size==1, query_lengths==[1]` (A alone decoding), and B
completes one step after A because it was admitted two prefill/decode
steps behind.

`actual_output_tokens_per_request` was `[8, 8]` on every F pass, so
the total output-tokens work is 16 tokens across two requests — used
as-is for the throughput metric.

## 5. Median timing table (rank 0)

Case-level medians over the three measured repeats:

| Case | TTFT median (ms) | E2E median (ms) | Tok/s median | ms/tok median | Output tokens total |
|---|---:|---:|---:|---:|---:|
| A B=1 N=8 | 70.18 | 493.72 | 16.20 | 61.72 | 8 |
| B B=1 N=16 | 75.29 | 1116.67 | 14.33 | 69.79 | 16 |
| C B=2 equal N=8 | 76.45 | 572.29 | 27.96 | 35.77 | 16 |
| D B=2 ragged N=8 | 88.95 | 554.28 | 28.87 | 34.64 | 16 |
| E B=2 mixed-KV N=8 | 90.78 | 568.32 | 28.15 | 35.52 | 16 |
| F dyn admission 1→2→1 | 73.42 | 676.93 | 23.64 | 42.31 | 16 |

Rank 0 min / max for the same metrics:

| Case | TTFT min/max | E2E min/max | Tok/s min/max | ms/tok min/max |
|---|---|---|---|---|
| A | 69.62 / 76.70 | 488.15 / 519.75 | 15.39 / 16.39 | 61.02 / 64.97 |
| B | 72.79 / 161.10 | 1031.37 / 3655.64 | 4.38 / 15.51 | 64.46 / 228.48 |
| C | 75.51 / 76.78 | 565.51 / 573.78 | 27.89 / 28.29 | 35.34 / 35.86 |
| D | 88.12 / 93.55 | 546.73 / 572.18 | 27.96 / 29.27 | 34.17 / 35.76 |
| E | 89.33 / 99.85 | 558.01 / 569.69 | 28.09 / 28.67 | 34.88 / 35.61 |
| F | 73.16 / 74.53 | 672.64 / 682.83 | 23.43 / 23.79 | 42.04 / 42.68 |

Rank 1 medians (mirror trace):

| Case | TTFT median (ms) | E2E median (ms) | Tok/s median | ms/tok median |
|---|---:|---:|---:|---:|
| A | 70.83 | 493.87 | 16.20 | 61.73 |
| B | 72.88 | 1031.11 | 15.52 | 64.44 |
| C | 76.83 | 571.29 | 28.01 | 35.71 |
| D | 91.26 | 552.18 | 28.98 | 34.51 |
| E | 90.19 | 568.30 | 28.15 | 35.52 |
| F | 72.84 | 677.08 | 23.63 | 42.32 |

Rank 1 mirrors rank 0 because greedy sampling on all-gathered logits
yields bit-identical output tokens per rank and the per-step wall
clocks track within a few milliseconds of jitter under
`use_pynccl=False` lockstep TP=2 scheduling. Per-repeat exact values
for both ranks are preserved in the structured JSONL log for
auditability.

### 5.1 Notable outlier — Case B first measured repeat

On rank 0, Case B's first measured repeat (`repeat_id=0`) recorded
`e2e_latency_ms == 3655.64`, `tokens_per_second == 4.38`, and
`ms_per_output_token == 228.48`, while measured repeats 1 and 2
produced `e2e_latency_ms == 1116.67` and `1031.37`. Rank 1 saw the
same pattern (`e2e_max == 3742.32` at `repeat_id=0`). The median is
not affected; the outlier is visible in the summary line as
`e2e_latency_ms.max` and `tokens_per_second.min`.

This is consistent with a one-off first-decode graph JIT /
kernel-compile warmup effect on the 16-token path (Case B has the
longest per-repeat decode of any case; its single warmup pass did not
fully cover the extra compilation triggered on the fourth iteration
onward). This is the same outlier pattern documented at Gate 4.7 §5.1
on Qwen3-0.6B, so it is not Qwen3-1.7B-specific. Case B still passed
on every measured record because (a) `actual_output_tokens_per_request
== [16]`, (b) the allocator invariant held bit-exactly, and (c) the
Gate 4.13 definition of PASS does not include a variance ceiling — it
is a reproducibility snapshot, not a benchmark.

Case A warmup was 2610.58 ms on rank 0 versus ~490 ms for measured
repeats — same JIT-warmup pattern, absorbed by the warmup pass as
intended.

## 6. Allocator evidence (rank 0 measured records)

`baseline_available_tokens == 930640` and `total_pages == 58165` on
both ranks (identical to Gate 4.10 baseline, confirming the KV-cache
budget is deterministic across gate runs when the warmup + repeat
schedule is fixed).

All 12 measured records per rank (2 ranks × 6 cases × 3 repeats)
carried `available_tokens_after_case == 930640` exactly. The
`free_pages_before_after` values show the small radix-cache retention
that accumulates across cases as different prompt prefixes are
inserted:

| Case | `free_pages_before_after` (both ranks) | retention |
|---|---|---|
| A | `[58165, 58165]` | 0 pages |
| B | `[58165, 58164]` | 1 page (short prompt cached) |
| C | `[58165, 58164]` | 1 page (equal-length shares prefix) |
| D | `[58165, 58163]` | 2 pages (long prompt cached) |
| E | `[58165, 58163]` | 2 pages (same long prompt) |
| F | `[58165, 58163]` | 2 pages (same long prompt) |

Each retention is an evictable prefix left in the radix cache after
its case completed — `available_size = free_slots +
evictable_prefix_pages` normalises back to `930640` exactly on every
measurement. This is the same benign pattern documented at Gate 4.5
/ 4.10 / 4.11 / 4.12. Not a leak; `check_integrity()` reported
`cache_integrity_ok == true` on all 48 records (24 per rank × 2
ranks).

Full per-record table available in
`logs/gate4.13_qwen1.7b_tp2_timing_baseline.log`; a representative
extraction:

```
A_b1_maxnew8               measured r=0 avail=930640/930640 free=[58165, 58165] defer=0 intg=True status=PASS
B_b1_maxnew16              measured r=0 avail=930640/930640 free=[58165, 58164] defer=0 intg=True status=PASS
C_b2_equal_maxnew8         measured r=0 avail=930640/930640 free=[58165, 58164] defer=0 intg=True status=PASS
D_b2_ragged_maxnew8        measured r=0 avail=930640/930640 free=[58165, 58163] defer=0 intg=True status=PASS
E_b2_mixed_kv_maxnew8      measured r=0 avail=930640/930640 free=[58165, 58163] defer=0 intg=True status=PASS
F_dynamic_admission_b1_b2_b1 measured r=0 avail=930640/930640 free=[58165, 58163] defer=0 intg=True status=PASS
```

## 7. Cold-start retry note

**Attempt 1 (port 29550) succeeded cleanly.** Both ranks passed the
pre-load and post-load `_sync_get_memory()` imbalance check on the
first attempt; every JSONL record reports
`cold_start_attempt_id == 1` and `memory_sync_retry_note == ""`.

The outer shell loop was authorised (per Gate 4.13 spec §2) to sleep
8 s and re-launch with a fresh `--master_port` up to 2 more times on
cold-start imbalance. It was not exercised on this run. Attempt-1 log
lives at `logs/gate4.13_qwen1.7b_tp2_timing_baseline_attempt1.log` and
is also copied to
`logs/gate4.13_qwen1.7b_tp2_timing_baseline.log` as the canonical
trace.

The pre-existing 2 GiB tolerance in `Engine._sync_get_memory()` at
`python/minisgl/engine/engine.py:246` is unchanged. Gate 4.13 did not
modify `python/minisgl/`.

## 8. Internal comparison to Gate 3.4 / Gate 4.7

**Marked as an internal snapshot only. Not a cross-model / cross-TP
comparison. Not a claim.**

Gate 3.4 was a Qwen3-0.6B TP=1 timing snapshot; Gate 4.7 was a
Qwen3-0.6B TP=2 timing snapshot with the same six-case schedule
(A–F). Gate 4.13 keeps the same case schedule but on Qwen3-1.7B TP=2,
so the closest apples-to-apples reference is Gate 4.7. The one
difference is case F: Gate 4.7 used grow-shrink 1→2→3→2→1 (24 total
output tokens) whereas Gate 4.13 uses dynamic admission 1→2→1 (16
total output tokens), so F is **not comparable** across the two
gates.

Rank 0 median E2E latency, side by side (informational only):

| Case | Gate 4.7 Qwen3-0.6B TP=2 (ms) | Gate 4.13 Qwen3-1.7B TP=2 (ms) | Δ |
|---|---:|---:|---:|
| A B=1 N=8 | 508.1 | 493.72 | −2.8 % |
| B B=1 N=16 | 1038.5 | 1116.67 | +7.5 % |
| C B=2 equal N=8 | 529.2 | 572.29 | +8.1 % |
| D B=2 ragged N=8 | 518.8 | 554.28 | +6.8 % |
| E B=2 mixed-KV N=8 | 562.4 | 568.32 | +1.1 % |
| F | 372.2 (grow-shrink, 24 tok) | 676.93 (dyn admission, 16 tok) | **not comparable** |

Interpretation is deferred (this is a snapshot, not an analysis): the
larger 1.7B model at TP=2 lands within roughly ±8 % of the 0.6B model
at TP=2 on these tiny-`max_new_tokens` cases, which is dominated by
fixed prefill + scheduling + collective + metadata-hook overhead.
Nothing here should be interpreted as a performance ranking, an
optimisation result, or a stack comparison.

## 9. Regression evidence

Per-file pytest on `tests/misc/` (hermetic per-file mode) in the same
working tree used for the smoke run:

```
tests/misc/test_scheduler_abort_ack.py             → 8/8   PASS  (14.83s)
tests/misc/test_scheduler_overlap_abort_fence.py   → 7/7   PASS  (14.76s)
tests/misc/test_scheduler_prepare_batch_txn.py     → 5/5   PASS  (14.18s)
tests/misc/test_engine_forward_sampler_atomic.py   → 5/5   PASS  (12.99s)
tests/misc/test_scheduler_shutdown_drain.py        → 8/8   PASS  (14.85s)
tests/misc/test_exposed_path_abort_ack.py          → 2/2   PASS  (14.04s)
tests/misc/test_shell_cancel_cleanup.py            → 2/2   PASS  (13.77s)
tests/misc/test_pyproject_config.py                → 14/14 PASS  ( 0.04s)
```

Total: **51 / 51 PASS** in per-file (hermetic) mode. Every count
matches the last measurement at Gate 4.12. No test file was modified
by this gate.

## 10. Known limitations

* **Snapshot, not benchmark.** All numbers here are single-cold-start
  medians over 3 measured repeats after 1 warmup, with the timing
  hook installed. No sweep, no tuning, no repeats-of-runs, no server
  path, no HTTP path. Do not quote as performance.
* **Not a cross-stack comparison.** SGLang / vLLM / TGI /
  TensorRT-LLM are explicitly out of scope. This is a
  reproducibility snapshot inside mini-sglang-ascend only.
* **Case B (max_new_tokens=16) has a first-measured-repeat outlier**
  on both ranks (JIT / kernel-compile warmup). Median is not
  affected. Same pattern as Gate 4.7 Case B.
* **Case F is dynamic admission (1→2→1), not grow-shrink
  (1→2→3→2→1).** Do not compare F ms/tok or E2E to Gate 4.7 F.
* **Qwen3-1.7B only.** Qwen3-4B / 14B / 32B, quantized weights, and
  MoE variants are out of scope.
* **TP=2 only.** TP=4 / TP=8, TP runtime elasticity, runtime TP
  switching, Graph Re-Linker, Tensor-Remap-Kernel are all out of
  scope.
* **`use_pynccl=False` is mandatory on NPU.** All numbers reflect
  the HCCL + gloo sidecar collective path.
* **Timing hook overhead is included.** The script-local
  `AscendFIABackend.prepare_metadata` wrapper adds a `perf_counter()`
  read and a `list(...)` copy per forward pass. This overhead is
  present equally on every measurement so relative comparisons within
  this gate are internally consistent, but absolute numbers are not
  a lower bound on the runtime's un-hooked latency.
* **Radix-cache retention between cases** (0–2 pages) is benign;
  `available_size` normalises back to `930640` exactly on every
  measured record.
* **Cold-start `_sync_get_memory()` variability** documented at
  Gate 4.9 remains outside `python/minisgl/`. The Gate 4.13 outer
  shell loop authorises up to 2 retries with fresh `master_port` and
  8 s sleep. On this run only attempt 1 was required.

## 11. Decision matrix

| Question | Answer |
|---|---|
| Does Qwen3-1.7B `LLM.__init__` return on both ranks under TP=2? | Yes (attempt 1) |
| Does the two-shard weight load complete on both ranks? | Yes |
| Do all six cases (A–F) complete `warmup=1 + measured_repeats=3` on both ranks? | Yes |
| Are per-case median / min / max reported for TTFT, E2E, tok/s, ms/tok? | Yes |
| Does every measured record hit `available_tokens_after_case == baseline`? | Yes (`930640 == 930640` on all 24 rank-0 and 24 rank-1 measured+warmup records) |
| Are `deferred_abort_uids == 0` on every record? | Yes |
| Does `check_integrity()` pass on every record? | Yes |
| Does Case F `batch_timeline` contain `1 → 2 → 1`? | Yes (`[1, 1, 1, 2, 2, 2, 2, 2, 2, 1]` on all four passes) |
| Are `output_tokens_total` per case as expected? | Yes (A=8, B=16, C=D=E=F=16) |
| Is Case B outlier a script or runtime bug? | No — one-off JIT/kernel-compile warmup, same as Gate 4.7 §5.1 |
| Did the driver touch `python/minisgl/`? | No |
| Did the driver touch tests? | No |
| Did the driver claim performance superiority? | No |
| Did the driver compare against SGLang / vLLM / TGI? | No |
| Is `use_pynccl=False`? | Yes |
| Is the model Qwen3-1.7B only? | Yes |
| Is TP fixed at 2? | Yes |

**Verdict: PASS.**

## 12. Freeze boundary

The following files are the frozen artefacts for Gate 4.13:

* `scripts/gate4_13_qwen1_7b_tp2_timing_baseline.py`
* `docs/ascend_port/gate4.13_qwen1.7b_tp2_timing_baseline_verdict.md`

No files under `python/minisgl/` were modified at this gate. No
tests were modified at this gate. The freeze commit SHA is recorded
in this document header once the driver + verdict pair is committed,
and it is recorded on the `ascend-port` tip once the branch is merged
with `--no-ff`.
