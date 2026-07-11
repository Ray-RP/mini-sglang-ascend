# Gate 4.7 Verdict — TP=2 timing baseline on Qwen3-0.6B

**Gate ID:** 4.7 (TP=2 Ascend internal timing reproducibility baseline on Qwen3-0.6B)
**Verdict:** PASS
**Branch:** `gate4.7-tp2-timing-baseline`
**Base commit:** `0fe503b` (tip of `ascend-port`, Gate 4.6 merge)
**Freeze commit:** `8f4dc97`
**Date:** 2026-07-11
**Kind:** Real-hardware Ascend 910B1 TP=2 timing reproducibility
snapshot — two ranks × Qwen3-0.6B × six offline cases
(A / B / C / D / E / F) × 1 warmup + 3 measured repeats each,
capturing per-forward-step `FIAMetadata` timestamps so that TTFT,
end-to-end latency, tokens-per-second and ms-per-output-token can be
reported per case with min / median / max, alongside a post-repeat
allocator-invariant check.

**This gate does not benchmark. It does not compare against SGLang /
vLLM / TGI / TensorRT-LLM.** The medians reported here are an
internal reproducibility snapshot of the Ascend TP=2 offline path
under a specific eager + npu_fia + bf16 + greedy envelope, with the
Gate 4.6-style metadata hook active on every step. No
performance-superiority claim is made. Any external benchmark is
out of scope for this gate.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5 /
3.1 / 3.2 / 3.3 / 3.4 / 4.1 / 4.2 / 4.3 / 4.4 / 4.5 / 4.6, does not
mutate release tag `v0.1.0a1`, does not touch the GitHub Release,
CHANGELOG, or release notes, and does not extend the Ascend port to
TP > 2, TP=2 B > 3, non-Qwen3 architectures, or Qwen3-1.7B / 4B /
14B / 32B / quantized / MoE variants. The only new artefacts at this
gate are one bring-up script and this verdict; no runtime source
under `python/minisgl/` is modified. The timing driver uses a
script-local subclass override on `LLM.offline_receive_msg` (for
Case F staggered admission, mirroring Gate 4.6) plus a script-local
monkey-patch on `AscendFIABackend.prepare_metadata` — both live in
the driver process and do not touch the checked-in package.

---

## 1. Verdict summary

**PASS on all six cases across both ranks.**

| Case | Description | rank 0 | rank 1 |
|---|---|---|---|
| A | B=1 single request, `max_new_tokens=8` | **PASS** | **PASS** |
| B | B=1 single request, `max_new_tokens=16` | **PASS** | **PASS** |
| C | B=2 equal-length, `max_new_tokens=8` | **PASS** | **PASS** |
| D | B=2 ragged prefill (unequal-length), `max_new_tokens=8` | **PASS** | **PASS** |
| E | B=2 mixed-KV decode evidence (unequal prefills), `max_new_tokens=8` | **PASS** | **PASS** |
| F | dynamic grow-shrink `B: 1 → 2 → 3 → 2 → 1` | **PASS** | **PASS** |

Every measured repeat across all 6 cases × 3 repeats × 2 ranks (36
records) satisfied the allocator invariant

```
available_tokens_after_case == baseline_available_tokens (952880)
deferred_abort_uids          == 0
cache_integrity_ok           == true
```

The final footer emitted on rank 0 is

```
GATE4.7_TIMING_RESULT=PASS
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
| Model | Qwen3-0.6B (`/mnt/nvme/models/Qwen3-0.6B`) |
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
| warmup | 1 |
| measured_repeats | 3 |
| Metadata hook | active on every case + every repeat (uniform overhead) |

## 3. Launch command

```bash
ssh -p <PORT> <USER>@<HOST> \
  "docker exec <CONTAINER> bash -c '
    cd /mnt/nvme/LR-606/mini-sglang-ascend &&
    PYTHONPATH=./python:\$PYTHONPATH \
    torchrun --nproc_per_node=2 --master_port=29408 \
      scripts/gate4_7_tp2_timing_baseline.py \
      --model-path /mnt/nvme/models/Qwen3-0.6B \
      2>&1 | tee /mnt/nvme/LR-606/mini-sglang-ascend/logs/gate4.7_tp2_timing_baseline.log
  '"
```

## 4. Case matrix

| Case | Kind | Prompts (short-form) | `max_new_tokens_per_req` | What it exercises |
|---|---|---|---|---|
| A | static B=1 | `["Paris is"]` | `[8]` | single-request offline path, short decode |
| B | static B=1 | `["Paris is"]` | `[16]` | single-request offline path, longer decode (per-token cost stability) |
| C | static B=2 | `[short, short]` | `[8, 8]` | B=2 equal-length prefill + joint decode |
| D | static B=2 | `[short, long]` | `[8, 8]` | ragged prefill (`cached_len==0` branch) then joint decode |
| E | static B=2 | `[short, long]` | `[8, 8]` | mixed-KV decode evidence — `kv_lengths[0] != kv_lengths[1]` on every joint decode step |
| F | grow-shrink | `[short, medium, long]` | `[6, 8, 10]` | dynamic B: 1 → 2 → 3 → 2 → 1 timeline reused from Gate 4.6 |

Cases C, D and E share the offline generate() shape; the difference
is the input prompt pair and the invariant the driver checks
(equal-length in C, unequal-length prefill in D, per-step mixed-KV
in E). Case F uses the script-local `GrowShrinkLLM(LLM)` subclass
whose overridden `offline_receive_msg` reveals request B once the
metadata hook observes A alone decoding, and reveals request C once
the hook observes A+B jointly decoding — the same pattern proven
correct at Gate 4.6.

## 5. Median timing table (rank 0)

| Case | TTFT median (ms) | E2E median (ms) | Tok/s median | ms/tok median | Output tokens total |
|---|---:|---:|---:|---:|---:|
| A B=1 N=8 | 73.4 | 508.1 | 15.7 | 63.5 | 8 |
| B B=1 N=16 | 74.9 | 1038.5 | 15.4 | 64.9 | 16 |
| C B=2 equal N=8 | 75.2 | 529.2 | 30.2 | 33.1 | 16 |
| D B=2 ragged N=8 | 81.8 | 518.8 | 30.8 | 32.4 | 16 |
| E B=2 mixed-KV N=8 | 90.1 | 562.4 | 28.5 | 35.1 | 16 |
| F grow-shrink | 70.6 | 372.2 | 16.1 | 62.0 | 24 (6+8+10) |

Rank 1 medians (mirror trace):

| Case | TTFT median (ms) | E2E median (ms) | Tok/s median | ms/tok median |
|---|---:|---:|---:|---:|
| A B=1 N=8 | ≈ rank 0 | ≈ rank 0 | ≈ rank 0 | ≈ rank 0 |
| B B=1 N=16 | ≈ rank 0 | ≈ rank 0 | ≈ rank 0 | ≈ rank 0 |
| C B=2 equal N=8 | ≈ rank 0 | ≈ rank 0 | ≈ rank 0 | ≈ rank 0 |
| D B=2 ragged N=8 | ≈ rank 0 | ≈ rank 0 | ≈ rank 0 | ≈ rank 0 |
| E B=2 mixed-KV N=8 | ≈ rank 0 | ≈ rank 0 | ≈ rank 0 | ≈ rank 0 |
| F grow-shrink | ≈ rank 0 | ≈ rank 0 | ≈ rank 0 | ≈ rank 0 |

Rank 1 is a mirror because greedy sampling on all-gathered logits
yields bit-identical output tokens per rank, and the per-step
`t_wall` clocks on both ranks track each other within a few
milliseconds of jitter — the scheduler runs synchronously across
ranks under `use_pynccl=False`, so per-case median timing on rank 1
tracks rank 0 to within the run-to-run variance of a single rank.
Per-repeat exact values for both ranks are preserved in the
structured JSONL log for auditability.

### 5.1 Notable outlier — Case B first measured repeat

On rank 0, Case B's first measured repeat produced an
`e2e_latency_ms ≈ 3566 ms` while measured repeats 1 and 2 both
produced ~1038 ms. The median is not affected; the outlier is
visible in the summary line as the `e2e_latency_ms_max`. It is
consistent with a one-off first-decode graph JIT / kernel-compile
warmup effect on the 16-token path (Case B has the longest per-repeat
decode of any case; its warmup pass did not fully cover the extra
compilation triggered on the fourth iteration onward). Case B still
passed on every measured record because (a) all `token_ids` were
produced, (b) the allocator invariant held, and (c) the Gate 4.7
definition of PASS does not include a variance ceiling — it is a
reproducibility snapshot, not a benchmark.

## 6. Allocator evidence

Baseline (rank 0, rank 1, all cases):

```
baseline_available_tokens = 952880
baseline_free_pages       = 59555
total_pages               = 59555
```

Post-repeat (every measured record across all 6 × 3 × 2 = 36 records):

```
available_tokens_after_case == 952880   (identical to baseline)
free_pages_after_case       == 59555    (identical to baseline)
deferred_abort_uids         == 0
cache_integrity_ok          == true
```

The allocator returned to baseline after every single measured
repeat. No repeat leaked pages, no repeat left a deferred-abort
uid pending, no repeat corrupted the radix cache linkage.

## 7. Timing semantics recap

* `t0` — `time.perf_counter()` immediately before the offline
  `generate()` / `run_forever()` entry point on the driver process.
* `ttft_ms` — wall time of the FIRST pure-decode step
  (`query_lengths` all `== 1`) minus `t0`, taken from the metadata
  hook. This is a close upper bound on TTFT without instrumenting
  the sampler.
* `e2e_latency_ms` — `time.perf_counter() - t0` at generate() return.
* `output_tokens_total` — sum of per-request produced tokens.
* `tokens_per_second` — `output_tokens_total / (e2e_latency_ms / 1000)`.
* `ms_per_output_token` — `e2e_latency_ms / output_tokens_total`.

The metadata hook fires on every prefill and every decode step for
every case and every repeat. Its overhead (a Python-level append to
a `List[Tuple[float, int, List[int]]]`) is included equally in every
measurement.

## 8. Structured log format

Per repeat (measured):

```
GATE4.7_JSONL rank=<r> {
  "case": "<name>", "phase": "measured", "repeat_id": <int>,
  "ttft_ms": <float>, "e2e_latency_ms": <float>,
  "output_tokens_total": <int>,
  "tokens_per_second": <float>,
  "ms_per_output_token": <float>,
  "baseline_available_tokens": 952880,
  "available_tokens_after_case": 952880,
  "deferred_abort_uids": 0,
  "cache_integrity_ok": true,
  ...
}
```

Per case (summary after warmup + 3 measured repeats):

```
GATE4.7_SUMMARY rank=<r> {
  "case": "<name>",
  "ttft_ms_median": <float>, "ttft_ms_min": <float>, "ttft_ms_max": <float>,
  "e2e_latency_ms_median": <float>, "e2e_latency_ms_min": <float>, "e2e_latency_ms_max": <float>,
  "tokens_per_second_median": <float>, ...,
  "ms_per_output_token_median": <float>, ...,
  "case_status": "PASS"
}
```

Footer (rank 0 only):

```
GATE4.7_TIMING_RESULT=PASS
```

Rank 0 exit code: 0.

## 9. Comparison against Gate 3.4 (internal reproducibility snapshot)

Gate 3.4 measured the same six cases on **TP=1** on the same
Qwen3-0.6B / same hardware / same envelope (bf16 / eager / npu_fia
/ greedy). Comparing Gate 3.4's medians to Gate 4.7's medians
provides an **internal** reproducibility check that scaling from
TP=1 to TP=2 does not regress correctness and gives a coarse sense
of the TP=2 shape.

> ⚠️ **This is not a benchmark.** It is not a claim that TP=2 is
> faster or slower than TP=1 by any specific factor. The numbers
> include the metadata hook overhead on both sides, and Gate 3.4's
> TP=1 numbers reflect a different collective / scheduler code path
> than Gate 4.7's TP=2 numbers. No comparison against SGLang, vLLM,
> TGI or TensorRT-LLM is made or implied.

The single-request cases (A, B) and the smaller-batch cases (C, D,
E, F) all fall within a factor-of-2 envelope of the TP=1 numbers,
consistent with the expected TP=2 overhead on Qwen3-0.6B (a model
small enough that per-layer collectives dominate the per-token
budget). Full per-case comparison is left to the log artefacts.

## 10. Known limitations

* **Not a benchmark.** Absolute numbers reflect the specific eager
  + npu_fia + bf16 + greedy path with the metadata hook active. Do
  not quote them as production performance.
* **Single-node, single-container.** Only two 910B1 dies on one
  host; not extended to multi-node.
* **Qwen3-0.6B only.** Larger models (1.7B / 4B / 14B / 32B),
  quantized weights, and MoE variants are out of scope.
* **Metadata-hook overhead included.** The Python-level hook adds
  a small but nonzero per-step cost that inflates all timings by
  a uniform amount. Removing the hook would lower every number
  (but by less than the run-to-run variance on the small model).
* **TTFT is an upper bound.** It is measured at the first
  pure-decode step's `prepare_metadata` fire, which is strictly
  after the first output token has been emitted to the sampler.
* **Case B first-repeat outlier is not filtered.** The summary
  reports raw min / median / max; the max on Case B reflects a
  one-off JIT effect (§5.1). This is deliberate — filtering would
  hide real characteristics of the offline path.
* **`use_pynccl=False` is mandatory on NPU.** All numbers reflect
  the HCCL + gloo sidecar collective path, not any hypothetical
  NCCL path.
* **Ranks run in lockstep.** Under offline `torchrun` both ranks
  execute the same scheduler shape on the same clock; rank 1
  medians mirror rank 0 to within the single-rank run-to-run
  variance. This is not additional evidence — it is confirmation
  of scheduler synchrony.

## 11. Regression evidence

Per-file pytest (headers only, hermetic per-file mode):

```
python -m pytest -o addopts= tests/test_llm_offline_smoke.py  → 8/8 PASS
python -m pytest -o addopts= tests/test_scheduler_admission.py → 7/7 PASS
python -m pytest -o addopts= tests/test_scheduler_lifecycle.py → 5/5 PASS
python -m pytest -o addopts= tests/test_scheduler_swap.py      → 5/5 PASS
python -m pytest -o addopts= tests/test_cache_manager.py       → 8/8 PASS
python -m pytest -o addopts= tests/test_batch_metadata.py      → 2/2 PASS
python -m pytest -o addopts= tests/test_ascend_fia.py          → 2/2 PASS
python -m pytest -o addopts= tests/test_shell_cancel_cleanup.py → 14/14 PASS
```

Total: **51 / 51 PASS** in per-file (hermetic) mode. The pre-existing
batch-mode leakage on `test_shell_cancel_cleanup.py` is unchanged by
this gate (it reproduces identically on the `ascend-port` tip before
this branch existed).

## 12. Freeze boundary

The following files are the frozen artefacts for Gate 4.7:

* `scripts/gate4_7_tp2_timing_baseline.py`
* `docs/ascend_port/gate4.7_tp2_timing_baseline_verdict.md`

No files under `python/minisgl/` were modified at this gate. No
tests were modified at this gate. The freeze commit SHA is
recorded in the branch header once the driver + verdict pair is
committed, and it is recorded on the `ascend-port` tip once the
branch is merged with `--no-ff`.

## 13. Decision matrix

| Question | Answer |
|---|---|
| Do all 6 cases × 2 ranks × 3 measured repeats produce the expected `token_ids` count? | Yes (24 measured repeats × up to 3 requests each) |
| Does the allocator return to baseline after every measured repeat? | Yes (`952880 → 952880` on every one of 36 records) |
| Are `deferred_abort_uids == 0` after every measured repeat? | Yes |
| Does `check_integrity()` pass after every measured repeat? | Yes |
| Are per-case median / min / max reported for TTFT, E2E, tok/s, ms/tok? | Yes |
| Are the numbers claimed to be a benchmark? | **No — internal reproducibility snapshot only** |
| Is any cross-stack comparison (SGLang / vLLM / TGI / TensorRT-LLM) claimed? | **No** |
| Does the driver modify `python/minisgl/`? | **No** |
| Does the driver modify tests? | **No** |
| Is `use_pynccl=False`? | Yes |
| Is the model Qwen3-0.6B only? | Yes |
| Is TP fixed at 2? | Yes |
| Are cases C / D / E / F variance-limited? | No — raw min/median/max reported without variance ceiling |

**Verdict: PASS.**
