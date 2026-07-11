# Gate 5.6 Verdict — Qwen3-4B Fixed-TP2 Timing Baseline

**Gate ID:** 5.6 (Qwen3-4B fixed-TP2 internal timing baseline snapshot on Ascend 910B1)
**Verdict:** PASS
**Branch:** `gate5.6-qwen4b-tp2-timing-baseline`
**Base commit:** `bb8e0fa` (tip of `ascend-port`, Gate 5.5a merge)
**Freeze commit:** `3b67fc3` (pre-amend; final SHA will be recorded on `ascend-port` tip via `--no-ff` merge; see §14)
**Date:** 2026-07-12
**Kind:** Internal reproducibility timing snapshot — Qwen3-4B on 2 ×
Ascend 910B1 under fixed TP=2, eager, `npu_fia`, bf16, greedy.
Records structured per-rank per-repeat JSONL for Cases A/B/C/D/E/F
with 1 warmup + 3 measured repeats each, plus per-case
median/min/max summary and a `GATE5.6_TIMING_RESULT=PASS`
footer. Neither runtime nor tests are touched.

> **This is fixed-TP2 Ascend adaptation.**
> **It is not TP elasticity.**
> **It is not TP switching.**
> **It is not a formal benchmark.**
> **It is not a cross-stack comparison.**
> **It is not a performance-superiority claim.**
> The timing numbers here are internal reproducibility numbers with
> the per-step metadata-snapshot hook active on every measured
> repeat. They are not compared against SGLang / vLLM / TGI. B > 2,
> TP > 2, TP elasticity, runtime TP switching, Graph Re-Linker,
> Tensor-Remap-Kernel, non-Qwen3 architectures, and Qwen3-14B / 32B
> / quantized / MoE variants are all out of scope. The Gate 2.2f
> documented "ragged + non-zero `cached_len` + `extend_len > 1`"
> unsupported FIA branch is intentionally avoided.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5 /
3.1 / 3.2 / 3.3 / 3.4 / 4.1 / 4.2 / 4.3 / 4.4 / 4.5 / 4.6 / 4.7 /
4.8 / 4.9 / 4.10 / 4.11 / 4.12 / 4.13 / 4.13a / 4.13b / 4.14 / 4.15
/ 4.15a / 5.1 / 5.2 / 5.3 / 5.4 / 5.4a / 5.5 / 5.5a, does not
mutate release tag `v0.1.0a1`, does not touch the GitHub Release /
CHANGELOG / release notes / README, and does not extend the Ascend
port beyond the fixed-TP2 Qwen3-4B timing-snapshot envelope. The
only new artefacts at this gate are one bring-up script and this
verdict; no runtime source under `python/minisgl/` is modified;
no test file is modified.

---

## 1. Verdict summary

**PASS on all six cases (A / B / C / D / E / F) across both TP
ranks. Every measured repeat produced timing metrics; every
allocator invariant held.**

| Case | Description | Rank 0 | Rank 1 |
|---|---|:---:|:---:|
| A | B=1 single request, `max_new_tokens=8` | **PASS** | **PASS** |
| B | B=1 single request, `max_new_tokens=16` | **PASS** | **PASS** |
| C | B=2 equal-length, `max_new_tokens=8` | **PASS** | **PASS** |
| D | B=2 ragged prefill (`[2, 12]`), `max_new_tokens=8` | **PASS** | **PASS** |
| E | B=2 mixed-KV decode, `max_new_tokens=8` | **PASS** | **PASS** |
| F | Dynamic admission B: 1 → 2 → 1 | **PASS** | **PASS** |

Every measured repeat on both ranks reported:

* `warmup_count == 1`, `measured_repeats == 3`
* `actual_output_tokens_per_request == requested_max_new_tokens`
* `available_tokens_after_case == baseline_available_tokens`
* `deferred_abort_uids == 0`
* `cache_integrity_ok == true`
* `status == "PASS"`

Footer:

```
GATE5.6_TIMING_RESULT=PASS
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
| Model | Qwen3-4B (`/mnt/nvme/models/Qwen3-4B`) |
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
| Warmup per case | 1 |
| Measured repeats per case | 3 |

## 3. Model path check

`/mnt/nvme/models/Qwen3-4B` present on the validation host
container. All measured JSONL records logged `model_path` == the
same path on both ranks. The on-disk config (`Qwen3ForCausalLM`,
`hidden_size=2560`, `num_hidden_layers=36`,
`num_attention_heads=32`, `num_key_value_heads=8` (GQA 32/8),
`head_dim=128`, `intermediate_size=9728`, `vocab_size=151936`,
`tie_word_embeddings=true`, `bf16`) was already documented at
Gate 5.1 §4 and is unchanged for this gate.

## 4. Launch command

Same outer-shell retry pattern as Gates 4.10 / 4.11 / 4.12 / 4.13 /
4.14 / 5.1 / 5.2 / 5.3 / 5.4 / 5.5: sleep 8 s between attempts,
bump `--master_port` (PORT_BASE=29900 + ATTEMPT*10), thread
`--cold-start-attempt-id` and `--memory-sync-retry-note` into the
driver. **Attempt 1 (port 29910) succeeded cleanly — no retry
needed.**

> **Public hygiene note:** remote host, username, password, and
> container identifiers are redacted in this document.

```bash
ssh -p <PORT> <USER>@<HOST> \
  "docker exec <CONTAINER> bash -c '
    set +e
    cd <REMOTE_PATH>/mini-sglang-ascend-gate5.6 &&
    mkdir -p logs &&
    PORT_BASE=29900
    NOTE=""
    for ATTEMPT in 1 2 3; do
      PORT=$((PORT_BASE + ATTEMPT * 10))
      LOG=logs/gate5.6_Qwen3-4B_attempt${ATTEMPT}.log
      PYTHONPATH=./python:$PYTHONPATH torchrun --nproc_per_node=2 --master_port=$PORT \
        scripts/gate5_6_qwen4b_tp2_timing_baseline.py \
        --model-path /mnt/nvme/models/Qwen3-4B \
        --cold-start-attempt-id $ATTEMPT \
        --memory-sync-retry-note "$NOTE" > $LOG 2>&1
      if grep -q GATE5.6_TIMING_RESULT= $LOG && ! grep -q "Memory across TP ranks are imbalanced" $LOG; then
        cp $LOG logs/gate5.6_Qwen3-4B.log
        break
      fi
      IMB=$(grep "Memory across TP ranks are imbalanced" $LOG | head -1)
      NOTE="$NOTE | attempt $ATTEMPT port $PORT: $IMB"
      sleep 8
    done
  '"
```

## 5. Case matrix

| Case | Kind | Prompts (tokenized lengths) | `max_new_tokens` |
|---|---|---|---|
| A | `static_b1` | `"Paris is"` (2) | `[8]` |
| B | `static_b1` | `"Paris is"` (2) | `[16]` |
| C | `static_b2` | `"The capital of France is"` (5), `"The capital of Italy is"` (5) | `[8, 8]` |
| D | `static_b2` | `"Paris is"` (2), `"The largest planet in our solar system by mass and volume is"` (12) | `[8, 8]` |
| E | `static_b2` | `"Paris is"` (2), `"The largest planet in our solar system by mass and volume is"` (12) | `[8, 8]` |
| F | `dynamic_admission` | `"Paris is"` (2), `"The largest planet in our solar system by mass and volume is"` (12) | `[8, 8]` |

Case E reuses D's prompt pair but is scored against the mixed-KV
invariant (`kv_lengths[0] != kv_lengths[1]` on every decode step)
documented at Gate 5.4 §6. Case F uses the same
`StaggeredLLM(LLM)` staged-admission pattern documented at
Gate 5.5 §6 — request B is revealed only after the timing hook
observes A alone decoding (`batch_size == 1`,
`query_lengths == [1]`).

## 6. Timing summary (median over 3 measured repeats)

Numbers below are the exact `GATE5.6_SUMMARY` payloads emitted per
rank from `logs/gate5.6_Qwen3-4B.log`.

### Rank 0

| Case | `ttft_ms` median | `e2e_latency_ms` median | `tokens_per_second` median | `ms_per_output_token` median |
|---|---:|---:|---:|---:|
| A | 92.70 | 650.13 | 12.31 | 81.27 |
| B | 101.51 | 1379.05 | 11.60 | 86.19 |
| C | 93.48 | 696.29 | 22.98 | 43.52 |
| D | 111.05 | 702.48 | 22.78 | 43.90 |
| E | 119.62 | 693.57 | 23.07 | 43.35 |
| F | 88.88 | 850.82 | 18.81 | 53.18 |

### Rank 1

| Case | `ttft_ms` median | `e2e_latency_ms` median | `tokens_per_second` median | `ms_per_output_token` median |
|---|---:|---:|---:|---:|
| A | 92.48 | 650.15 | 12.30 | 81.27 |
| B | 93.15 | 1279.66 | 12.50 | 79.98 |
| C | 92.85 | 696.59 | 22.97 | 43.54 |
| D | 111.79 | 702.95 | 22.76 | 43.93 |
| E | 119.99 | 693.41 | 23.07 | 43.34 |
| F | 88.74 | 851.39 | 18.79 | 53.21 |

Notes on the numbers:

* **Case B has a wide range on `e2e_latency_ms`** (min ≈ 1235 ms,
  max ≈ 3833 ms on rank 1; min ≈ 1235 ms, max ≈ 3734 ms on rank 0).
  One measured repeat on each rank paid an outlier ≈ 2.5 s spike
  that inflates the max; the median (≈ 1.28–1.38 s) is on the
  expected 16-token scale-up from Case A's 8-token e2e. Gate 5.6
  makes no claim of steady-state or outlier-free behaviour — this
  is a functional snapshot, not a statistical characterisation.
  The rank-0 median is 1379.05 ms and rank-1 median is 1279.66 ms
  (both consistent with 2× Case A e2e); the per-repeat records in
  `logs/gate5.6_Qwen3-4B.log` disclose the outlier explicitly.
* **Rank 0 vs rank 1 medians agree to within ±10 % on every
  metric except Case B `e2e_latency_ms`** (99 ms gap on the
  median, dominated by the outlier). HCCL lock-step keeps the
  arithmetic identical; the small median deltas reflect
  perf-counter jitter, not divergent work.
* **`tokens_per_second`** is `output_tokens_total / (e2e / 1000)`.
  Case A / B produce 8 / 16 tokens for one request; Case C–F
  produce 16 tokens across two requests. The two-request cases
  therefore report ≈ 2× the tokens/s of the single-request cases
  under the same wall-clock envelope.
* Every timing number includes the metadata-snapshot hook overhead.
  The hook is script-local and identical across all repeats, so
  its overhead is a constant additive term across the table.

## 7. Per-rank allocator evidence

Baseline captured after weight load, before Case A warmup. Every
measured-repeat record on both ranks logged
`available_tokens_after_case == baseline_available_tokens ==
689744` and `deferred_abort_uids == 0` and `cache_integrity_ok ==
true`.

| Field | Rank 0 | Rank 1 |
|---|---:|---:|
| `baseline_available_tokens` | 689744 | 689744 |
| `baseline_free_pages` | 43109 | 43109 |
| `available_tokens_after_case` (every measured repeat, all 6 cases) | 689744 | 689744 |
| `free_pages_after_case` (every measured repeat, all 6 cases) | 43107 | 43107 |
| `deferred_abort_uids` | 0 | 0 |
| `cache_integrity_ok` | true | true |

Baseline (`689744` / `43109`) matches Gates 5.1 / 5.4 exactly. The
2-page reduction in `free_pages_after_case` (43107 vs baseline
43109) is retained evictable radix-cache prefix for the pool of
distinct prompt strings the six cases traverse (Case A/B `"Paris
is"`, Case C's two equal-length prompts, Case D/E/F long prompt).
`available_size = free_slots + evictable_prefix_pages` absorbs the
retention, so `available_tokens_after_case == baseline` holds
exactly on every repeat and every case. Same benign pattern
documented at Gates 4.5 / 4.10 / 4.11 / 4.12 / 4.13 / 4.14 / 5.3 /
5.4 / 5.5. `check_integrity()` confirms radix-cache linkage; not a
page leak.

## 8. Cold-start retry note

**Attempt 1 succeeded cleanly.** The run did not trip the
pre/post-load `_sync_get_memory()` imbalance check (2 GiB tolerance
at `python/minisgl/engine/engine.py:246`, unchanged since Gate
4.14). Every JSONL record on both ranks logged
`cold_start_attempt_id == 1` and `memory_sync_retry_note == ""`.

The outer shell loop was authorised (per Gate 5.6 spec §5,
matching Gates 5.1 / 5.2 / 5.3 / 5.4 / 5.5) to sleep 8 s and
re-launch with a fresh `--master_port` up to 2 more times on
cold-start imbalance. It was not exercised on this run.

## 9. Public-hygiene grep summary

Post-authoring verification against the six known leaked substring
classes documented at Gate 4.13a / 4.13b (host / user / password /
container / IP / composite):

```
$ git grep -l -E "<OLD_SSHPASS_PATTERN>|<OLD_PASSWORD_PATTERN>|<OLD_HOST_PATTERN>|<OLD_CONTAINER_PATTERN>" \
    scripts/gate5_6_qwen4b_tp2_timing_baseline.py \
    docs/ascend_port/gate5.6_qwen4b_tp2_timing_baseline_verdict.md
(no output — zero files)
```

Loopback `127.0.0.1` and bind `0.0.0.0` do not appear in either
artefact. All host references in the verdict use placeholders
(`<HOST>`, `<PORT>`, `<USER>`, `<CONTAINER>`, `<REMOTE_PATH>`).

## 10. Regression evidence

Optional pytest per Gate 5.6 spec §9 was skipped — Gate 5.6
modifies zero runtime and zero test files. The Gate 4.13
regression measurement of `51 / 51 PASS` per-file pytest on
`tests/misc/` is unchanged by construction.

`git diff --check` on the freeze branch tip: clean.

## 11. Relationship to Gate 4.13

Gate 4.13 recorded the same six-case timing snapshot for Qwen3-1.7B
on the identical envelope (2 × Ascend 910B1, TP=2, `npu_fia`,
eager, bf16, greedy, warmup=1, measured_repeats=3). Gate 5.6
reuses the driver structure verbatim (script-local metadata hook,
`StaggeredLLM` subclass for Case F, per-repeat allocator invariant
check, result-flag PASS / PARTIAL / BLOCKED degradation) and swaps
only:

* Model path default → `/mnt/nvme/models/Qwen3-4B`
* Output tags → `GATE5.6_JSONL` / `GATE5.6_SUMMARY` /
  `GATE5.6_TIMING_RESULT`
* Docstring / description strings

Gate 5.6's numbers are **not** a Qwen3-4B vs Qwen3-1.7B comparison
and **not** part of any speedup / superiority claim. They are two
independent internal reproducibility snapshots on the same rig,
recorded to prove each model separately runs cleanly through the
same six-case matrix under the same functional envelope.

## 12. Known limitations

* **Fixed-TP2, single-model, six pre-set cases only.** Qwen3-4B
  has been proven at B=1 (Gate 5.1), B=2 equal-length (Gate 5.2),
  B=2 ragged prefill (Gate 5.3), B=2 mixed-KV decode (Gate 5.4),
  dynamic admission B: 1→2→1 (Gate 5.5), and now the fixed six-case
  timing snapshot (this gate). B > 2, TP > 2, TP elasticity,
  runtime TP switching, and other prompt / dynamic-admission
  shapes are **not** covered.
* **Not a benchmark.** No throughput, latency, or steady-state
  claim is made. No cross-stack comparison to SGLang / vLLM / TGI
  is made or implied. The metadata-snapshot hook adds
  unbudgeted per-step overhead on every measured repeat.
* **Case B outliers.** One measured repeat on each rank paid a
  ≈ 2.5 s spike on the 16-token single-request e2e; both are
  disclosed in the per-repeat JSONL and preserved verbatim in the
  min/max fields of the Case B summary. Gate 5.6 makes no
  steady-state or outlier-free claim.
* **Timing is captured under the hook.** Both the timing hook and
  the (Case F only) `StaggeredLLM.offline_receive_msg` override
  are script-local. Neither mutates runtime state; batch_timeline
  and `actual_output_tokens_per_request` are byte-identical to
  Gates 5.3 / 5.4 / 5.5 for the same prompt pairs, confirming
  non-perturbation.
* **TP=2 only.** TP=4 / TP=8, TP runtime elasticity, runtime TP
  switching, Graph Re-Linker, Tensor-Remap-Kernel are all out of
  scope.
* **One Qwen3 dense model.** Qwen3-14B, Qwen3-32B, quantized
  weights, and Qwen3 MoE variants are not part of this gate.
* **`use_pynccl=False` is mandatory on NPU.** The all-gather /
  all-reduce path exercised here is the HCCL + gloo sidecar
  collective path.
* **Radix-cache retention (2 pages) between baseline and post-case
  `free_pages`** is benign; `available_size` normalises back to
  baseline exactly on every measured repeat.
* **Cold-start `_sync_get_memory()` variability** documented at
  Gate 4.9 remains outside `python/minisgl/`. The Gate 5.6
  outer shell loop authorises up to 2 retries with fresh
  `master_port` and 8 s sleep. On this run only attempt 1 was
  required.

## 13. Decision matrix

| Question | Answer |
|---|---|
| Does `/mnt/nvme/models/Qwen3-4B` exist on the validation host? | Yes |
| Does `LLM.__init__` return on both ranks under TP=2? | Yes (attempt 1) |
| Does weight load succeed on both ranks under TP=2? | Yes |
| Does post-load `check_integrity()` pass on both ranks? | Yes |
| Did each of the 6 cases complete `warmup=1` and `measured_repeats=3`? | Yes on both ranks |
| Did every measured repeat record `ttft_ms`, `e2e_latency_ms`, `tokens_per_second`, `ms_per_output_token`? | Yes |
| Did every measured repeat produce `actual_output_tokens_per_request == requested_max_new_tokens`? | Yes |
| Did `available_tokens_after_case == baseline` hold on every measured repeat? | Yes on both ranks |
| Was `deferred_abort_uids == 0` on every measured repeat? | Yes on both ranks |
| Did `check_integrity()` pass on every measured repeat? | Yes on both ranks |
| Did the driver touch `python/minisgl/`? | No |
| Did the driver touch tests? | No |
| Did the driver claim performance superiority? | No |
| Did the driver compare against SGLang / vLLM / TGI? | No |
| Is `use_pynccl=False`? | Yes |
| Is the model limited to Qwen3-4B? | Yes |
| Is TP fixed at 2? | Yes |
| Are `warmup=1` / `measured_repeats=3` the only settings recorded? | Yes |

**Verdict: PASS.**

## 14. Freeze boundary

The frozen artefacts for Gate 5.6 are:

* `scripts/gate5_6_qwen4b_tp2_timing_baseline.py`
* `docs/ascend_port/gate5.6_qwen4b_tp2_timing_baseline_verdict.md`

No files under `python/minisgl/` were modified at this gate. No
tests were modified at this gate. The freeze commit SHA is
recorded in this document header once the driver + verdict pair is
committed, and it is recorded on the `ascend-port` tip once the
branch is merged with `--no-ff`.
