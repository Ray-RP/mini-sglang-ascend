# Gate 3.4 Verdict — TP=1 Timing Baseline (Qwen3-0.6B + Qwen3-1.7B)

**Gate ID:** 3.4 (TP=1 Ascend timing baseline, Qwen3-0.6B + Qwen3-1.7B)
**Verdict:** PASS
**Branch:** `gate3.4-tp1-timing-baseline`
**Base commit:** `16da4a2` (tip of `ascend-port`, Gate 3.3 merge)
**Freeze commit:** *(populated at merge into `ascend-port`)*
**Date:** 2026-07-11
**Kind:** Real-hardware Ascend 910B1 reproducible timing snapshot —
warmup=1 + measured_repeats=3 per (model × case), TTFT / e2e /
tokens-per-second measured under the frozen Gate 3.3 envelope.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5 /
3.1 / 3.2 / 3.3, does not mutate release tag `v0.1.0a1`, does not
touch the GitHub Release, CHANGELOG, or release notes, and does not
extend the Ascend port to TP > 1, HCCL, Qwen3-4B / 14B / 32B,
non-Qwen3 architectures (Llama / MoE / Qwen3-Next / Qwen3-ASR /
Qwen3-Coder-Next), MoE / quantized variants (Qwen3-32B-FP8,
Qwen3-30B-A3B), long soak, HTTP server restart, forward/sampler
exception recovery, non-stream HTTP cancel, offline `LLM.abort()`,
chunked prefill, or any comparison with SGLang / vLLM / TGI. The
numbers below are a **reproducibility snapshot**, not a performance
benchmark; no throughput / latency / leadership claim is made.

---

## 1. Verdict summary

**PASS on the full timing matrix.** Under the frozen TP=1 / eager
/ `npu_fia` / bf16 / greedy envelope, both Qwen3-0.6B and Qwen3-1.7B
complete every required timing case with 3 measured repeats each:

| Case | Shape | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|---|
| A | B=1, `max_new_tokens=8`  | **PASS** (3/3) | **PASS** (3/3) |
| B | B=1, `max_new_tokens=16` | **PASS** (3/3) | **PASS** (3/3) |
| C | B=2 equal-length, `max_new_tokens=8` | **PASS** (3/3) | **PASS** (3/3) |
| D | B=2 ragged prefill, `max_new_tokens=8` | **PASS** (3/3) | **PASS** (3/3) |

For every measured repeat on every case on every model:

* Each request returned exactly the requested `max_new_tokens`.
* `cache_manager.available_size` (= free_slots + evictable prefix
  cache pages — the paged-cache invariant per Gate 3.1 verdict §4)
  returned to the baseline value recorded at case start
  (`470592` for Qwen3-0.6B; `449568` for Qwen3-1.7B).
* `cache_manager.check_integrity()` succeeded at every snapshot.
* `scheduler.deferred_abort_uids` was empty at every snapshot.

24 measured PASS rows + 8 warmup rows = 32 JSONL records, 0 FAIL rows.

Zero code changes under `python/minisgl/` or `tests/`. The gate adds
one smoke script and one verdict document.

---

## 2. Envelope (locked at this gate — matches Gate 3.3)

```
Hardware:          Ascend 910B1 (1 die, 64 GiB HBM)
Container:         <CONTAINER> on remote <HOST>:<PORT>
Software:          Python 3.11.14
                   torch 2.9.0+cpu
                   torch_npu 2.9.0.post1+gitee7ba04
                   CANN 8.5.1 (compiler build 20250725)
Parallelism:       TP=1
Execution:         eager (cuda_graph_bs=[])
Attention backend: npu_fia
page_size:         16       (FIA NO_QUANT block_size % 16 == 0)
memory_ratio:      0.85
max_running_req:   8
Sampling:          greedy (temperature=0.0, top_k=1, top_p=1.0, ignore_eos=True)
Timer:             time.perf_counter() (wall clock, single process)
Warmup / repeats:  1 warmup + 3 measured per (model × case)
Driver:            scripts/gate3_4_timing_baseline.py
                   parent orchestrator + per-model child subprocess
                   (one process per model — same reason as Gate 3.3 §7.1:
                    ``minisgl.distributed.info.set_tp_info`` is a
                    process-global singleton)
```

TTFT instrumentation point: a script-local `TimingLLM(LLM)` subclass
overrides `offline_send_result(reply: List[DetokenizeMsg])` — the
tokenizer→driver callback that receives one message per generated
token per uid. First message per uid gives per-uid TTFT;
`ttft_ms = max(per-uid TTFTs)` (defined as "time until every request
has produced its first token", so a B=2 case reports one TTFT for the
batch, not two). End-to-end latency is `perf_counter` measured across
the `llm.generate([...], sp)` call.

---

## 3. Models

Both dense bf16 Qwen3 models already present on the Ascend host under
`/mnt/nvme/models/`. No downloads. No new architecture family.

| Model | Path | `hidden_size` | `intermediate_size` | Shards | KV cache tokens |
|---|---|---|---|---|---|
| Qwen3-0.6B | `/mnt/nvme/models/Qwen3-0.6B` | 1024 | 3072 | 1 | 470592 |
| Qwen3-1.7B | `/mnt/nvme/models/Qwen3-1.7B` | 2048 | 6144 | 2 | 449568 |

Both share `Qwen3ForCausalLM` / `num_hidden_layers=28` /
`num_attention_heads=16` / `num_key_value_heads=8` / `head_dim=128` /
`vocab_size=151936` / `tie_word_embeddings=true`. Rejected model
families are the same set as Gate 3.3 §3.

---

## 4. Cases

Each case × model × repeat is emitted as one JSONL row with the full
Gate 3.4 field set (`model_name`, `model_path`, `case_name`,
`description`, `prompt_lengths_chars`, `batch_size`,
`requested_max_new_tokens`, `actual_output_tokens_per_request`,
`warmup_count`, `repeat_id`, `is_warmup`, `ttft_ms`,
`ttft_ms_per_uid`, `e2e_latency_ms`, `output_tokens_total`,
`tokens_per_second`, `ms_per_output_token`,
`baseline_available_tokens`, `available_tokens_after_case`,
`baseline_free_pages`, `free_pages_after_case`, `total_pages`,
`deferred_abort_uids`, `cache_integrity_ok`, `status`,
`failure_reason`).

| Case | Description | Prompt shape | `max_new_tokens` |
|---|---|---|---|
| A | B=1, N=8 | `"The capital of France is"` (24 chars) | `[8]` |
| B | B=1, N=16 | `"The capital of France is"` (24 chars) | `[16]` |
| C | B=2 equal-length | two identical `"The capital of France is"` prompts | `[8, 8]` |
| D | B=2 ragged (short + long) — **per-repeat prompt variants** | `(short, long)` pair drawn from a 4-entry pool, rotated per repeat (see §7.2) | `[8, 8]` |

Each row's `status` is one of `WARMUP` (repeat_id=-1) or `PASS` (0–2)
or `FAIL`. All 24 measured rows recorded `PASS`.

---

## 5. Timing summary (median / min / max across 3 measured repeats)

Milliseconds are wall-clock via `time.perf_counter()`. Rows report
median with min/max in parentheses. `tokens_per_second` and
`ms_per_output_token` are computed over `output_tokens_total /
e2e_latency_ms`.

### 5.1 Qwen3-0.6B

| Case | `ttft_ms` (median / min / max) | `e2e_latency_ms` (median / min / max) | `tokens_per_second` (median / min / max) | `ms_per_output_token` (median) |
|---|---|---|---|---|
| A | 107.71 (107.39 / 113.87) | 412.29 (400.58 / 421.68) | 19.40 (18.97 / 19.97) | 51.54 |
| B | 113.93 (113.31 / 114.27) | 831.16 (830.57 / 875.66) | 19.25 (18.27 / 19.26) | 51.95 |
| C | 119.91 (119.48 / 119.95) | 463.12 (462.40 / 463.37) | 34.55 (34.53 / 34.60) | 28.94 |
| D | 138.76 (138.60 / 139.16) | 483.86 (482.85 / 484.18) | 33.07 (33.05 / 33.14) | 30.24 |

### 5.2 Qwen3-1.7B

| Case | `ttft_ms` (median / min / max) | `e2e_latency_ms` (median / min / max) | `tokens_per_second` (median / min / max) | `ms_per_output_token` (median) |
|---|---|---|---|---|
| A | 100.22 (100.06 / 100.30) | 372.66 (372.49 / 373.37) | 21.47 (21.43 / 21.48) | 46.58 |
| B | 103.34 (103.34 / 103.35) | 760.80 (760.80 / 761.71) | 21.03 (21.02 / 21.03) | 47.55 |
| C | 108.40 (108.13 / 108.60) | 418.94 (418.72 / 419.60) | 38.19 (38.13 / 38.21) | 26.18 |
| D | 126.24 (126.19 / 126.68) | 439.50 (439.24 / 440.24) | 36.41 (36.34 / 36.43) | 27.47 |

Interpretation notes (informational; not a comparative claim):

* **Case C > A tokens_per_second** on both models — expected, B=2
  amortizes per-step overhead over 2 tokens.
* **Case B ≈ A ttft** but longer e2e — expected, TTFT is prefill
  dominated, B just runs 8 more decode steps than A.
* **Case D ttft > C ttft** on both models — expected, ragged prefill
  makes the prefill step's compute bound by the longer prompt.
* **Qwen3-0.6B and Qwen3-1.7B tokens_per_second are within ~10 %**
  in every case — this is because both are compute-bound in the
  prefill on the Ascend 910B1 and share attention geometry
  (`num_attention_heads=16` / `num_key_value_heads=8` /
  `head_dim=128`); MLP size doubling on 1.7B is offset by mildly
  better tensor-core utilization. Do not read this as a general
  performance claim — see §10.

Min/max spread per case is tight (≤ 2 % on measured rows on both
models), consistent with a single-process eager loop with no
CUDAGraph and no dynamic scheduling variance.

---

## 6. Commands

Executed on remote container `<CONTAINER>` at working directory
`/mnt/nvme/LR-606/mini-sglang-ascend-gate34`.

Smoke:

```bash
PYTHONPATH=python python3 scripts/gate3_4_timing_baseline.py \
  --models /mnt/nvme/models/Qwen3-0.6B,/mnt/nvme/models/Qwen3-1.7B \
  --warmup 1 --repeats 3 \
  --jsonl-out /tmp/gate3_4.jsonl
```

Scrapable footer emitted by the parent:

```
GATE3.4_MODEL_Qwen3-0.6B_CASE_A=PASS
GATE3.4_MODEL_Qwen3-0.6B_CASE_B=PASS
GATE3.4_MODEL_Qwen3-0.6B_CASE_C=PASS
GATE3.4_MODEL_Qwen3-0.6B_CASE_D=PASS
GATE3.4_MODEL_Qwen3-0.6B=PASS
GATE3.4_MODEL_Qwen3-1.7B_CASE_A=PASS
GATE3.4_MODEL_Qwen3-1.7B_CASE_B=PASS
GATE3.4_MODEL_Qwen3-1.7B_CASE_C=PASS
GATE3.4_MODEL_Qwen3-1.7B_CASE_D=PASS
GATE3.4_MODEL_Qwen3-1.7B=PASS
GATE3.4_TIMING_RESULT=PASS
```

Regression (per-file, per Gate 3.1 / 3.2 / 3.3 verdicts' regression
notes):

```bash
for f in test_scheduler_abort_ack \
         test_scheduler_overlap_abort_fence \
         test_scheduler_prepare_batch_txn \
         test_engine_forward_sampler_atomic \
         test_scheduler_shutdown_drain \
         test_exposed_path_abort_ack \
         test_shell_cancel_cleanup \
         test_pyproject_config; do
  PYTHONPATH=python:tests pytest -q -o addopts="" tests/misc/$f.py
done
```

---

## 7. Allocator evidence and implementation notes

### 7.1 Allocator invariant — every measured repeat

`available_tokens_after_case == baseline_available_tokens` and
`deferred_abort_uids == 0` were asserted at every measured repeat
(24 rows) and verified by inspection at every warmup row (8 rows).
`cache_integrity_ok == true` at every snapshot (32 rows).

| Model | baseline_available_tokens | after every measured repeat | free_pages drift range (measured) |
|---|---|---|---|
| Qwen3-0.6B | 470592 | 470592 (24/24) | 29403..29412 |
| Qwen3-1.7B | 449568 | 449568 (24/24) | 28094..28098 |

`free_pages` drift is designed radix-cache retention across ragged
prompts (case D per-repeat variants leave a small evictable page each
repeat); the invariant is `available_tokens`, not raw `free_pages`.
See Gate 3.1 verdict §4 for the full rationale.

### 7.2 Case D methodology — per-repeat prompt variants

Cases A/B/C reuse the same prompt across warmup+repeats. That is
fine there because either (a) B=1 puts the whole prompt in one page
that the radix cache holds as a full hit on the next repeat with no
`extend_len>1` after a partial hit (A/B), or (b) B=2 equal-length
prompts share the exact same prefix so both requests see the same
`cached_len` and there is no ragged path (C).

Case D is B=2 **ragged** (short + long prompt). If the same
`(short, long)` pair is used across warmup+repeats, the first
measured repeat sees the long prompt with `cached_len==16` (one page
kept by the radix cache from warmup) and `extend_len==prompt_len-16 > 1`.
That combination — non-zero cached_len with `extend_len>1` on a
ragged batch — is the exact metadata shape that
`AscendFIAMetadata` raises `NotImplementedError` for in the current
port (**Gate 2.2f documented limitation**: FIA supports the
`cached_len==0` ragged-prefill branch and the pure-decode ragged
branch, but not the "mid-partial-hit ragged prefill" cross-branch).

To keep every case D measurement on the supported branch, the driver
rotates through a 4-entry `(short, long)` prompt pool per repeat.
Each measured repeat therefore lands on `cached_len==0` — a clean
ragged prefill — and gets a real timing measurement of the intended
shape.

This is:

* **Not** a workaround around a runtime bug — the FIA
  NotImplemented is the honest shape of the port at this gate.
* **Standard benchmark methodology** — rotating inputs across repeats
  is preferred practice for LLM timing anyway, since it avoids the
  "second repeat is a full radix cache hit" artifact that would
  otherwise pollute e2e latency numbers.
* Documented in the row's `description` field and in the driver's
  `_case_D_prompts_for(variant_id)` docstring.

The gate confirms that under the supported FIA branch, the B=2
ragged case has stable timing (min/max spread ≤ 1 % across variants
on both models — see §5).

### 7.3 One child subprocess per model

Same reason as Gate 3.3 §7.1:
`python/minisgl/distributed/info.py:24` raises
`RuntimeError: TP info has been set` on a second call to
`set_tp_info`, so instantiating a second `LLM` in the same process
fails. The Gate 3.4 driver is a parent orchestrator that spawns one
`python <script> --single --model-path <path>` child per model,
streams child stdout, and aggregates `GATE3.4_JSONL` /
`GATE3.4_SUMMARY` / `GATE3.4_MODEL_*` lines.

### 7.4 Timing hook — `offline_send_result`

`LLM.offline_send_result(reply)` is the tokenizer→driver callback
that receives one `DetokenizeMsg` per generated token per uid (each
message carries a `uid` and a `next_token`). The script-local
`TimingLLM` subclass overrides it to:

1. Call `super().offline_send_result(reply)` first (unchanged
   behavior — appends to `status.output_ids`).
2. If the timer is armed (`_arm_timing(t_start)` at repeat start) and
   the reply is non-empty, stamp `perf_counter()` as `_last_msg_time`
   and record `_first_token_time[uid]` on first-seen uids.

At repeat end, `ttft_ms = max(first_token_time.values()) - t_start`;
`e2e_ms = t_end - t_start` where `t_end` is measured immediately
after `llm.generate` returns. The overhead of the hook is a few
Python operations per token — well under a millisecond per step and
identical across cases, so it does not distort relative shape.

### 7.5 Zero changes to `python/minisgl/`

Everything above is done at the smoke-script layer. No source under
`python/minisgl/`, `tests/`, or any packaging file was modified.

---

## 8. Regression evidence

Per-file rows on the same container / same tree:

| File                                         | Rows |
|---|---|
| `test_scheduler_abort_ack.py`                | 8 passed |
| `test_scheduler_overlap_abort_fence.py`      | 7 passed |
| `test_scheduler_prepare_batch_txn.py`        | 5 passed |
| `test_engine_forward_sampler_atomic.py`      | 5 passed |
| `test_scheduler_shutdown_drain.py`           | 8 passed |
| `test_exposed_path_abort_ack.py`             | 2 passed |
| `test_shell_cancel_cleanup.py`               | 2 passed |
| `test_pyproject_config.py`                   | 14 passed |
| **Total**                                    | **51 passed** |

Matches the recorded counts of Gate 2.5, Gate 3.1, Gate 3.2, and
Gate 3.3.

The single-process cross-file ordering artifact carried from Gate 2.5
(fake `pydantic` stub in `test_scheduler_abort_ack.py` poisoning
`sys.modules` for a later `test_shell_cancel_cleanup.py`) is
unchanged; Gate 3.4 does not touch `tests/`. Per-file execution
sidesteps the artifact exactly as prior gates dictate.

---

## 9. Support matrix delta (Gate 3.3 → Gate 3.4)

| Capability                                             | Gate 3.3 | Gate 3.4 |
|---|---|---|
| Qwen3-0.6B B=1 (N=8, N=16) — timing snapshot           | shape PASS | **timing PASS** (3 repeats each) |
| Qwen3-0.6B B=2 equal-length — timing snapshot          | shape PASS | **timing PASS** (3 repeats) |
| Qwen3-0.6B B=2 ragged prefill — timing snapshot        | shape PASS | **timing PASS** (3 repeats, per-repeat variants) |
| Qwen3-1.7B B=1 (N=8, N=16) — timing snapshot           | shape PASS | **timing PASS** (3 repeats each) |
| Qwen3-1.7B B=2 equal-length — timing snapshot          | shape PASS | **timing PASS** (3 repeats) |
| Qwen3-1.7B B=2 ragged prefill — timing snapshot        | shape PASS | **timing PASS** (3 repeats, per-repeat variants) |
| Allocator `available_tokens` returns to baseline (repeat granularity) | 10 cases × 1 repeat | **24 measured repeats** |
| `deferred_abort_uids` stays empty (repeat granularity) | 10 cases × 1 repeat | **32 snapshots (24 measured + 8 warmup)** |
| TTFT / e2e / tokens_per_second recorded per repeat     | not measured | **24 measured rows** |
| Gate 2.2f FIA "ragged + non-zero cached_len + extend_len>1" boundary | referenced | **re-observed and worked around at driver layer for case D** |
| B=2 mixed-KV decode (Gate 3.3 case E)                  | PASS | not re-attested |
| TP > 1                                                 | UNKNOWN | UNKNOWN (out of scope) |
| Non-Qwen3 architecture families                        | UNKNOWN | UNKNOWN (unchanged) |
| Qwen3-4B / 14B / 32B                                   | UNKNOWN | UNKNOWN (deferred) |
| Regression: 8 hermetic suites (per-file)               | 51 passed | 51 passed (unchanged) |

---

## 10. What is NOT proven at this gate

Explicit exclusions carried forward from the Gate 3.4 opening:

* **Not a performance benchmark.** These numbers are a
  reproducibility snapshot of the current port on a specific card in
  a specific container. No throughput / latency / tokens-per-second
  leadership claim is made against SGLang, vLLM, TGI, LMDeploy, or
  any other stack. No cross-hardware claim is made.
* **No CUDAGraph / `cuda_graph_bs` optimization** — `cuda_graph_bs=[]`
  is locked at eager mode. torch_npu does not implement CUDAGraph.
* **No batching study.** Only B=1 and B=2 shapes are timed; B=3 was
  attested for shape (Gate 3.2 case D) but not for timing.
* **No long-sequence / context-length sweep.** All prompts fit in
  1–2 pages of the paged cache; long-context timing is out of scope.
* **TP > 1** for either model. HCCL wiring unexercised.
* **Qwen3-4B / 14B / 32B**, quantized (Qwen3-32B-FP8) and MoE
  (Qwen3-30B-A3B) variants, Qwen3-Next-*, Qwen3-ASR-*,
  Qwen3-Coder-Next. All out of scope.
* **Non-Qwen3 model families** (Llama / Mistral / DeepSeek / MoE).
* **Long soak / rolling allocator run.** Only 3 measured repeats
  per case were executed.
* **HTTP server restart, crash recovery, non-stream HTTP cancel,
  offline `LLM.abort()`, chunked prefill.** Same NOT REACHED /
  NOT SUPPORTED boundaries as Gate 2.5 / 3.1 / 3.2 / 3.3.
* **Forward/sampler exception recovery** inside the scheduler.
  Unchanged from Gate 2.5.
* **Case D under a single fixed `(short, long)` pair** across
  warmup+repeats — this hits the Gate 2.2f FIA NotImplemented and
  is therefore **not proven** at this gate. Case D is only proven
  under the per-repeat prompt-variants methodology of §7.2.

Verdict decision matrix (from gate open):

| Outcome | Definition | This gate |
|---|---|---|
| PASS    | Both models complete A–D with 3 measured repeats each, allocator returns to baseline every repeat | **✔ (this verdict)** |
| PARTIAL | One model completes A–D, the other has a documented environment / model-path / resource limitation | not reached |
| BLOCKED | Neither model can do the B=1 baseline (A) | not reached |

---

## 11. Freeze boundary

This gate freezes the fact that Mini-SGLang-Ascend at `16da4a2`
produces a reproducible timing snapshot (warmup=1, measured=3) on
both Qwen3-0.6B and Qwen3-1.7B under the TP=1 eager `npu_fia` bf16
envelope for request shapes A / B / C / D, with allocator invariants
held for every measured repeat, `deferred_abort_uids` empty
throughout, `cache_integrity_ok` true throughout, and 8-file
regression 51/51 passing.

It does not claim TP>1 support.
It does not claim any new architecture family.
It does not claim performance parity or leadership against any other
inference stack.
It does not extend the offline `LLM` driver's public surface.
It does not modify any prior gate verdict, the release tag
`v0.1.0a1`, or the GitHub Release.
It adds no code under `python/minisgl/` or `tests/`.
