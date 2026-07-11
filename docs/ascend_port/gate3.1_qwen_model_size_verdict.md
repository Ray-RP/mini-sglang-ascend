# Gate 3.1 Verdict — Qwen3 Model Size Expansion (TP=1)

**Gate ID:** 3.1 (Qwen3 same-family model size expansion, TP=1)
**Verdict:** PASS
**Branch:** `gate3.1-qwen-model-size-expansion`
**Base commit:** `d8b1fd4` (tip of `ascend-port`, Gate 2.5 merge)
**Date:** 2026-07-11
**Kind:** Read-only compatibility audit + real-hardware Ascend 910B1
smoke on a single Qwen3 model larger than the Qwen3-0.6B baseline.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5,
does not mutate release tag `v0.1.0a1`, does not touch the GitHub
Release, CHANGELOG, or release notes, and does not extend the Ascend
port to TP > 1, HCCL, non-`qwen3` architectures (Llama / MoE /
Qwen3-Next / Qwen3-ASR / Qwen3-Coder-Next), MoE variants (Qwen3-30B-A3B),
quantized variants (Qwen3-32B-FP8), performance benchmarks, long soak,
HTTP server restart, forward/sampler exception recovery, non-stream
HTTP cancel, offline `LLM.abort()`, or chunked prefill.

---

## 1. Verdict summary

**PASS.** Mini-SGLang-Ascend at `d8b1fd4` boots **Qwen3-1.7B** — the
smallest same-family model above the Qwen3-0.6B baseline present on
the Ascend host — under the frozen envelope (TP=1, eager, `npu_fia`,
bf16, greedy) and completes end-to-end prefill + multi-step decode on
a live Ascend 910B1 die. The paged allocator returns to the recorded
baseline after each request; `deferred_abort_uids` stays empty; no
new code was added under `python/minisgl/` (zero code diff — audit
Section 4 confirmed the code path already covers the shape change).

Gate 3.1 delivered:

* `docs/ascend_port/gate3.1_qwen_model_size_audit.md` — read-only
  audit that establishes Qwen3-1.7B as the smallest next-size dense
  bf16 Qwen3 model on disk and proves via grep + Read that no numeric
  literal in `python/minisgl/` encodes the 0.6B shape.
* `scripts/gate3_1_smoke.py` — the offline `minisgl.llm.LLM`-driven
  Ascend NPU smoke recorded as evidence for this gate.
* Real Ascend 910B1 execution log (Section 4 below) — three tests
  PASS on-hardware.

No files under `python/minisgl/`, no test files, no config, no
pyproject metadata, and no CHANGELOG were modified.

---

## 2. Envelope (locked at this gate)

```
Hardware:          Ascend 910B1 (1 die, 64 GiB HBM)
Container:         <CONTAINER> on remote <HOST>:<PORT>
Software:          Python 3.11.14
                   torch 2.9.0+cpu
                   torch_npu 2.9.0.post1+gitee7ba04
                   CANN 8.5.1 (compiler build 20250725)
Baseline model:    /mnt/nvme/models/Qwen3-0.6B  (prior gates)
Target model:      /mnt/nvme/models/Qwen3-1.7B
                   dense bf16, 2 safetensors shards, ~3.8 GiB on disk
Model geometry:    num_hidden_layers=28   hidden_size=2048
                   intermediate_size=6144 vocab_size=151936
                   num_attention_heads=16 num_key_value_heads=8
                   head_dim=128           max_position_embeddings=40960
                   tie_word_embeddings=true
Parallelism:       TP=1
Execution:         eager (cuda_graph_bs=[]; torch_npu has no CUDAGraph)
Attention backend: npu_fia
Page size:         16   (FIA NO_QUANT tiling requires block_size % 16 == 0)
memory_ratio:      0.85
max_running_req:   8
Sampling:          greedy (temperature=0.0, top_k=1, top_p=1.0, ignore_eos=True)
Prompt:            "The capital of France is"
Driver:            offline in-process minisgl.llm.LLM
```

Rejected model choices and why (see audit Section 2):

| Model              | Why rejected                                   |
|---|---|
| Qwen3-4B / 14B / 32B | Skip the "next-larger" step per gate spec.  |
| Qwen3-32B-FP8      | Quantized; out of `npu_fia` bf16 scope.       |
| Qwen3-30B-A3B      | MoE; explicitly excluded ("新模型族").          |
| Qwen3-Next-*       | `qwen3_next` architecture, not `qwen3` dense. |
| Qwen3-ASR-*        | Not `Qwen3ForCausalLM`.                       |
| Qwen3-Coder-Next   | `qwen3_next` architecture.                    |

---

## 3. Code-side compatibility (freeze at `d8b1fd4`)

Zero code changes under `python/minisgl/`. The audit
(`gate3.1_qwen_model_size_audit.md` Section 4) proves compatibility
by inspection:

* `python/minisgl/models/qwen3.py` — every layer dimension is drawn
  from `ModelConfig` (`config.hidden_size`, `config.rms_norm_eps`,
  `config.num_layers`, `config.vocab_size`); no numeric literal
  encodes 0.6B geometry.
* `python/minisgl/models/utils.py` — `GatedMLP` and `RopeAttn` read
  `config.hidden_size`, `config.intermediate_size`, `config.head_dim`,
  `config.num_qo_heads`, `config.num_kv_heads`, `config.rms_norm_eps`.
* `python/minisgl/models/config.py:50` — `head_dim` fallback branch
  is not exercised (Qwen3-1.7B ships `head_dim=128` explicitly).
* Grep for `0.6`, `1024`, `3072`, `Qwen3-0` under `python/minisgl/` —
  zero shape-related hits.
* KV cache page geometry (`num_kv_heads * head_dim = 8 * 128 = 1024`)
  is identical between 0.6B and 1.7B; paged allocator byte layout
  is unchanged.
* `ParallelLMHead` output width `vocab_size = 151936` is unchanged.
* Weight loader (`python/minisgl/models/weight.py`) consumes the
  identical HF-format key namespace.

Only knobs the smoke script tunes at LLM-constructor level (no code
change): `attention_backend="npu_fia"`, `page_size=16`,
`cuda_graph_bs=[]`, `memory_ratio=0.85`, `max_running_req=8`.
These are the same envelope knobs Gate 1 / 2.x already use.

---

## 4. Smoke evidence

Command (executed inside container `<CONTAINER>` at
`/mnt/nvme/LR-606/mini-sglang-ascend-gate31`):

```bash
PYTHONPATH=python python3 scripts/gate3_1_smoke.py \
  --model-path /mnt/nvme/models/Qwen3-1.7B
```

Salient log (trimmed):

```
[gate3.1][boot] model_path=/mnt/nvme/models/Qwen3-1.7B
[gate3.1][boot] attention_backend=npu_fia
[gate3.1][boot] LLM initialised in <boot_time>s
[gate3.1][alloc] baseline free_pages=28098 available_tokens=449568
                 total_pages=28098 deferred_abort_uids=0
[gate3.1][A] single-request prefill + 8-step decode
[gate3.1][A] generated_tokens=8
[gate3.1][A] tokens=[12095, 13, 576, 6722, 315, 279, 3639, 4180]
[gate3.1][A] text=' Paris. The capital of the United States'
[gate3.1][alloc] after_A free_pages=28098 available_tokens=449568
                 total_pages=28098 deferred_abort_uids=0
[gate3.1][B] single-request prefill + 16-step decode
[gate3.1][B] generated_tokens=16
[gate3.1][B] tokens=[12095, 13, 576, 6722, 315, 279, 3639, 4180, 374,
                    6515, 11, 422, 727, 13, 576, 6722]
[gate3.1][B] text=' Paris. The capital of the United States is
                   Washington, D.C. The capital'
[gate3.1][alloc] after_B free_pages=28097 available_tokens=449568
                 total_pages=28098 deferred_abort_uids=0
[gate3.1][C] best-effort B=2 equal-length prefill + decode
[gate3.1][C] generated_tokens=[8, 8]
[gate3.1][alloc] after_C free_pages=28097 available_tokens=449568
                 total_pages=28098 deferred_abort_uids=0
[gate3.1][summary] baseline_free_pages=28098
                   baseline_available_tokens=449568
                   total_pages=28098
[gate3.1][summary] A(B=1, N=8)  = PASS
[gate3.1][summary] B(B=1, N=16) = PASS
[gate3.1][summary] C(B=2, N=8)  = PASS
GATE3.1_SMOKE_RESULT=PASS
GATE3.1_SMOKE_BATCH2=PASS
GATE3.1_SMOKE_BASELINE_FREE=28098
GATE3.1_SMOKE_BASELINE_AVAILABLE=449568
GATE3.1_SMOKE_TOTAL_PAGES=28098
```

Observations:

* **Text quality.** Greedy decoding produces coherent English on the
  target model (`" Paris. The capital of the United States is
  Washington, D.C. The capital"`). This is a sanity check, not a
  benchmark; the model is behaving as a language model, not producing
  garbage.
* **Allocator invariant.** The correct allocator invariant per the
  paged cache design is `cache_manager.available_size` (free slots +
  evictable prefix cache pages), **not** raw `len(free_slots)`. The
  radix prefix cache is designed to retain 0 or more evictable pages
  across requests as an optimization — that retention is not a leak.
  * `available_tokens` returns to the exact baseline (`449568`) after
    every request A, B, and C.
  * `free_pages` drifts by 1 between the baseline and after-B/after-C
    snapshots (`28098 → 28097`) — that single page is retained by the
    prefix cache and remains evictable, i.e. `available_tokens` is
    unchanged.
  * `cache_manager.check_integrity()` is called at every snapshot and
    asserts `free_pages + cache_pages == total_pages` — never violated.
* **`deferred_abort_uids` empty.** No residual abort state after any
  request. The Gate 2.3 f AbortAck invariants are not disturbed by
  the wider model.
* **Best-effort B=2 succeeds.** The gate spec explicitly permits B=2
  to be reported as a limitation if HBM is tight; the target ran the
  B=2 equal-length prefill+decode cleanly and returned each request
  the requested 8 tokens.

---

## 5. Regression evidence

Command (same container / same working tree):

```bash
PYTHONPATH=python:tests pytest -q -o addopts="" \
  tests/misc/test_scheduler_abort_ack.py \
  tests/misc/test_scheduler_overlap_abort_fence.py \
  tests/misc/test_scheduler_prepare_batch_txn.py \
  tests/misc/test_engine_forward_sampler_atomic.py \
  tests/misc/test_scheduler_shutdown_drain.py \
  tests/misc/test_exposed_path_abort_ack.py \
  tests/misc/test_shell_cancel_cleanup.py \
  tests/misc/test_pyproject_config.py
```

Per-file results (each file invoked in isolation to keep the
attestation truthful):

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

The 51-row total matches Gate 2.5's recorded count exactly.

### 5.1 Pre-existing single-process ordering artifact (not a Gate 3.1 regression)

When the eight files above are run in the **same** pytest invocation
in the command's declared order, `test_shell_cancel_cleanup.py`'s
two tests fail with `AttributeError: 'Message' object has no
attribute 'model_dump'`. The failure reproduces exactly against
Gate 2.5's own checkout (`/mnt/nvme/LR-606/mini-sglang-ascend-gate25`
at `d8b1fd4`) and is **not introduced by Gate 3.1** (which added
zero code under `python/minisgl/` and zero code under `tests/`).

Root cause is a test-infrastructure ordering bug already present at
`d8b1fd4`: `test_scheduler_abort_ack.py`'s module-load-time
`_fake_module("pydantic", ...)` runs **unconditionally**, installing
a `_BM` stub class as `sys.modules["pydantic"].BaseModel` *before*
`minisgl.server.api_server` is imported for the first time. In the
gate-25 process, real `pydantic 2.12.5` is installed; but because
the stub is inserted into `sys.modules` first, `api_server`'s
`from pydantic import BaseModel, Field` binds `Message(BaseModel)`
to `_BM` — a class without `model_dump`. `test_shell_cancel_cleanup.py`'s
own stub logic then guards with `if "pydantic" not in sys.modules`
and (correctly) leaves the poisoned entry alone.

Fixing this ordering bug requires modifying `tests/misc/`, which is
**out of scope** for Gate 3.1's declared surface (this gate is
"Qwen3 model size expansion" — the audit and verdict live under
`docs/ascend_port/` and the smoke lives under `scripts/`; no test
file is added or edited). Per Gate 3.1's own PASS criteria
(§ verdict spec: real-hardware smoke PASS, allocator invariant held,
no code change under `python/minisgl/`, no regression *attributable
to this gate*), the pre-existing test-order artifact is recorded
here as a **known-issue carryover from Gate 2.5** and does not
change the verdict.

All 51 rows are green when each file is invoked in a fresh Python
process (documented above). The Ascend port's behavior under the
gate-3.1 envelope is unaffected.

---

## 6. Support matrix delta (Gate 2.5 → Gate 3.1)

| Capability                                             | Gate 2.5 | Gate 3.1 |
|---|---|---|
| Qwen3-0.6B, TP=1, eager, npu_fia                       | PASS     | PASS (unchanged) |
| Qwen3-1.7B, TP=1, eager, npu_fia                       | UNKNOWN  | **PASS** |
| Qwen3-4B / 14B / 32B                                   | UNKNOWN  | UNKNOWN (deferred) |
| Qwen3-32B-FP8, Qwen3-30B-A3B, Qwen3-Next-*             | n/a      | n/a (out of scope) |
| TP > 1 forward                                         | UNKNOWN  | UNKNOWN (unchanged) |
| Non-Qwen3 architecture families (Llama / MoE)          | UNKNOWN  | UNKNOWN (unchanged) |
| E1 HTTP `/generate` stream                             | PASS     | PASS (unchanged) |
| E2 HTTP `/v1/chat/completions` stream=true             | PASS     | PASS (unchanged) |
| E3 HTTP `/v1/chat/completions` stream=false            | NOT REACHED | NOT REACHED (unchanged) |
| E5 Shell cancel                                        | PASS     | PASS (unchanged) |
| E6 Offline `LLM.abort()`                               | NOT REACHED | NOT REACHED (unchanged) |
| Regression: 8 hermetic suites (per-file)               | 51 passed | 51 passed (unchanged) |

Aggregate model-family readiness moves from "one Qwen3 size proven"
to "two Qwen3 sizes proven (0.6B, 1.7B)" — a single-step expansion
on the smallest available next size, per gate spec.

---

## 7. What is NOT proven at this gate

Explicit exclusions carried forward from the Gate 3.1 opening:

* **TP > 1 for Qwen3-1.7B (or any model).** HCCL wiring not exercised.
* **Qwen3-4B / 14B / 32B on Ascend.** Deferred for a later gate if
  desired.
* **Non-Qwen3 model families** (Llama / DeepSeek / Mistral / MoE /
  Qwen3-Next / Qwen3-ASR / Qwen3-Coder-Next).
* **Quantized weights** (Qwen3-32B-FP8 or any FP8/INT8 variant).
* **Performance benchmark.** No throughput, latency, or leadership
  claim is made. Timings in the smoke log are for debugging only.
* **Long soak / rolling allocator run.** Only three requests were
  executed.
* **HTTP server restart, crash recovery, non-stream HTTP cancel,
  offline `LLM.abort()`, chunked prefill.** Same NOT REACHED /
  NOT SUPPORTED boundaries as Gate 2.5.
* **Forward/sampler exception recovery inside the scheduler.**
  Unchanged from Gate 2.5.
* **Fixing the pre-existing single-process test-ordering artifact**
  in `tests/misc/`. Recorded in § 5.1 as a Gate 2.5 carryover.

---

## 8. Freeze boundary

This gate freezes the fact that Mini-SGLang-Ascend at `d8b1fd4`
runs Qwen3-1.7B on Ascend 910B1 under the frozen TP=1 eager
`npu_fia` bf16 envelope, with correct allocator invariants held.
It does not claim TP>1 support.
It does not claim any new architecture family.
It does not extend the offline `LLM` driver's public surface.
It does not modify any prior gate verdict, the release tag
`v0.1.0a1`, or the GitHub Release.
It makes no performance claim.
It adds no code under `python/minisgl/` or `tests/`.
