# Gate 3.3 Verdict — TP=1 Ascend Capability Matrix

**Gate ID:** 3.3 (TP=1 Ascend capability matrix, Qwen3-0.6B + Qwen3-1.7B)
**Verdict:** PASS
**Branch:** `gate3.3-tp1-capability-matrix`
**Base commit:** `ce97ed4` (tip of `ascend-port`, Gate 3.2 merge)
**Freeze commit:** *(populated at merge into `ascend-port`)*
**Date:** 2026-07-11
**Kind:** Real-hardware Ascend 910B1 capability-boundary snapshot —
unified case matrix (A / B / C / D / E) executed against every
in-scope Qwen3 dense model already present on the Ascend host.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5 /
3.1 / 3.2, does not mutate release tag `v0.1.0a1`, does not touch the
GitHub Release, CHANGELOG, or release notes, and does not extend the
Ascend port to TP > 1, HCCL, Qwen3-4B / 14B / 32B, non-Qwen3
architectures (Llama / MoE / Qwen3-Next / Qwen3-ASR / Qwen3-Coder-Next),
MoE / quantized variants (Qwen3-32B-FP8, Qwen3-30B-A3B), performance
benchmarks, long soak, HTTP server restart,
forward/sampler exception recovery, non-stream HTTP cancel, offline
`LLM.abort()`, or chunked prefill.

---

## 1. Verdict summary

**PASS on the full capability matrix.** Under the frozen TP=1 / eager
/ `npu_fia` / bf16 / greedy envelope, both Qwen3-0.6B and Qwen3-1.7B
handle every required request-shape case:

| Case | Shape | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|---|
| A | B=1, `max_new_tokens=8`  | **PASS** | **PASS** |
| B | B=1, `max_new_tokens=16` | **PASS** | **PASS** |
| C | B=2 equal-length, `max_new_tokens=8` | **PASS** | **PASS** |
| D | B=2 ragged prefill, `max_new_tokens=8` | **PASS** | **PASS** |
| E | B=2 mixed-KV decode, `max_new_tokens=8` | **PASS** | **PASS** |

For every case on every model:

* Each request returned exactly the requested `max_new_tokens`.
* `cache_manager.available_size` (= free_slots + evictable prefix
  cache pages — the correct paged-cache invariant per Gate 3.1
  verdict §4) returned to the baseline value recorded at case start
  (`470592` for Qwen3-0.6B; `449568` for Qwen3-1.7B).
* `cache_manager.check_integrity()` was invoked at every snapshot
  and never raised.
* `scheduler.deferred_abort_uids` was empty at every snapshot.

Zero code changes under `python/minisgl/` or `tests/`. The gate adds
one smoke script and one verdict document.

---

## 2. Envelope (locked at this gate — matches Gate 3.1 / 3.2)

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
Driver:            scripts/gate3_3_capability_matrix.py
                   parent orchestrator + per-model child subprocess
                   (one process per model because
                    ``minisgl.distributed.info.set_tp_info`` is a
                    process-global singleton — see §7.1)
```

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
`vocab_size=151936` / `tie_word_embeddings=true`. See Gate 3.1 audit
§3 for the full config diff.

Rejected (why — matches Gate 3.1 §2):

| Model | Reason |
|---|---|
| Qwen3-4B, 14B, 32B | Multiple size steps ahead — deferred; not required by Gate 3.3. |
| Qwen3-32B-FP8 | Quantized — outside the bf16 `npu_fia` envelope. |
| Qwen3-30B-A3B | MoE — explicitly out of scope. |
| Qwen3-Next-*, Qwen3-Coder-Next | Different architecture family (`qwen3_next`). |
| Qwen3-ASR-* | Not `Qwen3ForCausalLM`. |
| Llama / Mistral / DeepSeek / other | Non-Qwen3 family — explicitly out of scope. |

---

## 4. Case matrix — per-case evidence

Each case row was emitted as a JSONL record with the full field set
required by the Gate 3.3 spec (`model_name`, `model_path`,
`case_name`, `prompt_lengths_chars`, `requested_max_new_tokens`,
`actual_output_tokens_per_request`, `batch_size_timeline`,
`baseline_available_tokens`, `available_tokens_after_case`,
`deferred_abort_uids`, `cache_integrity_ok`, `status`,
`failure_reason`). The tables below summarize the human-readable
subset of each row.

### 4.1 Qwen3-0.6B

Baseline / total pages: `free_pages=29412`, `total_pages=29412`,
`baseline_available_tokens=470592`, `deferred_abort_uids=0`.

| Case | Prompts (chars) | Requested N | Actual N | Batch | `available_tokens_after` | `deferred` | `check_integrity` | Verdict |
|---|---|---|---|---|---|---|---|---|
| A | `[24]` | `[8]`  | `[8]`  | `[1]` | 470592 | 0 | ok | **PASS** |
| B | `[24]` | `[16]` | `[16]` | `[1]` | 470592 | 0 | ok | **PASS** |
| C | `[24, 24]` | `[8, 8]` | `[8, 8]` | `[2]` | 470592 | 0 | ok | **PASS** |
| D | `[3, 118]` | `[8, 8]` | `[8, 8]` | `[2]` | 470592 | 0 | ok | **PASS** |
| E | `[6, 116]` | `[8, 8]` | `[8, 8]` | `[2]` | 470592 | 0 | ok | **PASS** |

Sample outputs (greedy, `ignore_eos=True`):

```
A req0:            ' Paris. The capital of France is also'
B req0 (N=16):     ' Paris. The capital of France is also the capital of the French Republic. The'
C req0 == req1:    ' Paris. The capital of France is also'    (identical prompts -> identical outputs)
D req0 (short):    ' I need to find the value of the'
D req1 (long):     ' It is a high-performance computing platform that'
E req0 (short):    ' I need to find the value of the'
E req1 (long):     ' The paragraph should be in the style of'
```

D and E both showed `reply_texts_equal=false` — the two ragged
prompts produced distinct outputs (guards against a logits-selection
swap bug where one request's first token comes from the other's
last-prompt position).

`free_pages` trended downward with each ragged-prefill case (29412
→ 29411 after B, still 29411 after C, 29409 after D, 29408 after E)
— evictable prefix-cache retention across cases, not a leak. The
`available_tokens` invariant returned to 470592 for every case.

### 4.2 Qwen3-1.7B

Baseline / total pages: `free_pages=28098`, `total_pages=28098`,
`baseline_available_tokens=449568`, `deferred_abort_uids=0`.

| Case | Prompts (chars) | Requested N | Actual N | Batch | `available_tokens_after` | `deferred` | `check_integrity` | Verdict |
|---|---|---|---|---|---|---|---|---|
| A | `[24]` | `[8]`  | `[8]`  | `[1]` | 449568 | 0 | ok | **PASS** |
| B | `[24]` | `[16]` | `[16]` | `[1]` | 449568 | 0 | ok | **PASS** |
| C | `[24, 24]` | `[8, 8]` | `[8, 8]` | `[2]` | 449568 | 0 | ok | **PASS** |
| D | `[3, 118]` | `[8, 8]` | `[8, 8]` | `[2]` | 449568 | 0 | ok | **PASS** |
| E | `[6, 116]` | `[8, 8]` | `[8, 8]` | `[2]` | 449568 | 0 | ok | **PASS** |

Sample outputs:

```
A req0:            ' Paris. The capital of the United States'
B req0 (N=16):     ' Paris. The capital of the United States is Washington, D.C. The capital'
C req0 == req1:    ' Paris. The capital of the United States'
D req0 (short):    " I'm trying to understand how to solve"
D req1 (long):     ' It is a 16-core processor'
E req0 (short):    " I'm trying to understand how to solve"
E req1 (long):     ' You may not use any technical terms or'
```

D and E both `reply_texts_equal=false`. `free_pages` trended 28098 →
28098 → 28097 → 28097 → 28095 → 28094 with the same evictable-
retention pattern as Qwen3-0.6B; `available_tokens` invariant stayed
at 449568 for every case.

---

## 5. Commands

Executed on remote container `<CONTAINER>` at working directory
`/mnt/nvme/LR-606/mini-sglang-ascend-gate33`.

Smoke:

```bash
PYTHONPATH=python python3 scripts/gate3_3_capability_matrix.py \
  --models /mnt/nvme/models/Qwen3-0.6B,/mnt/nvme/models/Qwen3-1.7B \
  --jsonl-out /tmp/gate3_3.jsonl
```

Scrapable footer emitted by the parent:

```
GATE3.3_MODEL_Qwen3-0.6B_CASE_A=PASS
GATE3.3_MODEL_Qwen3-0.6B_CASE_B=PASS
GATE3.3_MODEL_Qwen3-0.6B_CASE_C=PASS
GATE3.3_MODEL_Qwen3-0.6B_CASE_D=PASS
GATE3.3_MODEL_Qwen3-0.6B_CASE_E=PASS
GATE3.3_MODEL_Qwen3-0.6B=PASS
GATE3.3_MODEL_Qwen3-1.7B_CASE_A=PASS
GATE3.3_MODEL_Qwen3-1.7B_CASE_B=PASS
GATE3.3_MODEL_Qwen3-1.7B_CASE_C=PASS
GATE3.3_MODEL_Qwen3-1.7B_CASE_D=PASS
GATE3.3_MODEL_Qwen3-1.7B_CASE_E=PASS
GATE3.3_MODEL_Qwen3-1.7B=PASS
GATE3.3_MATRIX_RESULT=PASS
```

Regression (per-file, per Gate 3.1 / 3.2 verdicts' §5.1 note on the
carried-over sys.modules ordering artifact):

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

## 6. Allocator evidence

Baseline / after each case, per model.

### 6.1 Qwen3-0.6B

| Snapshot | free_pages | available_tokens | deferred_abort_uids | integrity |
|---|---|---|---|---|
| A_baseline | 29412 | 470592 | 0 | ok |
| A_after    | 29412 | 470592 | 0 | ok |
| B_baseline | 29412 | 470592 | 0 | ok |
| B_after    | 29411 | 470592 | 0 | ok |
| C_baseline | 29411 | 470592 | 0 | ok |
| C_after    | 29411 | 470592 | 0 | ok |
| D_baseline | 29411 | 470592 | 0 | ok |
| D_after    | 29409 | 470592 | 0 | ok |
| E_baseline | 29409 | 470592 | 0 | ok |
| E_after    | 29408 | 470592 | 0 | ok |

### 6.2 Qwen3-1.7B

| Snapshot | free_pages | available_tokens | deferred_abort_uids | integrity |
|---|---|---|---|---|
| A_baseline | 28098 | 449568 | 0 | ok |
| A_after    | 28098 | 449568 | 0 | ok |
| B_baseline | 28098 | 449568 | 0 | ok |
| B_after    | 28097 | 449568 | 0 | ok |
| C_baseline | 28097 | 449568 | 0 | ok |
| C_after    | 28097 | 449568 | 0 | ok |
| D_baseline | 28097 | 449568 | 0 | ok |
| D_after    | 28095 | 449568 | 0 | ok |
| E_baseline | 28095 | 449568 | 0 | ok |
| E_after    | 28094 | 449568 | 0 | ok |

Invariants held:

* `available_tokens_after_case == baseline_available_tokens` for every
  case on every model.
* `deferred_abort_uids == 0` at every snapshot on every model.
* `check_integrity()` (`free_pages + cache_pages == total_pages`)
  succeeded at every snapshot on every model.
* `free_pages` drift is designed radix-cache retention across
  ragged-prompt cases; the invariant is `available_tokens`, not raw
  `free_pages`. See Gate 3.1 verdict §4 for the full rationale.

---

## 7. Implementation notes

### 7.1 One child subprocess per model

`python/minisgl/engine/engine.py` invokes
`set_tp_info(rank=..., size=...)`, and
`python/minisgl/distributed/info.py:24` raises
`RuntimeError: TP info has been set` on a second call. Instantiating a
second `LLM` in the same process therefore fails.

Gate 3.3 needs to run the same case matrix against two different
model paths. The clean solution is a **parent orchestrator** that
spawns one child subprocess per model:

* Parent parses `--models path1,path2`, spawns
  `python <script> --single --model-path <path>` per entry, streams
  child stdout, aggregates `GATE3.3_JSONL` lines and per-model /
  per-case verdicts, and prints the overall matrix footer.
* Child (`--single`) boots a single `LLM`, runs cases A–E, prints
  JSONL rows, prints per-case + per-model verdict lines, exits.

This isolates the `set_tp_info` singleton to a single-model lifetime.

### 7.2 Case E design

The Gate 3.3 spec pins case E at `max_new_tokens=8` (equal to case
D). Under the frozen envelope "mixed-KV decode" at both N=8 is
achieved by giving the two requests **prompts with very different
token counts**, which causes them to carry different `cached_len` at
every shared decode step. Case E uses distinct prompt content from
case D (`"Hello."` + a long descriptive prompt) so the verdict has
two independent B=2 ragged-prefill records rather than one repeated
measurement.

The specific "solo-tail" variant (one request finishes earlier via
different `max_new_tokens`, the other keeps decoding alone) was
already recorded in Gate 3.2 case C and is not re-attested here.

### 7.3 No optional B=3 admission timeline

The Gate 3.3 spec lists B=3 grow-shrink as optional. Gate 3.2 case D
already recorded the exact dedup shape `[0, 1, 2, 3, 2, 1, 0]` for
Qwen3-1.7B and referenced Gate 2.2 as the Qwen3-0.6B equivalent, so
this gate does not re-attest B=3.

### 7.4 Zero changes to `python/minisgl/`

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

Matches the recorded counts of Gate 2.5, Gate 3.1, and Gate 3.2.

The single-process cross-file ordering artifact carried from Gate 2.5
(fake `pydantic` stub in `test_scheduler_abort_ack.py` poisoning
`sys.modules` for a later `test_shell_cancel_cleanup.py`) is
unchanged; Gate 3.3 does not touch `tests/`. Per-file execution
sidesteps the artifact exactly as prior gates dictate.

---

## 9. Support matrix delta (Gate 3.2 → Gate 3.3)

| Capability                                             | Gate 3.2 | Gate 3.3 |
|---|---|---|
| Qwen3-0.6B B=1 (N=8, N=16)                             | PASS (Gate 2.1) | PASS (Cases A, B — retested here) |
| Qwen3-0.6B B=2 equal-length                            | PASS (Gate 2.2) | PASS (Case C — retested here) |
| Qwen3-0.6B B=2 ragged prefill                          | UNKNOWN  | **PASS** (Case D) |
| Qwen3-0.6B B=2 mixed-KV decode                         | UNKNOWN  | **PASS** (Case E) |
| Qwen3-1.7B B=1 (N=8, N=16)                             | PASS (Gate 3.1) | PASS (Cases A, B — retested here) |
| Qwen3-1.7B B=2 equal-length                            | PASS (Gate 3.2 A) | PASS (Case C) |
| Qwen3-1.7B B=2 ragged prefill                          | PASS (Gate 3.2 B) | PASS (Case D) |
| Qwen3-1.7B B=2 mixed-KV decode                         | PASS (Gate 3.2 C, solo tail) | PASS (Case E, both N=8) |
| Qwen3-1.7B dynamic admission B=3                       | PASS (Gate 3.2 D) | PASS (unchanged) |
| Allocator `available_tokens` returns to baseline       | PASS (4 cases, 1 model) | PASS (10 cases, 2 models) |
| `deferred_abort_uids` stays empty                      | PASS (4 cases, 1 model) | PASS (10 cases, 2 models) |
| TP > 1                                                 | UNKNOWN | UNKNOWN (out of scope) |
| Non-Qwen3 architecture families                        | UNKNOWN | UNKNOWN (unchanged) |
| Qwen3-4B / 14B / 32B                                   | UNKNOWN | UNKNOWN (deferred) |
| Regression: 8 hermetic suites (per-file)               | 51 passed | 51 passed (unchanged) |

---

## 10. What is NOT proven at this gate

Explicit exclusions carried forward from the Gate 3.3 opening:

* **TP > 1** for either model. HCCL wiring unexercised.
* **Qwen3-4B / 14B / 32B**, quantized (Qwen3-32B-FP8) and MoE
  (Qwen3-30B-A3B) variants, Qwen3-Next-*, Qwen3-ASR-*,
  Qwen3-Coder-Next. All out of scope.
* **Non-Qwen3 model families** (Llama / Mistral / DeepSeek / MoE).
* **Performance benchmark.** No throughput, latency, or leadership
  claim is made. Timings in the smoke log are for debugging only.
* **Long soak / rolling allocator run.** Only 5 cases per model
  were executed.
* **HTTP server restart, crash recovery, non-stream HTTP cancel,
  offline `LLM.abort()`, chunked prefill.** Same NOT REACHED /
  NOT SUPPORTED boundaries as Gate 2.5 / 3.1 / 3.2.
* **Forward/sampler exception recovery** inside the scheduler.
  Unchanged from Gate 2.5.

Verdict decision matrix (from gate open):

| Outcome | Definition | This gate |
|---|---|---|
| PASS    | Both models pass required cases A–E, allocator returns to baseline after every case | **✔ (this verdict)** |
| PARTIAL | One model passes A–E, the other has a documented environment / model-path / resource limitation | not reached |
| BLOCKED | Neither model can run the basic B=1 smoke in the current environment | not reached |

---

## 11. Freeze boundary

This gate freezes the fact that Mini-SGLang-Ascend at `ce97ed4`
serves the required B=1 and B=2 request shapes on **both** Qwen3-0.6B
and Qwen3-1.7B under the TP=1 eager `npu_fia` bf16 envelope, with
allocator invariants held for every case and `deferred_abort_uids`
empty throughout.

It does not claim TP>1 support.
It does not claim any new architecture family.
It does not extend the offline `LLM` driver's public surface.
It does not modify any prior gate verdict, the release tag
`v0.1.0a1`, or the GitHub Release.
It makes no performance claim.
It adds no code under `python/minisgl/` or `tests/`.
