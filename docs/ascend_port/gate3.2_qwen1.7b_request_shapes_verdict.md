# Gate 3.2 Verdict — Qwen3-1.7B Request-Shape Expansion (TP=1)

**Gate ID:** 3.2 (Qwen3-1.7B request-shape expansion, TP=1)
**Verdict:** PASS
**Branch:** `gate3.2-qwen1.7b-request-shapes`
**Base commit:** `ad74ee1` (tip of `ascend-port`, Gate 3.1 merge)
**Freeze commit:** *(populated at merge into `ascend-port`)*
**Date:** 2026-07-11
**Kind:** Real-hardware Ascend 910B1 smoke of four request shapes on
Qwen3-1.7B under the Gate 3.1 frozen envelope.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5 /
3.1, does not mutate release tag `v0.1.0a1`, does not touch the GitHub
Release, CHANGELOG, or release notes, and does not extend the Ascend
port to TP > 1, HCCL, non-`qwen3` architectures (Llama / MoE /
Qwen3-Next / Qwen3-ASR / Qwen3-Coder-Next), MoE / quantized variants,
performance benchmarks, long soak, HTTP server restart,
forward/sampler exception recovery, non-stream HTTP cancel, offline
`LLM.abort()`, or chunked prefill.

---

## 1. Verdict summary

**PASS on all four request shapes.** Qwen3-1.7B on Ascend 910B1 under
TP=1 / eager / `npu_fia` / bf16 / greedy correctly handles:

* A. B=2 equal-length prefill + decode
* B. B=2 ragged prefill (short + long prompt)
* C. B=2 mixed-KV decode with solo tail (ragged + different `max_tokens`)
* D. Dynamic admission grow-shrink with observed batch-size timeline
     `1 → 2 → 3 → 2 → 1` (peak decode-running=3)

For every case:

* Each request returns exactly the requested `max_new_tokens`.
* Sibling requests do not interfere — B produced distinct outputs for
  the two prompts (`reply_texts_equal=false`), C's survivor continued
  decoding cleanly after the short-max sibling completed, D grew and
  shrank the running-decode set through 4 admissions and 3 completions.
* `cache_manager.available_size` (free slots + evictable prefix cache
  pages — the correct allocator invariant, see Gate 3.1 verdict §4)
  returns to the recorded baseline (`449568`) after every case.
* `cache_manager.check_integrity()` is called at every snapshot and
  never raised.
* `deferred_abort_uids` stays empty throughout.

Zero code changes under `python/minisgl/` or `tests/`. The gate adds
one smoke script and one verdict document.

---

## 2. Envelope (locked at this gate — same as Gate 3.1)

```
Hardware:          Ascend 910B1 (1 die, 64 GiB HBM)
Container:         <CONTAINER> on remote <HOST>:<PORT>
Software:          Python 3.11.14
                   torch 2.9.0+cpu
                   torch_npu 2.9.0.post1+gitee7ba04
                   CANN 8.5.1 (compiler build 20250725)
Model:             /mnt/nvme/models/Qwen3-1.7B
                   dense bf16, 2 safetensors shards
Model geometry:    num_hidden_layers=28   hidden_size=2048
                   intermediate_size=6144 vocab_size=151936
                   num_attention_heads=16 num_key_value_heads=8
                   head_dim=128           max_position_embeddings=40960
                   tie_word_embeddings=true
Parallelism:       TP=1
Execution:         eager (cuda_graph_bs=[])
Attention backend: npu_fia
page_size:         16       (FIA NO_QUANT block_size % 16 == 0)
memory_ratio:      0.85
max_running_req:   8
Sampling:          greedy (temperature=0.0, top_k=1, top_p=1.0, ignore_eos=True)
Driver:            scripts/gate3_2_request_shapes.py (in-process
                   minisgl.llm.LLM via a script-local DynamicLLM
                   subclass — see §7)
```

---

## 3. Cases

Every case is executed in the same LLM process instance because
`minisgl.distributed.info.set_tp_info` is a process-global singleton:
instantiating a second `LLM` in the same process raises
`RuntimeError: TP info has been set`. Cases A/B/C use the
`DynamicLLM` subclass through its transparent fast-path
(`_tick_schedule` empty → delegate straight to `LLM.offline_receive_msg`).
Case D installs a tick schedule and drives `run_forever()` directly.

### Case A — B=2 equal-length prefill + decode

| Field | Value |
|---|---|
| Prompts | `"The capital of France is"` × 2 |
| `max_new_tokens` | `[8, 8]` |
| Actual output tokens | `[8, 8]` |
| Batch size timeline | `[2]` |
| `alloc_before` | free_pages=28098 available_tokens=449568 deferred=0 |
| `alloc_after` | free_pages=28098 available_tokens=449568 deferred=0 |
| `check_integrity()` | ok |
| Verdict | **PASS** |

Text (identical between the two requests, as expected under greedy on
identical prompts):

```
req0: ' Paris. The capital of the United States'
req1: ' Paris. The capital of the United States'
```

### Case B — B=2 ragged prefill

| Field | Value |
|---|---|
| Prompt lengths (chars) | `[3, 118]` (`"Hi."`, `"The Ascend 910B1 accelerator has 64 gigabytes of high bandwidth memory and is used for large language model inference."`) |
| `max_new_tokens` | `[8, 8]` |
| Actual output tokens | `[8, 8]` |
| Batch size timeline | `[2]` |
| Reply texts equal? | **no** (guard against a swap bug) |
| `alloc_before` | free_pages=28098 available_tokens=449568 deferred=0 |
| `alloc_after` | free_pages=28096 available_tokens=449568 deferred=0 |
| `check_integrity()` | ok |
| Verdict | **PASS** |

Text:

```
req0 (short prompt): " I'm trying to understand how to solve"
req1 (long prompt):  ' It is a 16-core processor'
```

Note: `free_pages` drops by 2 vs baseline while `available_tokens`
stays at 449568 — the ragged prompts left two evictable pages in the
prefix cache, which is designed retention (not a leak). This is the
same invariant reasoning Gate 3.1 verdict §4 documents.

### Case C — B=2 mixed-KV decode with solo tail

| Field | Value |
|---|---|
| Prompt lengths (chars) | `[6, 116]` (`"Hello."`, `"In one short paragraph, describe the color of a clear afternoon sky over the ocean, using only common English words."`) |
| `max_new_tokens` | `[8, 16]` |
| Actual output tokens | `[8, 16]` |
| Batch size timeline | `[2, 1]` (both decoding → survivor alone after step 8) |
| `alloc_before` | free_pages=28096 available_tokens=449568 deferred=0 |
| `alloc_after` | free_pages=28094 available_tokens=449568 deferred=0 |
| `check_integrity()` | ok |
| Verdict | **PASS** |

Text:

```
req0 (short, N=8):  " I'm trying to understand how to solve"
req1 (long,  N=16): ' You may not use any technical terms or scientific jargon. Avoid using any markdown'
```

This case exercises two things in one shot:

1. **Mixed-KV decode during the shared decode phase** — the two
   requests have unequal `cached_len` at every decode step, so the
   attention metadata must select the correct per-request KV range
   from the paged cache.
2. **Solo-tail continuation** — once req0 finishes at step 8, req1
   keeps decoding alone for another 8 steps. The scheduler must
   release req0's pages without disturbing req1's live pages.

Verified via output length and allocator recovery.

### Case D — Dynamic admission grow-shrink 1 → 2 → 3 → 2 → 1

Timeline schedule: release req0 at tick 0, req1 at tick 3, req2 at
tick 6. `max_new_tokens` chosen so req2 finishes first, then req1,
then req0.

| Field | Value |
|---|---|
| `max_new_tokens` | `[24, 12, 4]` (req0, req1, req2) |
| Actual output tokens | `[24, 12, 4]` |
| Running-decode tick series | `[0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]` |
| **Dedup shape** | **`[0, 1, 2, 3, 2, 1, 0]`** |
| Peak running-decode | **3** |
| `alloc_before` | free_pages=28094 available_tokens=449568 deferred=0 |
| `alloc_after` | free_pages=28092 available_tokens=449568 deferred=0 |
| `check_integrity()` | ok |
| Verdict | **PASS** |

Text:

```
req uid=0 text=' made of cheese. What is the probability that the moon is made of cheese? - Brainly.com\nprofile\nprofile'
req uid=1 text=' Rain is a form of precipitation that occurs when water vapor in'
req uid=2 text=" I'm a new"
```

The dedup shape `[0, 1, 2, 3, 2, 1, 0]` is the exact grow-shrink
timeline the gate spec targets. Peak decode-batch=3 was achieved.
Every request returned its requested token budget.

---

## 4. Commands

Executed on remote container `<CONTAINER>` at working directory
`/mnt/nvme/LR-606/mini-sglang-ascend-gate32`.

Smoke:

```bash
PYTHONPATH=python python3 scripts/gate3_2_request_shapes.py \
  --model-path /mnt/nvme/models/Qwen3-1.7B \
  --jsonl-out /tmp/gate3_2.jsonl
```

Scrapable footer emitted by the smoke:

```
GATE3.2_SMOKE_RESULT=PASS
GATE3.2_CASE_A=PASS
GATE3.2_CASE_B=PASS
GATE3.2_CASE_C=PASS
GATE3.2_CASE_D=PASS
```

Regression (per-file, per Gate 3.1 verdict §5.1 note on the
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

## 5. Allocator and batch-timeline evidence

### 5.1 Allocator invariant (`available_tokens`)

| Snapshot | free_pages | available_tokens | deferred_abort_uids |
|---|---|---|---|
| A_baseline | 28098 | 449568 | 0 |
| A_after    | 28098 | 449568 | 0 |
| B_baseline | 28098 | 449568 | 0 |
| B_after    | 28096 | 449568 | 0 |
| C_baseline | 28096 | 449568 | 0 |
| C_after    | 28094 | 449568 | 0 |
| D_baseline | 28094 | 449568 | 0 |
| D_after    | 28092 | 449568 | 0 |

* `available_tokens` = free_slots + evictable prefix cache pages.
  This is the correct invariant — see Gate 3.1 verdict §4 for the
  full rationale. It returns to the exact baseline (`449568`) after
  every case.
* `free_pages` trends downward with each case because the radix
  prefix cache holds increasing evictable retention across ragged
  prompts (B → 2 pages, C → 2 more pages, D → 2 more pages). None
  are leaked — all remain evictable and count into `available_tokens`.
* `cache_manager.check_integrity()` — asserts
  `free_pages + cache_pages == total_pages` — passed at every snapshot.
* `deferred_abort_uids` was zero at every snapshot. No abort residue
  from any case.

### 5.2 Batch-size timeline (case D)

The DynamicLLM subclass samples `len(decode_manager.running_reqs)` at
every `offline_receive_msg` call. The unfiltered tick series (from the
JSONL row of case D) is:

```
[0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
```

Adjacent-dedup shape: `[0, 1, 2, 3, 2, 1, 0]` — the gate-spec target.

Interpretation:

* Tick 0: no running decodes yet; queued req0 admitted → prefill batch.
* Tick 1..3: req0 decoding.
* Tick 4..6: req0 + req1 decoding (grew to 2 after tick 3 admission).
* Tick 7..9: req0 + req1 + req2 decoding (grew to 3 after tick 6 admission).
* Tick 10..15: req2 finished (its 4 tokens exhausted), shrank to 2.
* Tick 16..25: req1 finished (its 12 tokens exhausted), shrank to 1.
* Tick 26..27: req0 finished (its 24 tokens exhausted), shrank to 0.

---

## 6. Regression evidence

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

Matches the recorded counts of Gate 2.5 and Gate 3.1.

The single-process cross-file ordering artifact carried from Gate 2.5
(fake `pydantic` stub in `test_scheduler_abort_ack.py` poisoning
`sys.modules` for a later `test_shell_cancel_cleanup.py`) is
unchanged; Gate 3.2 does not touch `tests/`. Per-file execution
sidesteps the artifact exactly as the gate spec §8 dictates.

---

## 7. Implementation notes

### 7.1 Why a single LLM instance for all four cases

`python/minisgl/engine/engine.py:53` calls
`set_tp_info(rank=..., size=...)`, and
`python/minisgl/distributed/info.py:24` raises
`RuntimeError: TP info has been set` on the second call. Re-booting
an `LLM` inside the same process therefore fails. All four cases
reuse one boot; the `DynamicLLM` subclass supplies transparent
behaviour for A/B/C and instrumentation for D.

### 7.2 Why DynamicLLM subclass rather than direct scheduler ticks

`LLM.generate` accepts a list of prompts up-front and calls
`run_forever` synchronously. That is exactly right for cases A/B/C.
For case D the gate spec asks for a dynamic timeline
(1→2→3→2→1), so we need to inject new pending prompts between
scheduler iterations. The cleanest hook is
`offline_receive_msg`, which the scheduler calls once per iteration.
Overriding it in a script-local subclass lets us stage new prompts
at chosen tick indices and sample the batch shape on the same call.

### 7.3 Zero changes to `python/minisgl/`

Everything above is done at the smoke-script layer. No source under
`python/minisgl/`, `tests/`, or any packaging file was modified.

---

## 8. Support matrix delta (Gate 3.1 → Gate 3.2)

| Capability                                             | Gate 3.1 | Gate 3.2 |
|---|---|---|
| Qwen3-1.7B B=1 prefill + decode                        | PASS     | PASS (unchanged) |
| Qwen3-1.7B B=2 equal-length                            | PASS     | PASS (retested here as Case A) |
| Qwen3-1.7B B=2 ragged prefill                          | UNKNOWN  | **PASS** (Case B) |
| Qwen3-1.7B B=2 mixed-KV decode with solo tail          | UNKNOWN  | **PASS** (Case C) |
| Qwen3-1.7B dynamic admission grow-shrink to B=3        | UNKNOWN  | **PASS** (Case D) |
| Allocator `available_tokens` returns to baseline       | PASS     | PASS (4 cases) |
| `deferred_abort_uids` stays empty                      | PASS     | PASS (4 cases) |
| TP > 1                                                 | UNKNOWN  | UNKNOWN (out of scope) |
| Non-Qwen3 architecture families                        | UNKNOWN  | UNKNOWN (unchanged) |
| Qwen3-4B / 14B / 32B                                   | UNKNOWN  | UNKNOWN (deferred) |
| Regression: 8 hermetic suites (per-file)               | 51 passed | 51 passed (unchanged) |

---

## 9. What is NOT proven at this gate

Explicit exclusions carried forward from the Gate 3.2 opening:

* **TP > 1** for Qwen3-1.7B (or any model). HCCL wiring unexercised.
* **Qwen3-4B / 14B / 32B**, quantized (Qwen3-32B-FP8) and MoE
  (Qwen3-30B-A3B) variants, Qwen3-Next-*, Qwen3-ASR-*,
  Qwen3-Coder-Next. All out of scope.
* **Non-Qwen3 model families** (Llama / Mistral / DeepSeek / MoE).
* **Performance benchmark.** No throughput, latency, or leadership
  claim is made. Timings in the smoke log are for debugging only.
* **Long soak / rolling allocator run.** Only 4 cases were executed.
* **HTTP server restart, crash recovery, non-stream HTTP cancel,
  offline `LLM.abort()`, chunked prefill.** Same NOT REACHED /
  NOT SUPPORTED boundaries as Gate 2.5 / 3.1.
* **Forward/sampler exception recovery** inside the scheduler.
  Unchanged from Gate 2.5.
* **B=3 in a purely synchronous batch** (as opposed to the dynamic
  admission of case D). Case D itself proves the scheduler can host
  three concurrent decoders; a pure synchronous B=3 through
  `LLM.generate([p, p, p], sp)` was not explicitly recorded — the
  gate spec calls for the dynamic-timeline variant and that is what
  was measured.

---

## 10. Freeze boundary

This gate freezes the fact that Mini-SGLang-Ascend at `ad74ee1` runs
the four request shapes A / B / C / D above on Qwen3-1.7B under the
Gate 3.1 frozen TP=1 eager `npu_fia` bf16 envelope, with allocator
invariants held for every case and `deferred_abort_uids` empty
throughout.

It does not claim TP>1 support.
It does not claim any new architecture family.
It does not extend the offline `LLM` driver's public surface.
It does not modify any prior gate verdict, the release tag
`v0.1.0a1`, or the GitHub Release.
It makes no performance claim.
It adds no code under `python/minisgl/` or `tests/`.
