# Gate 4.2 Verdict — TP=2 B=2 equal-length batching (Qwen3-0.6B)

**Gate ID:** 4.2 (TP=2 Ascend B=2 equal-length batching on Qwen3-0.6B)
**Verdict:** PASS
**Branch:** `gate4.2-tp2-b2-equal-length`
**Base commit:** `7e8b1aa` (tip of `ascend-port`, Gate 4.1 merge)
**Freeze commit:** `fcd84f0`
**Date:** 2026-07-11
**Kind:** Real-hardware Ascend 910B1 TP=2 B=2 equal-length proof — two
ranks × Qwen3-0.6B × 2 identical prompts × `max_new_tokens=8` completes
init → weight load → prefill → decode → symmetric shutdown with
bit-identical greedy output on both ranks and both uids, and per-rank
allocator invariants held.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5 /
3.1 / 3.2 / 3.3 / 3.4 / 4.1, does not mutate release tag `v0.1.0a1`,
does not touch the GitHub Release, CHANGELOG, or release notes, and
does not extend the Ascend port to TP > 2, TP=2 B > 2, TP=2 ragged
prefill, TP=2 mixed-KV decode, TP=2 dynamic admission /
grow-shrink, TP=2 timing benchmark, non-Qwen3 architectures, or
Qwen3-1.7B / Qwen3-4B / 14B / 32B / quantized / MoE variants. The
result produced (`actual_output_tokens_per_request=[8, 8]` per rank)
is a **correctness proof**, not a performance measurement; no
throughput / latency / tokens-per-second claim is made.

---

## 1. Verdict summary

**PASS on all three cases across both ranks.**

| Case | Description | rank 0 | rank 1 |
|---|---|---|---|
| A | init-only smoke — `_init_communication` returns, both ranks report `init_status=PASS` | **PASS** | **PASS** |
| B | model-load smoke — per-rank weight sharding + `check_integrity()` after boot | **PASS** | **PASS** |
| C | B=2 equal-length prefill + decode `max_new_tokens=8` — `llm.generate([p, p], sp)` returns two 8-token results with bit-identical text on both ranks | **PASS** | **PASS** |

Cases A / B still collapse into the same `LLM` boot (per §3 of the
Gate 4.1 opening carried forward — `set_tp_info` is a one-shot);
reaching a successful `_snapshot(cache_manager)` after `LLM.__init__`
returns proves init + load simultaneously.

Post-case allocator invariants held on both ranks:

* `available_tokens_after_case == baseline_available_tokens` (952880 on both ranks)
* `free_pages_after_case == baseline_free_pages == total_pages` (59555 on both ranks)
* `deferred_abort_uids == 0`
* `cache_integrity_ok == true`

Equal-length invariant on both ranks:

* `prompt_token_lengths == [5, 5]` (both prompts tokenized to 5 tokens each)
* `batch_size == 2`
* `actual_output_tokens_per_request == [8, 8]`
* `output_texts` matches element-wise across ranks — request 0 → request 0, request 1 → request 1.

Structured logs (both ranks):

```
GATE4.2_JSONL rank=0 {"rank": 0, "world_size": 2, "tp_size": 2, "model_path": "/mnt/nvme/models/Qwen3-0.6B", "device": "npu:0", "prompt_token_lengths": [5, 5], "batch_size": 2, "baseline_available_tokens": 952880, "baseline_free_pages": 59555, "total_pages": 59555, "init_status": "PASS", "load_status": "PASS", "prefill_status": "PASS", "decode_status": "PASS", "actual_output_tokens_per_request": [8, 8], "output_texts": [" Paris. The capital of Italy is Rome", " Paris. The capital of Italy is Rome"], "available_tokens_after_case": 952880, "free_pages_after_case": 59555, "deferred_abort_uids": 0, "cache_integrity_ok": true, "status": "PASS", "failure_stage": null, "failure_trace_summary": null}
GATE4.2_JSONL rank=1 {"rank": 1, "world_size": 2, "tp_size": 2, "model_path": "/mnt/nvme/models/Qwen3-0.6B", "device": "npu:1", "prompt_token_lengths": [5, 5], "batch_size": 2, "baseline_available_tokens": 952880, "baseline_free_pages": 59555, "total_pages": 59555, "init_status": "PASS", "load_status": "PASS", "prefill_status": "PASS", "decode_status": "PASS", "actual_output_tokens_per_request": [8, 8], "output_texts": [" Paris. The capital of Italy is Rome", " Paris. The capital of Italy is Rome"], "available_tokens_after_case": 952880, "free_pages_after_case": 59555, "deferred_abort_uids": 0, "cache_integrity_ok": true, "status": "PASS", "failure_stage": null, "failure_trace_summary": null}
```

Rank 0 and rank 1 produced **byte-identical** per-request `output_texts`.
Because the two prompts are identical strings, both requests
degenerate to the same greedy trajectory — expected in the frozen
envelope (temperature=0.0, top_k=1). What the gate proves is that the
scheduler and paged-cache path correctly walks a B=2 equal-length
batch under TP=2 without corrupting the allocator or diverging
across ranks.

---

## 2. Envelope (locked at this gate)

```
Hardware:          Ascend 910B1 (2 dies × 64 GiB HBM)
                   rank 0 → npu:0, rank 1 → npu:1
Container:         <CONTAINER> on remote <HOST>:<PORT>
Software:          Python 3.11.14
                   torch 2.9.0+cpu
                   torch_npu 2.9.0.post1+gitee7ba04
                   CANN 8.5.1 (compiler build 20250725)
Parallelism:       TP=2 (world_size=2, rank ∈ {0,1})
Execution:         eager (cuda_graph_bs=[]; torch_npu has no CUDAGraph)
Attention backend: npu_fia (FIA equal-length prefill branch)
page_size:         16
memory_ratio:      0.85
max_running_req:   4
Sampling:          greedy (temperature=0.0, top_k=1, top_p=1.0, ignore_eos=True)
Batching:          B=2 equal-length (two identical prompts)
Request:           N=8 per request (single prefill + 7 decode steps)
Model:             /mnt/nvme/models/Qwen3-0.6B (dense bf16 Qwen3ForCausalLM)
Launcher:          torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0
                            --master_addr=127.0.0.1 --master_port=29403
Distributed init:  MINISGL_DISTRIBUTED_ADDR=env:// (reuses torchrun store)
                   backend=hccl (primary) + gloo (sidecar via new_group)
                   use_pynccl=False (mandatory on NPU)
Driver:            scripts/gate4_2_tp2_b2_equal_length.py
                   per-rank worker under torchrun; both ranks
                   independently enqueue the SAME two prompts
                   (offline_mode short-circuits ZMQ fanout, so both
                   ranks drive their own generate() in lockstep).
```

---

## 3. Launch command

Executed on remote container `<CONTAINER>` at working directory
`/mnt/nvme/LR-606/mini-sglang-ascend-gate42`.

```bash
PYTHONPATH=python torchrun \
  --nproc_per_node=2 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29403 \
  scripts/gate4_2_tp2_b2_equal_length.py
```

The script's structured stdout is the primary evidence; each rank
emits exactly one `GATE4.2_JSONL rank=<r> {...}` line.

---

## 4. Per-rank output evidence

| Field | rank 0 | rank 1 |
|---|---|---|
| `device` | `npu:0` | `npu:1` |
| `prompt_token_lengths` | `[5, 5]` | `[5, 5]` |
| `batch_size` | `2` | `2` |
| `init_status` | `PASS` | `PASS` |
| `load_status` | `PASS` | `PASS` |
| `prefill_status` | `PASS` | `PASS` |
| `decode_status` | `PASS` | `PASS` |
| `actual_output_tokens_per_request` | `[8, 8]` | `[8, 8]` |
| `output_texts[0]` | `" Paris. The capital of Italy is Rome"` | `" Paris. The capital of Italy is Rome"` |
| `output_texts[1]` | `" Paris. The capital of Italy is Rome"` | `" Paris. The capital of Italy is Rome"` |
| `status` | `PASS` | `PASS` |
| `failure_stage` | `null` | `null` |

Cross-rank equality:

* `output_texts[0]` on rank 0 == `output_texts[0]` on rank 1 ✓
* `output_texts[1]` on rank 0 == `output_texts[1]` on rank 1 ✓
* `actual_output_tokens_per_request` on rank 0 == `actual_output_tokens_per_request` on rank 1 ✓

Wall-clock informational (not a timing claim): `generate()` returned
in 2433 ms on rank 0 and 2430 ms on rank 1 for the full 8-token B=2
batch. Not compared against Gate 4.1's B=1 number — this gate does
not report timing.

---

## 5. Per-rank allocator evidence

| rank | baseline_available_tokens | after case C | baseline_free_pages | after case C | total_pages | deferred_abort_uids | cache_integrity_ok |
|---|---|---|---|---|---|---|---|
| 0 | 952880 | 952880 | 59555 | 59555 | 59555 | 0 | true |
| 1 | 952880 | 952880 | 59555 | 59555 | 59555 | 0 | true |

Interpretation:

* Baseline `available_tokens = 952880` per rank — identical to
  Gate 4.1's baseline under the same TP=2 envelope. B=2 does not
  change the per-rank KV footprint budget (the budget is computed at
  boot, before any request is enqueued).
* `available_tokens` returned exactly to baseline on both ranks after
  the B=2 batch completed — matches the Gate 3.1 / 3.4 / 4.1
  allocator invariant. Two 5-token prefill pages + 7 decode-step
  extensions were fully released after the requests finished.
* `free_pages` returned to `total_pages` on both ranks — the radix
  cache did not retain any evictable pages after `generate()`
  returned (the batch was a fresh cold path with no prior prefix hit).
* `deferred_abort_uids == 0` — no abort path was entered (both
  requests completed normally on `max_tokens=8`).
* `cache_integrity_ok == true` — allocator invariants (free_slots ∪
  used_slots partition of num_pages, radix-tree ↔ page-table
  consistency) held throughout.

---

## 6. First failing stage

**None.** No failure surfaced on the smoke run. Cases A / B / C all
reported `PASS` on both ranks. `failure_stage` and
`failure_trace_summary` are `null` on both ranks.

No new source-file change under `python/minisgl/` was required at
this gate. The Gate 4.1 minimum fixes (LLM `tp_info` kwarg,
`EngineConfig.distributed_addr` env-var override) already inherited
from `ascend-port` at `7e8b1aa` provided the full driver-layer /
launcher-integration path.

---

## 7. Regression evidence

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

Matches the recorded counts of Gate 2.5, Gate 3.1, Gate 3.2, Gate
3.3, Gate 3.4, Gate 4.1.

Zero source-code changes under `python/minisgl/` at this gate — the
regression rows are expected to be unchanged by construction; the run
confirms it.

---

## 8. Support matrix delta (Gate 4.1 → Gate 4.2)

| Capability                                             | Gate 4.1 | Gate 4.2 |
|---|---|---|
| TP=2 driver support in offline `LLM`                   | PASS | PASS (unchanged) |
| TP=2 launcher rendezvous under torchrun (`env://`)     | PASS | PASS (unchanged) |
| TP=2 `Engine._init_communication` HCCL branch          | PASS | PASS (unchanged) |
| TP=2 Qwen3-0.6B B=1 single request N=8                 | PASS | not re-attested |
| TP=2 Qwen3-0.6B **B=2 equal-length** N=8               | UNKNOWN | **PASS** |
| TP=2 scheduler B=2 admission + prefill batching        | UNKNOWN | **PASS** (two uids share one prefill step) |
| TP=2 scheduler B=2 decode step (both uids advance in lockstep) | UNKNOWN | **PASS** (both uids reach N=8) |
| TP=2 allocator invariant under B=2                     | UNKNOWN | **PASS** (952880 → 952880 on both ranks) |
| TP=2 B=2 rank output determinism                       | UNKNOWN | **PASS** (bit-identical `output_texts` cross-rank per uid) |
| TP=2 B > 2                                             | UNKNOWN | UNKNOWN (out of scope) |
| TP=2 ragged prefill                                    | UNKNOWN | UNKNOWN (Gate 2.2f FIA boundary, out of scope) |
| TP=2 mixed-KV decode                                   | UNKNOWN | UNKNOWN (out of scope) |
| TP=2 dynamic admission / grow-shrink                   | UNKNOWN | UNKNOWN (out of scope) |
| TP=2 timing benchmark                                  | UNKNOWN | UNKNOWN (out of scope) |
| TP > 2                                                 | UNKNOWN | UNKNOWN (out of scope) |
| Non-Qwen3 architecture families under TP=2             | UNKNOWN | UNKNOWN (out of scope) |
| Qwen3-1.7B / 4B / 14B / 32B under TP=2                 | UNKNOWN | UNKNOWN (out of scope) |
| Regression: 8 hermetic suites (per-file)               | 51 passed | 51 passed (unchanged) |

---

## 9. What is NOT proven at this gate

Explicit exclusions carried forward from the Gate 4.2 opening:

* **B > 2 under TP=2.** Only B=2 exercised.
* **Ragged prefill under TP=2.** Prompts are equal-length by
  construction (same string twice → same tokenized length). The Gate
  2.2f FIA `NotImplementedError` on ragged + non-zero cached_len +
  extend_len > 1 is unchanged; ragged batches under TP=2 are not
  attested.
* **Mixed-KV decode under TP=2** (e.g. one uid in decode + one uid
  arriving with a different `cached_len`). Not exercised.
* **Dynamic admission / grow-shrink under TP=2** — both uids are
  enqueued together and complete together at the same
  `max_new_tokens`. There is no mid-batch admission or shrink.
* **TP=2 timing benchmark.** No TTFT / e2e / tokens-per-second is
  reported. The single wall-clock number observed (~2.43 s for the
  full generate call) is not a timing measurement.
* **CUDAGraph** under TP=2 — `cuda_graph_bs=[]` is locked at eager
  mode. torch_npu does not implement CUDAGraph.
* **Long-sequence / context-length sweep under TP=2.** Prompts fit in
  one page each.
* **TP > 2** and multi-node (NNODES > 1). Unexercised.
* **Qwen3-1.7B / Qwen3-4B / 14B / 32B under TP=2**, quantized
  (Qwen3-32B-FP8), MoE (Qwen3-30B-A3B), Qwen3-Next-*, Qwen3-ASR-*,
  Qwen3-Coder-Next. All out of scope.
* **Non-Qwen3 model families under TP=2** (Llama / Mistral / DeepSeek
  / MoE).
* **HTTP server under TP=2**, non-stream HTTP cancel, offline
  `LLM.abort()`, chunked prefill. Same NOT REACHED / NOT SUPPORTED
  boundaries as prior gates.
* **Forward/sampler exception recovery** inside the scheduler under
  TP=2. Unchanged from Gate 2.5.
* **Long soak / rolling allocator run under TP=2.** Only one B=2
  end-to-end is executed.

Verdict decision matrix (from gate open):

| Outcome | Definition | This gate |
|---|---|---|
| PASS    | TP=2 B=2 equal-length case returns exactly 8 tokens per request on both ranks; rank outputs match; allocator returns to baseline on both ranks | **✔ (this verdict)** |
| PARTIAL | init/load pass, but B=2 forward/decode fails with clear first failing stage | not reached |
| BLOCKED | TP=2 launch or model load no longer works | not reached |

---

## 10. Freeze boundary

This gate freezes the fact that Mini-SGLang-Ascend at `fcd84f0` —
descending from `7e8b1aa` with only the
new bring-up script and this verdict document added — completes a
TP=2 Qwen3-0.6B B=2 equal-length batching end-to-end on 2× Ascend
910B1 under the frozen eager `npu_fia` bf16 greedy `use_pynccl=False`
`MINISGL_DISTRIBUTED_ADDR=env://` envelope, with:

* `actual_output_tokens_per_request == [8, 8]` on both ranks
* `output_texts` bit-identical across ranks for both uids
* `available_tokens` returning to baseline (952880) on both ranks
* `deferred_abort_uids == 0` on both ranks
* `cache_integrity_ok == true` on both ranks
* 8-file regression 51/51 passing

It does not claim TP > 2 support.
It does not claim TP=2 B > 2 support.
It does not claim TP=2 ragged prefill support.
It does not claim TP=2 mixed-KV decode or dynamic admission.
It does not claim TP=2 timing / throughput / latency parity.
It does not claim TP=2 for any other model.
It does not modify any prior gate verdict, the release tag
`v0.1.0a1`, or the GitHub Release.
It adds no code under `python/minisgl/` or `tests/`.
