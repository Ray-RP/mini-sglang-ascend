# Gate 4.3 Verdict — TP=2 B=2 ragged prefill (Qwen3-0.6B)

**Gate ID:** 4.3 (TP=2 Ascend B=2 ragged prefill on Qwen3-0.6B)
**Verdict:** PASS
**Branch:** `gate4.3-tp2-b2-ragged-prefill`
**Base commit:** `f806659` (tip of `ascend-port`, Gate 4.2 merge)
**Freeze commit:** `d60d59f`
**Date:** 2026-07-11
**Kind:** Real-hardware Ascend 910B1 TP=2 B=2 ragged-prefill proof —
two ranks × Qwen3-0.6B × two unequal-length prompts ×
`max_new_tokens=8` completes init → weight load → ragged prefill →
decode → symmetric shutdown, with per-uid outputs matching across
ranks and per-rank allocator invariants held.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5 /
3.1 / 3.2 / 3.3 / 3.4 / 4.1 / 4.2, does not mutate release tag
`v0.1.0a1`, does not touch the GitHub Release, CHANGELOG, or release
notes, and does not extend the Ascend port to TP > 2, TP=2 B > 2,
TP=2 mixed-KV decode, TP=2 dynamic admission / grow-shrink, TP=2
ragged with non-zero cached_len + extend_len > 1 (Gate 2.2f
NotImplementedError), TP=2 timing benchmark, non-Qwen3
architectures, or Qwen3-1.7B / Qwen3-4B / 14B / 32B / quantized / MoE
variants. The result produced (`actual_output_tokens_per_request=[8,
8]` per rank) is a **correctness proof**, not a performance
measurement; no throughput / latency / tokens-per-second claim is
made.

---

## 1. Verdict summary

**PASS on all three cases across both ranks.**

| Case | Description | rank 0 | rank 1 |
|---|---|---|---|
| A | init-only smoke — `_init_communication` returns, both ranks report `init_status=PASS` | **PASS** | **PASS** |
| B | model-load smoke — per-rank weight sharding + post-load `check_integrity()` | **PASS** | **PASS** |
| C | B=2 ragged prefill + decode `max_new_tokens=8` — two unequal-length prompts drive a single `llm.generate([short, long], sp)` on the FIA cached_len==0 ragged branch; both requests produce 8 output tokens | **PASS** | **PASS** |

Cases A / B still collapse into the same `LLM` boot (`set_tp_info` is
a one-shot); reaching a successful `_snapshot(cache_manager)` after
`LLM.__init__` returns proves init + load simultaneously.

Ragged-batch invariants held on both ranks:

* `prompt_token_lengths == [2, 12]` — unequal by construction (short
  prompt `"Paris is"` → 2 tokens; long prompt `"The largest planet in
  our solar system by mass and volume is"` → 12 tokens)
* `batch_size == 2`
* `actual_output_tokens_per_request == [8, 8]`
* `output_texts` matches element-wise across ranks — request 0 →
  request 0, request 1 → request 1

Post-case allocator invariants held on both ranks:

* `available_tokens_after_case == baseline_available_tokens` (952880 on both ranks)
* `deferred_abort_uids == 0`
* `cache_integrity_ok == true`

Structured logs (both ranks):

```
GATE4.3_JSONL rank=0 {"rank": 0, "world_size": 2, "tp_size": 2, "model_path": "/mnt/nvme/models/Qwen3-0.6B", "device": "npu:0", "prompt_token_lengths": [2, 12], "batch_size": 2, "baseline_available_tokens": 952880, "baseline_free_pages": 59555, "total_pages": 59555, "init_status": "PASS", "load_status": "PASS", "prefill_status": "PASS", "decode_status": "PASS", "actual_output_tokens_per_request": [8, 8], "output_texts": [" a city in the southern part of France", "...? A. Mercury B. Venus"], "available_tokens_after_case": 952880, "free_pages_after_case": 59554, "deferred_abort_uids": 0, "cache_integrity_ok": true, "status": "PASS", "failure_stage": null, "failure_trace_summary": null}
GATE4.3_JSONL rank=1 {"rank": 1, "world_size": 2, "tp_size": 2, "model_path": "/mnt/nvme/models/Qwen3-0.6B", "device": "npu:1", "prompt_token_lengths": [2, 12], "batch_size": 2, "baseline_available_tokens": 952880, "baseline_free_pages": 59555, "total_pages": 59555, "init_status": "PASS", "load_status": "PASS", "prefill_status": "PASS", "decode_status": "PASS", "actual_output_tokens_per_request": [8, 8], "output_texts": [" a city in the southern part of France", "...? A. Mercury B. Venus"], "available_tokens_after_case": 952880, "free_pages_after_case": 59554, "deferred_abort_uids": 0, "cache_integrity_ok": true, "status": "PASS", "failure_stage": null, "failure_trace_summary": null}
```

Cross-rank per-uid equality (both ranks return the same `output_texts`
list in the same order):

* `output_texts[0]` on rank 0 == `output_texts[0]` on rank 1 → `" a city in the southern part of France"`
* `output_texts[1]` on rank 0 == `output_texts[1]` on rank 1 → `"...? A. Mercury B. Venus"`

The gate does not claim these are "correct" answers — Qwen3-0.6B is a
small base model with no instruction tuning. The gate claims that the
scheduler, paged cache, ragged-prefill dispatch, and TP=2
communication path correctly walk a B=2 ragged batch without
allocator corruption or cross-rank divergence.

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
Attention backend: npu_fia (FIA ragged prefill / cached_len==0 branch)
page_size:         16
memory_ratio:      0.85
max_running_req:   4
Sampling:          greedy (temperature=0.0, top_k=1, top_p=1.0, ignore_eos=True)
Batching:          B=2 ragged (two unequal-length prompts)
Request:           N=8 per request (single ragged prefill + 7 decode steps)
Model:             /mnt/nvme/models/Qwen3-0.6B (dense bf16 Qwen3ForCausalLM)
Launcher:          torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0
                            --master_addr=127.0.0.1 --master_port=29404
Distributed init:  MINISGL_DISTRIBUTED_ADDR=env:// (reuses torchrun store)
                   backend=hccl (primary) + gloo (sidecar via new_group)
                   use_pynccl=False (mandatory on NPU)
Driver:            scripts/gate4_3_tp2_b2_ragged_prefill.py
                   per-rank worker under torchrun; both ranks
                   independently enqueue the SAME two prompts.
                   Only one generate() call per process — the second
                   ragged batch that would trigger the Gate 2.2f
                   unsupported "ragged + non-zero cached_len +
                   extend_len > 1" branch is deliberately never sent.
Prompts:           short = "Paris is"                          (2 tokens)
                   long  = "The largest planet in our solar
                           system by mass and volume is"      (12 tokens)
```

---

## 3. Launch command

Executed on remote container `<CONTAINER>` at working directory
`/mnt/nvme/LR-606/mini-sglang-ascend-gate43`.

```bash
PYTHONPATH=python torchrun \
  --nproc_per_node=2 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29404 \
  scripts/gate4_3_tp2_b2_ragged_prefill.py
```

The script's structured stdout is the primary evidence; each rank
emits exactly one `GATE4.3_JSONL rank=<r> {...}` line.

---

## 4. Prompt token length evidence

Both ranks independently tokenize the same two prompts with the same
tokenizer, and both report:

```
prompt_token_lengths = [2, 12]
```

The inequality (`2 != 12`) is asserted by the driver before
`generate()` is dispatched — if the two prompts ever tokenized to the
same length the driver would abort with `RuntimeError: Gate 4.3
requires unequal-length prompts`, so this gate cannot silently regress
to an equal-length batch (which would defeat the ragged goal).

FIA branch selected: because both uids are freshly enqueued (no prior
generate() call has left a radix-cache prefix hit), the scheduler
sees `cached_len == 0` on both uids at the prefill step. This is the
`AscendFIAMetadata` supported branch — B ≥ 1 ragged prefill with
`cached_len == 0` (Gate 2.2f). The unsupported "ragged +
non-zero cached_len + extend_len > 1" branch (Gate 2.2f documented
`NotImplementedError`) is not touched because only one generate()
call is made per process.

---

## 5. Per-rank output evidence

| Field | rank 0 | rank 1 |
|---|---|---|
| `device` | `npu:0` | `npu:1` |
| `prompt_token_lengths` | `[2, 12]` | `[2, 12]` |
| `batch_size` | 2 | 2 |
| `init_status` | `PASS` | `PASS` |
| `load_status` | `PASS` | `PASS` |
| `prefill_status` | `PASS` | `PASS` |
| `decode_status` | `PASS` | `PASS` |
| `actual_output_tokens_per_request` | `[8, 8]` | `[8, 8]` |
| `output_texts[0]` | `" a city in the southern part of France"` | `" a city in the southern part of France"` |
| `output_texts[1]` | `"...? A. Mercury B. Venus"` | `"...? A. Mercury B. Venus"` |
| `status` | `PASS` | `PASS` |
| `failure_stage` | `null` | `null` |

Cross-rank equality is exact (same string) on both uids. This
confirms:

* The ragged-prefill dispatch correctly slices per-uid logits at the
  end-of-prompt boundary (position `[2, 14]` in a packed
  `sum_input_len=14` tensor) on both ranks.
* The `ParallelLMHead` all-gather reassembles a fully-replicated
  logits tensor identically on both ranks.
* The sampler picks bit-identical next-tokens on both ranks (greedy
  determinism).
* The decode step correctly advances each uid's per-page KV write on
  both ranks — divergence would have surfaced by token 2 or 3.

Wall-clock informational (not a timing claim): `generate()` returned
in 2451 ms on rank 0 and 2514 ms on rank 1 for the full B=2 ragged
+ 8-token decode. Not a benchmark.

---

## 6. Per-rank allocator evidence

| rank | baseline_available_tokens | after case C | baseline_free_pages | after case C | total_pages | deferred_abort_uids | cache_integrity_ok |
|---|---|---|---|---|---|---|---|
| 0 | 952880 | 952880 | 59555 | 59554 | 59555 | 0 | true |
| 1 | 952880 | 952880 | 59555 | 59554 | 59555 | 0 | true |

Interpretation:

* Baseline `available_tokens = 952880` per rank — identical to Gate
  4.1 and Gate 4.2 baselines under the same TP=2 envelope.
* `available_tokens` returned exactly to baseline on both ranks —
  the primary allocator invariant asserted by this gate.
  Ragged-prefill did not leak any per-request pages; the two uids'
  prefill+decode pages were fully released once each request hit
  `max_tokens=8`.
* `free_pages` drift of 1 (59555 → 59554) on both ranks — this is a
  single **evictable** radix-cache prefix page retained after the
  ragged batch. Per the Gate 3.1 §4 / Gate 3.4 §7.1 rationale,
  `available_size` is defined as `free_slots + evictable_prefix_pages`
  (tokens, not raw pages); one evictable page ≈ 16 tokens which is
  counted back into `available_tokens`, so the token-level invariant
  holds exactly while the raw `free_pages` count is off by 1. This
  is the expected shape for a batch that filled ≥ 1 page of a real
  prompt on a cold cache — Gate 3.4 case D exhibited the same
  behaviour (`free_pages` drift range 29403..29412 across 24 measured
  repeats vs baseline invariance in `available_tokens`).
* `deferred_abort_uids == 0` — no abort path was entered.
* `cache_integrity_ok == true` — allocator invariants (free_slots ∪
  used_slots partition of num_pages, radix-tree ↔ page-table
  consistency) held throughout.

Both ranks show the exact same drift pattern (identical baseline,
identical after-case, identical evictable-page retention). This
symmetry across ranks under a ragged batch is itself a signal that
the TP=2 allocator is not diverging between ranks.

---

## 7. First failing stage

**None.** No failure surfaced on the smoke run. Cases A / B / C all
reported `PASS` on both ranks. `failure_stage` and
`failure_trace_summary` are `null` on both ranks.

No new source-file change under `python/minisgl/` was required at
this gate. The Gate 4.1 minimum fixes (`LLM.__init__` `tp_info`
kwarg, `EngineConfig.distributed_addr` env-var override) already
inherited from `ascend-port` at `f806659` provided the full driver +
launcher path. Case C's ragged prefill exercised the same
`AscendFIAMetadata` `cached_len==0` branch that Gate 3.1 / 3.2 / 3.3
/ 3.4 have already attested at TP=1 — Gate 4.3 confirms the branch
holds under TP=2 as well.

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

Matches the recorded counts of Gate 2.5, Gate 3.1, Gate 3.2, Gate
3.3, Gate 3.4, Gate 4.1, Gate 4.2.

Zero source-code changes under `python/minisgl/` at this gate — the
regression rows are expected to be unchanged by construction; the run
confirms it.

---

## 9. Support matrix delta (Gate 4.2 → Gate 4.3)

| Capability                                                | Gate 4.2 | Gate 4.3 |
|---|---|---|
| TP=2 driver support in offline `LLM`                      | PASS | PASS (unchanged) |
| TP=2 launcher rendezvous under torchrun (`env://`)        | PASS | PASS (unchanged) |
| TP=2 Qwen3-0.6B B=1 single request N=8                    | Gate 4.1 PASS | not re-attested |
| TP=2 Qwen3-0.6B B=2 equal-length N=8                      | PASS | not re-attested |
| TP=2 Qwen3-0.6B **B=2 ragged prefill (cached_len==0)** N=8 | UNKNOWN | **PASS** |
| TP=2 FIA ragged-prefill dispatch (per-uid logits slice)   | UNKNOWN | **PASS** |
| TP=2 ragged-prefill allocator invariant (`available_size` returns) | UNKNOWN | **PASS** (952880 → 952880 on both ranks; 1 evictable page retained) |
| TP=2 ragged-prefill cross-rank per-uid determinism        | UNKNOWN | **PASS** (bit-identical `output_texts[i]` across ranks) |
| TP=2 ragged with non-zero cached_len + extend_len > 1     | UNKNOWN | UNKNOWN (Gate 2.2f `NotImplementedError`, not exercised) |
| TP=2 mixed-KV decode                                      | UNKNOWN | UNKNOWN (out of scope) |
| TP=2 dynamic admission / grow-shrink                      | UNKNOWN | UNKNOWN (out of scope) |
| TP=2 B > 2                                                | UNKNOWN | UNKNOWN (out of scope) |
| TP=2 timing benchmark                                     | UNKNOWN | UNKNOWN (out of scope) |
| TP > 2                                                    | UNKNOWN | UNKNOWN (out of scope) |
| Non-Qwen3 architecture families under TP=2                | UNKNOWN | UNKNOWN (out of scope) |
| Qwen3-1.7B / 4B / 14B / 32B under TP=2                    | UNKNOWN | UNKNOWN (out of scope) |
| Regression: 8 hermetic suites (per-file)                  | 51 passed | 51 passed (unchanged) |

---

## 10. What is NOT proven at this gate

Explicit exclusions carried forward from the Gate 4.3 opening:

* **Ragged + non-zero cached_len + extend_len > 1 under TP=2.**
  Deliberately not exercised — this is the Gate 2.2f documented FIA
  `NotImplementedError` boundary. The gate runs one generate() call
  per process, so the second-generate() radix-cache-hit path is
  never reached.
* **Mixed-KV decode under TP=2** (e.g. one uid in decode + one uid
  arriving with a different `cached_len`). Not exercised.
* **Dynamic admission / grow-shrink under TP=2** — both uids are
  enqueued together and complete together at the same
  `max_new_tokens`. There is no mid-batch admission or shrink.
* **B > 2 under TP=2.** Only B=2 exercised.
* **TP=2 timing benchmark.** No TTFT / e2e / tokens-per-second is
  reported. The single wall-clock numbers observed (~2.45–2.51 s for
  the full ragged batch + decode) are not timing measurements.
* **CUDAGraph** under TP=2 — `cuda_graph_bs=[]` is locked at eager
  mode.
* **Long-sequence / context-length sweep under TP=2.** The two
  prompts fit in one page each.
* **TP > 2** and multi-node (NNODES > 1).
* **Qwen3-1.7B / Qwen3-4B / 14B / 32B under TP=2**, quantized
  (Qwen3-32B-FP8), MoE (Qwen3-30B-A3B), Qwen3-Next-*, Qwen3-ASR-*,
  Qwen3-Coder-Next. All out of scope.
* **Non-Qwen3 model families under TP=2**.
* **HTTP server under TP=2**, non-stream HTTP cancel, offline
  `LLM.abort()`, chunked prefill. Same boundaries as prior gates.
* **Forward/sampler exception recovery** inside the scheduler under
  TP=2. Unchanged from Gate 2.5.
* **Long soak / rolling allocator run under TP=2.** Only one B=2
  ragged end-to-end is executed.

Verdict decision matrix (from gate open):

| Outcome | Definition | This gate |
|---|---|---|
| PASS    | TP=2 B=2 ragged prefill returns exactly 8 tokens per request on both ranks, prompt lengths are unequal, rank outputs match by uid, allocator returns to baseline on both ranks | **✔ (this verdict)** |
| PARTIAL | init/load pass, but ragged prefill or decode fails with clear first failing stage | not reached |
| BLOCKED | TP=2 launch or model load no longer works | not reached |

---

## 11. Freeze boundary

This gate freezes the fact that Mini-SGLang-Ascend at `d60d59f` —
descending from `f806659` with only the
new bring-up script and this verdict document added — completes a
TP=2 Qwen3-0.6B B=2 ragged (cached_len==0) prefill end-to-end on 2×
Ascend 910B1 under the frozen eager `npu_fia` bf16 greedy
`use_pynccl=False` `MINISGL_DISTRIBUTED_ADDR=env://` envelope, with:

* `prompt_token_lengths == [2, 12]` on both ranks (unequal)
* `actual_output_tokens_per_request == [8, 8]` on both ranks
* `output_texts` bit-identical cross-rank per uid
* `available_tokens` returning to baseline (952880) on both ranks
* `free_pages` drift of 1 page (retained as evictable radix-cache
  entry, counted back into `available_tokens`)
* `deferred_abort_uids == 0` on both ranks
* `cache_integrity_ok == true` on both ranks
* 8-file regression 51/51 passing

It does not claim TP > 2 support.
It does not claim TP=2 B > 2 support.
It does not claim TP=2 ragged with non-zero cached_len + extend_len > 1.
It does not claim TP=2 mixed-KV decode or dynamic admission.
It does not claim TP=2 timing / throughput / latency parity.
It does not claim TP=2 for any other model.
It does not modify any prior gate verdict, the release tag
`v0.1.0a1`, or the GitHub Release.
It adds no code under `python/minisgl/` or `tests/`.
