# Gate 4.4 Verdict — TP=2 B=2 mixed-KV decode explicit evidence (Qwen3-0.6B)

**Gate ID:** 4.4 (TP=2 Ascend B=2 mixed-KV decode explicit evidence on Qwen3-0.6B)
**Verdict:** PASS
**Branch:** `gate4.4-tp2-b2-mixed-kv-decode`
**Base commit:** `c063e21` (tip of `ascend-port`, Gate 4.3 merge)
**Freeze commit:** `c3cfecd`
**Date:** 2026-07-11
**Kind:** Real-hardware Ascend 910B1 TP=2 B=2 mixed-KV decode
explicit-evidence proof — two ranks × Qwen3-0.6B × two unequal-length
prompts × `max_new_tokens=8` completes init → weight load → ragged
prefill → decode with a script-only `AscendFIABackend.prepare_metadata`
snapshot hook capturing per-forward-pass `FIAMetadata`, proving the
mixed-KV decode invariant (`query_seq_lens == [1, 1]` with
`kv_seq_lens[0] != kv_seq_lens[1]`) fires on every one of the seven
decode steps, with per-rank allocator invariants held and cross-rank
per-uid output equality.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5 /
3.1 / 3.2 / 3.3 / 3.4 / 4.1 / 4.2 / 4.3, does not mutate release tag
`v0.1.0a1`, does not touch the GitHub Release, CHANGELOG, or release
notes, and does not extend the Ascend port to TP > 2, TP=2 B > 2,
TP=2 dynamic admission / grow-shrink / mid-batch arrival, TP=2 ragged
with non-zero cached_len + extend_len > 1 (Gate 2.2f
`NotImplementedError`), TP=2 timing benchmark, non-Qwen3
architectures, or Qwen3-1.7B / Qwen3-4B / 14B / 32B / quantized / MoE
variants. The result produced is a **correctness proof** for the
mixed-KV decode invariant end-to-end under TP=2; no throughput /
latency / tokens-per-second claim is made. The only new artefact
under `python/minisgl/` at this gate is nothing — Gate 4.4 adds one
new bring-up script and this verdict; no runtime source file is
modified. The metadata capture is a script-local monkey-patch on the
imported `AscendFIABackend.prepare_metadata` classmethod, wrapping the
original method for the duration of one process only.

---

## 1. Verdict summary

**PASS on all four cases across both ranks.**

| Case | Description | rank 0 | rank 1 |
|---|---|---|---|
| A | init-only smoke — `_init_communication` returns, both ranks report `init_status=PASS` | **PASS** | **PASS** |
| B | model-load smoke — per-rank weight sharding + post-load `check_integrity()` | **PASS** | **PASS** |
| C | B=2 unequal-length prompts + decode `max_new_tokens=8` — two unequal-length prompts drive a single `llm.generate([short, long], sp)`, both requests produce 8 output tokens | **PASS** | **PASS** |
| D | decode-step mixed-KV evidence — captured `FIAMetadata` snapshots show `query_seq_lens == [1, 1]` and `kv_seq_lens[0] != kv_seq_lens[1]` on **every** decode step | **PASS** | **PASS** |

Cases A / B still collapse into the same `LLM` boot (`set_tp_info` is
a one-shot); reaching a successful `_snapshot(cache_manager)` after
`LLM.__init__` returns proves init + load simultaneously.

Mixed-KV invariant proven on both ranks:

* `prompt_token_lengths == [2, 12]` — unequal by construction
* one prefill step (step 0) with `query_seq_lens == [2, 12]` and
  `kv_seq_lens == [2, 12]`
* seven decode steps (steps 1..7), each with:
  * `query_seq_lens == [1, 1]` (pure decode over both uids)
  * `kv_seq_lens[0] != kv_seq_lens[1]` — the KV length difference is
    exactly `long_prompt_len − short_prompt_len == 10` on every decode
    step (`[3,13] → [4,14] → [5,15] → [6,16] → [7,17] → [8,18] →
    [9,19]`)
* `mixed_kv_decode_step_count == 7` (all seven decode steps satisfy
  the invariant)

Post-case allocator invariants held on both ranks:

* `available_tokens_after_case == baseline_available_tokens` (952880 on both ranks)
* `deferred_abort_uids == 0`
* `cache_integrity_ok == true`

Structured logs (both ranks, JSON pretty-printed excerpt):

```
GATE4.4_JSONL rank=0 {
  "rank": 0, "world_size": 2, "tp_size": 2,
  "device": "npu:0", "prompt_token_lengths": [2, 12], "batch_size": 2,
  "baseline_available_tokens": 952880, "baseline_free_pages": 59555, "total_pages": 59555,
  "init_status": "PASS", "load_status": "PASS",
  "prefill_status": "PASS", "decode_status": "PASS", "mixed_kv_status": "PASS",
  "actual_output_tokens_per_request": [8, 8],
  "output_texts": [" a city in the southern part of France", "...? A. Mercury B. Venus"],
  "all_step_snapshots": [
    {"step_id": 0, "batch_size": 2, "query_lengths": [2, 12], "kv_lengths": [2, 12]},
    {"step_id": 1, "batch_size": 2, "query_lengths": [1, 1], "kv_lengths": [3, 13]},
    {"step_id": 2, "batch_size": 2, "query_lengths": [1, 1], "kv_lengths": [4, 14]},
    {"step_id": 3, "batch_size": 2, "query_lengths": [1, 1], "kv_lengths": [5, 15]},
    {"step_id": 4, "batch_size": 2, "query_lengths": [1, 1], "kv_lengths": [6, 16]},
    {"step_id": 5, "batch_size": 2, "query_lengths": [1, 1], "kv_lengths": [7, 17]},
    {"step_id": 6, "batch_size": 2, "query_lengths": [1, 1], "kv_lengths": [8, 18]},
    {"step_id": 7, "batch_size": 2, "query_lengths": [1, 1], "kv_lengths": [9, 19]}
  ],
  "decode_step_snapshots": [
    {"step_id": 1, "batch_size": 2, "query_lengths": [1, 1], "kv_lengths": [3, 13]},
    {"step_id": 2, "batch_size": 2, "query_lengths": [1, 1], "kv_lengths": [4, 14]},
    {"step_id": 3, "batch_size": 2, "query_lengths": [1, 1], "kv_lengths": [5, 15]},
    {"step_id": 4, "batch_size": 2, "query_lengths": [1, 1], "kv_lengths": [6, 16]},
    {"step_id": 5, "batch_size": 2, "query_lengths": [1, 1], "kv_lengths": [7, 17]},
    {"step_id": 6, "batch_size": 2, "query_lengths": [1, 1], "kv_lengths": [8, 18]},
    {"step_id": 7, "batch_size": 2, "query_lengths": [1, 1], "kv_lengths": [9, 19]}
  ],
  "mixed_kv_decode_step_count": 7,
  "available_tokens_after_case": 952880, "free_pages_after_case": 59554,
  "deferred_abort_uids": 0, "cache_integrity_ok": true,
  "status": "PASS", "failure_stage": null, "failure_trace_summary": null
}
```

Rank 1 emits a byte-identical snapshot trace (same `all_step_snapshots`
list, same `decode_step_snapshots` list, same
`mixed_kv_decode_step_count`, same `output_texts`, same allocator
numbers). Cross-rank symmetry across every snapshot is itself a
signal that the TP=2 scheduler / paged-cache path is not diverging
between ranks.

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
Attention backend: npu_fia (FIA prefill cached_len==0 branch + FIA
                   mixed-KV decode branch)
page_size:         16
memory_ratio:      0.85
max_running_req:   4
Sampling:          greedy (temperature=0.0, top_k=1, top_p=1.0, ignore_eos=True)
Batching:          B=2 unequal-length (2 tokens + 12 tokens)
Request:           N=8 per request (single ragged prefill + 7 decode steps)
Model:             /mnt/nvme/models/Qwen3-0.6B (dense bf16 Qwen3ForCausalLM)
Launcher:          torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0
                            --master_addr=127.0.0.1 --master_port=29405
Distributed init:  MINISGL_DISTRIBUTED_ADDR=env:// (reuses torchrun store)
                   backend=hccl (primary) + gloo (sidecar via new_group)
                   use_pynccl=False (mandatory on NPU)
Driver:            scripts/gate4_4_tp2_b2_mixed_kv_decode.py
                   per-rank worker under torchrun; both ranks
                   independently enqueue the SAME two prompts and
                   independently record snapshots via a script-local
                   monkey-patch on AscendFIABackend.prepare_metadata.
                   Only one generate() call per process — the Gate
                   2.2f-documented "ragged + non-zero cached_len +
                   extend_len > 1" branch is not exercised.
Prompts:           short = "Paris is"                          (2 tokens)
                   long  = "The largest planet in our solar
                           system by mass and volume is"      (12 tokens)
Metadata capture:  Script-only monkey-patch. Wraps
                   AscendFIABackend.prepare_metadata for the driver
                   process only; the runtime source under
                   python/minisgl/attention/ascend_fia.py is not
                   modified.
```

---

## 3. Launch command

Executed on remote container `<CONTAINER>` at working directory
`/mnt/nvme/LR-606/mini-sglang-ascend-gate44`.

```bash
PYTHONPATH=python torchrun \
  --nproc_per_node=2 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29405 \
  scripts/gate4_4_tp2_b2_mixed_kv_decode.py
```

The script's structured stdout is the primary evidence; each rank
emits exactly one `GATE4.4_JSONL rank=<r> {...}` line containing the
full per-forward-step snapshot list.

---

## 4. Prompt token length evidence

Both ranks independently tokenize the same two prompts with the same
tokenizer, and both report:

```
prompt_token_lengths = [2, 12]
```

The inequality (`2 != 12`) is asserted by the driver before
`generate()` is dispatched — if the two prompts ever tokenized to the
same length the driver would abort with `RuntimeError: Gate 4.4
requires unequal-length prompts`. This inequality is what makes the
per-decode-step KV lengths differ per uid: because the two uids
started prefill with different `cached_len==0` extend lengths, every
decode step sees each uid at a different running KV length, differing
by exactly `12 − 2 = 10` on every step.

FIA branches selected:

* **Prefill (step 0)** — B=2 ragged with `cached_len == 0` on both
  uids. Same branch attested by Gate 4.3.
* **Decode (steps 1..7)** — B=2 mixed-KV decode. Every uid has
  `query_seq_lens[i] == 1` and `kv_seq_lens[i]` differing per uid.
  This is the specific FIA branch Gate 4.4 exists to attest.

The Gate 2.2f-documented unsupported branch ("ragged with non-zero
cached_len + extend_len > 1") is not exercised — only one `generate()`
call is made per process, so no second ragged batch shares prefixes
with the first.

---

## 5. Decode-step metadata evidence (the primary Gate 4.4 evidence)

The script installs a monkey-patch on `AscendFIABackend.prepare_metadata`
before `LLM()` is constructed. The wrapper defers to the original
method (which sets `batch.attn_metadata = FIAMetadata(...)` at the end
of its work) and then reads `batch.attn_metadata.batch_size`,
`.query_seq_lens`, `.kv_seq_lens` and appends a `StepSnapshot` to a
list. The wrapper does not mutate the metadata — it only records it.

Captured per-forward-step snapshots (both ranks agree):

| step_id | batch_size | query_lengths | kv_lengths | interpretation |
|---:|---:|---|---|---|
| 0 | 2 | `[2, 12]` | `[2, 12]` | ragged prefill (cached_len==0 branch) |
| 1 | 2 | `[1, 1]` | `[3, 13]` | mixed-KV decode step 1 (kv diff = 10) |
| 2 | 2 | `[1, 1]` | `[4, 14]` | mixed-KV decode step 2 (kv diff = 10) |
| 3 | 2 | `[1, 1]` | `[5, 15]` | mixed-KV decode step 3 (kv diff = 10) |
| 4 | 2 | `[1, 1]` | `[6, 16]` | mixed-KV decode step 4 (kv diff = 10) |
| 5 | 2 | `[1, 1]` | `[7, 17]` | mixed-KV decode step 5 (kv diff = 10) |
| 6 | 2 | `[1, 1]` | `[8, 18]` | mixed-KV decode step 6 (kv diff = 10) |
| 7 | 2 | `[1, 1]` | `[9, 19]` | mixed-KV decode step 7 (kv diff = 10) |

Derived counts (both ranks agree):

* `len(all_step_snapshots)` = **8** (1 prefill + 7 decode)
* `len(decode_step_snapshots)` = **7** (steps 1..7, all with
  `query_lengths == [1, 1]`)
* `mixed_kv_decode_step_count` = **7** (all 7 decode steps satisfy
  `kv_lengths[0] != kv_lengths[1]`)
* KV-length delta per decode step: exactly `12 − 2 = 10` on every
  step — the invariant this gate exists to prove holds trivially and
  consistently.

Rank 0 and rank 1 produce byte-identical snapshot lists — the two
ranks see the same scheduler shape and pass identical
`(query_seq_lens, kv_seq_lens)` to their FIA operator on every step.
Because the scheduler is process-local (each rank runs its own copy),
this cross-rank agreement is a real signal that the paged-cache /
extend-len / device-len bookkeeping is deterministic and symmetric
across ranks.

Gate 4.4's mixed-KV assertion (`mixed_kv_status == PASS`) requires
`mixed_kv_decode_step_count >= 1`; the observed count of 7 exceeds
this bound on both ranks.

---

## 6. Per-rank output evidence

| Field | rank 0 | rank 1 |
|---|---|---|
| `device` | `npu:0` | `npu:1` |
| `prompt_token_lengths` | `[2, 12]` | `[2, 12]` |
| `batch_size` | 2 | 2 |
| `init_status` | `PASS` | `PASS` |
| `load_status` | `PASS` | `PASS` |
| `prefill_status` | `PASS` | `PASS` |
| `decode_status` | `PASS` | `PASS` |
| `mixed_kv_status` | `PASS` | `PASS` |
| `actual_output_tokens_per_request` | `[8, 8]` | `[8, 8]` |
| `output_texts[0]` | `" a city in the southern part of France"` | `" a city in the southern part of France"` |
| `output_texts[1]` | `"...? A. Mercury B. Venus"` | `"...? A. Mercury B. Venus"` |
| `mixed_kv_decode_step_count` | 7 | 7 |
| `status` | `PASS` | `PASS` |
| `failure_stage` | `null` | `null` |

Cross-rank per-uid output equality is exact (byte-identical string) on
both uids — same result attested by Gate 4.3 under the same envelope
minus the metadata capture. This confirms:

* Mixed-KV decode did not silently overwrite the "wrong" uid's KV
  page.
* Per-uid `device_len` bookkeeping stays correct across all 7 decode
  steps on both ranks.
* The FIA operator's per-request `actual_seq_lengths_kv` handoff
  handles unequal per-uid KV correctly at the operator boundary.
* The all-gathered `ParallelLMHead` continues to reassemble a
  fully-replicated logits tensor identically on both ranks.
* Greedy sampling remains bit-identical across ranks for each uid.

Wall-clock informational (not a timing claim): `generate()` returned
in 2662.53 ms on rank 0 and 2698.15 ms on rank 1 for the full B=2
ragged prefill + 7-step mixed-KV decode with metadata capture. Not
compared against Gate 4.3's numbers — this gate does not report
timing, and the monkey-patched wrapper adds a small per-step overhead
that is deliberately not budgeted.

The gate does not claim these are "correct" answers — Qwen3-0.6B is a
small base model with no instruction tuning. The gate claims that the
scheduler, paged cache, ragged-prefill dispatch, mixed-KV decode
dispatch, and TP=2 communication path correctly walk a B=2
unequal-length batch without allocator corruption or cross-rank
divergence, and that the FIA decode-step invariant is directly
observable in per-step metadata.

---

## 7. Per-rank allocator evidence

| rank | baseline_available_tokens | after case | baseline_free_pages | after case | total_pages | deferred_abort_uids | cache_integrity_ok |
|---|---|---|---|---|---|---|---|
| 0 | 952880 | 952880 | 59555 | 59554 | 59555 | 0 | true |
| 1 | 952880 | 952880 | 59555 | 59554 | 59555 | 0 | true |

Interpretation:

* Baseline `available_tokens = 952880` per rank — identical to Gate
  4.1, Gate 4.2, and Gate 4.3 baselines under the same TP=2 envelope
  (per-rank KV footprint budget is fixed at boot, before any request
  is enqueued).
* `available_tokens` returned exactly to baseline on both ranks —
  the primary allocator invariant. Mixed-KV decode did not leak any
  per-request pages; the two uids' prefill + decode pages were fully
  released once each request hit `max_tokens=8`.
* `free_pages` drift of 1 (59555 → 59554) on both ranks — this is a
  single **evictable** radix-cache prefix page retained after the
  ragged batch, exactly matching the Gate 4.3 evidence under the same
  batch shape. Per Gate 3.1 §4 / Gate 3.4 §7.1,
  `available_size = free_slots + evictable_prefix_pages` (in tokens),
  so 1 evictable page ≈ 16 tokens are counted back into
  `available_tokens`, keeping the token-level invariant exact while
  raw `free_pages` is off by 1.
* `deferred_abort_uids == 0` — no abort path was entered.
* `cache_integrity_ok == true` — allocator invariants (free_slots ∪
  used_slots partition of num_pages, radix-tree ↔ page-table
  consistency) held throughout, including through the 7 mixed-KV
  decode extension steps.

Both ranks show the exact same drift pattern (identical baseline,
identical after-case, identical evictable-page retention). This
symmetry across ranks under mixed-KV decode is itself a signal that
the TP=2 allocator is not diverging between ranks.

---

## 8. First failing stage

**None.** No failure surfaced on the smoke run. Cases A / B / C / D
all reported `PASS` on both ranks. `failure_stage` and
`failure_trace_summary` are `null` on both ranks.

No source-file change under `python/minisgl/` was required at this
gate. The Gate 4.1 minimum fixes (`LLM.__init__` `tp_info` kwarg,
`EngineConfig.distributed_addr` env-var override) already inherited
from `ascend-port` at `c063e21` provided the full driver + launcher
path. The FIA mixed-KV decode branch exercised at case D is the same
branch attested at TP=1 by Gate 3.1 / 3.2 / 3.3 / 3.4 (via the
functional decode path that reached `max_new_tokens > 1`) — Gate 4.4
turns that implicit attestation into explicit per-step evidence under
TP=2.

The metadata-snapshot hook is a script-only monkey-patch on the
imported `AscendFIABackend.prepare_metadata` classmethod. It runs
after the original method sets `batch.attn_metadata` and never mutates
the metadata; the underlying runtime source is not modified.

---

## 9. Regression evidence

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
3.3, Gate 3.4, Gate 4.1, Gate 4.2, Gate 4.3.

Zero source-code changes under `python/minisgl/` at this gate — the
regression rows are expected to be unchanged by construction; the run
confirms it.

---

## 10. Support matrix delta (Gate 4.3 → Gate 4.4)

| Capability                                                | Gate 4.3 | Gate 4.4 |
|---|---|---|
| TP=2 driver support in offline `LLM`                      | PASS | PASS (unchanged) |
| TP=2 launcher rendezvous under torchrun (`env://`)        | PASS | PASS (unchanged) |
| TP=2 Qwen3-0.6B B=2 ragged prefill (cached_len==0) N=8    | PASS | PASS (unchanged, re-attested by driver) |
| TP=2 FIA ragged-prefill dispatch (per-uid logits slice)   | PASS | PASS (unchanged) |
| TP=2 ragged-prefill allocator invariant                   | PASS | PASS (unchanged, 952880 → 952880 on both ranks) |
| TP=2 ragged-prefill cross-rank per-uid determinism        | PASS | PASS (unchanged) |
| **TP=2 mixed-KV decode step observability**               | UNKNOWN | **PASS** (per-forward `FIAMetadata` snapshotted via script hook) |
| **TP=2 mixed-KV decode invariant (query=[1,1], kv unequal)** | UNKNOWN | **PASS** (7/7 decode steps satisfy invariant on both ranks) |
| **TP=2 mixed-KV decode cross-rank snapshot symmetry**     | UNKNOWN | **PASS** (byte-identical snapshot trace across ranks) |
| **TP=2 mixed-KV decode allocator invariant**              | UNKNOWN | **PASS** (available_tokens returns to baseline on both ranks) |
| TP=2 ragged with non-zero cached_len + extend_len > 1     | UNKNOWN | UNKNOWN (Gate 2.2f `NotImplementedError`, not exercised) |
| TP=2 dynamic admission / grow-shrink                      | UNKNOWN | UNKNOWN (out of scope) |
| TP=2 B > 2                                                | UNKNOWN | UNKNOWN (out of scope) |
| TP=2 timing benchmark                                     | UNKNOWN | UNKNOWN (out of scope) |
| TP > 2                                                    | UNKNOWN | UNKNOWN (out of scope) |
| Non-Qwen3 architecture families under TP=2                | UNKNOWN | UNKNOWN (out of scope) |
| Qwen3-1.7B / 4B / 14B / 32B under TP=2                    | UNKNOWN | UNKNOWN (out of scope) |
| Regression: 8 hermetic suites (per-file)                  | 51 passed | 51 passed (unchanged) |

---

## 11. What is NOT proven at this gate

Explicit exclusions carried forward from the Gate 4.4 opening:

* **Ragged + non-zero cached_len + extend_len > 1 under TP=2.**
  Deliberately not exercised — this is the Gate 2.2f documented FIA
  `NotImplementedError` boundary. The gate runs one `generate()` call
  per process, so the second-generate() radix-cache-hit path is
  never reached.
* **Dynamic admission / grow-shrink / mid-batch arrival under TP=2** —
  both uids are enqueued together and complete together at the same
  `max_new_tokens`. There is no mid-batch admission or shrink.
* **B > 2 under TP=2.** Only B=2 exercised.
* **TP=2 timing benchmark.** No TTFT / e2e / tokens-per-second is
  reported. The wall-clock numbers observed (~2.66–2.70 s for the
  full generate call with metadata capture overhead) are not timing
  measurements. The metadata-snapshot hook adds unbudgeted per-step
  overhead by design.
* **CUDAGraph** under TP=2 — `cuda_graph_bs=[]` is locked at eager
  mode. torch_npu does not implement CUDAGraph.
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
  end-to-end is executed.
* **Runtime source refactor of `AscendFIABackend`.** The metadata
  capture is a script-local monkey-patch only; the runtime source
  under `python/minisgl/attention/ascend_fia.py` is untouched. The
  gate does not attest any new invariant on the source file itself.

Verdict decision matrix (from gate open):

| Outcome | Definition | This gate |
|---|---|---|
| PASS    | TP=2 B=2 unequal-length case returns exactly 8 tokens per request on both ranks; ≥1 decode step captured with `query_lengths == [1, 1]` and `kv_lengths[0] != kv_lengths[1]`; rank outputs match; allocator returns to baseline on both ranks | **✔ (this verdict; 7/7 decode steps satisfy the invariant)** |
| PARTIAL | init/load/prefill pass, but no mixed-KV decode step is captured or the metadata invariant fails | not reached |
| BLOCKED | TP=2 launch or model load no longer works | not reached |

---

## 12. Freeze boundary

This gate freezes the fact that Mini-SGLang-Ascend at the freeze
commit — descending from `c063e21` with only the new bring-up script
and this verdict document added — completes a TP=2 Qwen3-0.6B B=2
unequal-length ragged-prefill + 7-step mixed-KV decode end-to-end on
2× Ascend 910B1 under the frozen eager `npu_fia` bf16 greedy
`use_pynccl=False` `MINISGL_DISTRIBUTED_ADDR=env://` envelope, with:

* `prompt_token_lengths == [2, 12]` on both ranks (unequal)
* one prefill snapshot with `query_lengths == [2, 12]` and 7 decode
  snapshots with `query_lengths == [1, 1]` on both ranks
* every decode snapshot satisfies `kv_lengths[0] != kv_lengths[1]`
  with a constant delta of 10 on both ranks
  (`mixed_kv_decode_step_count == 7`)
* `actual_output_tokens_per_request == [8, 8]` on both ranks
* `output_texts` bit-identical cross-rank per uid
* `available_tokens` returning to baseline (952880) on both ranks
* `free_pages` drift of 1 page (retained as evictable radix-cache
  entry, counted back into `available_tokens`)
* `deferred_abort_uids == 0` on both ranks
* `cache_integrity_ok == true` on both ranks
* 8-file regression 51/51 passing
* Zero source-code changes under `python/minisgl/`

It does not claim TP > 2 support.
It does not claim TP=2 B > 2 support.
It does not claim TP=2 ragged with non-zero cached_len + extend_len > 1.
It does not claim TP=2 dynamic admission or grow-shrink.
It does not claim TP=2 timing / throughput / latency parity.
It does not claim TP=2 for any other model.
It does not modify any prior gate verdict, the release tag
`v0.1.0a1`, or the GitHub Release.
It adds no code under `python/minisgl/` or `tests/`; the only new
files are `scripts/gate4_4_tp2_b2_mixed_kv_decode.py` and this
verdict.
