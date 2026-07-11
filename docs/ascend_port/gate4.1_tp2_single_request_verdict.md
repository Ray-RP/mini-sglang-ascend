# Gate 4.1 Verdict — TP=2 Single-Request Bring-up (Qwen3-0.6B)

**Gate ID:** 4.1 (TP=2 Ascend single-request bring-up on Qwen3-0.6B)
**Verdict:** PASS
**Branch:** `gate4.1-tp2-single-request-bringup`
**Base commit:** `c651d91` (tip of `ascend-port`, Gate 3.4 merge)
**Freeze commit:** `8431743`
**Date:** 2026-07-11
**Kind:** Real-hardware Ascend 910B1 first-ever TP=2 end-to-end proof
— two ranks × Qwen3-0.6B × `max_new_tokens=8` single request completes
init → weight load → prefill → decode → symmetric shutdown with
bit-identical greedy output across ranks.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5 /
3.1 / 3.2 / 3.3 / 3.4, does not mutate release tag `v0.1.0a1`, does
not touch the GitHub Release, CHANGELOG, or release notes, and does
not extend the Ascend port to TP > 2, batching under TP=2 (B > 1),
ragged prefill under TP=2, timing benchmark under TP=2, non-Qwen3
architectures, or Qwen3-1.7B / Qwen3-4B / 14B / 32B / quantized / MoE
variants. The single number produced (`actual_output_tokens=8` per
rank) is a **correctness proof**, not a performance measurement; no
throughput / latency / tokens-per-second claim is made.

---

## 1. Verdict summary

**PASS on all three cases across both ranks.**

| Case | Description | rank 0 | rank 1 |
|---|---|---|---|
| A | init-only smoke — `_init_communication` returns, both ranks report `init_status=PASS` | **PASS** | **PASS** |
| B | model-load smoke — `load_weight` streams safetensors and applies per-rank sharding; both ranks report `load_status=PASS` and successful post-load `check_integrity()` | **PASS** | **PASS** |
| C | single-request `max_new_tokens=8` — `llm.generate([prompt], sp)` returns exactly 8 output tokens on both ranks with bit-identical greedy text | **PASS** | **PASS** |

Cases A / B collapse into the same `LLM` boot (per §3 of the Gate 4.1
opening — `set_tp_info` is a one-shot); reaching a successful
`_snapshot(cache_manager)` after `LLM.__init__` returns proves both
init and load simultaneously.

Post-case allocator invariants held on both ranks:

* `available_tokens_after_case == baseline_available_tokens` (952880 on both ranks)
* `free_pages_after_case == baseline_free_pages == total_pages` (59555 on both ranks)
* `deferred_abort_uids == 0`
* `cache_integrity_ok == true`

Structured logs (both ranks):

```
GATE4.1_JSONL rank=0 {"rank": 0, "world_size": 2, "model_path": "/mnt/nvme/models/Qwen3-0.6B", "tp_size": 2, "device": "npu:0", "baseline_available_tokens": 952880, "baseline_free_pages": 59555, "total_pages": 59555, "init_status": "PASS", "load_status": "PASS", "prefill_status": "PASS", "decode_status": "PASS", "actual_output_tokens": 8, "output_text": " Paris. The capital of Italy is Rome", "available_tokens_after_case": 952880, "free_pages_after_case": 59555, "deferred_abort_uids": 0, "cache_integrity_ok": true, "status": "PASS", "failure_stage": null, "failure_trace_summary": null}
GATE4.1_JSONL rank=1 {"rank": 1, "world_size": 2, "model_path": "/mnt/nvme/models/Qwen3-0.6B", "tp_size": 2, "device": "npu:1", "baseline_available_tokens": 952880, "baseline_free_pages": 59555, "total_pages": 59555, "init_status": "PASS", "load_status": "PASS", "prefill_status": "PASS", "decode_status": "PASS", "actual_output_tokens": 8, "output_text": " Paris. The capital of Italy is Rome", "available_tokens_after_case": 952880, "free_pages_after_case": 59555, "deferred_abort_uids": 0, "cache_integrity_ok": true, "status": "PASS", "failure_stage": null, "failure_trace_summary": null}
```

Rank 0 and rank 1 produced **byte-identical** `output_text` and
identical `actual_output_tokens=8`, confirming the greedy
determinism argument in §7 of the readiness audit:
`ParallelLMHead.forward` all-gathers vocab-split logits before
sampling, so both ranks pick the same next token independently.

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
Attention backend: npu_fia
page_size:         16       (FIA NO_QUANT block_size % 16 == 0)
memory_ratio:      0.85
max_running_req:   4
Sampling:          greedy (temperature=0.0, top_k=1, top_p=1.0, ignore_eos=True)
Batching:          B=1 (single request), one prompt enqueued per rank
Request:           N=8 (max_new_tokens=8; single prefill + 7 decode steps)
Model:             /mnt/nvme/models/Qwen3-0.6B (dense bf16 Qwen3ForCausalLM)
Launcher:          torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0
                            --master_addr=127.0.0.1 --master_port=29402
Distributed init:  MINISGL_DISTRIBUTED_ADDR=env:// (reuses torchrun store)
                   backend=hccl (primary) + gloo (sidecar via new_group)
                   use_pynccl=False (mandatory on NPU — PyNCCL is CUDA-only)
Timer:             time.perf_counter() (wall clock, per-rank)
Warmup / repeats:  n/a (bring-up smoke, not a timing gate)
Driver:            scripts/gate4_1_tp2_single_request.py
                   per-rank worker script executed under torchrun;
                   both ranks independently enqueue the same prompt
                   (offline_mode short-circuits ZMQ fanout, so both
                   ranks drive their own generate() in lockstep).
```

---

## 3. Launch command

Executed on remote container `<CONTAINER>` at working directory
`/mnt/nvme/LR-606/mini-sglang-ascend-gate41`.

```bash
PYTHONPATH=python torchrun \
  --nproc_per_node=2 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29402 \
  scripts/gate4_1_tp2_single_request.py
```

The script's structured stdout is the primary evidence; each rank
emits exactly one `GATE4.1_JSONL rank=<r> {...}` line.

---

## 4. Audit summary (from §9 of the readiness audit)

The read-only source audit
(`docs/ascend_port/gate4.1_tp2_readiness_audit.md`) identified three
predicted first-blockers at the driver / launcher layer:

1. `LLM.__init__` hardcoded `DistributedInfo(0, 1)` at
   `python/minisgl/llm/llm.py:32` — **fixed at this gate** (see §5.1).
2. `use_pynccl` defaults to `True` in `EngineConfig` — **avoided by
   passing `use_pynccl=False`** as a kwarg from the driver script (no
   source change needed).
3. Launcher env plumbing (`LOCAL_RANK` / `RANK` / `WORLD_SIZE` →
   `DistributedInfo`) — **handled by the script** (§5.3).

The audit also predicted the runtime would reach and pass:

* HCCL first-collective handshake in `VocabParallelEmbedding`.
* All-gather in `ParallelLMHead`.
* Sampler bit-identical greedy across ranks.
* Symmetric `synchronize_device → sync_all_ranks → drain →
  engine.shutdown`.

All predicted-pass sites did pass. One unpredicted blocker
surfaced (§6.1 below): `init_method="tcp://127.0.0.1:2333"` did not
stand up a server-side TCPStore on this torch build; the fix was
also a one-liner (§5.2).

---

## 5. Minimum fixes applied under `python/minisgl/`

Two source-file changes, both are strict extensions with a preserved
default. No behavioral change for TP=1 or for any existing offline
call site.

### 5.1 `python/minisgl/llm/llm.py` — accept `tp_info` kwarg

```python
def __init__(
    self,
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    tp_info: DistributedInfo | None = None,
    **kwargs,
):
    # Gate 4.1: allow the offline driver to run under TP > 1 by accepting
    # a caller-supplied ``tp_info``. Default preserves the historical
    # single-rank behaviour for every existing TP=1 call site (Gate 2.1 →
    # Gate 3.4). Each rank must instantiate its own ``LLM`` in its own
    # process (``set_tp_info`` is a process-global one-shot); the caller
    # is responsible for the launcher (e.g. torchrun) that sets
    # ``LOCAL_RANK`` / ``RANK`` / ``WORLD_SIZE`` and passes a matching
    # ``DistributedInfo(rank=RANK, size=WORLD_SIZE)`` here.
    if tp_info is None:
        tp_info = DistributedInfo(0, 1)
    config = SchedulerConfig(
        model_path=model_path,
        tp_info=tp_info,
        dtype=dtype,
        offline_mode=True,
        **kwargs,
    )
    super().__init__(config)
```

Every existing TP=1 call site (Gate 2.1 → 3.4 offline drivers) passes
no `tp_info` and continues to see `DistributedInfo(0, 1)`.

### 5.2 `python/minisgl/engine/config.py` — env-var rendezvous URI

```python
_DISTRIBUTED_ADDR_ENV = "MINISGL_DISTRIBUTED_ADDR"
...
@property
def distributed_addr(self) -> str:
    # Gate 4.1: honour an env-supplied rendezvous URI (e.g. ``env://`` when
    # launched under torchrun, so the launcher's own TCPStore is reused).
    # Falls back to the historical loopback URI when the env var is unset,
    # preserving every existing TP=1 offline call site (Gate 2.1 → 3.4).
    override = os.environ.get(_DISTRIBUTED_ADDR_ENV)
    if override:
        return override
    return "tcp://127.0.0.1:2333"
```

Every existing TP=1 call site does not set the env var and continues
to see `"tcp://127.0.0.1:2333"`. This override is only exercised by
the Gate 4.1 bring-up script.

### 5.3 Driver-layer glue — `scripts/gate4_1_tp2_single_request.py`

Reads `RANK` / `WORLD_SIZE` / `LOCAL_RANK` from the torchrun-supplied
env, constructs `DistributedInfo(rank=RANK, size=WORLD_SIZE)`, sets
`os.environ.setdefault("MINISGL_DISTRIBUTED_ADDR", "env://")`,
instantiates `LLM(..., tp_info=..., use_pynccl=False,
attention_backend="npu_fia", memory_ratio=0.85, page_size=16,
max_running_req=4, cuda_graph_bs=[])`. Runs A / B / C in one process
per rank, emits one JSONL row per rank.

---

## 6. First failure stage — resolved during bring-up

### 6.1 TCP rendezvous timeout — resolved

First torchrun attempt (before §5.2 was applied) failed at
`_init_communication` with both ranks timing out as TCP **clients**
against `127.0.0.1:2333`:

```
DistNetworkError: The client socket has timed out after 60000ms while
trying to connect to (127.0.0.1, 2333)
```

Root cause: on this torch build, the
`init_method="tcp://127.0.0.1:2333"` rendezvous handler did not stand
up a server-side `TCPStore` — rank 0 also went into client mode,
producing a double-client hang.

Resolution: the Gate 4.1-authorised minimum fix in §5.2 lets the
driver set `MINISGL_DISTRIBUTED_ADDR=env://`, reusing torchrun's own
`MASTER_ADDR` / `MASTER_PORT` store (which torchrun always stands up
correctly). The default `"tcp://..."` fallback is preserved for
every existing TP=1 call site.

### 6.2 Post-fix run — no further failures

Second run (with the env-var fix deployed): both ranks reached
`_init_communication`, HCCL init returned, gloo sidecar created,
`_sync_get_memory` all_reduce on gloo succeeded (60.60 GiB free on
both ranks, so no imbalance), weights streamed and sharded, KV cache
allocated (952880 tokens/rank — 2× the TP=1 baseline of 470592 tokens
because KV heads are split 8→4 per rank while `head_dim` and page
size are unchanged), first forward completed (2337 ms rank 0 / 2347
ms rank 1 wall clock for the full 8-token generate call), sampler
produced identical output on both ranks, symmetric shutdown drained
and destroyed the process group.

No stage after §6.1 required a source change.

---

## 7. Allocator evidence

Per rank, at the two allocator-invariant snapshot points (post-load
baseline and post-case):

| Rank | baseline_available_tokens | after case C | baseline_free_pages | after case C | total_pages | deferred_abort_uids | cache_integrity_ok |
|---|---|---|---|---|---|---|---|
| 0 | 952880 | 952880 | 59555 | 59555 | 59555 | 0 | true |
| 1 | 952880 | 952880 | 59555 | 59555 | 59555 | 0 | true |

Interpretation:

* **952880 tokens/rank** at TP=2 vs 470592 tokens at TP=1 (Gate 3.4
  §5.1 baseline) — consistent with the Gate 4.1 readiness audit §6:
  per-page KV footprint is halved when `num_kv_heads` is split
  8-way→4-way, so per-rank token capacity roughly doubles at the same
  0.85 memory ratio (952880 / 470592 ≈ 2.03, matching the 2× KV-head
  split plus the smaller per-rank `_sync_get_memory` minimum).
* **`available_tokens` returned exactly to baseline** on both ranks
  after case C completed — matches the Gate 3.1 / 3.4 allocator
  invariant.
* **`free_pages` returned to `total_pages`** on both ranks — the
  single request's prefix cache pages were fully evictable and
  released; this is expected for `ignore_eos=True` single request
  with no subsequent enqueue.
* **`deferred_abort_uids == 0`** on both ranks — no abort path was
  entered (the request completed normally on max_tokens).
* **`cache_integrity_ok == true`** on both ranks — the allocator
  invariants (free_slots ∪ used_slots partition of num_pages,
  radix-tree ↔ page-table consistency) held throughout.

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
3.3, Gate 3.4. Confirms the two source-file changes (§5.1, §5.2) do
not regress any existing offline / scheduler / config surface.

The single-process cross-file ordering artifact carried from Gate 2.5
(fake `pydantic` stub in `test_scheduler_abort_ack.py` poisoning
`sys.modules` for a later `test_shell_cancel_cleanup.py`) is
unchanged; Gate 4.1 does not touch `tests/`. Per-file execution
sidesteps the artifact exactly as prior gates dictate.

---

## 9. Support matrix delta (Gate 3.4 → Gate 4.1)

| Capability                                             | Gate 3.4 | Gate 4.1 |
|---|---|---|
| TP=1 timing baseline (Qwen3-0.6B + Qwen3-1.7B)         | PASS | PASS (not re-attested; unchanged) |
| TP=2 driver support in offline `LLM`                   | UNKNOWN | **wired via `tp_info` kwarg** |
| TP=2 launcher rendezvous under torchrun                | UNKNOWN | **wired via `MINISGL_DISTRIBUTED_ADDR=env://`** |
| TP=2 `Engine._init_communication` HCCL branch          | UNKNOWN | **PASS (first end-to-end proof on this port)** |
| TP=2 weight sharding for Qwen3-0.6B                    | UNKNOWN | **PASS** (Q/K/V dim 0, O dim 1, gate/up dim 0, down dim 1, embed/lm_head vocab dim 0) |
| TP=2 `_sync_get_memory` gloo-sidecar all_reduce        | UNKNOWN | **PASS** (60.60 GiB free on both ranks; no imbalance) |
| TP=2 single-request forward (Qwen3-0.6B B=1 N=8)       | UNKNOWN | **PASS** (both ranks bit-identical output) |
| TP=2 `VocabParallelEmbedding` all_reduce               | UNKNOWN | **PASS** (first HCCL forward collective; no handshake failure) |
| TP=2 `ParallelLMHead` all_gather                       | UNKNOWN | **PASS** (fully-replicated logits enable independent-greedy) |
| TP=2 sampler bit-identical greedy                      | UNKNOWN | **PASS** (identical output_text on rank 0 and rank 1) |
| TP=2 symmetric shutdown (drain + destroy_process_group) | UNKNOWN | **PASS** (both ranks reach shutdown; no hang) |
| TP=2 allocator invariant (per-rank `available_size` return) | UNKNOWN | **PASS** (952880 → 952880 on both ranks) |
| TP=2 batching (B > 1)                                  | UNKNOWN | UNKNOWN (out of scope — see §10) |
| TP=2 ragged prefill                                    | UNKNOWN | UNKNOWN (out of scope — Gate 2.2f FIA boundary) |
| TP=2 timing benchmark (TTFT / e2e / tokens/s)          | UNKNOWN | UNKNOWN (not a Gate 4.1 goal — see §10) |
| TP > 2                                                 | UNKNOWN | UNKNOWN (out of scope) |
| Non-Qwen3 architecture families under TP=2             | UNKNOWN | UNKNOWN (out of scope) |
| Qwen3-1.7B / 4B / 14B / 32B under TP=2                 | UNKNOWN | UNKNOWN (out of scope) |
| Regression: 8 hermetic suites (per-file)               | 51 passed | 51 passed (unchanged) |

---

## 10. What is NOT proven at this gate

Explicit exclusions carried forward from the Gate 4.1 opening:

* **Not a performance benchmark under TP=2.** No TTFT / e2e /
  tokens-per-second is reported. The single wall-clock number
  observed (2337 / 2347 ms for the full `generate()` call) is not a
  timing measurement — it includes single-process eager forward
  overhead, per-token tokenizer decode, and per-token `DetokenizeMsg`
  routing. A separate gate (4.2 or later) would be required to
  produce TP=2 timing.
* **No CUDAGraph** under TP=2 — `cuda_graph_bs=[]` is locked at eager
  mode. torch_npu does not implement CUDAGraph.
* **No B > 1 under TP=2.** Only B=1 single request is exercised.
  B=2 equal-length and B=2 ragged prefill are out of scope; Gate 2.2f
  FIA `NotImplementedError` boundary on ragged + non-zero cached_len
  is unchanged.
* **No long-sequence / context-length sweep under TP=2.** The single
  prompt (`"The capital of France is"`, 5 tokens) fits in one page.
* **TP > 2** and multi-node (NNODES > 1). HCCL / gloo wiring for
  wider TP is unexercised.
* **Qwen3-1.7B / Qwen3-4B / 14B / 32B under TP=2**, quantized
  (Qwen3-32B-FP8) and MoE (Qwen3-30B-A3B) variants, Qwen3-Next-*,
  Qwen3-ASR-*, Qwen3-Coder-Next. All out of scope.
* **Non-Qwen3 model families under TP=2** (Llama / Mistral / DeepSeek
  / MoE).
* **HTTP server under TP=2**, non-stream HTTP cancel, offline
  `LLM.abort()`, chunked prefill. Same NOT REACHED / NOT SUPPORTED
  boundaries as Gate 2.5 / 3.1 / 3.2 / 3.3 / 3.4.
* **Forward/sampler exception recovery** inside the scheduler under
  TP=2. Unchanged from Gate 2.5.
* **Long soak / rolling allocator run under TP=2.** Only one
  single-request end-to-end is executed.

Verdict decision matrix (from gate open):

| Outcome | Definition | This gate |
|---|---|---|
| PASS    | Both ranks complete A + B + C; single request returns exactly `max_new_tokens=8`; rank 0 and rank 1 produce bit-identical output; allocator invariants held per rank | **✔ (this verdict)** |
| PARTIAL | Init and load pass but generate() times out / errors / produces mismatched output across ranks — with a fully-diagnosed first-failing-stage | not reached |
| BLOCKED | Neither rank can complete case A (init-only) — HCCL init fails, or torch_npu / CANN environment prevents worker start | not reached |

---

## 11. Freeze boundary

This gate freezes the fact that Mini-SGLang-Ascend at `8431743` —
descending from `c651d91` with the two
minimum-fix commits under §5.1 and §5.2, plus the new bring-up
script and the two Gate 4.1 documents — completes a TP=2 Qwen3-0.6B
single-request bring-up end-to-end on 2× Ascend 910B1 under the
frozen eager `npu_fia` bf16 greedy `use_pynccl=False`
`MINISGL_DISTRIBUTED_ADDR=env://` envelope, with allocator
`available_size` returning to baseline on both ranks,
`deferred_abort_uids` empty on both ranks, `cache_integrity_ok` true
on both ranks, `output_text` bit-identical across ranks, and
8-file regression 51/51 passing.

It does not claim TP > 2 support.
It does not claim TP=2 support for any other model.
It does not claim TP=2 support for B > 1 or ragged prefill.
It does not claim TP=2 timing / throughput / latency parity or
leadership against any other inference stack.
It does not modify any prior gate verdict, the release tag
`v0.1.0a1`, or the GitHub Release.
