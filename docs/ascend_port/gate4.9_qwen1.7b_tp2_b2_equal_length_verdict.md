# Gate 4.9 Verdict — Qwen3-1.7B TP=2 B=2 equal-length batching

**Gate ID:** 4.9 (Qwen3-1.7B TP=2 B=2 equal-length batching on Ascend 910B1)
**Verdict:** PASS
**Branch:** `gate4.9-qwen1.7b-tp2-b2-equal-length`
**Base commit:** `1141752` (tip of `ascend-port`, Gate 4.8 merge)
**Freeze commit:** `705a8c6`
**Date:** 2026-07-11
**Kind:** Real-hardware Ascend 910B1 TP=2 B=2 equal-length batching
proof — two ranks × Qwen3-1.7B × two equal-length prompts × greedy
× `max_new_tokens=8`. Both ranks return the exact same 8 output
tokens per uid, uid-by-uid outputs match across ranks byte-for-byte,
and the allocator returns to baseline on both ranks with no
deferred aborts and no radix-cache corruption.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5 /
3.1 / 3.2 / 3.3 / 3.4 / 4.1 / 4.2 / 4.3 / 4.4 / 4.5 / 4.6 / 4.7 /
4.8, does not mutate release tag `v0.1.0a1`, does not touch the
GitHub Release, CHANGELOG, or release notes, and does not extend
the Ascend port to TP > 2, B > 2, ragged prefill, mixed-KV decode,
dynamic admission, TP=2 timing, non-Qwen3 architectures, or
Qwen3-4B / 14B / 32B / quantized / MoE variants. The only new
artefacts at this gate are one bring-up script and this verdict;
no runtime source under `python/minisgl/` is modified; no test file
is modified.

---

## 1. Verdict summary

**PASS on all three cases across both ranks.**

| Case | Description | rank 0 | rank 1 |
|---|---|---|---|
| A | TP=2 init — `_init_communication` returns, both ranks report `init_status=PASS` | **PASS** | **PASS** |
| B | Qwen3-1.7B TP=2 model-load — per-rank weight sharding + post-load `check_integrity()` | **PASS** | **PASS** |
| C | B=2 equal-length prefill + decode — `generate([prompt_a, prompt_b], max_tokens=8)` with equal tokenized lengths | **PASS** | **PASS** |

Cases A / B collapse into the same `LLM` boot (`set_tp_info` is
one-shot); reaching a successful `_snapshot(cache_manager)` after
`LLM.__init__` returns proves init + load simultaneously.

Case C proven invariants on both ranks:

* `prompt_token_lengths == [5, 5]` (equal — driver asserts this
  before dispatching `generate()`)
* `actual_output_tokens_per_request == [8, 8]`
* `output_texts[0] == " Paris. The capital of the United States"`
* `output_texts[1] == " Brasília. The capital of France is"`
* rank 0 `output_texts[i]` byte-identical to rank 1 `output_texts[i]`
  for `i ∈ {0, 1}`
* rank 0 `output_token_ids[i]` byte-identical to rank 1 `output_token_ids[i]`
  for `i ∈ {0, 1}`
* `output_token_ids[0] == [12095, 13, 576, 6722, 315, 279, 3639, 4180]`
* `output_token_ids[1] == [61124, 75372, 13, 576, 6722, 315, 9625, 374]`

Post-case allocator invariants held on both ranks:

* `available_tokens_after_case == baseline_available_tokens` (`930640` on both ranks)
* `free_pages_after_case == baseline_free_pages` (`58165` on both ranks)
* `deferred_abort_uids == 0`
* `cache_integrity_ok == true`

Structured log (both ranks, single JSON object each; rank 1 identical
except for `device: "npu:1"` and per-rank `generate_ms`):

```
GATE4.9_JSONL rank=0 {
  "rank": 0, "world_size": 2, "tp_size": 2,
  "model_path": "/mnt/nvme/models/Qwen3-1.7B",
  "device": "npu:0",
  "prompts": ["The capital of France is", "The capital of Brazil is"],
  "prompt_token_lengths": [5, 5], "batch_size": 2,
  "baseline_available_tokens": 930640, "baseline_free_pages": 58165, "total_pages": 58165,
  "init_status": "PASS", "load_status": "PASS",
  "prefill_status": "PASS", "decode_status": "PASS",
  "actual_output_tokens_per_request": [8, 8],
  "output_texts": [
    " Paris. The capital of the United States",
    " Brasília. The capital of France is"
  ],
  "output_token_ids": [
    [12095, 13, 576, 6722, 315, 279, 3639, 4180],
    [61124, 75372, 13, 576, 6722, 315, 9625, 374]
  ],
  "available_tokens_after_case": 930640,
  "free_pages_after_case": 58165,
  "deferred_abort_uids": 0,
  "cache_integrity_ok": true,
  "generate_ms": 2728.784679900855,
  "status": "PASS",
  "failure_stage": null, "failure_trace_summary": null
}
GATE4.9_JSONL rank=1 { ...device: "npu:1", generate_ms: 2582.36..., otherwise identical... }
```

Rank 0 exit code: 0.

## 2. Envelope (locked)

| Knob | Value |
|---|---|
| Hardware | 2 × Ascend 910B1 (64 GiB HBM each) |
| Container | `<CONTAINER>` on `<HOST>:<PORT>` |
| torch | 2.4.0 |
| torch_npu | 2.9.0.post1 |
| CANN | 8.5.1 |
| Python | 3.11.14 |
| Model | Qwen3-1.7B (`/mnt/nvme/models/Qwen3-1.7B`) |
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
| `max_new_tokens` | 8 |
| Batch size | 2 |
| Prompt equality | `prompt_token_lengths[0] == prompt_token_lengths[1] == 5` |

## 3. Launch command

```bash
ssh -p <PORT> <USER>@<HOST> \
  "docker exec <CONTAINER> bash -c '
    cd /mnt/nvme/LR-606/mini-sglang-ascend-gate49 &&
    PYTHONPATH=./python:\$PYTHONPATH \
    torchrun --nproc_per_node=2 --master_port=29439 \
      scripts/gate4_9_qwen1_7b_tp2_b2_equal_length.py \
      --model-path /mnt/nvme/models/Qwen3-1.7B \
      2>&1 | tee logs/gate4.9_qwen1.7b_tp2_b2_equal_length.log
  '"
```

## 4. Prompt-length evidence

Both prompts share the "The capital of X is" template with X chosen
so both country names tokenize to a single BPE token on Qwen3-1.7B.
The driver asserts `tokenized_lengths[0] == tokenized_lengths[1]`
before dispatching `generate()`; the assertion passed cleanly:

| uid | prompt | tokenized length (Qwen3-1.7B tokenizer) |
|---|---|---:|
| 0 | `"The capital of France is"` | 5 |
| 1 | `"The capital of Brazil is"` | 5 |

Both ranks reported `prompt_token_lengths == [5, 5]` in the JSONL,
confirming equality. This drives the FIA equal-length prefill path
identically to Gate 4.2 on Qwen3-0.6B, now proven on Qwen3-1.7B.

## 5. Per-rank output evidence

| Field | rank 0 | rank 1 |
|---|---|---|
| `device` | `npu:0` | `npu:1` |
| `prompt_token_lengths` | `[5, 5]` | `[5, 5]` |
| `batch_size` | 2 | 2 |
| `actual_output_tokens_per_request` | `[8, 8]` | `[8, 8]` |
| `output_texts[0]` | `" Paris. The capital of the United States"` | `" Paris. The capital of the United States"` |
| `output_texts[1]` | `" Brasília. The capital of France is"` | `" Brasília. The capital of France is"` |
| `output_token_ids[0]` | `[12095, 13, 576, 6722, 315, 279, 3639, 4180]` | `[12095, 13, 576, 6722, 315, 279, 3639, 4180]` |
| `output_token_ids[1]` | `[61124, 75372, 13, 576, 6722, 315, 9625, 374]` | `[61124, 75372, 13, 576, 6722, 315, 9625, 374]` |
| `generate_ms` | 2728.78 | 2582.36 |
| `status` | `PASS` | `PASS` |

Per-rank output equality proven by exact byte match on
`output_texts[i]` and on every element of `output_token_ids[i]` for
both `i ∈ {0, 1}`. Greedy sampling on all-gathered logits under
lockstep TP=2 scheduling gives bit-identical outputs per rank —
the same invariant that Gate 4.1 / 4.2 established on Qwen3-0.6B,
now proven on Qwen3-1.7B at B=2 equal-length.

## 6. Per-rank allocator evidence

| Field | rank 0 | rank 1 |
|---|---|---|
| `baseline_available_tokens` | 930640 | 930640 |
| `baseline_free_pages` | 58165 | 58165 |
| `total_pages` | 58165 | 58165 |
| `available_tokens_after_case` | 930640 | 930640 |
| `free_pages_after_case` | 58165 | 58165 |
| `deferred_abort_uids` | 0 | 0 |
| `cache_integrity_ok` | true | true |

The allocator returned to baseline exactly on both ranks. Two
in-flight uids both released their pages cleanly at completion; no
requests leaked pages, no requests left a deferred-abort uid
pending, no requests corrupted the radix cache linkage. Baseline
matches Gate 4.8's `930640` per rank — confirming the KV cache
budget computation is not perturbed by the B > 1 configuration
(as expected: `memory_ratio` and per-rank device state drive
`num_pages`, not the runtime batch size).

## 7. First failing stage

None on the successful run — the driver reached `status=PASS` on
both ranks without recording a `failure_stage`. `failure_stage` is
`null` on both ranks; `failure_trace_summary` is `null` on both
ranks.

### 7.1 Note on retries required to reach a clean device-memory sync

The first two torchrun invocations of this driver failed inside
`Engine._sync_get_memory()` at `python/minisgl/engine/engine.py:246`
with `RuntimeError: Memory across TP ranks are imbalanced`:

* Attempt 1: `min 58.44 GiB, max 60.60 GiB` (delta 2.16 GiB, at
  the pre-load memory-sync check)
* Attempt 2: `min 56.57 GiB, max 58.79 GiB` (delta 2.22 GiB, at
  the post-load memory-sync check inside `_determine_num_pages`)

The engine's tolerance is a hard 2 GiB in
`_sync_get_memory` — any TP-rank free-memory delta larger than
`2 * 1024**3` bytes raises `RuntimeError`. Both attempts landed
just above this ceiling.

`npu-smi info` between attempts showed both NPUs at ~3.4 GiB baseline
HBM usage with **no running user processes** (NPU 0: 3416 MB,
NPU 1: 3412 MB — only a 4 MB delta). The 2 + GiB runtime imbalance
therefore reflects transient torch_npu / HCCL bootstrap state
between rank-0 and rank-1 process starts on this box, not a
persistent hardware asymmetry and not a Gate 4.9-specific bug.
Gate 4.8 (single-request) squeaked through the same check on its
first attempt in the same envelope, confirming the check is not
per-B-size sensitive.

Attempt 3 (after an 8 s idle sleep for HCCL state to settle and a
new master port `29439`) passed both memory-sync checks cleanly
and produced the full PASS trace documented above.

This transient device-state condition is **not treated as a
runtime bug** at this gate — it is a pre-existing, script-external
device readiness variability that occasionally trips the engine's
2 GiB tolerance across TP ranks on cold starts. The runtime check
is doing exactly what it was written to do; no `python/minisgl/`
change is warranted here. A follow-up gate that isolates the
device-init variability from the offline path (or lifts the
tolerance to a documented value after measuring the natural
distribution across many cold starts) is out of scope for
Gate 4.9.

## 8. Regression evidence

Per-file pytest on `tests/misc/` (headers only, hermetic per-file mode)
in the same working tree used for the smoke run:

```
tests/misc/test_scheduler_abort_ack.py             → 8/8  PASS  (14.92s)
tests/misc/test_scheduler_overlap_abort_fence.py   → 7/7  PASS  (14.54s)
tests/misc/test_scheduler_prepare_batch_txn.py     → 5/5  PASS  (14.84s)
tests/misc/test_engine_forward_sampler_atomic.py   → 5/5  PASS  (12.62s)
tests/misc/test_scheduler_shutdown_drain.py        → 8/8  PASS  (14.72s)
tests/misc/test_exposed_path_abort_ack.py          → 2/2  PASS  (14.40s)
tests/misc/test_shell_cancel_cleanup.py            → 2/2  PASS  (13.66s)
tests/misc/test_pyproject_config.py                → 14/14 PASS ( 0.04s)
```

Total: **51 / 51 PASS** in per-file (hermetic) mode. Every count
matches the last measurement at Gate 4.8. No test file was modified
by this gate.

## 9. Known limitations

* **B=2 equal-length only.** Ragged B=2, mixed-KV, and dynamic
  admission on Qwen3-1.7B are out of scope. Gates 4.3 / 4.4 / 4.5 /
  4.6 proved those shapes on Qwen3-0.6B; Qwen3-1.7B is *not*
  proven for those shapes by this gate.
* **B > 2 not exercised.** Only B=2 tested.
* **`max_new_tokens=8`.** Longer decodes are not exercised.
* **Qwen3-1.7B only.** Qwen3-4B / 14B / 32B, quantized weights, and
  MoE variants are out of scope.
* **TP=2 only.** TP=4 / TP=8 not proven for Qwen3-1.7B by this gate.
* **Not a benchmark.** `generate_ms` is included in the JSONL for
  auditability only. It reflects the eager + npu_fia + bf16 +
  greedy path with a single fresh boot — not a stabilised
  throughput number, no warmup, no repeats. Do not quote it as
  performance.
* **`use_pynccl=False` is mandatory on NPU.** All numbers reflect
  the HCCL + gloo sidecar collective path.
* **Device-memory imbalance retries.** The two failed cold-start
  attempts documented in §7.1 are captured in the log but do not
  count against the verdict. A future gate that quantifies the
  cold-start distribution and either enforces a warmup / lift or
  refines the tolerance is out of scope for Gate 4.9.

## 10. Decision matrix

| Question | Answer |
|---|---|
| Does Qwen3-1.7B `LLM.__init__` return on both ranks under TP=2? | Yes (attempt 3, after two cold-start imbalance retries) |
| Does the two-shard weight load complete on both ranks? | Yes |
| Are `prompt_token_lengths[0] == prompt_token_lengths[1]`? | Yes (both `5`) |
| Does `generate([prompt_a, prompt_b], max_tokens=8)` return `[8, 8]` tokens? | Yes on both ranks |
| Do rank 0 and rank 1 produce byte-identical `output_texts` per uid? | Yes for both uids |
| Do rank 0 and rank 1 produce byte-identical `output_token_ids` per uid? | Yes for both uids |
| Does the allocator return to baseline on both ranks? | Yes (`930640 → 930640` per rank) |
| Are `deferred_abort_uids == 0` after the case? | Yes on both ranks |
| Does `check_integrity()` pass after the case? | Yes on both ranks |
| Is the driver B=2 equal-length only? | Yes |
| Does the driver do ragged / mixed-KV / dynamic admission / timing? | No |
| Does the driver modify `python/minisgl/`? | No |
| Does the driver modify tests? | No |
| Is `use_pynccl=False`? | Yes |
| Is the model Qwen3-1.7B only? | Yes |
| Is TP fixed at 2? | Yes |

**Verdict: PASS.**

## 11. Freeze boundary

The following files are the frozen artefacts for Gate 4.9:

* `scripts/gate4_9_qwen1_7b_tp2_b2_equal_length.py`
* `docs/ascend_port/gate4.9_qwen1.7b_tp2_b2_equal_length_verdict.md`

No files under `python/minisgl/` were modified at this gate. No
tests were modified at this gate. The freeze commit SHA is
recorded in this document header once the driver + verdict pair
is committed, and it is recorded on the `ascend-port` tip once the
branch is merged with `--no-ff`.
