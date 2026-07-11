# Gate 4.8 Verdict — Qwen3-1.7B TP=2 single-request bring-up

**Gate ID:** 4.8 (Qwen3-1.7B TP=2 B=1 single-request bring-up on Ascend 910B1)
**Verdict:** PASS
**Branch:** `gate4.8-qwen1.7b-tp2-single-request`
**Base commit:** `61fd9e5` (tip of `ascend-port`, Gate 4.7 merge)
**Freeze commit:** `dee4361`
**Date:** 2026-07-11
**Kind:** Real-hardware Ascend 910B1 TP=2 first-model-bring-up
proof — two ranks × Qwen3-1.7B × single prompt × greedy ×
`max_new_tokens=8`. Both ranks return the exact same 8 output
tokens; the allocator returns to baseline on both ranks; the
per-rank scheduler completes without deferred aborts or radix-cache
corruption.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5 /
3.1 / 3.2 / 3.3 / 3.4 / 4.1 / 4.2 / 4.3 / 4.4 / 4.5 / 4.6 / 4.7,
does not mutate release tag `v0.1.0a1`, does not touch the GitHub
Release, CHANGELOG, or release notes, and does not extend the
Ascend port to TP > 2, B > 1, ragged prefill, mixed-KV decode,
dynamic admission, TP=2 timing, non-Qwen3 architectures, or
Qwen3-4B / 14B / 32B / quantized / MoE variants. The only new
artefacts at this gate are one bring-up script and this verdict; no
runtime source under `python/minisgl/` is modified; no test file is
modified.

---

## 1. Verdict summary

**PASS on all three cases across both ranks.**

| Case | Description | rank 0 | rank 1 |
|---|---|---|---|
| A | init-only smoke — `_init_communication` returns, both ranks report `init_status=PASS` | **PASS** | **PASS** |
| B | Qwen3-1.7B model-load smoke — per-rank weight sharding + post-load `check_integrity()` | **PASS** | **PASS** |
| C | B=1 single request — `generate(["The capital of France is"], max_tokens=8)` | **PASS** | **PASS** |

Cases A / B collapse into the same `LLM` boot (`set_tp_info` is
one-shot); reaching a successful `_snapshot(cache_manager)` after
`LLM.__init__` returns proves init + load simultaneously.

Case C proven invariants on both ranks:

* `actual_output_tokens == 8`
* `output_token_ids == [12095, 13, 576, 6722, 315, 279, 3639, 4180]`
* `output_text == " Paris. The capital of the United States"`
* rank 0 `output_text` byte-identical to rank 1 `output_text`
* rank 0 `output_token_ids` byte-identical to rank 1 `output_token_ids`
* `prompt_token_length == 5`

Post-case allocator invariants held on both ranks:

* `available_tokens_after_case == baseline_available_tokens` (930640 on both ranks)
* `free_pages_after_case == baseline_free_pages` (58165 on both ranks)
* `deferred_abort_uids == 0`
* `cache_integrity_ok == true`

Structured log (both ranks, single JSON object each; rank 1 identical
except for `device: "npu:1"` and per-rank `generate_ms`):

```
GATE4.8_JSONL rank=0 {
  "rank": 0, "world_size": 2, "tp_size": 2,
  "model_path": "/mnt/nvme/models/Qwen3-1.7B",
  "device": "npu:0",
  "prompt": "The capital of France is",
  "prompt_token_length": 5, "batch_size": 1,
  "baseline_available_tokens": 930640, "baseline_free_pages": 58165, "total_pages": 58165,
  "init_status": "PASS", "load_status": "PASS",
  "prefill_status": "PASS", "decode_status": "PASS",
  "actual_output_tokens": 8,
  "output_text": " Paris. The capital of the United States",
  "output_token_ids": [12095, 13, 576, 6722, 315, 279, 3639, 4180],
  "available_tokens_after_case": 930640,
  "free_pages_after_case": 58165,
  "deferred_abort_uids": 0,
  "cache_integrity_ok": true,
  "generate_ms": 2366.756970062852,
  "status": "PASS",
  "failure_stage": null, "failure_trace_summary": null
}
GATE4.8_JSONL rank=1 { ...device: "npu:1", generate_ms: 2571.71..., otherwise identical... }
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
| Batch size | 1 |

## 3. Qwen3-1.7B config summary

Read directly from `/mnt/nvme/models/Qwen3-1.7B/config.json`:

| Key | Value |
|---|---|
| `architectures` | `["Qwen3ForCausalLM"]` |
| `model_type` | `qwen3` |
| `hidden_size` | 2048 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 (GQA, 2:1 Q:KV ratio) |
| `head_dim` | 128 |
| `intermediate_size` | 6144 |
| `max_position_embeddings` | 40960 |
| `rope_theta` | 1_000_000 |
| `rms_norm_eps` | 1e-06 |
| `hidden_act` | `silu` |
| `vocab_size` | 151936 |
| `tie_word_embeddings` | `true` |
| `torch_dtype` | `bfloat16` |
| `bos_token_id` / `eos_token_id` | 151643 / 151645 |

Loaded weight shards observed at runtime:

```
Loading weights: 100%|██████████| 2/2 [00:01<00:00, 1.26 it/s]
Free memory before loading model: 60.60 GiB (per rank)
Free memory after initialization: 9.09 GiB (per rank)
Allocating 930640 tokens for KV cache, K + V = 49.70 GiB (per rank)
```

Two safetensors shards (`model-00001-of-00002.safetensors` +
`model-00002-of-00002.safetensors`; ~3.4 GiB + ~623 MiB) loaded by
both ranks in ~1.6 s each. Per-rank baseline KV cache budget:
`930640` tokens (`58165` pages × `page_size=16`), vs Qwen3-0.6B's
`952880` at the same envelope — the Qwen3-1.7B model consumes more
per-rank device memory for weights, leaving less for KV cache, as
expected.

The `model_config_summary` field in the JSONL is empty (`{}`)
because the LLM class does not expose the HF config at any of the
three defensive attribute names probed by the driver. This does not
affect the gate outcome — the config summary is a defensive
best-effort logging field, not a required invariant — and the
authoritative config summary is the table above, read directly
from disk on the same box.

## 4. Launch command

```bash
ssh -p <PORT> <USER>@<HOST> \
  "docker exec <CONTAINER> bash -c '
    cd /mnt/nvme/LR-606/mini-sglang-ascend-gate48 &&
    PYTHONPATH=./python:\$PYTHONPATH \
    torchrun --nproc_per_node=2 --master_port=29418 \
      scripts/gate4_8_qwen1_7b_tp2_single_request.py \
      --model-path /mnt/nvme/models/Qwen3-1.7B \
      2>&1 | tee logs/gate4.8_qwen1.7b_tp2_single_request.log
  '"
```

## 5. Per-rank output evidence

| Field | rank 0 | rank 1 |
|---|---|---|
| `device` | `npu:0` | `npu:1` |
| `prompt` | `"The capital of France is"` | `"The capital of France is"` |
| `prompt_token_length` | 5 | 5 |
| `actual_output_tokens` | 8 | 8 |
| `output_token_ids` | `[12095, 13, 576, 6722, 315, 279, 3639, 4180]` | `[12095, 13, 576, 6722, 315, 279, 3639, 4180]` |
| `output_text` | `" Paris. The capital of the United States"` | `" Paris. The capital of the United States"` |
| `generate_ms` | 2366.76 | 2571.71 |
| `status` | `PASS` | `PASS` |

Per-rank output equality proven by exact byte match on `output_text`
and on every element of `output_token_ids`. Greedy sampling on
all-gathered logits under lockstep TP=2 scheduling gives
bit-identical outputs per rank — the same invariant that Gate 4.1
established on Qwen3-0.6B, now proven on Qwen3-1.7B.

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

The allocator returned to baseline exactly on both ranks. No
requests leaked pages, no requests left a deferred-abort uid
pending, no requests corrupted the radix cache linkage. This
matches the Gate 4.1 allocator invariant on Qwen3-0.6B, confirming
that the per-request lifecycle is model-size-independent at TP=2.

## 7. First failing stage

None — the driver reached `status=PASS` on both ranks without
recording a `failure_stage`. `failure_stage` is `null` on both
ranks; `failure_trace_summary` is `null` on both ranks.

## 8. Regression evidence

Per-file pytest on `tests/misc/` (headers only, hermetic per-file mode)
in the same working tree used for the smoke run:

```
tests/misc/test_scheduler_abort_ack.py             → 8/8  PASS  (15.16s)
tests/misc/test_scheduler_overlap_abort_fence.py   → 7/7  PASS  (15.15s)
tests/misc/test_scheduler_prepare_batch_txn.py     → 5/5  PASS  (14.55s)
tests/misc/test_engine_forward_sampler_atomic.py   → 5/5  PASS  (12.97s)
tests/misc/test_scheduler_shutdown_drain.py        → 8/8  PASS  (15.62s)
tests/misc/test_exposed_path_abort_ack.py          → 2/2  PASS  (14.44s)
tests/misc/test_shell_cancel_cleanup.py            → 2/2  PASS  (14.07s)
tests/misc/test_pyproject_config.py                → 14/14 PASS ( 0.04s)
```

Total: **51 / 51 PASS** in per-file (hermetic) mode. Every count
matches the last measurement at Gate 4.7. No test file was modified
by this gate.

## 9. Known limitations

* **B=1 only.** B > 1 on Qwen3-1.7B is out of scope. Gates 4.2–4.6
  proved B=2 / B=3 shapes on Qwen3-0.6B; Qwen3-1.7B is *not*
  proven at B > 1 by this gate.
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
* **Model config summary via LLM attributes is empty.** The driver's
  best-effort `_extract_model_config_summary()` returned `{}`
  because the LLM class does not expose the HF config at any of the
  three probed attribute names (`hf_config`, `model_config`,
  `config`, plus the three nested containers). Fixing this would
  require touching `python/minisgl/`, which is out of scope for
  this gate. Verdict §3 supplies the config summary read directly
  from disk instead.

## 10. Decision matrix

| Question | Answer |
|---|---|
| Does Qwen3-1.7B `LLM.__init__` return on both ranks under TP=2? | Yes |
| Does the two-shard weight load complete on both ranks? | Yes |
| Does `generate(["The capital of France is"], max_tokens=8)` return exactly 8 tokens on both ranks? | Yes |
| Do rank 0 and rank 1 produce byte-identical `output_text`? | Yes (`" Paris. The capital of the United States"`) |
| Do rank 0 and rank 1 produce byte-identical `output_token_ids`? | Yes (`[12095, 13, 576, 6722, 315, 279, 3639, 4180]`) |
| Does the allocator return to baseline on both ranks? | Yes (`930640 → 930640` per rank) |
| Are `deferred_abort_uids == 0` after the case? | Yes on both ranks |
| Does `check_integrity()` pass after the case? | Yes on both ranks |
| Is the driver B=1 only? | Yes |
| Does the driver do ragged / mixed-KV / dynamic admission / timing? | No |
| Does the driver modify `python/minisgl/`? | No |
| Does the driver modify tests? | No |
| Is `use_pynccl=False`? | Yes |
| Is the model Qwen3-1.7B only? | Yes |
| Is TP fixed at 2? | Yes |

**Verdict: PASS.**

## 11. Freeze boundary

The following files are the frozen artefacts for Gate 4.8:

* `scripts/gate4_8_qwen1_7b_tp2_single_request.py`
* `docs/ascend_port/gate4.8_qwen1.7b_tp2_single_request_verdict.md`

No files under `python/minisgl/` were modified at this gate. No
tests were modified at this gate. The freeze commit SHA is
recorded in this document header once the driver + verdict pair is
committed, and it is recorded on the `ascend-port` tip once the
branch is merged with `--no-ff`.
