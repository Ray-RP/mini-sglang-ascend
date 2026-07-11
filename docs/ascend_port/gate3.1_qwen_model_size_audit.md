# Gate 3.1 — Qwen3 Model Size Expansion Audit (TP=1)

**Gate ID:** 3.1 (Qwen3 same-family model size expansion, TP=1)
**Branch:** `gate3.1-qwen-model-size-expansion`
**Base commit:** `d8b1fd4` (tip of `ascend-port`, Gate 2.5 merge)
**Date:** 2026-07-11
**Kind:** Read-only audit of the available Qwen3 weights on the Ascend
host + minimum compatibility check of the checked-in code against a
target model larger than Qwen3-0.6B. Real-hardware smoke evidence is
recorded in the Gate 3.1 verdict.

This audit does not touch TP > 1, HCCL, performance benchmarks, long
soak, server restart, exception recovery, non-stream HTTP cancel,
offline `LLM.abort()`, chunked prefill, or any non-Qwen3 model family.
It does not mutate release tag `v0.1.0a1`, the GitHub Release, or any
prior gate verdict.

---

## 1. Scope

Answer exactly one question:

> Given the checked-in Ascend port at `d8b1fd4` and the model weights
> already present on the Ascend 910B1 host, can Mini-SGLang-Ascend
> serve a Qwen3 model larger than Qwen3-0.6B under the frozen envelope
> (`TP=1`, `eager`, `npu_fia`)?

Out of scope:

* Any new architecture family (Llama, Mistral, MoE, DeepSeek).
* Multi-tile / TP > 1 shape.
* Performance benchmark or leadership claim.
* Long soak / rolling allocator run.
* HTTP server restart, exception recovery, non-stream HTTP cancel,
  offline `LLM` abort.
* Chunked prefill.
* Downloading any additional weights or introducing new dependencies.

---

## 2. Available Qwen3-family weights on the Ascend host

Listing of `/mnt/nvme/models/` on container `<CONTAINER>`:

```
Qwen3-0.6B          (baseline, prior gates)
Qwen3-1.7B          (dense, bf16, non-quantized)
Qwen3-4B            (dense, bf16, non-quantized)
Qwen3-14B           (dense, bf16, non-quantized)
Qwen3-32B           (dense, bf16, non-quantized)
Qwen3-32B-FP8       (quantized, out-of-scope for eager npu_fia bf16 path)
Qwen3-30B-A3B       (MoE, explicitly out-of-scope per gate spec)
Qwen3-Next-*        (Qwen3-Next architecture, not Qwen3 dense)
Qwen3-ASR-*         (audio, non-CausalLM, out-of-scope)
Qwen3-Coder-Next    (Qwen3-Next architecture, out-of-scope)
```

Per gate spec — "**优先最小的下一档**" — the chosen target is
**Qwen3-1.7B**. Its architecture is identical to Qwen3-0.6B under
`Qwen3ForCausalLM`, only larger widths.

Rejected (why):

| Model | Reason |
|---|---|
| Qwen3-4B | Not the *next* size up; deferred for a later gate if 1.7B PASSes. |
| Qwen3-14B / 32B | Multiple size steps ahead; out of the "next-larger" spec. |
| Qwen3-32B-FP8 | Quantized weights; not covered by the frozen `npu_fia` bf16 path. |
| Qwen3-30B-A3B | MoE variant; explicitly excluded ("新模型族，如 Llama / MoE"). |
| Qwen3-Next-* | Different architecture family (`qwen3_next`), not `qwen3`. |
| Qwen3-ASR-*, Qwen3-Coder-Next | Not `Qwen3ForCausalLM` scope. |

---

## 3. Baseline vs target config diff

Both configs live under `/mnt/nvme/models/{Qwen3-0.6B,Qwen3-1.7B}/config.json`.

| Field | Qwen3-0.6B (baseline) | Qwen3-1.7B (target) | Same? |
|---|---|---|---|
| `architectures` | `Qwen3ForCausalLM` | `Qwen3ForCausalLM` | ✅ |
| `model_type` | `qwen3` | `qwen3` | ✅ |
| `head_dim` | 128 | 128 | ✅ |
| `num_attention_heads` | 16 | 16 | ✅ |
| `num_key_value_heads` | 8 | 8 | ✅ |
| `num_hidden_layers` | 28 | 28 | ✅ |
| `hidden_size` | 1024 | 2048 | ✱ larger |
| `intermediate_size` | 3072 | 6144 | ✱ larger |
| `vocab_size` | 151936 | 151936 | ✅ |
| `max_position_embeddings` | 40960 | 40960 | ✅ |
| `tie_word_embeddings` | true | true | ✅ |
| `torch_dtype` | bfloat16 | bfloat16 | ✅ |
| `rope_theta` | 1000000 | 1000000 | ✅ |
| `rope_scaling` | null | null | ✅ |
| `sliding_window`, `use_sliding_window` | null / false | null / false | ✅ |
| `attention_bias` | false | false | ✅ |
| `rms_norm_eps` | 1e-06 | 1e-06 | ✅ |
| `eos_token_id` | 151645 | 151645 | ✅ |
| `bos_token_id` | 151643 | 151643 | ✅ |

Only `hidden_size` and `intermediate_size` differ. Everything the
attention / KV / sampler path depends on (`head_dim`,
`num_attention_heads`, `num_key_value_heads`, `num_hidden_layers`,
`vocab_size`, `max_position_embeddings`) is byte-identical.

Weight footprint on disk:

```
Qwen3-0.6B          1.5 GiB    (2 safetensors shards)
Qwen3-1.7B          3.8 GiB    (2 safetensors shards, safetensors index)
```

Both fit comfortably in a single 910B1 die (64 GiB HBM), leaving room
for KV cache + activations.

---

## 4. Code-side compatibility check

The Ascend port's Qwen3 support lives in:

* `python/minisgl/models/qwen3.py` — `Qwen3ForCausalLM`, `Qwen3Model`,
  `Qwen3DecoderLayer`.
* `python/minisgl/models/utils.py` — `RopeAttn`, `GatedMLP`.
* `python/minisgl/models/config.py` — `ModelConfig` from HF `config.json`.

Read-only inspection (grep + Read):

* `qwen3.py:19–56` — every layer size is drawn from `ModelConfig`
  (`config.hidden_size`, `config.rms_norm_eps`, `config.num_layers`,
  `config.vocab_size`). No numeric literal encodes the 0.6B shape.
* `utils.py:GatedMLP` and `utils.py:RopeAttn` — same:
  `config.hidden_size`, `config.intermediate_size`, `config.head_dim`,
  `config.num_qo_heads`, `config.num_kv_heads`, `config.rms_norm_eps`.
* `config.py:50` — `head_dim = getattr(config, "head_dim", None) or
  config.hidden_size // config.num_attention_heads`. For 1.7B the HF
  config carries `head_dim=128` explicitly, so the fallback branch is
  not exercised.
* Global grep for `0.6`, `1024`, `3072`, `Qwen3-0` under
  `python/minisgl/` — zero hits that encode a shape assumption
  (only unrelated defaults such as `SHELL_TEMPERATURE=0.6` in
  `env.py`, and NCCL alignment constants). No hardcoded reference to
  the 0.6B geometry anywhere.
* KV cache page geometry (`num_kv_heads * head_dim` per layer) is
  identical between 0.6B and 1.7B (`8 * 128 = 1024`); the paged
  allocator's per-layer per-page byte count therefore does not change
  and the page-size arithmetic is unaffected.
* Sampler / logits shape (`ParallelLMHead`) — output width
  `config.vocab_size = 151936` is identical; the sampler works on the
  vocab dimension only.
* Weight-loader path (`python/minisgl/models/weight.py`) — reads
  `model.safetensors.index.json` and maps HF-format keys onto Qwen3
  submodules; both models ship the identical HF key naming
  (`model.embed_tokens.weight`, `model.layers.N.self_attn.*`,
  `model.norm.weight`, `lm_head` via `tie_word_embeddings`).

Conclusion: no code change is required to add Qwen3-1.7B as a target.
The remaining question is a real-hardware smoke.

---

## 5. Planned smoke envelope

Frozen for this gate (all match Gate 1 / 2.1 / 2.2 envelope):

```
Hardware:          Ascend 910B1 (1 die)
Model:             Qwen/Qwen3-1.7B (bf16, safetensors)
Model path:        /mnt/nvme/models/Qwen3-1.7B
Parallelism:       TP=1
Execution:         eager
Attention backend: npu_fia
Sampling:          greedy (temperature=0.0, top_k=1)
Prompt:            short, single English sentence
max_new_tokens:    8 for single-request, then 16 for a decode run
Batch:             B=1 single-request (prefill + multi-step decode)
                   B=2 equal-length prefill+decode if HBM allows
```

Driver: the offline in-process `LLM` API (same as prior gate smokes),
because the exposed HTTP path was not extended by this gate. Allocator
baseline is captured before and after each request; the invariant is
that `cache_manager` returns to its baseline free-page count with the
per-request `Req` freed and no `deferred_abort_uids` residue.

Not planned in this gate: B=2 mixed-KV decode, ragged batch, HTTP
serving, cancellation, TP > 1.

---

## 6. Risks and mitigations

* **HBM headroom for Qwen3-1.7B bf16.** Weights ≈ 3.8 GiB. KV cache
  at 40960 max positions × 28 layers × 8 KV heads × 128 head_dim × 2
  bytes = ~2.35 GiB per full-length request; the paged allocator
  configures a fixed number of pages independent of any single
  request. Total footprint well below one 910B1 die's 64 GiB.
* **B=2 batch fitting.** If B=2 fails on OOM at the current
  allocator page count, the gate reports it as a limitation and does
  NOT attempt to retune allocator sizing. B=1 is sufficient for PASS
  per the criteria.
* **Weight load speed.** 3.8 GiB from local NVMe is fast. No
  network dependency.
* **`configuration.json` present alongside `config.json`.** Qwen3-1.7B
  ships an extra ModelScope-flavoured `configuration.json`. HF
  `AutoConfig` will still prefer `config.json`. No change needed.

---

## 7. Freeze boundary

This audit freezes the read-only assessment that Qwen3-1.7B is the
smallest same-family model above the baseline, has an architecture
byte-compatible with the checked-in Qwen3 support, and is expected to
run end-to-end under TP=1 eager `npu_fia`. Whether it actually does is
recorded in `docs/ascend_port/gate3.1_qwen_model_size_verdict.md`.

It does not claim TP>1 support.
It does not claim any new architecture family.
It does not extend the offline `LLM` driver's public surface.
It does not modify any prior gate verdict, the release tag `v0.1.0a1`,
or the GitHub Release.
It makes no performance claim.
