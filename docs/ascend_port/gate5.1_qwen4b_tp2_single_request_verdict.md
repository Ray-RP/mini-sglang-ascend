# Gate 5.1 Verdict — Qwen3-4B Fixed-TP2 Single-Request Bring-Up

**Gate ID:** 5.1 (Qwen3-4B fixed-TP2 B=1 single-request bring-up on Ascend 910B1)
**Verdict:** PASS
**Branch:** `gate5.1-qwen4b-tp2-single-request`
**Base commit:** `46548de` (tip of `ascend-port`, Gate 4.15a merge)
**Freeze commit:** `01b8fe9`
**Date:** 2026-07-12
**Kind:** Functional single-request bring-up — Qwen3-4B on 2 × Ascend
910B1 under fixed TP=2, eager, `npu_fia`, bf16, greedy. Records
structured JSONL per rank for Cases A/B/C plus a
`GATE5.1_RESULT=PASS` footer. No timing statistics, no repeats, no
warmup. Neither runtime nor tests are touched.

> **This is a Qwen3-4B fixed-TP2 single-request bring-up.**
> **It is not a benchmark.** **It is not a cross-stack comparison.**
> No timing statistics are collected; the wall-clock in the JSONL
> `generate_ms` field is diagnostic only. B > 1, ragged prefill,
> mixed-KV decode, dynamic admission, TP > 2, TP elasticity, runtime
> TP switching, Graph Re-Linker, Tensor-Remap-Kernel, non-Qwen3
> architectures, and Qwen3-14B / 32B / quantized / MoE variants are
> all out of scope.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5 /
3.1 / 3.2 / 3.3 / 3.4 / 4.1 / 4.2 / 4.3 / 4.4 / 4.5 / 4.6 / 4.7 /
4.8 / 4.9 / 4.10 / 4.11 / 4.12 / 4.13 / 4.13a / 4.13b / 4.14 / 4.15
/ 4.15a, does not mutate release tag `v0.1.0a1`, does not touch the
GitHub Release / CHANGELOG / release notes, and does not extend the
Ascend port beyond the fixed-TP2 single-request Qwen3-4B envelope.
The only new artefacts at this gate are one bring-up script and
this verdict; no runtime source under `python/minisgl/` is modified;
no test file is modified.

---

## 1. Verdict summary

**PASS on all three cases (A / B / C) across both TP ranks.**

| Case | Description | Rank 0 | Rank 1 |
|---|---|:---:|:---:|
| A | TP=2 init (`LLM.__init__` returns) | **PASS** | **PASS** |
| B | Qwen3-4B TP=2 weight load + post-load `check_integrity()` | **PASS** | **PASS** |
| C | B=1 single request, `max_new_tokens=8`, greedy | **PASS** | **PASS** |

Every record on both ranks reported:

* `actual_output_tokens == 8`
* `available_tokens_after_case == baseline_available_tokens`
* `deferred_abort_uids == 0`
* `cache_integrity_ok == true`
* `output_text` and `output_token_ids` byte-identical across rank 0
  and rank 1

Footer:

```
GATE5.1_RESULT=PASS
```

## 2. Envelope (locked)

| Knob | Value |
|---|---|
| Hardware | 2 × Ascend 910B1 (64 GiB HBM each) |
| Container | `<CONTAINER>` on `<HOST>:<PORT>` |
| torch | 2.4.0 |
| torch_npu | 2.9.0.post1 |
| CANN | 8.5.1 |
| Python | 3.11.14 |
| Model | Qwen3-4B (`/mnt/nvme/models/Qwen3-4B`) |
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
| Batch size | 1 (single-request) |
| Repeats | 1 (functional, no warmup, no measured repeats) |

## 3. Model path existence

`/mnt/nvme/models/Qwen3-4B` present on the validation host container.
Directory listing (informational):

```
LICENSE
README.md
config.json
configuration.json
generation_config.json
merges.txt
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
model.safetensors.index.json
tokenizer.json
tokenizer_config.json
vocab.json
```

Both ranks logged `model_exists == true`.

## 4. Model config summary

Read directly from `/mnt/nvme/models/Qwen3-4B/config.json` on the
validation host (Qwen3-4B is a dense Qwen3 model):

| Key | Value |
|---|---|
| `architectures` | `["Qwen3ForCausalLM"]` |
| `model_type` | `qwen3` |
| `hidden_size` | 2560 |
| `intermediate_size` | 9728 |
| `num_hidden_layers` | 36 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 (GQA 32/8) |
| `head_dim` | 128 |
| `max_position_embeddings` | 40960 |
| `vocab_size` | 151936 |
| `tie_word_embeddings` | true |
| `rope_theta` | 1000000 |
| `torch_dtype` | `bfloat16` |
| `transformers_version` | `4.51.0` |

Weight footprint: 3 safetensor shards, ~8 GB bf16 total. Under
TP=2, each rank holds ~4 GB of sharded weights plus its half of KV
state; both fit comfortably inside 64 GiB HBM per die.

The JSONL `model_config_summary` field was emitted empty ({}) on
both ranks — the LLM object exposed no `hf_config` / `model_config`
/ `config` attribute at the paths the driver probes. This is
diagnostic metadata only; it is **not** a Gate 5.1 pass/fail
predicate, and the runtime capability path (init → weight load →
prefill → decode) succeeded end-to-end regardless.

## 5. Launch command

Same outer-shell retry pattern as Gates 4.10 / 4.11 / 4.12 / 4.13 /
4.14: sleep 8 s between attempts, bump `--master_port` (PORT_BASE=29650
+ ATTEMPT*10), thread `--cold-start-attempt-id` and
`--memory-sync-retry-note` into the driver. **Attempt 1 (port
29660) succeeded cleanly — no retry needed.**

> **Public hygiene note:** remote host, username, password, and
> container identifiers are redacted in this document.

```bash
ssh -p <PORT> <USER>@<HOST> \
  "docker exec <CONTAINER> bash -c '
    set +e
    cd <REMOTE_PATH>/mini-sglang-ascend-gate5.1 &&
    mkdir -p logs &&
    PORT_BASE=29650
    NOTE=""
    for ATTEMPT in 1 2 3; do
      PORT=$((PORT_BASE + ATTEMPT * 10))
      LOG=logs/gate5.1_Qwen3-4B_attempt${ATTEMPT}.log
      PYTHONPATH=./python:$PYTHONPATH torchrun --nproc_per_node=2 --master_port=$PORT \
        scripts/gate5_1_qwen4b_tp2_single_request.py \
        --model-path /mnt/nvme/models/Qwen3-4B \
        --model-name Qwen3-4B \
        --cold-start-attempt-id $ATTEMPT \
        --memory-sync-retry-note "$NOTE" > $LOG 2>&1
      if grep -q GATE5.1_RESULT= $LOG && ! grep -q "Memory across TP ranks are imbalanced" $LOG; then
        cp $LOG logs/gate5.1_Qwen3-4B.log
        break
      fi
      IMB=$(grep "Memory across TP ranks are imbalanced" $LOG | head -1)
      NOTE="$NOTE | attempt $ATTEMPT port $PORT: $IMB"
      sleep 8
    done
  '"
```

## 6. Prompts

| Case | Prompt | `prompt_token_length` | `requested_max_new_tokens` |
|---|---|---|---|
| C | `"Paris is"` | 2 | 8 |

Cases A and B do not enqueue any request; they exercise
`LLM.__init__` (which internally performs TP=2 rendezvous, weight
sharding, KV pool allocation, and post-load `check_integrity()`).

## 7. Per-rank output evidence (Case C)

Both ranks returned byte-identical outputs:

| Field | Rank 0 | Rank 1 |
|---|---|---|
| `actual_output_tokens` | 8 | 8 |
| `output_text` | `" the capital of France, and the capital"` | `" the capital of France, and the capital"` |
| `output_token_ids` | `[279, 6722, 315, 9625, 11, 323, 279, 6722]` | `[279, 6722, 315, 9625, 11, 323, 279, 6722]` |

Rank-0 vs rank-1 equality was verified by exact byte match on
`output_text` and every element of `output_token_ids`.

## 8. Per-rank allocator evidence

Baseline captured after weight load, before Case C. Post-case
allocator returned to baseline exactly on both ranks.

| Field | Rank 0 | Rank 1 |
|---|---:|---:|
| `total_pages` | 43109 | 43109 |
| `baseline_free_pages` | 43109 | 43109 |
| `baseline_available_tokens` | 689744 | 689744 |
| `free_pages_before_after` (Case C) | `[43109, 43109]` | `[43109, 43109]` |
| `available_tokens_after_case` | 689744 | 689744 |
| `deferred_abort_uids` | 0 | 0 |
| `cache_integrity_ok` | true | true |

Baseline is bit-identical across ranks. Case C returned every
allocated page and the radix cache reported no dangling reference:
`available_tokens_after_case == baseline_available_tokens` on both
ranks, `check_integrity()` clean.

`generate_ms` (diagnostic only — **not** a Gate 5.1 evidence
field): rank 0 = 2717 ms, rank 1 = 2675 ms. Gate 5.1 makes no
timing claim.

## 9. Cold-start retry note

**Attempt 1 succeeded cleanly.** The run did not trip the
pre/post-load `_sync_get_memory()` imbalance check (2 GiB tolerance
at `python/minisgl/engine/engine.py:246`, unchanged since Gate
4.14). Both ranks reported `cold_start_attempt_id == 1` and
`memory_sync_retry_note == ""`.

The outer shell loop was authorised (per Gate 5.1 spec §5) to
sleep 8 s and re-launch with a fresh `--master_port` up to 2 more
times on cold-start imbalance. It was not exercised on this run.

## 10. Public-hygiene grep summary

Post-authoring verification against the six known leaked
substring classes documented at Gate 4.13a / 4.13b (host / user /
password / container / IP / composite):

```
$ git grep -l -E "<OLD_SSHPASS_PATTERN>|<OLD_PASSWORD_PATTERN>|<OLD_HOST_PATTERN>|<OLD_CONTAINER_PATTERN>" \
    scripts/gate5_1_qwen4b_tp2_single_request.py \
    docs/ascend_port/gate5.1_qwen4b_tp2_single_request_verdict.md
(no output — zero files)
```

Loopback `127.0.0.1` and bind `0.0.0.0` do not appear in either
artefact. All host references in the verdict use placeholders
(`<HOST>`, `<PORT>`, `<USER>`, `<CONTAINER>`, `<REMOTE_PATH>`).

## 11. Regression evidence

Optional pytest per Gate 5.1 spec §8 was skipped — Gate 5.1
modifies zero runtime and zero test files. The Gate 4.13
regression measurement of `51 / 51 PASS` per-file pytest on
`tests/misc/` is unchanged by construction.

`git diff --check` on the freeze branch tip: clean.

## 12. Known limitations

* **Single-model, single-batch, single-request functional probe.**
  Qwen3-4B has been proven only under B=1 with `max_new_tokens=8`.
  B > 1, ragged prefill, mixed-KV decode, and dynamic admission
  are **not** covered by this gate. Those capabilities on Qwen3-4B
  will require their own follow-up gates modelled on Gate 4.2 /
  4.3 / 4.4 / 4.5.
* **No timing evidence.** `generate_ms` is a diagnostic dump only.
  Gate 5.1 makes no throughput, latency, or steady-state claim.
* **TP=2 only.** TP=4 / TP=8, TP runtime elasticity, runtime TP
  switching, Graph Re-Linker, Tensor-Remap-Kernel are all out of
  scope.
* **One Qwen3 dense model.** Qwen3-14B, Qwen3-32B, quantized
  weights, and Qwen3 MoE variants are not part of this gate.
* **`use_pynccl=False` is mandatory on NPU.** The all-gather /
  all-reduce path exercised here is the HCCL + gloo sidecar
  collective path.
* **`model_config_summary` field emitted empty.** The runtime's
  HF-config accessor did not surface at the paths the driver
  probes on Qwen3-4B. Config is documented in §4 from the on-disk
  `config.json` instead. This does not affect the Gate 5.1 PASS
  predicate, which depends solely on the runtime capability path
  (init → load → generate → allocator invariants).
* **Not a benchmark.** No throughput, latency, or cross-stack
  comparisons are made or implied.

## 13. Decision matrix

| Question | Answer |
|---|---|
| Does `/mnt/nvme/models/Qwen3-4B` exist on the validation host? | Yes |
| Does `LLM.__init__` return on both ranks under TP=2? | Yes (attempt 1) |
| Does weight load succeed on both ranks under TP=2? | Yes |
| Does post-load `check_integrity()` pass on both ranks? | Yes |
| Does Case C `generate()` return? | Yes |
| Is `actual_output_tokens == 8` on both ranks? | Yes |
| Is rank 0 `output_text` byte-identical to rank 1? | Yes |
| Is rank 0 `output_token_ids` byte-identical to rank 1? | Yes |
| Does `available_tokens_after_case == baseline`? | Yes on both ranks |
| Is `deferred_abort_uids == 0`? | Yes on both ranks |
| Does `check_integrity()` pass post-case? | Yes on both ranks |
| Did the driver touch `python/minisgl/`? | No |
| Did the driver touch tests? | No |
| Did the driver claim performance superiority? | No |
| Did the driver compare against SGLang / vLLM / TGI? | No |
| Is `use_pynccl=False`? | Yes |
| Is the model limited to Qwen3-4B? | Yes |
| Is TP fixed at 2? | Yes |

**Verdict: PASS.**

## 14. Freeze boundary

The frozen artefacts for Gate 5.1 are:

* `scripts/gate5_1_qwen4b_tp2_single_request.py`
* `docs/ascend_port/gate5.1_qwen4b_tp2_single_request_verdict.md`

No files under `python/minisgl/` were modified at this gate. No
tests were modified at this gate. The freeze commit SHA is
recorded in this document header once the driver + verdict pair
is committed, and it is recorded on the `ascend-port` tip once
the branch is merged with `--no-ff`.
