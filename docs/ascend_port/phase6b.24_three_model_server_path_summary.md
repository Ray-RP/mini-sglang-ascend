# Phase 6B.24 Three-Model Server-Path Capability Summary

**Kind:** Documentation-only cross-model roll-up of the Ascend
fixed-TP2 server-path bring-up work already completed for the
three Qwen3 sizes tested to date (0.6B, 1.7B, 4B). This file
introduces no runtime, script, test, tag, GitHub Release, or
`CHANGELOG.md` change; it launches no server, calls no endpoint,
and prints no credentials. Its sole purpose is to consolidate the
three per-model summaries into one cross-model capability table
so the v0.2.0a1 technical preview can be talked about as a single
envelope rather than three parallel arcs.

Sources (each is itself a per-model roll-up of the underlying
per-endpoint verdicts, cited via those source summaries — this
file does not re-cite the leaf verdicts individually):

* Qwen3-0.6B:
  [`phase6b.11_qwen06b_server_path_summary.md`](./phase6b.11_qwen06b_server_path_summary.md)
* Qwen3-1.7B:
  [`phase6b.17_qwen17b_server_path_summary.md`](./phase6b.17_qwen17b_server_path_summary.md)
* Qwen3-4B:
  [`phase6b.23_qwen4b_server_path_summary.md`](./phase6b.23_qwen4b_server_path_summary.md)

Envelope: fixed TP=2, eager, `npu_fia`, bf16, `page_size=16`,
greedy — v0.2.0a1.

**Overall verdict: PASS.** Every readiness route on every model
passes under the greedy contract. The four known constraints
apply uniformly across the three tiers. The single expected-blocked
non-greedy chat path is the same on all three tiers and remains
out of scope for the v0.2.0a1 envelope.

---

## 1. Launch envelope (shared, unchanged across all three models)

Frozen at Phase 6B.2, amended once at Phase 6B.7 to add
`--page-size 16`. Required flags are identical on 0.6B, 1.7B, and
4B; only the model path changes:

```
--tp-size 2
--attention-backend npu_fia
--disable-pynccl
--cuda-graph-max-bs 0
--page-size 16
```

Launch skeleton (redacted, `<PORT>` chosen at bring-up time):

```
python -m minisgl.server.launch \
  --model-path /mnt/nvme/models/Qwen3-<SIZE> \
  --tp-size 2 \
  --attention-backend npu_fia \
  --disable-pynccl \
  --cuda-graph-max-bs 0 \
  --page-size 16 \
  --host 0.0.0.0 \
  --port <PORT>
```

Preflight: same `prompt_toolkit` runtime-dependency preflight
across all three arcs (declared in `pyproject.toml` at Phase
6B.4; environments provisioned before that merge must
`pip install prompt_toolkit`). No per-model preflight amendment.

## 2. Three-model endpoint matrix

| Route | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|---|---|---|---|
| Server bring-up (HCCL + Gloo, Uvicorn LISTEN, per-rank NPU pinning, HBM headroom) | **PASS** | **PASS** | **PASS** |
| `GET /v1/models` | **PASS** | **PASS** | **PASS** |
| `POST /generate` (SSE, greedy defaults, `max_tokens=8`, `ignore_eos=true`) | **PASS** | **PASS** | **PASS** |
| `POST /v1/chat/completions` `stream=false`, `temperature=0` | **PASS** | **PASS** | **PASS** |
| `POST /v1/chat/completions` `stream=true`, `temperature=0` | **PASS** | **PASS** | **PASS** |

Cross-model invariants observed:

* **Chat-templated greedy prefix is byte-identical across all
  three models.** Both non-stream and stream `/v1/chat/completions`
  emit `"<think>\nOkay, the user is"` under
  `max_tokens=8, temperature=0` on 0.6B, 1.7B, and 4B — a strong
  determinism signal for the sampler / detokenizer / frontend
  chain and the Qwen3 chat template.
* **Raw `/generate` diverges at the 4B tier.** 0.6B and 1.7B both
  emit `" Paris. The capital of the United"` under
  `max_tokens=8, ignore_eos=true`; 4B emits
  `" Paris. The capital of Germany is"`. Divergence begins at
  token 5 and reflects genuine 4B-tier greedy behaviour on the
  raw (non-chat-templated) prompt, not sampler drift — the greedy
  `torch.argmax` codepath is shared across all three tiers.
* **Same cleanup signature on every gate on every model.**
  Uvicorn shutdown sequence completes, SIGTERM sweep clears
  children without SIGKILL, `NO_RESIDUAL`, port released. A
  benign `resource_tracker` semaphore-leak warning at parent exit
  is present on every gate on every model.

## 3. Resource footprint at bring-up (per model, per rank)

| Metric | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|---|---|---|---|
| KV cache tokens allocated | 772,720 | 772,720 | 566,928 |
| KV cache size per rank | 41.27 GiB | 41.27 GiB | 38.93 GiB |
| Scheduler `proc-mem` per rank | ~44 GiB | ~44 GiB | ~44.3 GiB |
| Free memory after initialization, per rank | ~4.74 GiB | ~4.74 GiB | ~4.74 GiB |
| HBM-Usage per active NPU (bring-up) | ~60.7 GiB / 64 GiB | ~60.7 GiB / 64 GiB | ~60.7 GiB / 64 GiB |
| Parse-to-ready wall-clock | ~20 s | ~20 s | ~20 s |
| Rank / NPU pinning | rank 0 → NPU 0, rank 1 → NPU 1 | same | same |
| NPUs 2–7 | container baseline (~16.4 GiB) | same | same |

Observation: the KV-cache autosizer absorbs the growing
weight-tensor footprint at higher model tiers while holding
`Free memory after initialization` at ~4.74 GiB per rank
(driven by `memory_ratio=0.9`). The 4B tier fits at the same
~92% HBM-Usage pressure as the smaller tiers because the
autosizer traded ~206k KV-cache tokens (~2.3 GiB per rank) for
the extra weight footprint. No `--memory-ratio` amendment was
needed at any tier.

## 4. Common constraints (apply uniformly to all three models)

Any client of the server path on any of the three tiers must
honour all four:

1. **`--page-size 16` required for `npu_fia`.** CANN
   constraint — the default `page_size=1` is rejected by the
   BF16 no-quant `aclnnFusedInferAttentionScoreV3` kernel with
   error `561002`. `--page-size 16` held across every gate on
   every model with zero
   `block_size` / `561002` / `CheckFeatureNoquant` matches.
2. **`model` field must equal the absolute path returned by
   `/v1/models`.** The server reports its model as the absolute
   filesystem path passed to `--model-path` (0.6B →
   `/mnt/nvme/models/Qwen3-0.6B`, 1.7B →
   `/mnt/nvme/models/Qwen3-1.7B`, 4B →
   `/mnt/nvme/models/Qwen3-4B`). Short aliases are not resolved.
   Any downstream request must send that exact string in its
   `model` field.
3. **`temperature=0` (or `top_k=1`) required for the greedy chat
   path.** `SamplingParams.is_greedy` (`python/minisgl/core.py:30`)
   evaluates
   `(temperature <= 0.0 or top_k == 1) and top_p == 1.0`.
   Requests that fail this predicate route through `sample_impl`
   (`python/minisgl/engine/sample.py:30`), which imports the
   CUDA-only `flashinfer.sampling` module and would crash both
   scheduler ranks (Phase 6B.8 root cause). Every gate that
   exercised the sampler on any of the three models reported zero
   `flashinfer` matches under this constraint.
4. **`usage` token counters currently unpopulated.**
   `prompt_tokens`, `completion_tokens`, `total_tokens` all
   report `0` on every non-stream `/v1/chat/completions` response
   observed across the three models. The `usage` block is
   present and correctly typed in the non-stream envelope but
   not accounted. The streaming chunk envelope omits `usage`
   entirely on all three models.

Same-shape `text_completion.chunk` streaming envelope + trailing
`data: [DONE]` sentinel applies across all three models — the
Phase 6B.10 / 6B.16 / 6B.22 stream bodies share the exact same
delta structure.

## 5. Single known blocked path (uniform across all three models)

**`POST /v1/chat/completions` with default OpenAI sampling
(`temperature=1.0`, `top_k=-1`, `top_p=1.0`)** — expected
**BLOCKED** on 0.6B, 1.7B, and 4B. The root cause is device-level
(CUDA-only `flashinfer.sampling` import in `sample_impl`), not
model-specific; the failure documented at Phase 6B.8 for
Qwen3-0.6B is expected to reproduce identically on 1.7B and 4B if
the request body drops the `temperature=0` override. Only 0.6B has
been experimentally observed to fail this way (Phase 6B.8); the
1.7B and 4B expected-blocked verdicts are inferences from the
shared device-level root cause, not new experiments.

Resolution options (unchanged across all three per-model
summaries, all out of scope for the v0.2.0a1 envelope):

* Recipe-level constraint (`temperature=0` in every request) —
  the workaround exercised on every model in this envelope.
* NPU-native sampling backend — requires runtime code changes.
* Optional-import guard around `flashinfer.sampling` with a
  CUDA-free fallback — requires runtime code changes.

## 6. Cleanup / invariant summary (cross-model)

Across every gate on every model:

* HCCL + Gloo present as the sole comm backend
  (`ProcessGroupHCCL` watchdog warnings + `[Gloo] Rank {0,1} is
  connected to 1 peer ranks.`).
* Zero external `pynccl` / `nccl` / `cuda` matches outside the
  intentional `use_pynccl=False` argument-echo,
  `cuda_graph_bs=None`, `cuda_graph_max_bs=0` argument-echoes,
  and the deliberate `CUDA graph is disabled.` notice.
* Zero `block_size` / `561002` / `CheckFeatureNoquant` matches
  under `--page-size 16`.
* Zero `flashinfer` matches on any sampler-exercising gate under
  the `temperature=0` constraint.
* `NO_RESIDUAL` after SIGTERM on every gate; port released; no
  `SIGKILL` fallback needed.
* Same benign `resource_tracker` semaphore-leak notice at parent
  exit on every gate.

## 7. Explicit non-goals of the v0.2.0a1 server envelope

The three-model matrix in §2 does **not** cover, and this roll-up
does **not** claim, the following:

* **Non-greedy sampling support.** All chat gates on all three
  models use the `temperature=0` recipe-level workaround; the
  underlying CUDA-only `flashinfer.sampling` blocker is not
  fixed. Any client that needs stochastic sampling is out of
  envelope.
* **Benchmarking.** Wall-clock numbers cited in the per-model
  summaries (e.g. ~2.7–2.9 s for an 8-token chat streamed
  response, ~20 s parse-to-ready) are anecdotal single-run
  observations, not benchmark measurements. No latency or
  throughput claim is made.
* **Multi-request / server concurrency.** Every gate on every
  model exercised exactly one request per launch. Multi-request,
  batching, and concurrent-client behaviour on the server path
  are not covered.
* **TP elasticity / TP switching.** The envelope is fixed at
  TP=2. Bring-up under other TP topologies (TP=1 offline path,
  TP=4, TP=8) on the server path is not covered by this
  envelope. Only the offline TP=1 path has separate coverage
  (out of this roll-up's scope); TP=4 / TP=8 have no gate at all.

## 8. Next recommended gate

Not decided by this file. The three-model server envelope is now
consolidated; the next unit of work is a follow-up decision (a
larger model that would force a `--memory-ratio` recipe amendment;
a code-side fix that removes the `temperature=0` constraint; a
multi-request behaviour probe; or an entirely different
sub-project). This summary does not commit to any one of those
paths.

## 9. What this summary does NOT establish

* No new experiment was run for this gate. Every claim above is
  reference-only, cited to the three source per-model summaries
  in the header (which in turn cite their leaf verdicts).
* No 4B → 8B / 14B extrapolation is asserted; the ~92% HBM-Usage
  pressure at 4B is the ceiling on this envelope for KV-cache
  autosizing at TP=2 with `memory_ratio=0.9`, and a larger tier
  would likely require a recipe amendment.
* No claim is made about model-quality parity, semantic accuracy
  of generations, or task performance on any of the three
  models — only that the server path emits the expected
  wire-level envelopes under the greedy contract.
* No client-disconnect / abort-ack behaviour has been exercised
  on any of the three models.
* No `usage` accounting fix has been validated; §4 item 4 remains
  open.
