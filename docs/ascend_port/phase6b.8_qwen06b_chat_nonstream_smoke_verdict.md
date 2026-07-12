# Phase 6B.8 Qwen3-0.6B Server `/v1/chat/completions` Non-Stream Smoke Verdict

**Kind:** Documentation-only verdict for the Qwen3-0.6B fixed-TP2
server `/v1/chat/completions` (non-stream) smoke against the Ascend
host. Launched with the Phase 6B.7 recipe
([`phase6b.2_server_launch_recipe.md`](./phase6b.2_server_launch_recipe.md)
as amended at Phase 6B.7 to include `--page-size 16`), issued one
`curl` to `POST /v1/chat/completions` with `stream=false`, observed
the outcome, then terminated. This file introduces no runtime,
script, test, tag, GitHub Release, or `CHANGELOG.md` change and does
not print credentials.

Envelope: fixed TP=2, eager, `npu_fia`, bf16, `page_size=16` —
v0.2.0a1.

**Overall verdict: BLOCKED** — the server started cleanly and
survived every earlier readiness gate, but on the first
`/v1/chat/completions` request both scheduler ranks crashed inside
the sampler because the default OpenAI-style request body
(`temperature=1.0`, `top_p=1.0`, `top_k=-1`) does **not** satisfy
`SamplingParams.is_greedy` and therefore takes the non-greedy code
path, which imports the CUDA-only `flashinfer.sampling` module.
`flashinfer` is not installed on the NPU container, so the import
raises `ModuleNotFoundError` and the forward loop dies before any
tokens can be produced.

Root cause is a **request-shape / sampler-routing gap**, not the
Phase 6B.7 recipe. `--page-size 16` held; no `block_size` /
`561002` error appeared. The failure is not present under
`/generate` (Phase 6B.7) because that gate hit the sampler with
`SamplingParams` whose defaults keep `is_greedy=True`. Any Phase
6B.9+ fix must either (a) constrain the recipe to require
`temperature=0` (or `top_k=1`) in the request body, (b) provide an
NPU-compatible sampling backend for non-greedy paths, or (c) route
non-greedy sampling through a CUDA-free implementation.

---

## 1. Environment

* Host: Ascend NPU host (8 × 910B1).
* Container: `998ce5ba6e5e`.
* Repo tree: `/mnt/nvme/LR-606/mini-sglang-ascend` (unchanged since
  Phase 6B.3).
* Model: `/mnt/nvme/models/Qwen3-0.6B`.
* Working dir: `/mnt/nvme/LR-606/phase6b8/` (launch script + log +
  empty `chat.headers`).
* Port: `1919`.

## 2. Launch invocation (as executed)

```
python -m minisgl.server.launch \
  --model-path /mnt/nvme/models/Qwen3-0.6B \
  --tp-size 2 \
  --attention-backend npu_fia \
  --disable-pynccl \
  --cuda-graph-max-bs 0 \
  --page-size 16 \
  --host 0.0.0.0 \
  --port 1919
```

Parsed `ServerArgs` log line confirmed every intended flag landed:

```
attention_backend='npu_fia',
cuda_graph_bs=None, cuda_graph_max_bs=0,
page_size=16,
use_pynccl=False,
tp_info=DistributedInfo(rank=0, size=2),
dtype=torch.bfloat16
```

## 3. Server ready state

* Parent PID `42266`; three `multiprocessing.spawn` children
  (`42276`, `42277` = 2 scheduler ranks + 1 detokenizer) +
  resource-tracker.
* Log excerpt:
  ```
  [Gloo] Rank 1 is connected to 1 peer ranks.
  [Gloo] Rank 0 is connected to 1 peer ranks.
  CUDA graph is disabled.
  Scheduler is idle, waiting for new reqs...
  Scheduler is ready
  API server is ready to serve on 0.0.0.0:1919
  INFO:     Uvicorn running on http://0.0.0.0:1919 (Press CTRL+C to quit)
  ```
* HCCL + Gloo handshakes present (2× `ProcessGroupHCCL` watchdog
  warnings + `[Gloo] Rank {0,1} is connected to 1 peer ranks.`).

## 4. `/v1/chat/completions` request

Command executed inside the container:

```
curl -sS -D /mnt/nvme/LR-606/phase6b8/chat.headers \
  http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/mnt/nvme/models/Qwen3-0.6B",
       "messages":[{"role":"user","content":"What is the capital of France?"}],
       "max_tokens":8,
       "stream":false}'
```

The request body carries no explicit `temperature` / `top_k` /
`top_p`, so `OpenAICompletionRequest` defaults apply:

```
temperature = 1.0   # python/minisgl/server/api_server.py:73
top_k       = -1    # python/minisgl/server/api_server.py:75
top_p       = 1.0   # python/minisgl/server/api_server.py:76
```

The resulting `SamplingParams.is_greedy`
(`python/minisgl/core.py:30`) evaluates to
`(1.0 <= 0.0 or -1 == 1) and 1.0 == 1.0`, which is `False and True`
→ `False`. The sampler therefore chooses the non-greedy code path.

## 5. Response — what the client got

| metric | value |
|---|---|
| HTTP status | **none observed** — no status line was written to `chat.headers` (0 bytes) |
| response headers file size | `0` bytes |
| response body | **none** — the client never received a JSON envelope |
| `choices` | not applicable — no body |
| `choices[0].message.role` / `.content` | not applicable — no body |
| `finish_reason` | not applicable — no body |
| `usage.prompt_tokens` / `.completion_tokens` / `.total_tokens` | not applicable — no body |
| curl exit | client-side hang, terminated when server was torn down |

The scheduler ranks crashed during the very first forward pass, so
the server never emitted a completion payload. The FastAPI request
task was still waiting on the ZMQ response channel when the curl
was aborted at server shutdown time.

## 6. Root-cause evidence (both ranks)

Log lines quoted verbatim from
`/mnt/nvme/LR-606/phase6b8/server.log`:

```
Process minisgl-TP0-scheduler:
Process minisgl-TP1-scheduler:
Traceback (most recent call last):
Traceback (most recent call last):
  File ".../python/minisgl/server/launch.py", line 31, in _run_scheduler
    scheduler.run_forever()
  ...
  File ".../python/minisgl/scheduler/scheduler.py", line 250, in _forward
    forward_output = self.engine.forward_batch(batch, sample_args)
  File ".../python/minisgl/engine/engine.py", line 261, in forward_batch
    next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
  File ".../python/minisgl/engine/sample.py", line 77, in sample
    return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
  File ".../python/minisgl/engine/sample.py", line 30, in sample_impl
    import flashinfer.sampling as sampling
ModuleNotFoundError: No module named 'flashinfer'
```

The identical traceback appears once per scheduler rank (both TP0
and TP1). Both `Process minisgl-TP0-scheduler` and
`Process minisgl-TP1-scheduler` printed the exception via
`multiprocessing.Process._bootstrap` and exited. Immediately after
these tracebacks the log shows:

```
INFO:     Shutting down
INFO:     Waiting for background tasks to complete. (CTRL+C to force quit)
```

which is the parent Uvicorn responding to the operator's SIGTERM
after the scheduler ranks died.

### Interpretation

* The frontend accepted the JSON body without any per-field
  override, so the tokenizer server passed default sampling
  parameters through to the scheduler.
* At the sampler, `Sampler.prepare`
  (`python/minisgl/engine/sample.py:54-55`) sets
  `BatchSamplingArgs.temperatures = None` **only** when every
  `SamplingParams.is_greedy` is `True`. Because the request was
  non-greedy, `temperatures` is a tensor, so `Sampler.sample`
  falls through to `sample_impl(...)`
  (`python/minisgl/engine/sample.py:77`).
* `sample_impl` starts with
  `import flashinfer.sampling as sampling`
  (`python/minisgl/engine/sample.py:30`). `flashinfer` is a
  CUDA-only sampling library and is not installed on the NPU
  container — expected, given the Ascend port explicitly avoids
  CUDA-only wheels.
* The forward call that raises is
  `self.engine.forward_batch(batch, sample_args)` at
  `python/minisgl/engine/engine.py:261`, i.e. exactly the first
  sampler call after the request lands.

Nothing about `--attention-backend npu_fia`, `--page-size 16`, or
the HCCL/Gloo backend contributed to this failure; the attention
kernel produced logits successfully and the crash is one line into
sampling.

## 7. `block_size` / CANN error status

* `grep -inE "block_size|561002|CheckFeatureNoquant"` on the log:
  **zero matches**.
* The Phase 6B.6 `aclnnFusedInferAttentionScoreV3` blocker is
  **not** present. `--page-size 16` from the Phase 6B.7 recipe
  held: attention BF16 no-quant tiling accepted the 16-aligned
  paged-KV blocks, and the failure moved down the pipeline to the
  sampler.

## 8. No CUDA / NCCL / pynccl fallback (with a caveat)

Grep over the full server log:

* `pynccl` / `nccl` external matches (excluding
  `use_pynccl=False` in the `ServerArgs` echo line): **zero**.
* `cuda` external matches (excluding `cuda_graph_bs=`,
  `cuda_graph_max_bs=`, and the deliberate
  `CUDA graph is disabled.` notice): **zero**.
* HCCL + Gloo evidence present as in prior gates: 2×
  `ProcessGroupHCCL` watchdog warnings + `[Gloo] Rank {0,1} is
  connected to 1 peer ranks.`.

**Caveat.** The failure itself is on a CUDA-only sampling library
import (`flashinfer.sampling`) inside `sample_impl`. This is not a
comm-backend fallback (no NCCL, no pynccl was loaded) and it did
not fabricate a CUDA process group — the module simply is not
present, so the import raises before any CUDA symbol resolution.
The invariant checked in prior gates (no CUDA/NCCL/pynccl comm
fallback) still holds; separately, the invariant "the server can
sample a non-greedy request without touching CUDA-only code" does
**not** hold. That gap is what pushes the overall verdict to
BLOCKED.

## 9. Scheduler rank health after the request

* Both scheduler processes (`42276`, `42277`) exited with the
  `ModuleNotFoundError: No module named 'flashinfer'` traceback
  above and reached `<defunct>` before the operator issued
  SIGTERM.
* The Uvicorn parent (`42266`) stayed alive with its LISTEN
  socket bound — same supervisor behaviour observed at Phase
  6B.6 (dead scheduler ranks do not cascade into a server exit).
* Post-crash, no `Scheduler is idle, waiting for new reqs...`
  line appears again; the scheduler never re-entered the idle
  state.

## 10. Termination and residual check

* `kill -TERM 42266` (parent). Uvicorn shutdown sequence
  completed:
  ```
  INFO:     Shutting down
  INFO:     Waiting for background tasks to complete. (CTRL+C to force quit)
  ```
* Follow-up SIGTERM sweep over any surviving
  `multiprocessing.spawn` / `resource_tracker` children exited on
  the signal path (no `SIGKILL` fallback).
* Final `ps -eo pid,cmd | grep -E "multiprocessing.(spawn|resource)|minisgl.server"`:
  `NO_RESIDUAL`.
* Listening port `:1919` released; `netstat`: no match.

## 11. Verdict — per required checklist

| Check | Result |
|---|---|
| `/v1/chat/completions` HTTP status | FAIL — no status line was written; response headers file is 0 bytes |
| response JSON shape | FAIL — no body was returned |
| `choices[0].message.content` observed | FAIL — no body |
| `finish_reason`, if present | FAIL — no body |
| `usage` fields, if present | FAIL — no body |
| rank health after request | FAIL — both scheduler ranks (`42276`, `42277`) died with `ModuleNotFoundError: No module named 'flashinfer'` at `python/minisgl/engine/sample.py:30`; ranks were `<defunct>` before shutdown |
| no `block_size` / `561002` error | PASS — `--page-size 16` from Phase 6B.7 held; zero `block_size` / `561002` / `CheckFeatureNoquant` matches |
| no CUDA / NCCL / pynccl fallback | PASS (comm backend) — zero non-argument `pynccl` / `nccl` / `cuda` matches; HCCL + Gloo backends confirmed. Caveat: the crash itself is on a CUDA-only sampling library import (`flashinfer.sampling`), not on a comm backend fallback |
| clean shutdown | PASS — Uvicorn `Shutting down` completed on SIGTERM |
| no residual server processes | PASS — final scan reported `NO_RESIDUAL`, port released |

**Overall verdict: BLOCKED.**

## 12. Recommended next gate

**Phase 6B.9 — decide the sampler-routing contract for non-greedy
OpenAI requests on NPU.**

Three mutually-exclusive resolutions are visible today:

1. **Recipe-level constraint.** Amend the Phase 6B.7 recipe to
   require `temperature=0` (or `top_k=1`, or `top_p<1.0` in a
   combination that yields `is_greedy=True`) in every
   `/v1/chat/completions` / `/generate` request body until a
   non-greedy NPU sampler exists. Docs-only, no code change.
2. **NPU sampling backend.** Introduce an NPU-compatible sampling
   path so `sample_impl` no longer requires `flashinfer`. Runtime
   code change; out of scope for the v0.2.0a1 envelope.
3. **Optional-import guard.** Gate the `flashinfer.sampling`
   import behind a device check in `sample_impl` and fall back to
   a CUDA-free reference implementation. Runtime code change; out
   of scope for the v0.2.0a1 envelope.

The docs-only path (option 1) is the minimum-scope Phase 6B.9. It
must (a) update the Phase 6B.7 recipe with an explicit sampler
constraint, (b) rerun the `/v1/chat/completions` non-stream smoke
with an explicit `temperature=0` request body, and (c) record the
resulting `choices[0].message.content` and `usage` fields.

## 13. What this gate does NOT establish

* No `/v1/chat/completions` **with `stream=true`** was attempted.
* No `/generate` re-run was performed in this gate.
* No performance claim of any kind is made.
* The observation that the Uvicorn parent stays alive after both
  scheduler ranks die is unchanged from Phase 6B.6 and remains a
  supervisor observation, not a scope item here.
* Whether short model names (e.g. `qwen3-0.6b`) would be
  auto-resolved to the filesystem-path model id is still
  untested; the request in this gate used the exact `id` string
  reported by `/v1/models` at Phase 6B.5.
