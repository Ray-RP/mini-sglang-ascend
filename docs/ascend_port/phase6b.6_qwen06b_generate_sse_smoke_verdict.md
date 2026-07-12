# Phase 6B.6 Qwen3-0.6B Server `/generate` SSE Smoke Verdict

**Kind:** Documentation-only verdict for the Qwen3-0.6B fixed-TP2
server `/generate` SSE smoke against the Ascend host. Launched
with the Phase 6B.2 recipe
([`phase6b.2_server_launch_recipe.md`](./phase6b.2_server_launch_recipe.md)),
issued one `curl` to `POST /generate`, observed the outcome, then
terminated. This file introduces no runtime, script, test, tag,
GitHub Release, or `CHANGELOG.md` change and does not print
credentials.

Envelope: fixed TP=2, eager, `npu_fia`, bf16, greedy — v0.2.0a1.

**Overall verdict: BLOCKED** — server started cleanly, `POST /generate`
returned HTTP 200 and opened the SSE stream, but both scheduler ranks
crashed on the first NPU forward pass with `aclnnFusedInferAttentionScoreV3`
error `561002`. Zero SSE data events were delivered. Root cause is a
gap in the Phase 6B.2 recipe: `--page-size` defaults to `1`, but
`npu_fia` (`aclnn` no-quant path) requires block size aligned to
`16`.

---

## 1. Environment

* Host: Ascend NPU host (8 × 910B1).
* Container: `998ce5ba6e5e`.
* Repo tree: `/mnt/nvme/LR-606/mini-sglang-ascend` (unchanged since
  Phase 6B.3).
* Model: `/mnt/nvme/models/Qwen3-0.6B`.
* Working dir: `/mnt/nvme/LR-606/phase6b6/` (launch script copied
  from Phase 6B.3; log saved locally in the container).
* Port: `1919`.

## 2. Launch invocation

```
python -m minisgl.server.launch \
  --model-path /mnt/nvme/models/Qwen3-0.6B \
  --tp-size 2 \
  --attention-backend npu_fia \
  --disable-pynccl \
  --cuda-graph-max-bs 0 \
  --host 0.0.0.0 \
  --port 1919
```

Parsed `ServerArgs` recorded `page_size=1` (the `ServerArgs` /
`SchedulerConfig` default). This flag is not part of the Phase 6B.2
recipe.

## 3. Server ready state

* Parent PID `41411`; scheduler ranks `41421` (TP0) and `41422`
  (TP1); detokenizer `41423`.
* Log excerpt:
  ```
  [Gloo] Rank 0 is connected to 1 peer ranks.
  [Gloo] Rank 1 is connected to 1 peer ranks.
  Allocating 795005 tokens for KV cache, K + V = 42.46 GiB   (rank=0)
  Allocating 795005 tokens for KV cache, K + V = 42.46 GiB   (rank=1)
  CUDA graph is disabled.
  Scheduler is idle, waiting for new reqs...
  Scheduler is ready
  API server is ready to serve on 0.0.0.0:1919
  INFO:     Uvicorn running on http://0.0.0.0:1919 (Press CTRL+C to quit)
  ```
* `netstat` at ready: `tcp 0.0.0.0:1919 LISTEN 41411/python`.

## 4. `/generate` request and response

Command executed inside the container:

```
curl -N -sS http://127.0.0.1:1919/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_tokens":8,"ignore_eos":true}'
```

Server access log:

```
INFO:     127.0.0.1:41654 - "POST /generate HTTP/1.1" 200 OK
```

A follow-up `curl -v` against the same running server confirmed
the response headers verbatim:

```
< HTTP/1.1 200 OK
< date: Sun, 12 Jul 2026 17:33:18 GMT
< server: uvicorn
< content-type: text/event-stream; charset=utf-8
< transfer-encoding: chunked
```

The chunked body then never delivered any bytes — the SSE stream
opened but produced **zero data events** because both scheduler
ranks crashed on the first attention forward pass before any token
could be scheduled back.

## 5. Root-cause evidence (both ranks)

Log lines quoted verbatim from
`/mnt/nvme/LR-606/phase6b6/server.log`:

* Rank 0 (PID 41421, Device 0):
  ```
  File ".../python/minisgl/attention/ascend_fia.py", line 266, in forward
      result = torch_npu.npu_fused_infer_attention_score(...)
  RuntimeError: npu_fused_infer_attention_score_symint:...:415
      NPU function error: call aclnnFusedInferAttentionScoreV3 failed,
      error code is 561002
  [ERROR] 2026-07-12-17:20:56 (PID:41421, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
  EZ9999[PID: 41421] ...: In NO_QUANT situation, block_size should aligned to 16, but got 1.
      [FUNC:CheckFeatureNoquantBlockSize]
      [FILE:fused_infer_attention_score_tiling_check_feature.cpp]
      [LINE:159]
  ```
* Rank 1 (PID 41422, Device 1): identical stack, identical
  `error code 561002`, identical
  `block_size should aligned to 16, but got 1` message.
* Both `Process minisgl-TP0-scheduler` and
  `Process minisgl-TP1-scheduler` printed the exception via
  `multiprocessing.Process._bootstrap` and exited.

Interpretation:

* `--attention-backend npu_fia` routes to
  `python/minisgl/attention/ascend_fia.py:forward` which calls
  `torch_npu.npu_fused_infer_attention_score` — the CANN
  `aclnnFusedInferAttentionScoreV3` kernel.
* That kernel's `CheckFeatureNoquantBlockSize` (no-quant BF16 path)
  requires the paged-KV block size to be a multiple of `16`.
* Under the Phase 6B.2 recipe, `ServerArgs.page_size` defaults to
  `1` (verified in the `ServerArgs(...)` log line: `page_size=1`).
  The offline driver at v0.2.0a1 was proven with `page_size=16`;
  the server recipe inherited the default and thus a
  block-size-16-only kernel is invoked with `block_size=1`, which
  is what the kernel rejects.
* This is a **recipe gap**, not a code bug. The fix is one added
  flag: `--page-size 16`.

## 6. What the client saw

* HTTP status: `200`.
* Content-Type: `text/event-stream; charset=utf-8`.
* Transfer-Encoding: `chunked`.
* SSE data events delivered: `0`.
* Final generated text: none.
* Actual output token count (visible from a returned event): none;
  no completion payload was ever emitted.

## 7. No CUDA / NCCL / pynccl fallback

Grep over the full server log:

* `pynccl` / `nccl` external matches (excluding
  `use_pynccl=False` in the `ServerArgs` line): **zero**.
* `cuda` external matches (excluding
  `cuda_graph_bs=`, `cuda_graph_max_bs=`, and the
  `CUDA graph is disabled.` notice): **zero**.
* HCCL + Gloo backend evidence identical to Phase 6B.3 / 6B.5:
  2× `ProcessGroupHCCL` watchdog warnings +
  `[Gloo] Rank {0,1} is connected to 1 peer ranks.`.

The failure is purely a per-op tiling constraint in the CANN
`aclnnFusedInferAttentionScoreV3` kernel; it did not trigger any
CUDA or NCCL fallback path.

## 8. Termination and residual check

* After the scheduler ranks died, the parent Uvicorn stayed alive
  and the port stayed in `LISTEN` — the process supervisor did not
  cascade the child failure into a server exit. A subsequent
  `curl` was accepted (HTTP 200 headers returned) but hung
  indefinitely because no scheduler could service it. This
  behaviour is an observation, not a scope item; a later doc gate
  may lock it as an intentional watchdog contract or file it as a
  supervisor gap.
* `kill -TERM 41411` (parent) + child SIGTERM sweep:
  * All `minisgl.server.launch`, `multiprocessing.spawn`, and
    `multiprocessing.resource_tracker` processes exited on the
    signal path. No `SIGKILL` fallback was needed for the
    remaining processes.
* Final `ps -eo pid,cmd | grep -E "multiprocessing.(spawn|resource)|minisgl.server"`:
  `NO_RESIDUAL`.
* Listening port `:1919` released; `netstat`: no match.

## 9. Verdict — per required checklist

| Check | Result |
|---|---|
| server reaches LISTEN | PASS — `tcp 0.0.0.0:1919 LISTEN 41411/python` |
| `/generate` HTTP status | PASS — `200` returned, SSE `content-type: text/event-stream` |
| SSE event count | FAIL — 0 data events; only the HTTP 200 headers of an empty chunked stream |
| final generated text | FAIL — none; no completion payload emitted |
| actual output token count (if visible) | FAIL — none |
| no traceback/error in server log | FAIL — 2 `RuntimeError` tracebacks (one per scheduler rank); CANN error `561002`; `block_size should aligned to 16, but got 1` |
| no CUDA/NCCL/pynccl fallback | PASS — zero non-argument matches; failure is on the CANN NPU op path |
| clean shutdown | PASS — parent + all children exited on SIGTERM; port released |
| no residual server processes | PASS — final scan reported `NO_RESIDUAL` |

**Overall verdict: BLOCKED** — server bring-up and OpenAI-metadata
route are already PASS from Phase 6B.3 and Phase 6B.5, but the
`/generate` end-to-end path is blocked on the `--page-size 16`
recipe gap. Cleanup and no-fallback invariants held.

## 10. Recommended next gate

**Phase 6B.7 — add `--page-size 16` to the Phase 6B.2 recipe and
re-run `/generate` SSE smoke.**

Concretely: a docs-only gate that appends `--page-size 16` to the
required-flags list in
`docs/ascend_port/phase6b.2_server_launch_recipe.md` (with
`aclnnFusedInferAttentionScoreV3` block-size rationale), plus a
re-run of §4 above. Expected outcome: SSE stream delivers 8 data
events for `max_tokens=8, ignore_eos=true`, `final_text` present,
zero rank-side tracebacks.

## 11. What this gate does NOT establish

* No `/v1/chat/completions` call (streaming or non-streaming) was
  attempted.
* No performance claim of any kind is made.
* The observation that the Uvicorn parent stays alive after both
  scheduler ranks die is recorded but not fixed here — it is a
  supervisor observation, not part of this gate's scope.
