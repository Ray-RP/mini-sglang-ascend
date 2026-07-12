# Phase 6B.7 Qwen3-0.6B Server `/generate` SSE Smoke with `--page-size 16`

**Kind:** Documentation-only verdict for the corrective re-run of
the Phase 6B.6 `/generate` SSE smoke against the Ascend host. The
Phase 6B.2 recipe was amended in the same gate to add
`--page-size 16`; this file records the successful rerun. No
runtime, script, test, tag, GitHub Release, or `CHANGELOG.md`
change; no credential printing.

Envelope: fixed TP=2, eager, `npu_fia`, bf16, greedy, `page_size=16`
— v0.2.0a1.

**Overall verdict: PASS.**

---

## 1. Environment

* Host: Ascend NPU host (8 × 910B1).
* Container: `998ce5ba6e5e`.
* Repo tree: `/mnt/nvme/LR-606/mini-sglang-ascend` (unchanged since
  Phase 6B.3).
* Model: `/mnt/nvme/models/Qwen3-0.6B`.
* Working dir: `/mnt/nvme/LR-606/phase6b7/` (launch script + log +
  response body/headers).
* Port: `1919`.

## 2. Recipe change

The Phase 6B.2 recipe
([`phase6b.2_server_launch_recipe.md`](./phase6b.2_server_launch_recipe.md))
now lists `--page-size 16` in its required-flags block and in its
launch skeleton. Rationale, quoted verbatim from that file:

> `--page-size 16` — the `npu_fia` backend routes to the CANN
> `aclnnFusedInferAttentionScoreV3` kernel. Its BF16 no-quant path
> requires `block_size` (paged-KV page size) aligned to `16`, per
> `CheckFeatureNoquantBlockSize`
> (`fused_infer_attention_score_tiling_check_feature.cpp:159`).
> `ServerArgs.page_size` defaults to `1`, which the kernel rejects
> with error `561002`
> (`In NO_QUANT situation, block_size should aligned to 16, but got 1`).

## 3. Launch invocation (as executed)

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

Parsed `ServerArgs` log line confirmed the flag landed as intended:

```
attention_backend='npu_fia',
cuda_graph_bs=None, cuda_graph_max_bs=0,
page_size=16,
use_pynccl=False,
tp_info=DistributedInfo(rank=0, size=2),
dtype=torch.bfloat16
```

## 4. Server ready state

* Parent PID `41869`; three `multiprocessing.spawn` children
  (`41879`, `41880`, `41881` = 2 scheduler ranks + 1 detokenizer)
  + resource-tracker `41878`.
* Log excerpt:
  ```
  [Gloo] Rank 0 is connected to 1 peer ranks.
  [Gloo] Rank 1 is connected to 1 peer ranks.
  CUDA graph is disabled.
  Scheduler is idle, waiting for new reqs...
  Scheduler is ready
  API server is ready to serve on 0.0.0.0:1919
  INFO:     Uvicorn running on http://0.0.0.0:1919 (Press CTRL+C to quit)
  ```
* `netstat`: `tcp 0.0.0.0:1919 LISTEN 41869/python`.

## 5. `/generate` request and response

Command executed inside the container:

```
curl -N -sS http://127.0.0.1:1919/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_tokens":8,"ignore_eos":true}'
```

Curl instrumentation:

| metric | value |
|---|---|
| HTTP status | `200` |
| content-type | `text/event-stream; charset=utf-8` |
| transfer-encoding | `chunked` |
| response bytes | `93` |
| wall-clock | `~2.90 s` |

Response headers verbatim:

```
HTTP/1.1 200 OK
date: Sun, 12 Jul 2026 17:44:11 GMT
server: uvicorn
content-type: text/event-stream; charset=utf-8
transfer-encoding: chunked
```

Response body verbatim (SSE stream):

```
data:  Paris
data: .
data:  The
data:  capital
data:  of
data:  Italy
data:  is
data: [DONE]
```

* SSE data events: **8** (`7` decoded tokens + one `[DONE]`
  sentinel).
* Final generated text (concatenated tokens):
  `" Paris. The capital of Italy is"`.
* Under `max_tokens=8, ignore_eos=true, greedy`, the model
  produced 7 whitespace-prefixed token strings then closed with
  the `[DONE]` sentinel. Whether the eighth token was truncated
  by an internal `<eos>`/`[DONE]` sentinel handling detail (given
  `ignore_eos=true`) is an observation only; the request completed
  the SSE stream cleanly.

Server access log:

```
INFO:     127.0.0.1:57648 - "POST /generate HTTP/1.1" 200 OK
```

## 6. Scheduler rank health after the request

* All four child processes (`41878` resource-tracker, `41879`,
  `41880`, `41881`) remained alive after the SSE completed.
* Post-completion log line:
  ```
  [2026-07-12|17:44:15|core|rank=0] Scheduler is idle, waiting for new reqs...
  ```
  This is proof rank 0 returned to the idle state — the request
  went all the way through the forward loop and back to the
  scheduler, not just up to the HTTP layer.
* Rank 1 also survived (parent still had all three
  `spawn_main` children when queried immediately after the
  response completed).

## 7. `block_size` / CANN error status

* `grep -iE "block_size|acl.*561002|traceback|error code"` on
  `/mnt/nvme/LR-606/phase6b7/server.log`: **zero matches**.
* No `RuntimeError`, no
  `Process minisgl-TP*-scheduler` bootstrap traceback.
* The Phase 6B.6 blocker
  (`In NO_QUANT situation, block_size should aligned to 16, but got 1`)
  is resolved by the `--page-size 16` flag.

## 8. No CUDA / NCCL / pynccl fallback

* `pynccl` / `nccl` external matches (excluding the intended
  `use_pynccl=False` argument-echo line): **zero**.
* `cuda` external matches (excluding
  `cuda_graph_bs=`, `cuda_graph_max_bs=`, and the deliberate
  `CUDA graph is disabled.` notice): **zero**.
* HCCL + Gloo backend evidence as before: 2× `ProcessGroupHCCL`
  watchdog warnings + `[Gloo] Rank {0,1} is connected to 1 peer ranks.`.

## 9. Termination and residual check

* `kill -TERM 41869` (parent). Uvicorn shutdown sequence completed:
  ```
  Shutting down
  Waiting for application shutdown.
  Application shutdown complete.
  Finished server process [41869]
  ```
* One benign
  `multiprocessing.resource_tracker: There appear to be 2 leaked semaphore objects`
  warning at parent exit — same benign notice as Phase 6B.5. Not
  root-caused here.
* Follow-up SIGTERM sweep over any surviving
  `multiprocessing.spawn` / `resource_tracker` children exited on
  the signal path (no `SIGKILL` fallback).
* Final `ps -eo pid,cmd | grep -E "multiprocessing.(spawn|resource)|minisgl.server"`:
  `NO_RESIDUAL`.
* Listening port `:1919` released; `netstat`: no match.

## 10. Verdict — per required checklist

| Check | Result |
|---|---|
| launch includes `--page-size 16` | PASS — `page_size=16` in `ServerArgs` log line; `launch.sh` records the flag on line 8 |
| server reaches LISTEN | PASS — `tcp 0.0.0.0:1919 LISTEN 41869/python` |
| `/generate` HTTP status | PASS — `200`, `content-type: text/event-stream; charset=utf-8`, `transfer-encoding: chunked` |
| SSE data event count | PASS — 8 events emitted (7 token payloads + `[DONE]`) |
| final generated text (if emitted) | PASS — `" Paris. The capital of Italy is"` |
| scheduler rank health after request | PASS — parent + 3 spawn children still alive; rank 0 logged `Scheduler is idle, waiting for new reqs...` post-completion |
| no aclnn `block_size` error | PASS — zero `block_size`/`561002`/`RuntimeError` matches |
| no CUDA/NCCL/pynccl fallback | PASS — zero non-argument matches; HCCL + Gloo present |
| clean shutdown | PASS — Uvicorn `Application shutdown complete.` + `Finished server process [41869]`; one benign `resource_tracker` semaphore cleanup notice |
| no residual processes | PASS — final scan `NO_RESIDUAL`, port released |

**Overall verdict: PASS.**

## 11. What this gate does NOT establish

* No `/v1/chat/completions` call (streaming or non-streaming) was
  attempted.
* No multi-request or batch behaviour was tested.
* The supervisor behaviour observed at Phase 6B.6 (Uvicorn parent
  survives dead scheduler ranks) is out of scope for this gate.
* The `resource_tracker` semaphore-leak warning at parent exit is
  a documented benign observation, not a root-caused finding.
* No performance claim of any kind is made.
