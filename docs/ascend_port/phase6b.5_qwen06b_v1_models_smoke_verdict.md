# Phase 6B.5 Qwen3-0.6B Server `/v1/models` Smoke Verdict

**Kind:** Documentation-only verdict for the Qwen3-0.6B fixed-TP2
server `/v1/models` smoke against the Ascend host. Launched with
the Phase 6B.2 recipe
([`phase6b.2_server_launch_recipe.md`](./phase6b.2_server_launch_recipe.md)),
issued exactly one `curl` to `GET /v1/models`, then terminated. No
generation route (`/generate`, `/v1/chat/completions`) was
exercised, no streaming was tested. This file introduces no
runtime, script, test, tag, GitHub Release, or `CHANGELOG.md`
change and does not print credentials.

Envelope: fixed TP=2, eager, `npu_fia`, bf16, greedy — v0.2.0a1.

---

## 1. Environment

* Host: Ascend NPU host (8 × 910B1).
* Container: `998ce5ba6e5e`.
* Repo tree: `/mnt/nvme/LR-606/mini-sglang-ascend`
  (server-path source matches audit tree; unchanged since
  Phase 6B.3).
* Model: `/mnt/nvme/models/Qwen3-0.6B`.
* Working dir: `/mnt/nvme/LR-606/phase6b5/`
  (launch script copied from Phase 6B.3; log and response body
  saved locally in the container).
* Port: `1919`.
* `prompt_toolkit` already installed from Phase 6B.3 preflight.

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

Executed via `setsid /mnt/nvme/LR-606/phase6b5/launch.sh` inside
the container.

## 3. Server ready state

Parent PID observed: `41238`.

Log excerpt:

```
[2026-07-12|15:48:22|initializer] Tokenize server 0 is ready
[Gloo] Rank 0 is connected to 1 peer ranks.
[Gloo] Rank 1 is connected to 1 peer ranks.
[2026-07-12|15:48:24|core|rank=0] Free memory before loading model: 47.86 GiB
[2026-07-12|15:48:25|core|rank=0] Allocating 795005 tokens for KV cache, K + V = 42.46 GiB
[2026-07-12|15:48:25|core|rank=1] Allocating 795005 tokens for KV cache, K + V = 42.46 GiB
[2026-07-12|15:48:27|core|rank=0] CUDA graph is disabled.
[2026-07-12|15:48:28|core|rank=0] Scheduler is idle, waiting for new reqs...
[2026-07-12|15:48:28|initializer] Scheduler is ready
[2026-07-12|15:48:28|FrontendAPI] API server is ready to serve on 0.0.0.0:1919
INFO:     Uvicorn running on http://0.0.0.0:1919 (Press CTRL+C to quit)
```

`netstat -tlnp` snapshot at ready time:

```
tcp   0   0 0.0.0.0:1919   0.0.0.0:*   LISTEN   41238/python
```

Wall-clock parse → ready: ~20 s.

## 4. `/v1/models` request and response

Command executed inside the container:

```
curl -sS http://127.0.0.1:1919/v1/models
```

Curl instrumentation (`-w`):

| metric | value |
|---|---|
| HTTP status | `200` |
| content-type | `application/json` |
| response bytes | `163` |
| wall-clock | `~2.6 ms` |

Response body (verbatim):

```json
{
  "object": "list",
  "data": [
    {
      "id": "/mnt/nvme/models/Qwen3-0.6B",
      "object": "model",
      "created": 1783871547,
      "owned_by": "mini-sglang",
      "root": "/mnt/nvme/models/Qwen3-0.6B"
    }
  ]
}
```

* Envelope: OpenAI-compatible `ModelList` — `object=list`, `data`
  is a list of `Model` records.
* Model id: `"/mnt/nvme/models/Qwen3-0.6B"` — the server reports
  the model by its `--model-path` filesystem path rather than a
  short handle. This is the OpenAI-`model` string clients must
  send to any later `/v1/chat/completions` call.
* `owned_by`: `"mini-sglang"`.
* Single-entry list — the fixed-TP2 server exposes exactly one
  model, as expected for this launch envelope.

## 5. Server-side access log

The parent Uvicorn access log confirmed the request landed and
returned 200 without going through any request-generation code
path:

```
INFO:     127.0.0.1:43022 - "GET /v1/models HTTP/1.1" 200 OK
```

## 6. No CUDA / NCCL / pynccl fallback

Grep results against `/mnt/nvme/LR-606/phase6b5/server.log`
(matches on `use_pynccl=False`, `cuda_graph_bs=`,
`cuda_graph_max_bs=`, and the intentional `CUDA graph is disabled.`
notice are excluded because they are the deliberate envelope
signals, not fallback markers):

* `pynccl` / `nccl` external matches: **zero**.
* `cuda` external matches: **zero**.
* `traceback` / `error` / `exception` / `fail`: **zero**.

HCCL primary + Gloo sidecar handshake evidence is the same as
Phase 6B.3: 2× `ProcessGroupHCCL` watchdog warnings +
`[Gloo] Rank {0,1} is connected to 1 peer ranks.` — proof the
NPU communication backend actually ran.

## 7. Clean shutdown and residual check

* `kill -TERM 41238` (parent).
* Uvicorn shutdown sequence completed:
  ```
  Shutting down
  Waiting for application shutdown.
  Application shutdown complete.
  Finished server process [41238]
  ```
* One benign Python-runtime notice at parent exit:
  ```
  multiprocessing/resource_tracker.py:254: UserWarning:
    resource_tracker: There appear to be 2 leaked semaphore
    objects to clean up at shutdown
  ```
  This is a `multiprocessing.resource_tracker` cleanup notice, not
  a code error. Not investigated in this gate.
* Follow-up SIGTERM sweep across surviving `multiprocessing.spawn`
  children exited on the same signal path.
* Final `ps -eo pid,cmd | grep -E "multiprocessing.(spawn|resource)|minisgl.server"`:
  `NO_RESIDUAL`.
* Listening port `:1919` released; `netstat`: no match.

## 8. Verdict — per required checklist

| Check | Result |
|---|---|
| server reaches LISTEN | PASS — `tcp 0.0.0.0:1919 LISTEN 41238/python` |
| `/v1/models` HTTP status | PASS — `200` |
| response body shape | PASS — OpenAI `ModelList` with one `Model` entry (`object=list`, `data=[Model]`) |
| model id/name observed | PASS — `"id": "/mnt/nvme/models/Qwen3-0.6B"`, `"owned_by": "mini-sglang"` |
| no traceback/error in server log | PASS — zero `traceback`/`error`/`exception`/`fail` markers |
| no CUDA/NCCL/pynccl fallback | PASS — zero `pynccl`/`nccl` matches; zero `cuda` matches outside the deliberate `--cuda-graph-max-bs 0` notice; HCCL + Gloo handshake logged |
| clean shutdown | PASS — Uvicorn `Application shutdown complete.` + `Finished server process [41238]`; one benign `resource_tracker` semaphore cleanup notice |
| no residual server processes | PASS — final scan reported `NO_RESIDUAL`, port released |

**Overall verdict: PASS.**

## 9. What this gate does NOT establish

* No generation call was issued. `/generate`,
  `/v1/chat/completions` (both `stream=false` and `stream=true`)
  remain unverified against the server path.
* The `resource_tracker` semaphore-leak warning at parent exit
  is documented but not root-caused. A later doc gate may pin it
  to a specific ZMQ or shared-memory handle if needed.
* `/v1/models` returns the filesystem path as the `model` id.
  Whether clients that send a short model name (e.g. `qwen3-0.6b`)
  are auto-resolved is not tested here. Any Phase 6B.6
  `/v1/chat/completions` gate must send the exact id string
  observed in §4.
