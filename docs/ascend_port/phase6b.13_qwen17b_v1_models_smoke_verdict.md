# Phase 6B.13 Qwen3-1.7B Server `/v1/models` Smoke Verdict

**Kind:** Documentation-only verdict for the Qwen3-1.7B fixed-TP2
server `/v1/models` smoke against the Ascend host. Launched with
the Phase 6B.7 recipe
([`phase6b.2_server_launch_recipe.md`](./phase6b.2_server_launch_recipe.md)
as amended at Phase 6B.7 to include `--page-size 16`) but with
`--model-path /mnt/nvme/models/Qwen3-1.7B`, issued one `curl` to
`GET /v1/models`, observed the outcome, then terminated. This file
introduces no runtime, script, test, tag, GitHub Release, or
`CHANGELOG.md` change and does not print credentials.

Envelope: fixed TP=2, eager, `npu_fia`, bf16, `page_size=16` —
v0.2.0a1 recipe, second model.

**Overall verdict: PASS.**

The `/v1/models` endpoint returned an OpenAI-compatible
`ModelList` envelope whose sole entry advertises the launched
model's absolute filesystem path as `id` and `root`, matches the
Phase 6B.5 shape observed for Qwen3-0.6B, and includes an
`owned_by` value of `mini-sglang`. Both scheduler ranks remained
alive after the request; the Uvicorn parent shut down cleanly on
SIGTERM.

---

## 1. Environment

* Host: Ascend NPU host (8 × 910B1).
* Container: `998ce5ba6e5e`.
* Repo tree: `/mnt/nvme/LR-606/mini-sglang-ascend` (unchanged since
  Phase 6B.3).
* Model: `/mnt/nvme/models/Qwen3-1.7B`.
* Working dir: `/mnt/nvme/LR-606/phase6b13/` (launch script + log
  + response headers/body).
* Port: `1919`.

## 2. Launch invocation (as executed)

```
python -m minisgl.server.launch \
  --model-path /mnt/nvme/models/Qwen3-1.7B \
  --tp-size 2 \
  --attention-backend npu_fia \
  --disable-pynccl \
  --cuda-graph-max-bs 0 \
  --page-size 16 \
  --host 0.0.0.0 \
  --port 1919
```

Same wrapper skeleton used at Phase 6B.12 (`setsid nohup` inside
the container; `PYTHONPATH=python`), only the phase-specific
working directory changed to `/mnt/nvme/LR-606/phase6b13/`.

## 3. Server ready state

* Parent PID `44088`; three `multiprocessing.spawn` children
  (`44098`, `44099`, `44100` = 2 scheduler ranks + 1 detokenizer)
  + resource-tracker `44097`.
* Log excerpt:
  ```
  [Gloo] Rank 1 is connected to 1 peer ranks.
  [Gloo] Rank 0 is connected to 1 peer ranks.
  [core|rank=0] Free memory before loading model: 47.86 GiB
  [core|rank=0] Allocating 772720 tokens for KV cache, K + V = 41.27 GiB
  [core|rank=1] Allocating 772720 tokens for KV cache, K + V = 41.27 GiB
  [core|rank=0] Free memory after initialization: 4.74 GiB
  [core|rank=0] CUDA graph is disabled.
  [core|rank=0] Scheduler is idle, waiting for new reqs...
  Scheduler is ready
  API server is ready to serve on 0.0.0.0:1919
  INFO:     Uvicorn running on http://0.0.0.0:1919 (Press CTRL+C to quit)
  ```
* `netstat`: `tcp 0.0.0.0:1919 LISTEN 44088/python`.
* Consistent with Phase 6B.12: KV cache 41.27 GiB per rank,
  ready state ~20 s after launch (parse at 19:33:10; ready at
  19:33:18).

## 4. `/v1/models` request

Command executed inside the container:

```
curl -sS -o models.body -D models.headers \
  -w "HTTP_STATUS=%{http_code}\nSIZE=%{size_download}\nTIME=%{time_total}\nCT=%{content_type}\n" \
  http://127.0.0.1:1919/v1/models
```

No request body; simple `GET`. No sampling or forward pass
involved — `/v1/models` is served by the FastAPI frontend from
the model registry without reaching the scheduler.

## 5. Response — what the client got

Curl instrumentation:

| metric | value |
|---|---|
| HTTP status | `200` |
| content-type | `application/json` |
| content-length | `163` |
| response bytes | `163` |
| wall-clock | `~0.0028 s` |

Response headers verbatim:

```
HTTP/1.1 200 OK
date: Sun, 12 Jul 2026 19:35:33 GMT
server: uvicorn
content-length: 163
content-type: application/json
```

Response body verbatim (JSON, pretty-printed here for readability;
wire form is a single line):

```json
{
  "object": "list",
  "data": [
    {
      "id": "/mnt/nvme/models/Qwen3-1.7B",
      "object": "model",
      "created": 1783884933,
      "owned_by": "mini-sglang",
      "root": "/mnt/nvme/models/Qwen3-1.7B"
    }
  ]
}
```

Server access log:

```
INFO:     127.0.0.1:54706 - "GET /v1/models HTTP/1.1" 200 OK
```

### Per-field verdict

| field | observed | verdict |
|---|---|---|
| top-level `object` | `"list"` | PASS — OpenAI ModelList wrapper |
| `data` | list of length 1 | PASS |
| `data[0].id` | `"/mnt/nvme/models/Qwen3-1.7B"` | PASS — absolute path of the launched model |
| `data[0].object` | `"model"` | PASS |
| `data[0].created` | `1783884933` (Unix seconds) | PASS — populated timestamp |
| `data[0].owned_by` | `"mini-sglang"` | PASS — matches Phase 6B.5 Qwen3-0.6B observation |
| `data[0].root` | `"/mnt/nvme/models/Qwen3-1.7B"` | PASS — equals `id` |

### Cross-check against Phase 6B.5 (Qwen3-0.6B)

The envelope shape is identical to Phase 6B.5, which reported the
same fields for `/mnt/nvme/models/Qwen3-0.6B`. Only the two path
strings (`id`, `root`) and the `created` timestamp change with the
launched model — confirming the `/v1/models` handler simply
reflects the value of `--model-path` back to the client and does
not maintain a short-alias registry. Any client that targets this
server must send this exact absolute-path string in downstream
`model` fields (`/v1/chat/completions`, `/v1/completions`).

## 6. `block_size` / CANN error status

* `grep -inE "block_size|561002|CheckFeatureNoquant"` on the log:
  **zero matches**.
* `--page-size 16` from the Phase 6B.7 recipe held.

## 7. No CUDA / NCCL / pynccl fallback

Grep over the full server log:

* `pynccl` / `nccl` external matches (excluding the intentional
  `use_pynccl=False` argument-echo line): **zero**.
* `cuda` external matches (excluding `cuda_graph_bs=`,
  `cuda_graph_max_bs=`, and the deliberate
  `CUDA graph is disabled.` notice): **zero**.
* HCCL + Gloo evidence present: 2× `ProcessGroupHCCL` watchdog
  warnings + `[Gloo] Rank {0,1} is connected to 1 peer ranks.`.
* `flashinfer` matches: **zero** — expected, since `/v1/models`
  never invokes the sampler.

## 8. Scheduler rank health after the request

* All three `spawn_main` children (`44098`, `44099`, `44100`) and
  the resource-tracker `44097` remained alive after the `200`
  response.
* No `Traceback` / `Error` / `RuntimeError` matches in the server
  log.
* Rank 0 stayed in its pre-request idle state
  (`Scheduler is idle, waiting for new reqs...`) throughout —
  consistent with `/v1/models` being a frontend-only route.

## 9. Termination and residual check

* `kill -TERM 44088` (parent). Uvicorn shutdown sequence
  completed:
  ```
  INFO:     Shutting down
  INFO:     Waiting for application shutdown.
  INFO:     Application shutdown complete.
  INFO:     Finished server process [44088]
  ```
* One benign
  `multiprocessing.resource_tracker: There appear to be 2 leaked
  semaphore objects to clean up at shutdown` warning at parent
  exit — same benign notice recorded across Phases 6B.5, 6B.7,
  6B.9, 6B.10, 6B.12.
* Follow-up SIGTERM sweep across the surviving spawn/resource
  children required no `SIGKILL` fallback.
* Final residual scan
  (`ps -eo pid,cmd | grep -E "minisgl.server|multiprocessing.(spawn|resource)"`):
  `NO_RESIDUAL`.
* Listening port `:1919` released; `netstat`: no match.

## 10. Verdict — per required checklist

| Check | Result |
|---|---|
| HTTP status | PASS — `200`, `content-type: application/json`, `content-length: 163` |
| response body shape | PASS — `{"object":"list","data":[{...}]}` OpenAI ModelList envelope |
| model id / name observed | PASS — `id = root = "/mnt/nvme/models/Qwen3-1.7B"`, `owned_by = "mini-sglang"`, `object = "model"` |
| rank health after request | PASS — parent `44088` + 3 spawn children (`44098`, `44099`, `44100`) + resource-tracker (`44097`) still alive; zero `Traceback`/`Error`/`RuntimeError` in log |
| no `block_size` / `561002` error | PASS — zero matches; `--page-size 16` held |
| no CUDA / NCCL / pynccl fallback | PASS — zero `pynccl`/`nccl`/`cuda`/`flashinfer` non-argument matches; HCCL + Gloo handshakes present |
| clean shutdown | PASS — Uvicorn `Application shutdown complete.` + `Finished server process [44088]`; one benign `resource_tracker` semaphore cleanup notice |
| no residual server processes | PASS — final scan `NO_RESIDUAL`, port released |

**Overall verdict: PASS.**

## 11. What this gate does NOT establish

* No `/generate`, `/v1/completions`, or `/v1/chat/completions`
  request was issued against Qwen3-1.7B — those are follow-up
  gates (Phases 6B.14+).
* No inference-path validation: `/v1/models` returns registry
  metadata without touching the sampler, tokenizer, or forward
  loop, so this gate does not exercise HCCL comm traffic beyond
  the initial handshake, does not test the greedy-only sampling
  contract (Phase 6B.9 §1) against the new model, and does not
  probe the `--page-size 16` requirement against the new model's
  attention path.
* No short-alias resolution is claimed. The launched model is
  advertised only by its absolute filesystem path; whether short
  aliases (e.g. `qwen3-1.7b`) would resolve remains untested and
  is out of scope for the v0.2.0a1 envelope.
* No `usage` accounting, streaming envelope, or `finish_reason`
  semantics were exercised.
* The benign `resource_tracker` semaphore-leak warning at parent
  exit remains a documented observation, not a root-caused
  finding.
* No performance claim of any kind is made. Handler latency
  (~2.8 ms wall-clock) is anecdotal, not benchmarked.
