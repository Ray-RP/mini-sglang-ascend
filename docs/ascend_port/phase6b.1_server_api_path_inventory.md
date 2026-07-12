# Phase 6B.1 Server / OpenAI API Path Inventory

**Kind:** Documentation-only audit of the HTTP server and OpenAI-compatible
API path in `python/minisgl/`. This file introduces no runtime, script, test,
tag, GitHub Release, or `CHANGELOG.md` change. It does not launch the server,
call `torchrun`, curl any endpoint, run experiments, or print credentials.
Its sole purpose is to inventory the server code path in preparation for
extending Ascend fixed-TP2 coverage from the offline driver into the HTTP +
ZMQ server path.

Scope: fixed TP=2, eager, `npu_fia`, bf16, greedy — same envelope as the
v0.2.0a1 technical preview.

---

## 1. Server entrypoints

Two operational entrypoints reach the same `launch_server()` function:

* `python -m minisgl` — `python/minisgl/__main__.py` imports
  `launch_server` from `python/minisgl/server/__init__.py` and calls it.
* `python -m minisgl.server.launch` — `python/minisgl/server/launch.py`
  exposes `launch_server()` directly.

`python/minisgl/entrypoints/` **does not exist**. There is no separate
`entrypoints` package; all HTTP/CLI entry lives under
`python/minisgl/server/`.

`launch_server()` in `python/minisgl/server/launch.py`:

1. Parses CLI via `parse_args()` → `ServerArgs`.
2. Spawns child processes with `mp.Process`, **not** `torchrun`:
   * `world_size` scheduler ranks (target `_run_scheduler`, per-rank
     `DistributedInfo(i, world_size)`).
   * `1` detokenizer process.
   * `num_tokenizer` tokenizer processes.
3. Waits on an ack queue for `num_tokenizer + 2` ready acks.
4. Calls `run_api_server(...)` in `python/minisgl/server/api_server.py`,
   which either:
   * `uvicorn.run(app, host, port)` — HTTP server mode, or
   * `asyncio.run(shell())` — interactive `--shell-mode`.

## 2. `/generate` status

Route: `POST /generate` in `python/minisgl/server/api_server.py:341`.

* Request model: `GenerateRequest{prompt: str, max_tokens: int,
  ignore_eos: bool}`.
* Response: streaming SSE only (`StreamingResponse(media_type=
  "text/event-stream")`) driven by `stream_generate(...)`.
* Cancellation: routed through `FrontendManager` abort-pending state
  machine (Gate 2.3f) via `stream_with_cancellation`.
* Verdict: **present and wired**. No non-streaming variant of `/generate`.

## 3. `/v1/chat/completions` status

Route: `POST /v1/chat/completions` in
`python/minisgl/server/api_server.py:368`.

* Request model: `OpenAICompletionRequest` supports both `messages` (chat)
  and `prompt` (completions-style), plus `model`, `max_tokens`,
  `temperature`, `top_k`, `top_p`, `n`, `stream`, `stop`,
  `presence_penalty`, `frequency_penalty`, `ignore_eos`.
* Response: both streaming (`stream=true` → SSE) and non-streaming
  (JSON) supported.
* Adjacent OpenAI-shaped routes:
  * `GET/POST /v1` at `api_server.py:363` — health probe.
  * `GET /v1/models` at `api_server.py:426` — returns `ModelList`.
* Verdict: **present with both streaming and non-streaming**. Chat and
  raw-prompt inputs both accepted.

## 4. Streaming status

* Both `/generate` and `/v1/chat/completions` use FastAPI
  `StreamingResponse(media_type="text/event-stream")`.
* Underlying transport between frontend and scheduler is ZMQ IPC; the
  FastAPI handler pulls tokens from the frontend and forwards them as
  SSE frames.
* Cancellation: `FrontendManager.stream_with_cancellation` implements
  the Gate 2.3f abort-pending → abort-ack contract so a client
  disconnect propagates an abort into the scheduler without leaking
  the request slot.
* Verdict: **streaming supported end-to-end** for both `/generate`
  and `/v1/chat/completions`.

## 5. Ascend fixed-TP2 readiness — server path

The offline driver `python/minisgl/llm/llm.py` (`LLM(Scheduler)`) is
the path that the v0.2.0a1 six-case functional matrix was proven
against. The server path (`server/launch.py` → `mp.Process` → per-rank
scheduler → `Engine._init_communication`) has **not** been exercised
under fixed TP=2 on Ascend in any recorded gate.

Readiness risks identified by inventory (not by execution):

1. **`use_pynccl=True` is the default.** `EngineConfig.use_pynccl`
   defaults to `True` (`python/minisgl/engine/config.py`). At TP≥2,
   `Engine._init_communication`
   (`python/minisgl/engine/engine.py:148-191`) then calls
   `enable_pynccl_distributed`, which loads a CUDA-only pynccl
   module. On NPU this path fails.
   * **Mitigation:** must launch the server with `--disable-pynccl`
     (defined in `python/minisgl/server/args.py`, `dest=use_pynccl`,
     `action="store_false"`) so the code falls through to
     `get_distributed_backend(device_type)`, which maps `npu→hccl`
     and adds a gloo sidecar via
     `torch.distributed.new_group(backend="gloo")`.
2. **Rendezvous does not come from `MINISGL_DISTRIBUTED_ADDR`.**
   `EngineConfig.distributed_addr` honours the env var, but
   `ServerArgs` overrides it to
   `tcp://127.0.0.1:{server_port+1}`. So the offline-driver env-var
   recipe used at Phase 6A does not apply to the server path;
   the server picks its own TCP rendezvous relative to
   `--port`.
3. **Server uses `mp.Process`, not `torchrun`.** Per-rank environment
   (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT`)
   is not populated by `torchrun`; whatever `mp.Process` sets is
   what each scheduler rank sees. Ascend-specific per-rank device
   pinning (`ASCEND_RT_VISIBLE_DEVICES` per child, or explicit
   `torch.npu.set_device(local_rank)` inside `_run_scheduler`) is
   not obviously configured by `launch.py`. This must be verified
   in the follow-up gate — two ranks sharing NPU 0 will crash HCCL
   init.
4. **`cuda_graph_max_bs` default.** `ServerArgs` inherits the
   default from `SchedulerConfig` / `EngineConfig`. The v0.2.0a1
   envelope is eager (`cuda_graph_bs=[]`); on NPU the server must
   be launched with `--cuda-graph-max-bs 0` (or the equivalent
   flag that empties the graph list) to stay inside the proven
   envelope.
5. **`--attention-backend` default.** `ServerArgs` accepts
   `--attention-backend/--attn` but the CLI default is not
   `npu_fia`. To stay inside the v0.2.0a1 envelope, the server
   must be launched with `--attention-backend npu_fia`.
6. **Health / readiness ack semantics under NPU init.** `launch.py`
   waits on `ack_queue` for `num_tokenizer + 2` acks before
   opening the HTTP port. If HCCL init hangs on either rank, the
   server never binds — no HTTP-level probe is possible during
   bring-up. Bring-up must be attempted with the ack queue as the
   only readiness signal.
7. **`_v1` health probe.** `GET /v1` is a shallow probe over
   FastAPI; it does not attempt a scheduler round-trip. It cannot
   substitute for a real `/generate` smoke request during
   Ascend bring-up.

## 6. Required Ascend fixed-TP2 CLI recipe (proposed, not executed)

Based purely on the inventory above, the minimal Ascend fixed-TP2
server invocation to stay inside the v0.2.0a1 envelope would be:

```bash
python -m minisgl \
  --model-path /mnt/nvme/models/Qwen3-0.6B \
  --tp-size 2 \
  --attention-backend npu_fia \
  --disable-pynccl \
  --dtype bfloat16 \
  --memory-ratio 0.85 \
  --page-size 16 \
  --max-running-requests 4 \
  --cuda-graph-max-bs 0 \
  --host 127.0.0.1 \
  --port 1919
```

This recipe is **proposed only** — it has not been executed by this
gate. The follow-up gate is responsible for locking, tightening, or
correcting it against observed behaviour on the Ascend host.

## 7. Residual CUDA / NCCL / GPU defaults

Grepping `python/minisgl/server/args.py` and adjacent config:

* `EngineConfig.use_pynccl` default `True` (CUDA-only path if not
  overridden with `--disable-pynccl`).
* `cuda_graph_max_bs` CLI flag exists; default is non-empty for the
  CUDA envelope.
* No per-rank NPU device pinning is visible in `launch.py` /
  `_run_scheduler` from the inventory reads.
* `_init_communication` in `engine/engine.py` still assumes the
  CUDA branch by default; NPU is reachable only via `use_pynccl=False`.

These are not bugs relative to the offline-driver envelope, but they
are the specific defaults that must be overridden or investigated
before the server path can be declared Ascend fixed-TP2 ready.

## 8. Recommended next gate

**Phase 6B.2 — Ascend fixed-TP2 server bring-up dry-run.**

A follow-up gate should attempt exactly one thing on the Ascend host:
launch `python -m minisgl` with the CLI proposed in §6 against
`/mnt/nvme/models/Qwen3-0.6B`, observe whether both scheduler ranks
complete HCCL init (via the `ack_queue` signal and per-rank logs),
and, if the HTTP port binds, serve a single `/generate` request with
`max_tokens=8` under greedy. The gate should record either the
successful transcript or the first failure mode (pynccl load, HCCL
init hang, device-visibility collision, missing rendezvous env, etc.)
and stop.

A preceding documentation-only gate may be advisable to lock the
exact CLI + env recipe, per-rank device-pinning strategy, and
readiness contract before executing the bring-up.

---

## Appendix A — files consulted

* `python/minisgl/__main__.py`
* `python/minisgl/server/__init__.py`
* `python/minisgl/server/launch.py`
* `python/minisgl/server/api_server.py`
* `python/minisgl/server/args.py`
* `python/minisgl/engine/config.py`
* `python/minisgl/engine/engine.py` (lines 148-191, `_init_communication`)
* `python/minisgl/scheduler/config.py`
* `python/minisgl/llm/llm.py`
* `README.md`
* `docs/`
