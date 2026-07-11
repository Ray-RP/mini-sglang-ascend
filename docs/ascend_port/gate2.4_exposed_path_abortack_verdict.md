# Gate 2.4 Verdict — Exposed-Path AbortAck Audit

**Gate ID:** 2.4 (exposed-path abort/cancel audit for Mini-SGLang-Ascend)
**Verdict:** PARTIAL
**Branch:** `gate2.4-exposed-path-abortack-audit`
**Audit base commit:** `5540116` (tip of `ascend-port`, `v0.1.0a1`
release commit family)
**Date:** 2026-07-11
**Kind:** Read-only audit + minimal hermetic exposed-path proof.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3, does not
mutate release tag `v0.1.0a1`, does not touch the GitHub release, and
does not extend the Ascend port to server crash / restart, client
replay, TP > 1, sampler exception recovery, CPU copy failure recovery,
long soak, chunked-prefill abort, or performance benchmarking.

---

## 1. Verdict summary

* **PASS on E1 (HTTP `POST /generate`, stream).** The AbortAck chain
  proven in Gate 2.3f is reachable end-to-end from a real HTTP
  client. Client disconnect → `stream_with_cancellation` →
  `abort_user` → `AbortMsg` on the wire → tokenizer worker →
  `AbortBackendMsg` → scheduler → `AbortAckMsg` → tokenizer worker →
  `AbortAckReply` → `FrontendManager.listen`. Late-token suppression,
  four-bucket cleanup, and one-shot event fire are all covered.

* **PASS on E2 (HTTP `POST /v1/chat/completions`, `stream=true`).**
  Same wrapper as E1; same conclusion.

* **NOT REACHED on E3 (HTTP `POST /v1/chat/completions`,
  `stream=false`).** No `request.is_disconnected()` polling around
  the non-streaming await. A closed client socket does not raise
  `asyncio.CancelledError` on the await, and no `abort_user` call is
  ever made. The request runs to natural finish. The gap is a code
  wiring change, deliberately not applied at this audit gate.

* **NOT REACHED on E5 (shell).** `shell_completion` builds the SSE
  response without `stream_with_cancellation`, and the
  `BackgroundTask(lambda: _abort)` on the response passes the
  coroutine function (not a coroutine) so nothing is awaited even if
  a cancel path existed. The shell's whole-server teardown
  (`shutdown()` + `psutil.Process.kill`) does terminate requests, but
  it is not the AbortAck contract. Fix requires code changes in
  `api_server.py:shell_completion`; not applied here.

* **NOT REACHED on E6 (`minisgl.llm.LLM`).** The offline driver has
  no `abort()` / `cancel()` method; `offline_receive_msg` cannot
  produce `AbortBackendMsg`. The AbortAck chain is out of the offline
  driver's public surface. This is a scope decision, not a wiring
  bug. Not extended here.

* **n/a on E4** (`GET /v1/models`, read-only metadata).

* **PASS on E7** (hermetic tests). Gate 2.3f test surface still
  green (47 tests) after the audit; two Gate 2.4 tests added and
  green.

Because the AbortAck chain is fully reachable on the streaming HTTP
paths E1/E2 that this release actually documents in the Ascend Quick
Start, but is *not* reachable on the non-streaming HTTP path E3, the
shell E5, or the offline driver E6, the overall verdict is
**PARTIAL** per the criteria supplied at gate open:

> PASS = 当前真实入口支持 abort/cancel，并能证明 AbortAck、
> late-token suppression、资源回收
> **PARTIAL = AbortAck 只在 in-process/hermetic path 成立，
> 真实入口尚未暴露 abort/cancel**
> BLOCKED = 当前仓库没有可审计的真实入口或测试无法启动

The PARTIAL wording in the gate opening spec was written expecting
"only in-process/hermetic path" — the audit found that streaming HTTP
paths are actually wired end-to-end but the non-streaming HTTP, the
shell, and the offline driver are not. PARTIAL is the honest label:
some real entries expose the chain, others do not.

---

## 2. Support matrix (frozen at this verdict)

| Entry | cross-proc | via ZMQ | AbortBackendMsg reachable | AbortAckReply observed | late-token suppression | resource reclaim evidence |
|---|---|---|---|---|---|---|
| E1 HTTP `/generate` (stream) | Y | Y | Y | Y | Y | Y (Gate 2.3f + `_free_req_resources`) |
| E2 HTTP `/v1/chat/completions` stream=true | Y | Y | Y | Y | Y | Y |
| E3 HTTP `/v1/chat/completions` stream=false | Y | Y | **N** | n/a | n/a | request runs to natural finish |
| E4 HTTP `/v1/models` | Y | n/a | n/a | n/a | n/a | n/a |
| E5 Shell | Y | Y | **N** | n/a | n/a | whole-server shutdown only |
| E6 Offline `LLM` | N | N | **N** | n/a | n/a | natural finish only |
| E7 Hermetic pytest | N | N | Y | Y | Y | Y |

Cells marked **N** are gaps documented in the audit doc
(`gate2.4_exposed_path_abortack_audit.md` §6). Nothing on this matrix
is a regression of Gate 2.3.

---

## 3. Evidence

### 3.1 Read-only audit

* `docs/ascend_port/gate2.4_exposed_path_abortack_audit.md` — full
  entry inventory, per-entry request-lifecycle traces, support
  matrix, minimal gap analysis, and description of the two hermetic
  tests below.

### 3.2 Frontend-side over-the-wire AbortAck proof

Committed at `tests/misc/test_exposed_path_abort_ack.py`:

* `test_exposed_path_abort_ack_single_request` —
  `FrontendManager.__new__` with a real `ZmqAsyncPushQueue` and a
  real `ZmqAsyncPullQueue` on loopback `ipc://` addresses, a stub
  tokenizer counter-party in the same process, and the following
  assertions:
    * `new_user()` populates `ack_map` + `event_map`, does not touch
      `abort_pending`.
    * `send_one(TokenizeMsg)` produces an on-wire msgpack-encoded
      `TokenizeMsg` decoded by the counter-party.
    * `abort_user(uid)` moves the uid into `abort_pending` BEFORE
      the wire send.
    * `AbortMsg(uid)` arrives on the counter-party's `PULL` socket.
    * A `UserReply(uid)` pushed back after abort-pending is set is
      dropped by `listen` (no growth in `ack_map[uid]`).
    * An `AbortAckReply(uid)` on the recv wire unblocks
      `wait_for_abort_ack(uid)` and cleans all four buckets.

* `test_exposed_path_abort_duplicate_idempotent` — same rig:
    * Two consecutive `abort_user(uid)` calls emit two `AbortMsg` on
      the wire (frontend must not silently coalesce; the scheduler
      side dedupes at free/ack, per Gate 2.3f scenario E).
    * A second `AbortAckReply(uid)` on the wire is a strict no-op
      inside `listen` — the listener task is still running and no
      exception has surfaced.

Both tests run against real `pyzmq` sockets, real `msgpack` framing,
and the checked-in `FrontendManager` code path — no mocks below the
optional-dependency stubs for `prompt_toolkit` / `fastapi` /
`starlette` / `pydantic` (identical to Gate 2.3f scenario F).

### 3.3 No-regression evidence

Prior Gate 2.3 hermetic surface re-run on the same commit + gate 2.4
files:

```
pytest -q -o addopts="" \
    tests/misc/test_scheduler_abort_ack.py \
    tests/misc/test_scheduler_overlap_abort_fence.py \
    tests/misc/test_scheduler_prepare_batch_txn.py \
    tests/misc/test_engine_forward_sampler_atomic.py \
    tests/misc/test_scheduler_shutdown_drain.py \
    tests/misc/test_pyproject_config.py

47 passed
```

### 3.4 Gate 2.4 exposed-path proof result

```
pytest -q -o addopts="" tests/misc/test_exposed_path_abort_ack.py

2 passed
```

Environment: containerised Linux aarch64, Python 3.11.14, `pyzmq
27.1.0`, `msgpack 1.1.2`, `torch 2.9.0+cpu` (CPU torch is fine — no
NPU or model is touched by these tests). Same container image the
Gate 2.3f suite runs in.

---

## 4. Freeze boundary

This gate freezes exposed-path AbortAck audit evidence.

It does not claim production server restart support.
It does not claim TP>1 support.
It does not claim in-process forward/sampler exception recovery.
It does not add new Ascend model execution evidence beyond prior gates.

---

## 5. What this verdict does not claim

* It does not claim that E3 (non-streaming chat completions), E5
  (shell), or E6 (offline `LLM`) have working abort/cancel. They do
  not; §2 of this verdict says so explicitly. The audit documents
  each gap and the minimal follow-up code change; none of those code
  changes are applied here.
* It does not claim that HTTP request replay works across a server
  restart.
* It does not claim any TP > 1 behaviour.
* It does not claim any sampler / forward exception recovery.
* It does not claim any long-soak stability behaviour.
* It does not claim that the HTTP path's *end-to-end* subprocess-based
  wiring (uvicorn + real tokenizer worker + NPU scheduler) has been
  frozen at this commit. `v0.1.0a1` explicitly declines that freeze,
  and Gate 2.4 does not change it. What Gate 2.4 adds is a
  frontend-side proof over real ZMQ transports; the tokenizer worker
  and scheduler sides are already proven by Gate 2.3f scenarios H and
  A–E.
* It does not modify or supersede any earlier gate verdict.

---

## 6. Minimal follow-ups (not part of this gate)

Ordered by "smallest closable next", carried over from the audit
doc §6:

1. **E3 non-streaming cancel wiring** — wrap the `wait_for_ack`
   collector in a `is_disconnected()` polling loop, or share the
   `stream_with_cancellation` guard with a buffered collector. No
   scheduler-side change required (Gate 2.3f already handles what
   this would push).
2. **E5 shell cancel wiring** — (a) route `shell_completion`'s
   generator through `stream_with_cancellation`, (b) fix the
   `BackgroundTask` argument to call the coroutine (`lambda: _abort()`
   instead of `lambda: _abort`).
3. **E6 offline `LLM.abort(uid)`** — inject `AbortBackendMsg` into
   the next `offline_receive_msg` batch. Requires a public method on
   `LLM` and a small addition to `offline_receive_msg` — nothing
   else in the scheduler needs to change (the code path taken is
   already Gate 2.3f scenario A / B).

Each of the above is a separate gate; none of them is opened here.

---

## 7. References

* Audit — `docs/ascend_port/gate2.4_exposed_path_abortack_audit.md`
* Gate 2.3 verdict — `docs/ascend_port/gate2_3_request_lifecycle_verdict.md`
* Release notes — `docs/ascend_port/release_notes_0.1.0a1.md`
* New tests — `tests/misc/test_exposed_path_abort_ack.py`
* Frontend state machine — `python/minisgl/server/api_server.py`
* Tokenizer worker dispatch — `python/minisgl/tokenizer/server.py`
* Scheduler abort handling — `python/minisgl/scheduler/scheduler.py`
* Offline driver — `python/minisgl/llm/llm.py`
* ZMQ transport — `python/minisgl/utils/mp.py`
