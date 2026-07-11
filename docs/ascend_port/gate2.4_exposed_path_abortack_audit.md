# Gate 2.4 — Exposed-Path AbortAck Audit

**Gate ID:** 2.4 (exposed-path abort/cancel audit for Mini-SGLang-Ascend)
**Branch:** `gate2.4-exposed-path-abortack-audit`
**Base commit:** `5540116` (tip of `ascend-port`, `v0.1.0a1` release commit
family)
**Date:** 2026-07-11
**Kind:** Read-only audit + minimal hermetic exposed-path proof for
FrontendManager over real ZMQ IPC. Not a new scheduler gate; strictly a
mapping of what Gate 2.3f actually reaches through the code paths a real
user of this repository can invoke.

This document does not modify Gate 1 / 2.1 / 2.2 / 2.3 verdicts. It does
not extend release `v0.1.0a1`. It records where the AbortAck chain
proven in Gate 2.3f is exposed to real callers, where it is not, and
what the minimal next step is.

---

## 1. Scope

Gate 2.3f (part of Gate 2.3) proved that:

* `Scheduler._process_one_msg` handles `AbortBackendMsg` in three code
  branches (waiting-only, running non-inflight, overlap-deferred) and
  emits exactly one `AbortAckMsg` per abort msg regardless of branch.
* The tokenizer worker translates `AbortAckMsg` (backend-facing) into
  `AbortAckReply` (frontend-facing) verbatim.
* `FrontendManager.listen` dispatches `AbortAckReply` independently of
  `UserReply`, gates late `UserReply` messages on `abort_pending`
  membership, and cleans four bookkeeping buckets (`ack_map`,
  `event_map`, `abort_pending`, `abort_pending_events`) idempotently.

Those proofs used the in-process offline driver (`LLM`), the hermetic
scheduler skeleton (`Scheduler.__new__`), and stubbed `send_result`
spies. They did **not** exercise the transport wires as a real caller
would drive them.

Gate 2.4 answers exactly one question:

> Given the shape this repository exposes today (HTTP server, shell,
> offline LLM, tests) — are those AbortAck guarantees actually
> reachable from a real request path, or do they stop at the seams
> proven in isolation by Gate 2.3f?

Explicitly out of scope (per user directive):

* Server crash / restart lifecycle.
* Client reconnect and request replay.
* TP > 1 lifecycle audit.
* Sampler / forward exception recovery.
* CPU detokenize copy or event-record failure recovery.
* Long soak.
* Chunked-prefill abort audit.
* Performance benchmarks.
* Any package rename or release-tag mutation.

---

## 2. Real entry points at `5540116`

Enumerated by reading `pyproject.toml`, `python/minisgl/__main__.py`,
`python/minisgl/shell.py`, `python/minisgl/server/`, `python/minisgl/llm/`
and cross-referencing every `if __name__ == "__main__"` block plus every
usage of `from minisgl.llm import LLM` in the repo:

| # | Entry | How invoked | Where |
|---|-------|-------------|-------|
| E1 | HTTP `POST /generate` (SSE stream) | `python -m minisgl --model ... --host 0.0.0.0 --port 1919` then `curl -N http://.../generate` | `python/minisgl/server/api_server.py:342` |
| E2 | HTTP `POST /v1/chat/completions`, `stream=true` | same server process, OpenAI-style client with `"stream": true` | `python/minisgl/server/api_server.py:369` |
| E3 | HTTP `POST /v1/chat/completions`, `stream=false` | same server process, OpenAI-style client with `"stream": false` (or omitted) | `python/minisgl/server/api_server.py:369` |
| E4 | HTTP `GET /v1/models` | trivial metadata endpoint | `python/minisgl/server/api_server.py:427` |
| E5 | Interactive shell | `python -m minisgl.shell --model ... --shell` (or `--shell-mode`) | `python/minisgl/shell.py`, `python/minisgl/server/api_server.py:465` |
| E6 | Offline in-process driver `minisgl.llm.LLM` | `LLM(model_path=...).generate(prompts, sampling_params)` from Python | `python/minisgl/llm/llm.py:28` |
| E7 | Hermetic pytest suite | `pytest tests/misc/test_scheduler_abort_ack.py …` | `tests/misc/*.py` |

There is **no** additional CLI, gRPC entry, native SDK entry, or
public HTTP endpoint. `pyproject.toml` declares no `[project.scripts]`
console entry point; the only user-facing script is `python -m minisgl`,
which resolves to `python/minisgl/__main__.py` and calls
`server.launch_server()`.

The upstream Mini-SGLang README preserved in `README.md` describes some
CUDA-target commands (e.g. `--tp 4`, `--cache naive` benchmarking) which
are not attested by the Ascend port and are not counted as Ascend real
entries; the Ascend Quick Start (`README.md:73–150`) lists exactly E1,
E2, E3, E4, and E5 above, plus the pytest command surface for E7.

---

## 3. Request lifecycle by entry

Notation: `→` denotes an in-process call, `⇒ ZMQ` denotes a
`msgpack`-encoded message across a `zmq.PUSH/PULL` IPC socket.

### 3.1 HTTP streaming (E1, E2)

```
client
  POST /generate | POST /v1/chat/completions {stream:true}
  → FrontendManager.new_user() → uid
  → FrontendManager.send_one(TokenizeMsg(uid, text, sp))
     ⇒ ZMQ frontend → tokenizer
  → StreamingResponse(stream_with_cancellation(
        stream_generate(uid) | stream_chat_completions(uid),
        request, uid))

tokenizer worker
  ⇒ ZMQ tokenizer → backend  UserMsg(uid, input_ids, sp)

scheduler
  _process_one_msg(UserMsg) → prefill_manager.add_one_req
  overlap_loop / normal_loop → forward, sample, _process_last_data
     → send_result([DetokenizeMsg(...)])
     ⇒ ZMQ backend → tokenizer

tokenizer worker
  detokenize_manager.detokenize → UserReply(uid, text, finished)
     ⇒ ZMQ tokenizer → frontend

FrontendManager.listen
  dispatch on isinstance:
    UserReply(uid): append to ack_map[uid], set event_map[uid]
  wait_for_ack(uid) yields each ack to stream_generate

  (client TCP disconnects — stream_with_cancellation raises)
  → asyncio.create_task(abort_user(uid))
     → abort_pending.add(uid)
     → send_one(AbortMsg(uid))
        ⇒ ZMQ frontend → tokenizer

tokenizer worker
  partition pending_msg on isinstance:
    AbortMsg(uid) → AbortBackendMsg(uid)
     ⇒ ZMQ tokenizer → backend

scheduler
  _process_one_msg(AbortBackendMsg):
    if uid in inflight_uids: deferred_abort_uids.add(uid)
       (freed by _apply_deferred_aborts after _process_last_data)
    else: _free_req_resources; _pending_abort_acks.append(uid)
  _apply_deferred_aborts / _flush_pending_acks
     → send_result([AbortAckMsg(uid)])
     ⇒ ZMQ backend → tokenizer

tokenizer worker
  partition on isinstance AbortAckMsg → AbortAckReply(uid)
     ⇒ ZMQ tokenizer → frontend

FrontendManager.listen
  isinstance(msg, AbortAckReply):
    abort_pending.discard(uid); ack_map.pop; event_map.pop; …
    abort_pending_events[uid].set()

stream_with_cancellation completes; client sees TCP FIN (already gone).
```

Every arrow is present in the checked-in code path. The cancel trigger
(`request.is_disconnected()` + `asyncio.CancelledError` re-raise + the
`asyncio.create_task(self.abort_user(uid))` call) lives in
`api_server.py:265–275`.

### 3.2 HTTP non-streaming chat completions (E3)

```
client POST /v1/chat/completions {stream: false}
  → FrontendManager.new_user() → uid
  → send_one(TokenizeMsg)
  → for ack in state.wait_for_ack(uid): full_content += ...
  → return JSON when ack.finished
```

There is **no** `is_disconnected()` polling around the non-streaming
loop, **no** call to `stream_with_cancellation`, **no** call to
`abort_user`. A client that closes the TCP connection mid-generation is
not observed by the frontend; the scheduler runs the request to natural
finish, `wait_for_ack` collects tokens into `full_content`, and the
JSON response is discarded when uvicorn eventually notices the dead
socket.

Conclusion for E3: no path from the exposed request to
`AbortBackendMsg`. AbortAck chain is unreachable from this entry.

### 3.3 Shell (E5)

`shell()` in `api_server.py:465` calls `shell_completion(req)` which
builds a `StreamingResponse` **without** wrapping the generator in
`stream_with_cancellation`, and attaches an abort as
`BackgroundTask(lambda: _abort)`.

Two issues in the wiring:

* `StreamingResponse(state.stream_generate(uid), …)` — no
  `stream_with_cancellation`, so a shell Ctrl-C / EOF never routes to
  `abort_user`. The shell's own `finally` block instead performs
  `get_global_state().shutdown()` and kills subprocesses; that unwinds
  the whole server, not the individual request.
* `background=BackgroundTask(lambda: _abort)` — the lambda returns the
  coroutine **function** `_abort` (not a coroutine); Starlette will
  invoke the lambda, receive a callable back, and not await it. The
  abort is never actually issued even in the (non-existent) cancel
  path.

Conclusion for E5: the AbortAck chain is unreachable from the shell.
The wiring is present in code but has never been exercised end-to-end;
it is dormant. The only cancel semantics the shell today provides is
whole-server tear-down via `shutdown()` + `psutil.Process.kill`.

### 3.4 Offline LLM driver (E6)

`LLM(Scheduler)` runs the scheduler in `offline_mode=True`:

* Message intake is `offline_receive_msg` — reads from
  `self.pending_requests`, emits `UserMsg`. It has **no** branch for
  producing `AbortBackendMsg`.
* Message egress is `offline_send_result` — writes to
  `status_map[uid].output_ids`. It has **no** handling for
  `AbortAckMsg` (would raise `KeyError` on `status_map[msg.uid]` if
  one arrived, since no ack path can produce one anyway).
* Public API is `generate(prompts, sampling_params)` — no `abort()`
  method, no `cancel()` method, no way to inject an `AbortBackendMsg`
  from the caller.

Conclusion for E6: the AbortAck chain is unreachable — not because of
a wiring bug but because no abort API is exposed. Requests either
complete or `run_forever` unwinds via `RequestAllFinished`.

### 3.5 Metadata endpoint (E4)

`/v1/models` returns a `ModelList` synchronously. No request lifecycle
of the kind Gate 2.3 covers; abort/cancel is not applicable.

### 3.6 Hermetic tests (E7)

`tests/misc/test_scheduler_abort_ack.py` scenarios A–H already exercise
every branch of the scheduler and the tokenizer/frontend dispatch with
`Scheduler.__new__`, `FrontendManager.__new__`, and stubbed
`send_result`. This is the Gate 2.3f evidence surface; not a real
user-facing entry.

---

## 4. Abort/cancel support matrix

For each entry, the audited answer is either checked-in code that would
run today (Y) or a specific gap (N + reason).

| Entry | cross-proc | goes via ZMQ | can trigger AbortBackendMsg | receives AbortAckMsg (as AbortAckReply) | late-token suppression | resource reclaim evidence |
|---|---|---|---|---|---|---|
| E1 HTTP `/generate` (stream) | Y | Y | **Y** (`stream_with_cancellation` → `abort_user`) | **Y** (`FrontendManager.listen` isinstance-dispatch) | **Y** (`abort_pending` set gates `UserReply`) | **Y** (`_free_req_resources` in scheduler + Gate 2.3d drain path) |
| E2 HTTP `/v1/chat/completions` stream=true | Y | Y | **Y** (same wrapper) | **Y** | **Y** | **Y** |
| E3 HTTP `/v1/chat/completions` stream=false | Y | Y | **N** — no `is_disconnected()` polling, `wait_for_ack` never sees a cancel | n/a | n/a | request runs to natural finish |
| E4 HTTP `/v1/models` | Y | n/a | n/a — read-only metadata | n/a | n/a | n/a |
| E5 Shell | Y | Y | **N** — no `stream_with_cancellation`; `BackgroundTask(lambda: _abort)` is dead code (returns the coroutine function, not a coroutine) | n/a | n/a | shell-side cleanup is whole-server `shutdown()` + `psutil.kill` |
| E6 Offline `minisgl.llm.LLM` | N (single process) | N | **N** — no public `abort()`/`cancel()` on `LLM`; `offline_receive_msg` cannot emit `AbortBackendMsg` | n/a | n/a | natural finish only |
| E7 Hermetic tests | N | N | Y (direct method call on scheduler skeleton) | Y (test spy on `send_result`) | Y (test F on `FrontendManager.__new__`) | Y (test B/C/E on `_free_req_resources` spy) |

Summary: of the six user-facing entries (E1–E6), only **E1 and E2**
route a client-side cancel through the AbortAck chain proven by Gate
2.3f. E3, E5 have wiring gaps; E6 has no abort API at all; E4 is not
applicable.

---

## 5. What Gate 2.3f already proves vs. what E1/E2 needs

Gate 2.3f test surface (already committed, unchanged in this gate):

* `test_A_waiting_abort_no_resources_one_ack`
* `test_B_running_non_inflight_free_then_ack`
* `test_C_overlap_deferred_ack_after_apply`
* `test_D_unknown_uid_idempotent_ack`
* `test_E_duplicate_abort_single_free_single_ack`
* `test_F_frontend_abort_pending_drops_late_userreply`
* `test_G_natural_finish_no_ack`
* `test_H_tokenizer_forwards_abort_ack_to_frontend`

Coverage those tests provide, per seam:

* Scheduler side: A/B/C/D/E/G — all branches of `_process_one_msg` +
  `_apply_deferred_aborts` + `_flush_pending_acks`.
* Tokenizer worker side: H — `AbortAckMsg → AbortAckReply` isinstance
  dispatch on `send_frontend`.
* Frontend state machine: F — `abort_pending` gate + `AbortAckReply`
  cleanup + duplicate ack idempotency, all against
  `FrontendManager.__new__` (no real ZMQ).

What Gate 2.3f does **not** exercise, and what the E1/E2 exposed path
needs:

* `FrontendManager.abort_user` running against a real
  `ZmqAsyncPushQueue` and producing an `AbortMsg` on the wire that a
  real ZMQ pull socket receives.
* `FrontendManager.listen` running against a real `ZmqAsyncPullQueue`
  and reacting to an `AbortAckReply` that was actually msgpack-encoded
  and shipped over a ZMQ push socket.
* `wait_for_abort_ack(uid)` completing on that ack arrival.
* Duplicate `AbortAckReply` on the wire being a strict no-op.

Two of the audit's committed tests (§7 of this document) fill exactly
those gaps and nothing more. They do not model the scheduler side, the
tokenizer worker, the model, or the NPU; those are already proven by
Gate 2.3f test surface, and re-proving them is out of scope.

---

## 6. Current minimal gap

Ranking gaps by "how far the observed exposed path is from
end-to-end AbortAck", smallest first:

1. **HTTP streaming E1/E2 lack a checked-in over-the-wire AbortAck
   proof.** All the seams are wired; only the frontend side of the
   real ZMQ transport is untested end-to-end. This is the closable
   gap in this gate and is closed in §7 below by two hermetic
   subprocess-free tests that talk to a stub tokenizer over real
   `zmq.PULL/PUSH` sockets on the loopback ipc.
2. **HTTP non-streaming E3 has no cancel wiring.** Fix would add
   `is_disconnected()` polling around the non-stream await path (or
   split the non-stream path into a wrapped generator + collector).
   This is a real code change and is deferred: it expands scope beyond
   an audit, and the release notes for `v0.1.0a1` already declare the
   HTTP cross-process path as unfrozen.
3. **Shell E5 has broken cancel wiring.** Fix would (a) call the
   coroutine in the `BackgroundTask` (`lambda: _abort()`), (b) route
   `shell_generate` through `stream_with_cancellation`. Deferred for
   the same reason as E3; the shell today relies on whole-server
   teardown, which is a working escape hatch even though not the
   AbortAck contract.
4. **Offline LLM E6 has no abort API.** Fix would add
   `LLM.abort(uid)` that injects an `AbortBackendMsg` into the next
   `offline_receive_msg` batch. Deferred: no downstream consumer of
   the offline driver requests this today (the only in-tree caller is
   the benchmark script which submits, runs to completion, and exits).

Nothing in gaps 2–4 is a regression of Gate 2.3f; they are entries
whose owning code was written before the AbortAck contract landed and
was never retro-wired. Their fix is out of scope for a read-only
audit + minimal proof.

---

## 7. Minimal exposed-path evidence added at this gate

Two tests live at `tests/misc/test_exposed_path_abort_ack.py`. They
exercise `FrontendManager` over real `zmq.PULL/PUSH` IPC pairs on
loopback (via `ipc:///tmp/...`), representing the transport wires
that HTTP streaming (E1/E2) actually uses at runtime. The scheduler
and tokenizer worker are not modelled; the tokenizer's isinstance
dispatch and the scheduler's ack emission are already proven by Gate
2.3f tests H and A–E.

`test_exposed_path_abort_ack_single_request`:

* Constructs a `FrontendManager` with a real `ZmqAsyncPushQueue` bound
  to a scratch IPC address and a real `ZmqAsyncPullQueue` bound to a
  second scratch IPC address.
* Counter-parties: a `ZmqPullQueue` on the send address (receives what
  the frontend emits toward the tokenizer) and a `ZmqPushQueue` on the
  recv address (delivers what the tokenizer would emit toward the
  frontend).
* Drives `state.new_user()`; asserts uid bookkeeping is initialised.
* Drives `state.send_one(TokenizeMsg)`; asserts a `TokenizeMsg`
  arrives on the pull counter-party with the expected uid.
* Drives `state.abort_user(uid)`; asserts uid enters `abort_pending`
  before the msg is enqueued, and asserts an `AbortMsg` with the
  expected uid arrives on the pull counter-party.
* Pushes a late `UserReply(uid)` back over the recv counter-party;
  asserts `listen` drops it (no growth in `ack_map[uid]`).
* Pushes an `AbortAckReply(uid)`; asserts `wait_for_abort_ack(uid)`
  returns, `abort_pending` is emptied, `ack_map`/`event_map`/
  `abort_pending_events` for the uid are all gone.

`test_exposed_path_abort_duplicate_idempotent`:

* Same rig.
* Calls `state.abort_user(uid)` twice; asserts two `AbortMsg` on the
  pull counter-party (idempotency of caller-side retry is a *no drop*
  at the frontend; the scheduler-side idempotency of Gate 2.3f test E
  covers what happens next).
* Pushes one `AbortAckReply`; asserts `wait_for_abort_ack` returns.
* Pushes a second `AbortAckReply` for the same uid; asserts `listen`
  does not raise (strict no-op per Gate 2.3f spec).

These two tests deliberately avoid launching a real tokenizer worker
subprocess: doing so would require a HF tokenizer, an NPU-side model
runtime, and — crucially — repeat the Gate 2.3f scheduler proof under a
different disguise. Scope keeps the added surface at exactly what E1/E2
uses that Gate 2.3f skipped: the frontend-side msgpack encode/decode
across real `pyzmq` sockets, and the `abort_user → send_one`/
`listen → AbortAckReply` state-machine transitions over those sockets.

---

## 8. Not modified in this gate

* `python/minisgl/scheduler/*` — no scheduler edits.
* `python/minisgl/tokenizer/server.py` — no tokenizer edits.
* `python/minisgl/server/api_server.py` — no server edits (E3 gap
  documented, not patched).
* `python/minisgl/llm/llm.py` — no offline-driver edits (E6 gap
  documented, not patched).
* `python/minisgl/message/*` — no message-schema edits.
* Release tag `v0.1.0a1` — untouched.
* GitHub release — untouched.
* Gate 1 / 2.1 / 2.2 / 2.3 verdicts — untouched.
* `CHANGELOG.md` — untouched (no new capability delivered by this
  audit).

---

## 9. References

* Gate 2.3 verdict — `docs/ascend_port/gate2_3_request_lifecycle_verdict.md`
* Gate 2.3f test surface — `tests/misc/test_scheduler_abort_ack.py`
* Frontend state machine — `python/minisgl/server/api_server.py:102–327`
* Tokenizer isinstance dispatch — `python/minisgl/tokenizer/server.py:69–133`
* Scheduler abort handling — `python/minisgl/scheduler/scheduler.py:110–158, 334–482`
* Offline driver — `python/minisgl/llm/llm.py`
* Real ZMQ transports — `python/minisgl/utils/mp.py:12–102`
