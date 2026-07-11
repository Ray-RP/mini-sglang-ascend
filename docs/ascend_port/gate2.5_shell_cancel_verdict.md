# Gate 2.5 Verdict — Shell Cancel Path Cleanup

**Gate ID:** 2.5 (shell cancel path cleanup for Mini-SGLang-Ascend)
**Verdict:** PASS
**Branch:** `gate2.5-shell-cancel-cleanup`
**Base commit:** `6aa1413` (tip of `ascend-port`, Gate 2.4 merge)
**Date:** 2026-07-11
**Kind:** Minimal scoped fix (Option A) + hermetic proof.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4, does not
mutate release tag `v0.1.0a1`, does not touch the GitHub Release,
CHANGELOG, or release notes, and does not extend the Ascend port to
server crash / restart, TP > 1 forward, forward/sampler exception
recovery, long soak, chunked-prefill abort, offline `LLM.abort()`,
non-streaming HTTP cancel, or benchmarking.

---

## 1. Verdict summary

**PASS on E5 (shell entry).** The AbortAck chain proven by Gate 2.3f
and end-to-end wired for E1/E2 by Gate 2.4 is now reachable from the
interactive shell entry as well. `KeyboardInterrupt` or
`asyncio.CancelledError` observed while consuming
`shell_completion`'s byte generator routes into
`FrontendManager.abort_user(uid)` → wire `AbortMsg` → tokenizer worker
`AbortBackendMsg` → scheduler AbortAck path → `AbortAckMsg` →
tokenizer worker `AbortAckReply` → `FrontendManager.listen` cleanup,
followed by `wait_for_abort_ack(uid)` returning in the shell loop.
Late `UserReply` chunks arriving after the cancel are dropped by two
independent gates: `FrontendManager.listen`'s `abort_pending` check
and the shell inner loop's `if aborted: continue` guard.

The dead `BackgroundTask(lambda: _abort)` and the corresponding
`starlette.background` import are removed. The whole-server
`finally` block in `shell()` (`shutdown()` + `psutil.Process.kill`)
is unchanged — it is only invoked on shell exit (`/exit`, Ctrl-D,
uncaught exception), not on per-request cancel.

Because Gate 2.5 was framed as a boolean cleanup of a single entry
(E5) and both required tests pass with all other suites green, the
verdict is **PASS** per the criteria supplied at gate open:

> PASS = shell cancel wiring restored to the AbortAck contract with
> two hermetic proofs; no regression in Gate 2.3 / 2.4 suites.
> PARTIAL = shell cancel wired but only one of the two required
> proofs green.
> BLOCKED = shell cancel cannot be routed into the AbortAck chain
> without touching scheduler / tokenizer / IPC seams outside
> `api_server.py`.

---

## 2. Support matrix delta (Gate 2.4 → Gate 2.5)

| Entry | Gate 2.4 | Gate 2.5 |
|---|---|---|
| E1 HTTP `/generate` (stream) | PASS | PASS (unchanged) |
| E2 HTTP `/v1/chat/completions` stream=true | PASS | PASS (unchanged) |
| E3 HTTP `/v1/chat/completions` stream=false | NOT REACHED | NOT REACHED (out of scope) |
| E4 HTTP `/v1/models` | n/a | n/a |
| E5 Shell | NOT REACHED | **PASS** |
| E6 Offline `minisgl.llm.LLM` | NOT REACHED | NOT REACHED (out of scope) |
| E7 Hermetic tests | PASS | PASS (2 tests added) |

Aggregate exposed-path readiness moves from 2/6 (E1, E2) to 3/6
(E1, E2, E5). E3 and E6 remain deferred by the Gate 2.5 opening
scope — this gate is deliberately narrow.

---

## 3. Change surface

Confined to `python/minisgl/server/api_server.py`:

* Removed `from starlette.background import BackgroundTask` (previously
  line 28) — unused after the rewrite.
* Widened `typing` import with `AsyncIterator`.
* `shell_completion(req)` now returns
  `Tuple[int, AsyncIterator[bytes]]` (uid + plain byte generator)
  instead of a `StreamingResponse` with a dead `BackgroundTask`.
  The uid is exposed to the caller so the shell loop can route a
  cancel through `abort_user`/`wait_for_abort_ack` on the same uid.
* `shell()`'s inner loop wraps the `async for chunk in gen:` iteration
  in a `try / except (KeyboardInterrupt, asyncio.CancelledError):`
  block. On cancel: set `aborted = True`, call
  `state.abort_user(uid)`, await `state.wait_for_abort_ack(uid)`, do
  not append the cancelled request to `history`.
* An `if aborted: continue` guard is added inside the render loop as
  a belt-and-braces defence against a `UserReply` slipping between the
  `abort_user()` call and the tokenizer's next dispatch tick.

No other file was modified. `stream_with_cancellation`,
`FrontendManager.abort_user`, `FrontendManager.wait_for_abort_ack`,
`FrontendManager.listen`, the tokenizer worker, and the scheduler are
untouched.

---

## 4. Test surface (locked at this gate)

New file: `tests/misc/test_shell_cancel_cleanup.py`, two tests:

* `test_shell_cancel_triggers_abort_backend_msg` — drives the real
  rewritten `shell_completion` + a local repro of the Gate 2.5 shell
  inner loop against a real `pyzmq` PUSH/PULL IPC pair.
  * Emits two `UserReply` chunks, renders them, cancels mid-stream.
  * Asserts `AbortMsg` on the tokenizer-inbound wire (which the
    tokenizer worker translates to `AbortBackendMsg` — the
    transformation itself is validated by
    `test_scheduler_abort_ack.py::test_H_tokenizer_forwards_abort_ack_to_frontend`
    and the Gate 2.3f dispatch code path).
  * Asserts `abort_pending` set before the wire msg, cleared exactly
    when the `AbortAckReply` is emitted by the stub.
  * Asserts all four bookkeeping buckets clean after
    `wait_for_abort_ack` returns.

* `test_shell_cancel_suppresses_late_output` — same rig, but pushes a
  late `UserReply` *after* the cancel is raised.
  * Asserts the collected `cur_msg` does not contain the late chunk.
  * Asserts terminal bookkeeping is still clean.

Regression suites re-run at the same commit:

```
tests/misc/test_shell_cancel_cleanup.py            2 passed
tests/misc/test_exposed_path_abort_ack.py          2 passed
tests/misc/test_scheduler_abort_ack.py            <suite> passed
tests/misc/test_scheduler_prepare_batch_txn.py    <suite> passed
tests/misc/test_engine_forward_sampler_atomic.py  <suite> passed
tests/misc/test_scheduler_shutdown_drain.py       <suite> passed
tests/misc/test_scheduler_overlap_abort_fence.py  <suite> passed
tests/misc/test_pyproject_config.py               <suite> passed
```

Combined row count from a single pytest invocation:
`51 passed in 21.02s` (Gate 2.3f 47 + Gate 2.4 2 + Gate 2.5 2 = 51).

Run environment: remote container `<CONTAINER>` on the Ascend host
described in the private CLAUDE.md; the tests themselves are pure-CPU
and hermetic (real `pyzmq` on `ipc:///tmp/…`, no NPU, no HF
tokenizer, no model weights).

---

## 5. What is NOT proven at this gate

Per the Gate 2.5 opening spec, the following are explicitly out of
scope:

* E3 non-streaming HTTP cancel — still `NOT REACHED`.
* E6 offline `LLM.abort()` — still not part of the public surface.
* Server crash / restart lifecycle.
* TP > 1 forward or cancel.
* Forward / sampler exception recovery inside the scheduler.
* CPU detokenize copy or event-record failure recovery.
* Long-duration soak.
* Chunked-prefill abort audit.
* Performance benchmarks.
* Any change outside `python/minisgl/server/api_server.py`.
* The interactive shell's actual `prompt_toolkit` REPL is not
  exercised end-to-end on a live NPU; the shell inner-loop *body* is
  what Gate 2.5 changed and what the tests exercise. Booting the
  full `python -m minisgl --shell` process on a live 910B1 host is
  the same NPU-side attestation already covered by prior gates and
  not re-attested here.

---

## 6. Freeze boundary

This gate freezes shell cancel routing into the AbortAck chain.
It does not claim production server restart support.
It does not claim TP>1 support.
It does not claim in-process forward/sampler exception recovery.
It does not extend the offline `LLM` driver's public surface.
It does not add new Ascend model execution evidence beyond Gate 1 –
Gate 2.4.
