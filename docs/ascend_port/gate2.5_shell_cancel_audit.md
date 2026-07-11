# Gate 2.5 — Shell Cancel Path Audit

**Gate ID:** 2.5 (shell cancel path cleanup for Mini-SGLang-Ascend)
**Branch:** `gate2.5-shell-cancel-cleanup`
**Base commit:** `6aa1413` (tip of `ascend-port`, Gate 2.4 merge)
**Date:** 2026-07-11
**Kind:** Read-only audit of the shell entry (E5) followed by a minimal
scoped fix (Option A). Runtime changes are confined to
`python/minisgl/server/api_server.py`.

This gate does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 verdicts, does
not mutate release tag `v0.1.0a1`, does not touch the GitHub release,
CHANGELOG, or release notes. It only closes the shell-side gap called
out by Gate 2.4:

> E5 (shell) NOT REACHED — `shell_completion` builds the SSE response
> without `stream_with_cancellation`, and the
> `BackgroundTask(lambda: _abort)` on the response passes the coroutine
> function (not a coroutine) so nothing is awaited even if a cancel
> path existed.

---

## 1. Scope

Only E5 is in scope. Not in scope for this gate:

* E3 (HTTP non-streaming) — still deferred.
* E6 (offline `LLM`) — no `abort()` API; scope decision unchanged.
* Server restart / crash lifecycle.
* TP > 1 shell cancel.
* Forward / sampler exception recovery inside the scheduler.
* Long soak, chunked-prefill abort, benchmarks.
* Any change outside `api_server.py`.

The single question this audit answers is:

> Given the shell (`--shell`) entry, does the code today route a
> per-request cancel (Ctrl-C on the shell prompt, an EOF during
> streaming, or a client-side task cancellation) into the AbortAck
> chain Gate 2.3f proved and Gate 2.4 partly wired?

Answer today: **no**. The finding below is the same shape as Gate 2.4
Section 3.3, expanded with the exact code frames.

---

## 2. Shell entry — real call graph at `6aa1413`

Entry point:

```
python -m minisgl --model ... --shell
  → minisgl.__main__ → server.launch_server(run_shell=True)
    → api_server.run_api_server(..., run_shell=True)
      → asyncio.run(shell())
```

`python/minisgl/shell.py` is a thin wrapper (`launch_server(run_shell=True)`);
the real shell body lives in `python/minisgl/server/api_server.py:465`.

Inside `shell()` the per-line request loop is:

```
while True:
    cmd = (await session.prompt_async()).strip()
    ...
    req = OpenAICompletionRequest(..., stream=True)
    cur_msg = ""
    async for chunk in (await shell_completion(req)).body_iterator:
        ...
        print(msg, end="", flush=True)
    print("", flush=True)
    history.append((cmd, cur_msg))
```

`shell_completion(req)` at `api_server.py:433` builds:

```python
uid = state.new_user()
await state.send_one(TokenizeMsg(uid, ...))

async def _abort():
    await state.abort_user(uid)

return StreamingResponse(
    state.stream_generate(uid),
    media_type="text/event-stream",
    background=BackgroundTask(lambda: _abort),
)
```

Two independent wiring defects here:

* **`StreamingResponse` wrapper is bypassed.** The shell reads
  `body_iterator` directly (`async for chunk in ... .body_iterator`).
  It never runs the ASGI response cycle. That means the response's
  `background` is never fired, regardless of what it contains — no
  `finally` in `body_iterator`, no `Response.__call__` invoking
  `background`.
* **`BackgroundTask(lambda: _abort)` is malformed.** Even if it were
  triggered, `lambda: _abort` returns the coroutine **function**
  `_abort` (unbound), not a coroutine. Starlette's `BackgroundTask`
  expects a callable that returns an awaitable; calling the lambda
  yields a callable back and awaits nothing. The abort would not
  execute even on the HTTP path.

The generator itself (`state.stream_generate(uid)`) is *not* wrapped
in `stream_with_cancellation`, so cancellation of the outer task never
becomes an `abort_user` call the way E1/E2 do.

Additionally, `shell()`'s `finally` performs:

```python
finally:
    print("Exiting shell...")
    await asyncio.sleep(0.1)
    get_global_state().shutdown()
    import psutil
    parent = psutil.Process()
    for child in parent.children(recursive=True):
        child.kill()
```

That is whole-server tear-down, not per-request abort. It runs on
`/exit`, Ctrl-D, or any exception escaping the loop. It has no
relationship to the AbortAck contract.

### 2.1 What can cancel a shell request today?

Scanning the shell body:

* **`/exit`** — normal exit, no request in flight to abort.
* **`/reset`** — history clear only, no request in flight.
* **`EOFError` (Ctrl-D at the prompt)** — same as `/exit`.
* **Ctrl-C during token streaming** — `asyncio.run(shell())` catches
  `KeyboardInterrupt` at the process boundary; the running task is
  cancelled, the `async for chunk in ...` raises
  `CancelledError`, propagates up through `shell()`, and hits the
  `finally`. No `abort_user`. The shell teardown kills the whole
  server. If the user re-enters the shell, they must reboot the
  process.
* **`session.prompt_async()` returning while a previous
  `body_iterator` is still being consumed** — not possible under the
  present linear loop; there is no concurrent input.
* **Backend natural finish** — no cancel involved; `[DONE]` chunk
  ends the loop and `history.append` records the reply.

There is currently no line of code that reaches `abort_user(uid)`
from the shell — the `_abort` coroutine defined in `shell_completion`
is dead code.

---

## 3. Symbol usage sweep

Grep evidence supporting the audit:

* `BackgroundTask` — imported once (`api_server.py:28`), referenced
  once (`api_server.py:460`). No other consumer in the repo.
* `shell_completion` — defined once (`api_server.py:433`), called once
  (`api_server.py:498`, inside `shell()`).
* `stream_with_cancellation` — called only from E1 and E2
  (`api_server.py:359, 396`), never from the shell.
* `abort_user` — called only from `stream_with_cancellation` and from
  the (dead) `_abort` closure in `shell_completion`.
* `wait_for_abort_ack` — never called by any real code path today; used
  only by Gate 2.4 tests.

Because `shell_completion` has a single caller, its signature and
return type are safe to change locally.

---

## 4. Option analysis

Two options were considered per the gate opening.

### 4.1 Option A — real per-request cancel in the shell

Change `shell_completion` to return a plain async byte generator (not
a `StreamingResponse` with dead background), and change `shell()` to
handle `KeyboardInterrupt` / cancellation around its
`async for chunk in gen:` loop by calling `state.abort_user(uid)`,
awaiting `state.wait_for_abort_ack(uid)`, and *not* printing any
tokens that arrive after the abort has been requested.

Cost:

* One rewrite of `shell_completion` (drop `BackgroundTask`, drop
  `StreamingResponse` — the shell never needed the HTTP wrapper).
* One rewrite of `shell()`'s inner loop (try/except for `CancelledError`
  and `KeyboardInterrupt`, propagate `abort_user` + `wait_for_abort_ack`
  before returning to prompt).
* No scheduler / tokenizer / IPC changes.
* No change to `stream_with_cancellation` (only used by HTTP paths).
* Hermetically provable using the Gate 2.4 fixture (real
  `pyzmq` IPC + `FrontendManager.__new__`) plus a scheduler stub.

Benefit:

* E5 becomes wired end-to-end, closing the Gate 2.4 gap.
* `BackgroundTask` and its (unused) import go away.
* Per-request cancel no longer requires tearing down the entire server.
* Behaviour matches the Gate 2.3f contract that already governs E1/E2.

### 4.2 Option B — freeze shell cancel as unsupported

Delete the dead `BackgroundTask(lambda: _abort)` and `_abort` closure,
change `StreamingResponse` to a plain generator (or leave it — since
`body_iterator` bypass makes it moot), and document that shell cancel
is not implemented. Leave the whole-server teardown as-is.

Cost:

* Minimal (delete 6 lines).
* No new capability.
* Gate 2.4's PARTIAL boundary stays PARTIAL for E5.

Benefit:

* Smaller diff, no test surface expansion.

### 4.3 Choice

**Option A.** The change is confined to two short function bodies in
one file. The Gate 2.4 test fixture is already available for
hermetic proof. Wiring E5 into the AbortAck chain lifts one of the
two "NOT REACHED" entries flagged by Gate 2.4 without touching
`_process_one_msg`, the tokenizer worker, or the ZMQ transport.

Option B would not raise the overall Ascend port from the current
PARTIAL and would leave `BackgroundTask` as a footgun in the file
for the next contributor.

---

## 5. Planned change surface (Option A)

Bounded to `python/minisgl/server/api_server.py`:

* Drop `from starlette.background import BackgroundTask` (line 28) —
  unused after the fix.
* Rewrite `shell_completion(req)` to return an
  `AsyncIterator[bytes]` together with the `uid` used to issue the
  request, so `shell()` can call `abort_user`/`wait_for_abort_ack` on
  the same uid without reaching into private state. Concretely:

  ```python
  async def shell_completion(req):
      state = get_global_state()
      ...
      uid = state.new_user()
      await state.send_one(TokenizeMsg(...))
      return uid, state.stream_generate(uid)
  ```

* Rewrite the inner loop of `shell()` to consume that generator with
  cancel handling:

  ```python
  uid, gen = await shell_completion(req)
  cur_msg = ""
  aborted = False
  try:
      async for chunk in gen:
          # (existing chunk parsing)
          if aborted:
              continue
          cur_msg += msg
          print(msg, end="", flush=True)
  except (KeyboardInterrupt, asyncio.CancelledError):
      aborted = True
      await state.abort_user(uid)
      await state.wait_for_abort_ack(uid)
      print("", flush=True)
      # do NOT append to history — request was cancelled
      continue
  print("", flush=True)
  history.append((cmd, cur_msg))
  ```

  The `if aborted: continue` inside the loop is defensive: even if the
  cancellation is thrown between `abort_user`'s `abort_pending.add`
  and the arrival of `AbortAckReply`, the abort_pending gate in
  `FrontendManager.listen` already drops late `UserReply`s. The
  in-loop guard covers the (tight) window in which the shell task
  is still iterating the generator while the abort is being enqueued.

* No changes to `stream_with_cancellation`, `abort_user`,
  `wait_for_abort_ack`, `FrontendManager.listen`, tokenizer worker, or
  the scheduler. Those already implement the contract.

The `finally` block in `shell()` (whole-server shutdown on shell exit)
is kept unchanged. It runs on `/exit` and EOF, which are still
whole-server exits.

---

## 6. Hermetic test plan

Two new tests will land in `tests/misc/test_shell_cancel_cleanup.py`,
reusing the Gate 2.4 rig (`FrontendManager.__new__` + real
`pyzmq` IPC on `ipc:///tmp/…`):

* **`test_shell_cancel_triggers_abort_backend_msg`** — start a
  `FrontendManager`, a stub tokenizer that echoes `TokenizeMsg`
  through to the frontend as `UserReply` slowly, and a stub scheduler
  that receives `AbortBackendMsg` and produces `AbortAckMsg`. Drive
  the shell inner loop directly (call `shell_completion`, iterate,
  raise `CancelledError` mid-stream, assert:
    * an `AbortMsg` arrived on the tokenizer-inbound queue,
    * `abort_pending.add(uid)` happened before the AbortMsg was
      sent (invariant recorded by Gate 2.3f),
    * `wait_for_abort_ack(uid)` returned after the AbortAckReply
      landed,
    * `abort_pending`, `ack_map`, `event_map`, `abort_pending_events`
      are all empty for that uid.

* **`test_shell_cancel_suppresses_late_output`** — same rig, but the
  stub tokenizer produces one `UserReply` *after* the cancel is
  raised. Assert that no bytes from the late reply were written to
  the caller's collected output (i.e. `cur_msg` stops accumulating at
  the cancel point).

Both tests will use `Scheduler.__new__`-style stubbing so no NPU is
required and the suite still runs on the CI host.

---

## 7. Ascend-side runtime note

The scheduler / tokenizer / ZMQ transports involved are the same
seams already frozen by Gate 2.3d/e/f and re-attested end-to-end in
Gate 2.4. This gate does not require any new NPU-side execution; the
Ascend 910B1 forward path is unchanged and no new attestation is
being introduced against it.

---

## 8. Freeze boundary

This audit freezes the exposed-path shell cancel wiring.
It does not claim production server restart support.
It does not claim TP>1 support.
It does not claim in-process forward/sampler exception recovery.
It does not extend the offline `LLM` driver's public surface.
It does not add new Ascend model execution evidence beyond Gate 1 –
Gate 2.4.
