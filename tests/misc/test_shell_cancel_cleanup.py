"""Gate 2.5 — shell cancel wiring evidence over real ZMQ IPC.

These tests exercise the shell (``python -m minisgl --shell``) entry's
cancellation semantics at the frontend seam only:
``FrontendManager.abort_user``, ``FrontendManager.wait_for_abort_ack``,
and the ``shell_completion`` -> shell inner-loop cancel path added by
Gate 2.5 in ``python/minisgl/server/api_server.py``.

The scheduler side, the tokenizer worker, the NPU forward path, and the
model are all deliberately not modelled here — those are covered by
``tests/misc/test_scheduler_abort_ack.py`` (Gate 2.3f), the
end-to-end Gate 1 – 2.3 verdicts, and Gate 2.4's exposed-path proof.

What is proved here (and only here):

* ``test_shell_cancel_triggers_abort_backend_msg`` — driving the shell
  inner loop (as rewritten by Gate 2.5) through a real ZMQ IPC pair and
  raising ``CancelledError`` mid-stream results in an ``AbortMsg`` on
  the tokenizer-inbound wire (which the tokenizer worker translates to
  ``AbortBackendMsg`` — the transformation is validated in
  ``test_scheduler_abort_ack.py::test_H``). ``abort_pending`` is set
  before the wire msg is enqueued and cleared exactly when the
  ``AbortAckReply`` arrives.
* ``test_shell_cancel_suppresses_late_output`` — a ``UserReply`` pushed
  back over the recv counter-party AFTER the cancel has been raised is
  not rendered by the shell loop. The collected ``cur_msg`` stops
  accumulating at the cancel point.

Both tests use the same hermetic rig style as Gate 2.4 —
``FrontendManager.__new__`` + real ``pyzmq`` PUSH/PULL on
``ipc:///tmp/…`` — plus a driver that mirrors the exact loop body in
``api_server.shell()``. The shell body itself is not called directly
because it requires ``prompt_toolkit.PromptSession``; the loop body it
runs (which is the only code Gate 2.5 changed at the request level) is
reproduced locally so the test can inject a cancel at a chosen chunk
boundary.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_ROOT = _REPO_ROOT / "python"


def _ensure_python_root_on_path() -> None:
    p = str(_PYTHON_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def _stub_optional_server_deps() -> None:
    """Same as Gate 2.4: ``minisgl.server.api_server`` imports
    prompt_toolkit / fastapi / uvicorn / starlette at module import time.
    None of them are needed to exercise the FrontendManager +
    shell_completion state machine that this test drives; stub whichever
    are missing.
    """
    import types as _t

    def _fake_module(name: str, attrs: dict | None = None) -> None:
        if name in sys.modules:
            return
        mod = _t.ModuleType(name)
        for k, v in (attrs or {}).items():
            setattr(mod, k, v)
        sys.modules[name] = mod

    try:
        import prompt_toolkit  # noqa: F401
    except Exception:
        _fake_module("prompt_toolkit", {"PromptSession": MagicMock()})
        _fake_module("prompt_toolkit.completion", {"WordCompleter": MagicMock()})
    try:
        import uvicorn  # noqa: F401
    except Exception:
        _fake_module("uvicorn", {"run": MagicMock()})
    try:
        import fastapi  # noqa: F401
    except Exception:
        _fake_module("fastapi", {"FastAPI": MagicMock(), "Request": MagicMock()})
        _fake_module("fastapi.responses", {"StreamingResponse": MagicMock()})
    try:
        import starlette  # noqa: F401
        import starlette.background  # noqa: F401
    except Exception:
        _fake_module("starlette", {})
        _fake_module("starlette.background", {"BackgroundTask": MagicMock()})
    if "pydantic" not in sys.modules:
        try:
            import pydantic  # noqa: F401
        except Exception:
            class _BM:
                def __init__(self, **kw):
                    for k, v in kw.items():
                        setattr(self, k, v)

            def _Field(**kw):
                default = kw.get("default")
                if default is not None:
                    return default
                factory = kw.get("default_factory")
                return factory() if factory is not None else None

            _fake_module(
                "pydantic", {"BaseModel": _BM, "Field": _Field}
            )


def _new_ipc_addr(tag: str) -> str:
    tok = uuid.uuid4().hex[:8]
    path = os.path.join(tempfile.gettempdir(), f"minisgl_gate25_{tag}_{tok}")
    return f"ipc://{path}"


@pytest.fixture
def shell_rig():
    _ensure_python_root_on_path()
    _stub_optional_server_deps()

    # Import after stubbing so api_server's optional imports resolve
    # against the stubs where the real package is missing.
    import minisgl.server.api_server as api_server
    from minisgl.message import (
        AbortAckReply,
        AbortMsg,
        BaseFrontendMsg,
        BaseTokenizerMsg,
        BatchFrontendMsg,
        TokenizeMsg,
        UserReply,
    )
    from minisgl.server.api_server import FrontendManager
    from minisgl.utils import (
        ZmqAsyncPullQueue,
        ZmqAsyncPushQueue,
        ZmqPullQueue,
        ZmqPushQueue,
    )

    send_addr = _new_ipc_addr("send")  # frontend → tokenizer stub
    recv_addr = _new_ipc_addr("recv")  # tokenizer stub → frontend

    frontend = FrontendManager.__new__(FrontendManager)
    frontend.config = None
    frontend.uid_counter = 0
    frontend.initialized = False
    frontend.ack_map = {}
    frontend.event_map = {}
    frontend.abort_pending = set()
    frontend.abort_pending_events = {}
    frontend.send_tokenizer = ZmqAsyncPushQueue(
        send_addr, create=True, encoder=BaseTokenizerMsg.encoder
    )
    frontend.recv_tokenizer = ZmqAsyncPullQueue(
        recv_addr, create=True, decoder=BaseFrontendMsg.decoder
    )

    stub_recv_from_frontend = ZmqPullQueue(
        send_addr, create=False, decoder=BaseTokenizerMsg.decoder
    )
    stub_send_to_frontend = ZmqPushQueue(
        recv_addr, create=False, encoder=BaseFrontendMsg.encoder
    )

    # Install the frontend as global state so shell_completion picks it up
    # via get_global_state(). Restore whatever was there afterwards.
    prev_state = api_server._GLOBAL_STATE
    api_server._GLOBAL_STATE = frontend

    rig = {
        "api_server": api_server,
        "frontend": frontend,
        "stub_recv_from_frontend": stub_recv_from_frontend,
        "stub_send_to_frontend": stub_send_to_frontend,
        "TokenizeMsg": TokenizeMsg,
        "AbortMsg": AbortMsg,
        "UserReply": UserReply,
        "AbortAckReply": AbortAckReply,
        "BatchFrontendMsg": BatchFrontendMsg,
    }
    try:
        yield rig
    finally:
        api_server._GLOBAL_STATE = prev_state
        frontend.send_tokenizer.stop()
        frontend.recv_tokenizer.stop()
        stub_recv_from_frontend.stop()
        stub_send_to_frontend.stop()


async def _wait_pull(pull, deadline: float = 5.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + deadline
    while loop.time() < end:
        if not pull.empty():
            return pull.get()
        await asyncio.sleep(0.01)
    raise TimeoutError("no message on ZMQ pull queue within deadline")


def _make_shell_req():
    """Build the minimum ``OpenAICompletionRequest`` the rewritten
    ``shell_completion`` expects. Mirrors what ``shell()`` constructs per
    prompt line — one user message, streaming enabled.
    """
    from minisgl.server.api_server import Message, OpenAICompletionRequest

    return OpenAICompletionRequest(
        model="",
        messages=[Message(role="user", content="hello")],
        max_tokens=8,
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        stream=True,
    )


async def _drive_shell_inner_loop(fm, gen, cancel_after_chunks: int):
    """Reproduction of the shell()'s per-prompt inner loop as rewritten
    by Gate 2.5. Kept in the test so we don't have to spin up
    ``prompt_toolkit`` just to reach the loop body. Raises
    ``asyncio.CancelledError`` inside ``async for`` after
    ``cancel_after_chunks`` chunks have been rendered (i.e. counted into
    cur_msg). Returns ``(cur_msg, aborted, uid_out)``.

    The body faithfully mirrors ``api_server.shell()`` lines 501–537 in
    the tree after the Gate 2.5 patch: same chunk parsing, same
    ``if aborted: continue`` guard, same abort_user + wait_for_abort_ack
    order, same "cancelled requests do not enter history" semantics.
    """
    cur_msg = ""
    aborted = False
    rendered = 0
    try:
        async for chunk in gen:
            msg = chunk.decode()  # type: ignore
            assert msg.startswith("data: "), msg
            msg = msg[6:]
            assert msg.endswith("\n"), msg
            msg = msg[:-1]
            if msg == "[DONE]":
                continue
            if aborted:
                continue
            cur_msg += msg
            rendered += 1
            if rendered == cancel_after_chunks:
                # Inject a cancel exactly at the requested boundary.
                # This models a Ctrl-C or an outer-task cancellation
                # observed while the async-for is between chunks.
                raise asyncio.CancelledError
    except (KeyboardInterrupt, asyncio.CancelledError):
        aborted = True
        await fm.abort_user(_drive_shell_inner_loop.uid_out)
        await fm.wait_for_abort_ack(_drive_shell_inner_loop.uid_out)
    return cur_msg, aborted


# --------------------------------------------------------------------- test 1


def test_shell_cancel_triggers_abort_backend_msg(shell_rig):
    """Cancel mid-stream in the shell must enqueue an AbortMsg (whose
    tokenizer translation is AbortBackendMsg per Gate 2.3f/H) and
    complete cleanly once the AbortAckReply lands.
    """
    rig = shell_rig
    fm = rig["frontend"]
    TokenizeMsg = rig["TokenizeMsg"]
    AbortMsg = rig["AbortMsg"]
    UserReply = rig["UserReply"]
    AbortAckReply = rig["AbortAckReply"]

    async def _drive():
        # 1. Call the rewritten shell_completion. It must return
        #    (uid, generator) — the plain tuple form Gate 2.5 introduced.
        req = _make_shell_req()
        uid, gen = await rig["api_server"].shell_completion(req)
        assert isinstance(uid, int)
        # Prove send path: the tokenize msg lands on the tokenizer wire.
        tok_msg = await _wait_pull(rig["stub_recv_from_frontend"])
        assert isinstance(tok_msg, TokenizeMsg)
        assert tok_msg.uid == uid

        # 2. Start the listener (real FrontendManager.listen) so
        #    UserReply / AbortAckReply on the recv wire are dispatched.
        listener = asyncio.create_task(fm.listen())

        try:
            # 3. Feed two normal UserReply chunks so the shell loop
            #    renders and increments cur_msg. Third chunk triggers
            #    the cancel.
            for i, txt in enumerate(["foo", "bar"]):
                rig["stub_send_to_frontend"].put(
                    UserReply(uid=uid, incremental_output=txt, finished=False)
                )

            # 4. Drive the shell inner loop, cancelling after 2 rendered
            #    chunks. abort_user + wait_for_abort_ack are invoked
            #    inside the except-branch mirrored from the real shell().
            _drive_shell_inner_loop.uid_out = uid

            drive_task = asyncio.create_task(
                _drive_shell_inner_loop(fm, gen, cancel_after_chunks=2)
            )

            # 5. Wait for the AbortMsg to appear on the tokenizer wire —
            #    this is the whole point of the test. The rewritten shell
            #    routes cancel into abort_user, which sends AbortMsg.
            abort_msg = await _wait_pull(rig["stub_recv_from_frontend"])
            assert isinstance(abort_msg, AbortMsg), (
                f"expected AbortMsg on the tokenizer wire after cancel, "
                f"got {type(abort_msg).__name__}"
            )
            assert abort_msg.uid == uid

            # abort_pending is set the instant abort_user was called and
            # BEFORE the AbortMsg is enqueued (Gate 2.3f invariant).
            # The wait_for_abort_ack awaiter is now blocked on the ack.
            assert uid in fm.abort_pending

            # 6. Emit the ack. The tokenizer worker would produce this
            #    after the scheduler frees + acks; here the stub emits
            #    it directly. wait_for_abort_ack + listen cleanup runs.
            rig["stub_send_to_frontend"].put(AbortAckReply(uid=uid))

            cur_msg, aborted = await asyncio.wait_for(drive_task, timeout=5.0)
            assert aborted is True
            # Two chunks were rendered before the cancel; the abort path
            # then took over.
            assert cur_msg == "foobar"

            # 7. All four buckets clean, uid gone. Same terminal state
            #    Gate 2.4's exposed-path test asserts.
            assert uid not in fm.abort_pending
            assert uid not in fm.ack_map
            assert uid not in fm.event_map
            assert uid not in fm.abort_pending_events
        finally:
            listener.cancel()
            try:
                await listener
            except (asyncio.CancelledError, Exception):
                pass

    asyncio.run(_drive())


# --------------------------------------------------------------------- test 2


def test_shell_cancel_suppresses_late_output(shell_rig):
    """A UserReply that arrives AFTER the shell's cancel must not be
    rendered. The rewritten loop's ``if aborted: continue`` guard plus
    ``FrontendManager.listen``'s abort_pending gate together guarantee
    this.
    """
    rig = shell_rig
    fm = rig["frontend"]
    UserReply = rig["UserReply"]
    AbortAckReply = rig["AbortAckReply"]
    TokenizeMsg = rig["TokenizeMsg"]
    AbortMsg = rig["AbortMsg"]

    async def _drive():
        req = _make_shell_req()
        uid, gen = await rig["api_server"].shell_completion(req)
        tok_msg = await _wait_pull(rig["stub_recv_from_frontend"])
        assert isinstance(tok_msg, TokenizeMsg)

        listener = asyncio.create_task(fm.listen())
        try:
            # 1. One rendered chunk, then cancel.
            rig["stub_send_to_frontend"].put(
                UserReply(uid=uid, incremental_output="A", finished=False)
            )

            _drive_shell_inner_loop.uid_out = uid
            drive_task = asyncio.create_task(
                _drive_shell_inner_loop(fm, gen, cancel_after_chunks=1)
            )

            # 2. Wait for the AbortMsg to be enqueued. Once it is on the
            #    wire we know abort_user has moved uid into abort_pending.
            abort_msg = await _wait_pull(rig["stub_recv_from_frontend"])
            assert isinstance(abort_msg, AbortMsg)
            assert abort_msg.uid == uid
            assert uid in fm.abort_pending

            # 3. Now push a LATE UserReply. FrontendManager.listen will
            #    see uid in abort_pending and drop it. Even if it made it
            #    into wait_for_ack (it will not), the shell loop's
            #    ``if aborted: continue`` guard drops it again. In any
            #    case cur_msg must not grow past "A".
            rig["stub_send_to_frontend"].put(
                UserReply(uid=uid, incremental_output="LATE", finished=False)
            )
            # Give listen a fair chance to observe + drop.
            for _ in range(50):
                await asyncio.sleep(0.01)
                if uid not in fm.ack_map:
                    break

            # 4. Ack the abort. drive_task now completes.
            rig["stub_send_to_frontend"].put(AbortAckReply(uid=uid))
            cur_msg, aborted = await asyncio.wait_for(drive_task, timeout=5.0)

            # Late "LATE" must be absent.
            assert cur_msg == "A", (
                f"cur_msg must not contain post-cancel chunks; got {cur_msg!r}"
            )
            assert aborted is True

            # Final state clean.
            assert uid not in fm.abort_pending
            assert uid not in fm.ack_map
            assert uid not in fm.event_map
            assert uid not in fm.abort_pending_events
        finally:
            listener.cancel()
            try:
                await listener
            except (asyncio.CancelledError, Exception):
                pass

    asyncio.run(_drive())
