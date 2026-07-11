"""Gate 2.4 — exposed-path AbortAck evidence over real ZMQ IPC.

These tests exercise the *exposed* HTTP-streaming request path
(``/generate`` and ``/v1/chat/completions`` with ``stream=true``) at the
frontend seam only: ``FrontendManager.abort_user`` and
``FrontendManager.listen`` running against real ``pyzmq`` PUSH/PULL
sockets on loopback IPC. The scheduler side, the tokenizer worker, the
NPU forward path, and the model are all deliberately not modelled —
those are covered by ``tests/misc/test_scheduler_abort_ack.py`` scenarios
A–H and by Gates 1 / 2.1 / 2.2 / 2.3.

What is proved here (and only here):

* ``TokenizeMsg`` emitted by ``FrontendManager.send_one`` reaches a real
  ``ZmqPullQueue`` counter-party correctly msgpack-encoded and decoded.
* ``FrontendManager.abort_user`` places the uid into ``abort_pending``
  BEFORE the ``AbortMsg`` is enqueued, and the ``AbortMsg`` reaches the
  same counter-party correctly typed and uid-tagged.
* A ``UserReply`` pushed back over the recv counter-party AFTER the
  frontend has moved the uid into ``abort_pending`` is dropped by
  ``FrontendManager.listen`` — no growth in ``ack_map[uid]``.
* An ``AbortAckReply`` arriving on the recv counter-party unblocks
  ``wait_for_abort_ack(uid)`` and cleans all four bookkeeping buckets
  (``ack_map``, ``event_map``, ``abort_pending``,
  ``abort_pending_events``).
* A duplicate ``abort_user`` sends a second ``AbortMsg`` (idempotency
  is a scheduler-side property, per Gate 2.3f scenario E — the
  frontend must not silently drop the retry) and a duplicate
  ``AbortAckReply`` on the wire is a strict no-op that does not raise.

These are hermetic: they need only ``pyzmq`` + ``msgpack`` + ``pytest``
plus the minisgl package sources on ``sys.path``. No HF tokenizer,
no NPU, no model weights, no HTTP server socket, no uvicorn.

The two tests are named exactly as the Gate 2.4 audit doc requires:
``test_exposed_path_abort_ack_single_request`` and
``test_exposed_path_abort_duplicate_idempotent``.
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
    """``minisgl.server.api_server`` imports prompt_toolkit / fastapi /
    starlette at module import time. The FrontendManager state machine
    exercised here does not touch any of them; stub whatever is missing
    so the import goes through in a container that only has the
    scheduler-side wheel installed. This mirrors Gate 2.3f scenario F.
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
    # ipc:// under /tmp so it survives on containers that mask $TMPDIR.
    tok = uuid.uuid4().hex[:8]
    path = os.path.join(tempfile.gettempdir(), f"minisgl_gate24_{tag}_{tok}")
    return f"ipc://{path}"


@pytest.fixture
def frontend_rig():
    _ensure_python_root_on_path()
    _stub_optional_server_deps()

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

    send_addr = _new_ipc_addr("send")  # frontend → (tokenizer stub)
    recv_addr = _new_ipc_addr("recv")  # (tokenizer stub) → frontend

    # Bind order matches the running server (api_server.run_api_server):
    # the frontend binds both ends of its own ZMQ pair, then the tokenizer
    # counter-party connects. Under real deployment the tokenizer worker
    # would connect via ``create=False``; the stubs below take the same
    # role.
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

    # Tokenizer-side stubs. These are synchronous ZMQ queues because the
    # test drives them from the same asyncio loop the FrontendManager uses;
    # for a real subprocess the tokenizer worker uses the sync flavour too
    # (see minisgl.tokenizer.server.tokenize_worker).
    stub_recv_from_frontend = ZmqPullQueue(
        send_addr, create=False, decoder=BaseTokenizerMsg.decoder
    )
    stub_send_to_frontend = ZmqPushQueue(
        recv_addr, create=False, encoder=BaseFrontendMsg.encoder
    )

    rig = {
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
        frontend.send_tokenizer.stop()
        frontend.recv_tokenizer.stop()
        stub_recv_from_frontend.stop()
        stub_send_to_frontend.stop()


async def _wait_pull(pull, deadline: float = 5.0):
    """Poll a sync ZmqPullQueue from an async task without blocking the
    loop. Returns the decoded object or raises TimeoutError.
    """
    loop = asyncio.get_running_loop()
    end = loop.time() + deadline
    while loop.time() < end:
        if not pull.empty():
            return pull.get()
        await asyncio.sleep(0.01)
    raise TimeoutError("no message on ZMQ pull queue within deadline")


def test_exposed_path_abort_ack_single_request(frontend_rig):
    rig = frontend_rig
    fm = rig["frontend"]
    TokenizeMsg = rig["TokenizeMsg"]
    AbortMsg = rig["AbortMsg"]
    UserReply = rig["UserReply"]
    AbortAckReply = rig["AbortAckReply"]

    async def _drive():
        # ------------------------------------------------------------------
        # 1. new_user + tokenize round-trip: prove the send path works over
        #    real msgpack + real ZMQ before we touch the abort machinery.
        # ------------------------------------------------------------------
        uid = fm.new_user()
        assert uid in fm.ack_map
        assert uid in fm.event_map
        assert uid not in fm.abort_pending

        # SamplingParams import is deferred to test bodies to avoid
        # pulling torch into the fixture's stub-detection.
        from minisgl.core import SamplingParams

        await fm.send_one(
            TokenizeMsg(
                uid=uid,
                text="hello",
                sampling_params=SamplingParams(
                    ignore_eos=False, max_tokens=8
                ),
            )
        )
        msg = await _wait_pull(rig["stub_recv_from_frontend"])
        assert isinstance(msg, TokenizeMsg)
        assert msg.uid == uid
        assert msg.text == "hello"

        # ------------------------------------------------------------------
        # 2. abort_user: uid moves into abort_pending BEFORE the wire msg
        #    is enqueued (Gate 2.3f spec — otherwise a UserReply could race
        #    the abort_pending mutation and slip past the gate).
        # ------------------------------------------------------------------
        pre_pending_snapshot = uid in fm.abort_pending
        abort_task = asyncio.create_task(fm.abort_user(uid))
        # abort_user's very first statement (after the log) is the set-add;
        # give it a scheduler tick, but no I/O yet.
        await asyncio.sleep(0)
        assert uid in fm.abort_pending, (
            "abort_pending must be set before AbortMsg is enqueued"
        )
        # Now let abort_user complete its send_one.
        await abort_task
        assert not pre_pending_snapshot
        assert uid in fm.abort_pending_events

        abort_msg = await _wait_pull(rig["stub_recv_from_frontend"])
        assert isinstance(abort_msg, AbortMsg)
        assert abort_msg.uid == uid

        # ------------------------------------------------------------------
        # 3. Late UserReply for an abort-pending uid must be dropped by
        #    listen; no growth in ack_map[uid].
        # ------------------------------------------------------------------
        listener = asyncio.create_task(fm.listen())
        try:
            rig["stub_send_to_frontend"].put(
                UserReply(uid=uid, incremental_output="ghost", finished=False)
            )
            # Give listen a fair chance to observe and drop it.
            for _ in range(50):
                await asyncio.sleep(0.01)
                if uid not in fm.ack_map:
                    break
            assert uid in fm.ack_map, (
                "abort_pending should not have cleaned ack_map yet"
            )
            assert fm.ack_map[uid] == [], (
                "late UserReply on abort-pending uid must be dropped"
            )

            # ------------------------------------------------------------------
            # 4. AbortAckReply on the wire → wait_for_abort_ack returns and
            #    all four buckets are cleaned.
            # ------------------------------------------------------------------
            rig["stub_send_to_frontend"].put(AbortAckReply(uid=uid))
            await asyncio.wait_for(fm.wait_for_abort_ack(uid), timeout=5.0)
            # listen's cleanup runs before wait_for_abort_ack returns:
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


def test_exposed_path_abort_duplicate_idempotent(frontend_rig):
    rig = frontend_rig
    fm = rig["frontend"]
    AbortMsg = rig["AbortMsg"]
    AbortAckReply = rig["AbortAckReply"]

    async def _drive():
        uid = fm.new_user()

        # Two abort_user calls: the frontend must NOT silently coalesce
        # the retry — the scheduler-side idempotency (Gate 2.3f scenario E)
        # is what dedupes at the free/ack layer. Both AbortMsgs travel the
        # wire; both hit abort_pending harmlessly (set-add is idempotent).
        await fm.abort_user(uid)
        await fm.abort_user(uid)
        assert uid in fm.abort_pending

        pulled: List = []
        pulled.append(await _wait_pull(rig["stub_recv_from_frontend"]))
        pulled.append(await _wait_pull(rig["stub_recv_from_frontend"]))
        assert all(isinstance(m, AbortMsg) for m in pulled)
        assert [m.uid for m in pulled] == [uid, uid]

        listener = asyncio.create_task(fm.listen())
        try:
            # First ack — completes wait_for_abort_ack, cleans state.
            rig["stub_send_to_frontend"].put(AbortAckReply(uid=uid))
            await asyncio.wait_for(fm.wait_for_abort_ack(uid), timeout=5.0)
            assert uid not in fm.abort_pending
            assert uid not in fm.ack_map
            assert uid not in fm.event_map
            assert uid not in fm.abort_pending_events

            # Second ack — must be a strict no-op. listen tolerates a uid
            # already gone from every bookkeeping bucket.
            rig["stub_send_to_frontend"].put(AbortAckReply(uid=uid))
            # A short settle period lets listen consume + no-op. If any
            # branch KeyError'd we'd catch it via the task's exception.
            for _ in range(50):
                await asyncio.sleep(0.01)
                if listener.done():
                    break
            assert not listener.done(), (
                "duplicate AbortAckReply must not raise inside listen"
            )
            # Buckets still clean, uid never re-emerged.
            assert uid not in fm.abort_pending
            assert uid not in fm.ack_map
        finally:
            listener.cancel()
            try:
                await listener
            except (asyncio.CancelledError, Exception):
                pass

    asyncio.run(_drive())
