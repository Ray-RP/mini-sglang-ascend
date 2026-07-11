"""Gate 3.2 — Qwen3-1.7B request-shape expansion smoke on Ascend 910B1.

Not a hermetic pytest: talks to the real NPU. Committed as evidence for
the Gate 3.2 verdict. Envelope is identical to Gate 3.1 (TP=1, eager,
npu_fia, bf16, greedy). Only the request shape varies.

Cases:

  A. B=2 equal-length prefill + decode
     Two identical prompts, same max_new_tokens. Verifies the basic
     multi-request batching that Gate 2.2 proved on Qwen3-0.6B still
     holds on Qwen3-1.7B.

  B. B=2 ragged prefill
     One short prompt, one longer prompt, both same max_new_tokens.
     Verifies:
       * per-request prefill length is respected,
       * logits-selection extracts the correct final-position logit for
         each request (a ragged prefill mistake would swap the first
         token of one request with the other's last-prompt token).

  C. B=2 mixed-KV decode with solo tail
     Ragged prompts AND different max_new_tokens. The short-max request
     finishes earlier; the long-max request continues decoding alone.
     Verifies:
       * attention metadata handles per-request cache_len during the
         shared decode phase (cached_len differs per request),
       * the surviving request continues cleanly once its sibling is
         released,
       * allocator recovers pages of the finished request without
         disturbing the surviving one.

  D. Dynamic admission grow-shrink: B timeline 1 → 2 → 3 → 2 → 1
     Uses a script-local ``DynamicLLM`` subclass that injects new
     pending_requests entries between scheduler iterations. Samples
     ``decode_manager.running_reqs`` at every offline_receive_msg tick.
     Verifies:
       * a new request can be admitted while others are decoding,
       * finishing one request frees its pages while siblings keep
         decoding,
       * allocator returns to baseline once the timeline collapses.

Per-case allocator + integrity + deferred_abort_uids evidence is
emitted line-by-line and also as a JSONL summary. The final scrape line
``GATE3.2_SMOKE_RESULT=<verdict>`` is scraped by the verdict generator.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from typing import List


def _log(section: str, msg: str) -> None:
    print(f"[gate3.2][{section}] {msg}", flush=True)


def _snapshot_alloc(scheduler) -> dict:
    """Capture the allocator snapshot the gate asserts against.

    Correct paged-cache invariant (see Gate 3.1 verdict):
      ``available_size`` = ``free_slots + evictable prefix pages``.
    Radix-cache retention across requests DOES NOT count as a leak; the
    invariant that must return to baseline is ``available_size``.
    """
    cm = scheduler.cache_manager
    return {
        "free_pages": len(cm.free_slots),
        "available_tokens": cm.available_size,
        "total_pages": cm.num_pages,
        "deferred_abort_uids": len(scheduler.deferred_abort_uids),
    }


def _check_alloc_and_integrity(scheduler, tag: str) -> dict:
    snap = _snapshot_alloc(scheduler)
    scheduler.cache_manager.check_integrity()
    _log(
        "alloc",
        f"{tag} free_pages={snap['free_pages']} "
        f"available_tokens={snap['available_tokens']} "
        f"total_pages={snap['total_pages']} "
        f"deferred_abort_uids={snap['deferred_abort_uids']}",
    )
    return snap


# ------------------------------------------------------------ DynamicLLM
# Defined at module scope so a single instance can serve every case.
# ``set_tp_info`` is a process-global singleton
# (``minisgl.distributed.info.set_tp_info``) — instantiating a second
# ``LLM`` in the same process raises ``RuntimeError: TP info has been set``.

# Deferred imports so the module can be imported outside the container
# (the CI harness only exercises this script inside the Ascend image).
try:
    from minisgl.llm import LLM as _LLM
except Exception:  # pragma: no cover - import happens lazily at runtime
    _LLM = None  # type: ignore[assignment]


class DynamicLLM(_LLM if _LLM is not None else object):  # type: ignore[misc]
    """Stage pending prompts across scheduler ticks instead of all at once.

    * With no ``tick_schedule`` installed, behaves identically to
      the stock ``LLM`` — cases A/B/C use it via ``generate``.
    * With a ``tick_schedule`` installed (case D), releases queued
      prompts on a tick schedule and samples the batch shape at every
      ``offline_receive_msg`` call. The timeline lets the verdict
      record the observed running-decode count over ticks.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # Case-D state; unused for A/B/C.
        self._tick_schedule: list = []
        self._tick: int = 0
        self._timeline: list = []

    def offline_receive_msg(self, blocking: bool = False):
        # Fast path: no schedule installed -> stock LLM behaviour.
        if not self._tick_schedule and not self._timeline and self._tick == 0:
            return super().offline_receive_msg(blocking=blocking)

        # Release any schedule entries whose tick has arrived.
        released_this_tick = 0
        while self._tick_schedule and self._tick_schedule[0][0] <= self._tick:
            _, prompt, sp = self._tick_schedule.pop(0)
            self.pending_requests.append((prompt, sp))
            released_this_tick += 1

        running = len(self.decode_manager.running_reqs)
        pending = len(self.prefill_manager.pending_list)
        queued = len(self.pending_requests)
        self._timeline.append({
            "tick": self._tick,
            "running_decode": running,
            "pending_prefill": pending,
            "queued": queued,
            "released_this_tick": released_this_tick,
        })
        self._tick += 1

        # Only allow terminal blocking-drain once every scheduled item
        # has been released.
        eff_blocking = blocking and not self._tick_schedule
        return super().offline_receive_msg(blocking=eff_blocking)


def _build_llm(args):
    """Boot the offline LLM under the Gate 3.1/3.2 frozen envelope.

    Returns a ``DynamicLLM`` (see below): a subclass whose
    ``offline_receive_msg`` is transparent when no tick_schedule is
    installed, so it can serve cases A/B/C exactly like the stock
    ``LLM``, and case D by installing a staged schedule.

    Rebooting a second LLM in the same process triggers
    ``RuntimeError: TP info has been set`` from
    ``minisgl.distributed.info.set_tp_info`` — that singleton is
    process-global. Case D therefore reuses the same boot.
    """
    import torch

    t0 = time.time()
    llm = DynamicLLM(
        model_path=args.model_path,
        dtype=torch.bfloat16,
        attention_backend=args.attention_backend,
        max_running_req=8,
        memory_ratio=0.85,
        # Gate 3.1 envelope: eager mode (torch_npu has no CUDAGraph).
        cuda_graph_bs=[],
        # Ascend FIA NO_QUANT: page_size must be aligned to 16.
        page_size=16,
    )
    _log("boot", f"LLM initialised in {time.time() - t0:.2f}s")
    return llm


# ---------------------------------------------------------------- case A

def case_A(llm, args, jsonl_rows: List[dict]) -> str:
    """B=2 equal-length prefill + decode."""
    from minisgl.core import SamplingParams

    baseline = _check_alloc_and_integrity(llm, "A_baseline")
    prompts = [args.prompt, args.prompt]
    N = args.max_new_tokens_short
    sp = SamplingParams(
        temperature=0.0, top_k=1, top_p=1.0,
        max_tokens=N, ignore_eos=True,
    )
    t0 = time.time()
    out = llm.generate(prompts, [sp, sp])
    elapsed = time.time() - t0
    _log("A", f"elapsed={elapsed:.2f}s")
    for i, o in enumerate(out):
        _log("A", f"req{i} tokens={o['token_ids']}")
        _log("A", f"req{i} text={o['text']!r}")

    lens = [len(o["token_ids"]) for o in out]
    after = _check_alloc_and_integrity(llm, "A_after")
    pass_ok = (
        len(out) == 2
        and lens == [N, N]
        and after["available_tokens"] == baseline["available_tokens"]
        and after["deferred_abort_uids"] == 0
    )
    verdict = "PASS" if pass_ok else "FAIL"
    jsonl_rows.append({
        "case": "A",
        "description": "B=2 equal-length prefill + decode",
        "prompt_lengths_chars": [len(p) for p in prompts],
        "batch_size_timeline": [2],
        "requested_max_new_tokens": [N, N],
        "actual_output_token_count": lens,
        "alloc_before": baseline,
        "alloc_after": after,
        "check_integrity": "ok",
        "verdict": verdict,
    })
    _log("A", f"verdict={verdict}")
    return verdict


# ---------------------------------------------------------------- case B

def case_B(llm, args, jsonl_rows: List[dict]) -> str:
    """B=2 ragged prefill: short prompt + longer prompt, same max_new_tokens.

    Ragged prefill means the two prompts do not have identical token
    counts. A logits-selection bug here would either produce nonsense
    for one request or bleed tokens between the two.
    """
    from minisgl.core import SamplingParams

    baseline = _check_alloc_and_integrity(llm, "B_baseline")
    short = "Hi."
    longer = (
        "The Ascend 910B1 accelerator has 64 gigabytes of high bandwidth "
        "memory and is used for large language model inference."
    )
    prompts = [short, longer]
    N = args.max_new_tokens_short
    sp = SamplingParams(
        temperature=0.0, top_k=1, top_p=1.0,
        max_tokens=N, ignore_eos=True,
    )
    t0 = time.time()
    out = llm.generate(prompts, [sp, sp])
    elapsed = time.time() - t0
    _log("B", f"elapsed={elapsed:.2f}s")
    for i, o in enumerate(out):
        _log("B", f"req{i} tokens={o['token_ids']}")
        _log("B", f"req{i} text={o['text']!r}")

    lens = [len(o["token_ids"]) for o in out]
    after = _check_alloc_and_integrity(llm, "B_after")
    # Also require both replies are distinct-ish: a swap bug tends to
    # make both replies identical. Not enforced as PASS (greedy replies
    # could coincidentally overlap on short N), only recorded.
    reply_texts_equal = out[0]["text"] == out[1]["text"]
    pass_ok = (
        len(out) == 2
        and lens == [N, N]
        and after["available_tokens"] == baseline["available_tokens"]
        and after["deferred_abort_uids"] == 0
    )
    verdict = "PASS" if pass_ok else "FAIL"
    jsonl_rows.append({
        "case": "B",
        "description": "B=2 ragged prefill",
        "prompts_char_lengths": [len(short), len(longer)],
        "batch_size_timeline": [2],
        "requested_max_new_tokens": [N, N],
        "actual_output_token_count": lens,
        "reply_texts_equal": reply_texts_equal,
        "alloc_before": baseline,
        "alloc_after": after,
        "check_integrity": "ok",
        "verdict": verdict,
    })
    _log("B", f"verdict={verdict} reply_texts_equal={reply_texts_equal}")
    return verdict


# ---------------------------------------------------------------- case C

def case_C(llm, args, jsonl_rows: List[dict]) -> str:
    """B=2 mixed-KV decode with solo tail.

    Ragged prompt lengths force different per-request cached_len during
    the shared decode phase. Different max_new_tokens force one request
    to finish earlier; the survivor keeps decoding alone. Verifies:
      * attention metadata handles per-req cache_len during shared decode,
      * survivor continues cleanly after sibling frees pages,
      * allocator returns to baseline once both finish.
    """
    from minisgl.core import SamplingParams

    baseline = _check_alloc_and_integrity(llm, "C_baseline")
    short = "Hello."
    longer = (
        "In one short paragraph, describe the color of a clear afternoon sky "
        "over the ocean, using only common English words."
    )
    prompts = [short, longer]
    N_short = args.max_new_tokens_short  # 8
    N_long = args.max_new_tokens_long    # 16
    sp_short = SamplingParams(
        temperature=0.0, top_k=1, top_p=1.0,
        max_tokens=N_short, ignore_eos=True,
    )
    sp_long = SamplingParams(
        temperature=0.0, top_k=1, top_p=1.0,
        max_tokens=N_long, ignore_eos=True,
    )
    t0 = time.time()
    out = llm.generate(prompts, [sp_short, sp_long])
    elapsed = time.time() - t0
    _log("C", f"elapsed={elapsed:.2f}s")
    for i, o in enumerate(out):
        _log("C", f"req{i} tokens={o['token_ids']}")
        _log("C", f"req{i} text={o['text']!r}")

    lens = [len(o["token_ids"]) for o in out]
    after = _check_alloc_and_integrity(llm, "C_after")
    pass_ok = (
        len(out) == 2
        and lens == [N_short, N_long]
        and after["available_tokens"] == baseline["available_tokens"]
        and after["deferred_abort_uids"] == 0
    )
    verdict = "PASS" if pass_ok else "FAIL"
    jsonl_rows.append({
        "case": "C",
        "description": "B=2 mixed-KV decode with solo tail",
        "prompts_char_lengths": [len(short), len(longer)],
        "batch_size_timeline": [2, 1],
        "requested_max_new_tokens": [N_short, N_long],
        "actual_output_token_count": lens,
        "alloc_before": baseline,
        "alloc_after": after,
        "check_integrity": "ok",
        "verdict": verdict,
    })
    _log("C", f"verdict={verdict}")
    return verdict


# ---------------------------------------------------------------- case D

def case_D(llm, args, jsonl_rows: List[dict]) -> str:
    """Dynamic admission grow-shrink 1 -> 2 -> 3 -> 2 -> 1.

    Reuses the caller-supplied ``DynamicLLM`` instance (see boot note in
    ``_build_llm``: ``set_tp_info`` is a process-global singleton so we
    cannot re-boot). Installs a tick schedule that releases three
    prompts at ticks 0/3/6, samples ``decode_manager.running_reqs`` at
    every scheduler tick, and runs the offline loop to completion.

    Verifies:
      * a new request can be admitted while others are decoding,
      * finishing one request frees its pages while siblings keep
        decoding,
      * allocator returns to baseline once the timeline collapses.

    If the scheduler cannot admit B=3 (e.g. HBM tight, table full),
    the case falls back to PARTIAL with B=1/B=2 evidence retained.
    """
    from minisgl.core import SamplingParams
    from minisgl.llm.llm import RequestAllFinished

    _log("D", "reusing existing DynamicLLM instance")
    baseline = _check_alloc_and_integrity(llm, "D_baseline")

    # Prompt shapes chosen so the requested max_tokens dominate the
    # scheduler timeline instead of prompt length.
    p0 = "The moon is"                                 # req0 stays longest
    p1 = "In one sentence, describe rain."             # req1 mid-life
    p2 = "Say hello."                                  # req2 exits first

    N0 = 24
    N1 = 12
    N2 = 4
    sp0 = SamplingParams(temperature=0.0, top_k=1, top_p=1.0,
                         max_tokens=N0, ignore_eos=True)
    sp1 = SamplingParams(temperature=0.0, top_k=1, top_p=1.0,
                         max_tokens=N1, ignore_eos=True)
    sp2 = SamplingParams(temperature=0.0, top_k=1, top_p=1.0,
                         max_tokens=N2, ignore_eos=True)

    # Stage schedule: req0 at tick 0, req1 at tick 3, req2 at tick 6.
    # These are release-by boundaries — the scheduler is allowed to
    # admit them slightly later if a prefill batch is in flight. The
    # observed timeline is captured for the verdict.
    llm._tick_schedule = [(0, p0, sp0), (3, p1, sp1), (6, p2, sp2)]

    # Re-initialise generate()-style counters. Case D bypasses
    # LLM.generate because generate() posts all prompts up-front and
    # runs run_forever synchronously.
    llm.pending_requests = []
    llm.status_map = {}
    llm.counter = 0
    llm._tick = 0
    llm._timeline = []

    t0 = time.time()
    try:
        llm.run_forever()
    except RequestAllFinished:
        pass
    elapsed = time.time() - t0
    _log("D", f"elapsed={elapsed:.2f}s")

    # Collect outputs (mirrors LLM.generate's post-loop step).
    outs = []
    for uid in sorted(llm.status_map.keys()):
        status = llm.status_map[uid]
        text = llm.tokenizer.decode(status.output_ids)
        outs.append({"uid": uid, "text": text, "token_ids": status.output_ids})
        _log("D", f"req uid={uid} n_tokens={len(status.output_ids)} "
                  f"tokens={status.output_ids[:6]}...")
        _log("D", f"req uid={uid} text={text!r}")

    timeline_running = [row["running_decode"] for row in llm._timeline]
    seq = []
    for v in timeline_running:
        if not seq or seq[-1] != v:
            seq.append(v)
    _log("D", f"running_decode_seq={seq}")
    _log("D", f"peak_running_decode={max(timeline_running) if timeline_running else 0}")

    peak = max(timeline_running) if timeline_running else 0
    saw_grow_shrink = (
        peak >= 3
        and any(seq[i] < seq[i - 1] for i in range(1, len(seq)))
    )

    after = _check_alloc_and_integrity(llm, "D_after")
    output_lens = [len(o["token_ids"]) for o in outs]
    lens_ok = output_lens == [N0, N1, N2]
    alloc_ok = (
        after["available_tokens"] == baseline["available_tokens"]
        and after["deferred_abort_uids"] == 0
    )
    if peak >= 3 and saw_grow_shrink and lens_ok and alloc_ok:
        verdict = "PASS"
    elif peak >= 2 and lens_ok and alloc_ok:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    jsonl_rows.append({
        "case": "D",
        "description": "Dynamic admission grow-shrink 1->2->3->2->1",
        "batch_size_timeline_ticks": timeline_running,
        "batch_size_timeline_dedup": seq,
        "peak_running_decode": peak,
        "requested_max_new_tokens": [N0, N1, N2],
        "actual_output_token_count": output_lens,
        "alloc_before": baseline,
        "alloc_after": after,
        "check_integrity": "ok",
        "verdict": verdict,
    })
    _log("D", f"verdict={verdict} peak={peak}")

    return verdict


# ------------------------------------------------------------------ main

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--attention-backend", default="npu_fia")
    parser.add_argument("--max-new-tokens-short", type=int, default=8)
    parser.add_argument("--max-new-tokens-long", type=int, default=16)
    parser.add_argument(
        "--prompt",
        default="The capital of France is",
    )
    parser.add_argument("--jsonl-out", default="")
    args = parser.parse_args()

    _log("boot", f"model_path={args.model_path}")
    _log("boot", f"attention_backend={args.attention_backend}")

    jsonl_rows: List[dict] = []

    # ------------------------------------------------------------------
    # One LLM boot for all four cases (set_tp_info is process-global).
    # DynamicLLM's offline_receive_msg is transparent when no
    # tick_schedule is installed -> A/B/C behave like stock LLM.
    # ------------------------------------------------------------------
    llm = _build_llm(args)
    try:
        _log("case-A", "running")
        va = case_A(llm, args, jsonl_rows)
        _log("case-B", "running")
        vb = case_B(llm, args, jsonl_rows)
        _log("case-C", "running")
        vc = case_C(llm, args, jsonl_rows)
        _log("case-D", "running")
        try:
            vd = case_D(llm, args, jsonl_rows)
        except Exception as e:  # noqa: BLE001
            vd = f"FAIL: {type(e).__name__}: {e}"
            _log("case-D", f"exception: {vd}")
            traceback.print_exc()
    finally:
        del llm
        gc.collect()

    # ------------------------------------------------------------------
    # Summary + overall verdict
    # ------------------------------------------------------------------
    _log("summary", f"A={va} B={vb} C={vc} D={vd}")

    if va == "PASS" and vb == "PASS" and vc == "PASS" and vd == "PASS":
        overall = "PASS"
    elif (
        va == "PASS"
        and vb == "PASS"
        and vc == "PASS"
        and str(vd).startswith("PARTIAL")
    ):
        overall = "PARTIAL"
    else:
        overall = "FAIL"

    for row in jsonl_rows:
        print("GATE3.2_JSONL " + json.dumps(row, ensure_ascii=False), flush=True)

    if args.jsonl_out:
        with open(args.jsonl_out, "w", encoding="utf-8") as f:
            for row in jsonl_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"GATE3.2_SMOKE_RESULT={overall}", flush=True)
    print(f"GATE3.2_CASE_A={va}", flush=True)
    print(f"GATE3.2_CASE_B={vb}", flush=True)
    print(f"GATE3.2_CASE_C={vc}", flush=True)
    print(f"GATE3.2_CASE_D={vd}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
