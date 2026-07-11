"""Gate 3.4 — TP=1 timing baseline on Ascend 910B1.

Not a hermetic pytest: talks to the real NPU. Committed as evidence for
the Gate 3.4 verdict. This gate records a **reproducible timing
snapshot**, not a performance optimisation and not a comparison with
any other framework.

Envelope (identical to Gate 3.1 / 3.2 / 3.3, frozen):

    Hardware:          Ascend 910B1 (1 die)
    Parallelism:       TP=1
    Execution:         eager (cuda_graph_bs=[])
    Attention backend: npu_fia
    Sampling:          greedy (temperature=0.0, top_k=1, top_p=1.0,
                       ignore_eos=True)

Models (only local paths already on the Ascend host):

    Qwen3-0.6B
    Qwen3-1.7B

Case matrix (per model):

    A. B=1  max_new_tokens=8    prefill + 8-step decode
    B. B=1  max_new_tokens=16   prefill + 16-step decode
    C. B=2  max_new_tokens=8    equal-length batching
    D. B=2  max_new_tokens=8    ragged prefill (short + long)

Per model, per case:
    warmup_count      = 1
    measured_repeats  = 3

Per-repeat JSONL row captures (spec §3):

    model_name, model_path, case_name, prompt_lengths, batch_size,
    requested_max_new_tokens, actual_output_tokens_per_request,
    warmup_count, repeat_id,
    ttft_ms, e2e_latency_ms,
    output_tokens_total, tokens_per_second, ms_per_output_token,
    baseline_available_tokens, available_tokens_after_case,
    deferred_abort_uids, cache_integrity_ok,
    status, failure_reason

Per-repeat PASS predicate:

    len(output) == B and lens == requested and
    available_tokens_after_case == baseline_available_tokens and
    deferred_abort_uids == 0 and cache_integrity_ok

Summary printed after all cases per model:

    median / min / max of ttft_ms, e2e_latency_ms,
    tokens_per_second, ms_per_output_token — over the 3 measured
    repeats only (warmup excluded).

Process model:

    ``minisgl.distributed.info.set_tp_info`` is a process-global
    singleton. Each model runs in its own child subprocess.

Footer lines (scraped by the verdict generator):

    GATE3.4_MODEL_<name>=PASS|PARTIAL|FAIL
    GATE3.4_TIMING_RESULT=PASS|PARTIAL|BLOCKED

TTFT instrumentation:

    A ``TimingLLM`` subclass overrides ``offline_send_result``. That
    method is called every time the tokenizer worker delivers one or
    more ``DetokenizeMsg`` chunks to the offline driver. For each
    delivered msg, we record ``time.perf_counter()`` per uid the very
    first time we see a token appended to that request's output_ids.
    TTFT[uid] = t_first_token[uid] - t_start, where t_start is sampled
    just before ``llm.generate()`` returns control to the scheduler
    loop. This is a real per-request wall-time measurement (not an
    ex-post derivation from prefill_budget).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from typing import List, Dict


def _log(section: str, msg: str) -> None:
    print(f"[gate3.4][{section}] {msg}", flush=True)


# ---------------------------------------------------------------- helpers

def _snapshot_alloc(scheduler) -> dict:
    cm = scheduler.cache_manager
    return {
        "free_pages": len(cm.free_slots),
        "available_tokens": cm.available_size,
        "total_pages": cm.num_pages,
        "deferred_abort_uids": len(scheduler.deferred_abort_uids),
    }


def _check_alloc_and_integrity(scheduler, tag: str) -> dict:
    snap = _snapshot_alloc(scheduler)
    ok = True
    try:
        scheduler.cache_manager.check_integrity()
    except Exception as e:  # noqa: BLE001
        ok = False
        _log("alloc", f"{tag} check_integrity FAILED: {type(e).__name__}: {e}")
    snap["check_integrity_ok"] = ok
    _log(
        "alloc",
        f"{tag} free_pages={snap['free_pages']} "
        f"available_tokens={snap['available_tokens']} "
        f"total_pages={snap['total_pages']} "
        f"deferred_abort_uids={snap['deferred_abort_uids']} "
        f"integrity_ok={ok}",
    )
    return snap


def _model_name_from_path(path: str) -> str:
    return os.path.basename(os.path.normpath(path))


# ---------------------------------------------------------------- TimingLLM

# Deferred import: this module can be imported outside the container for
# static checks. The Ascend runtime imports occur when the script boots.
try:
    from minisgl.llm.llm import LLM as _LLM
except Exception:  # pragma: no cover
    _LLM = None  # type: ignore[assignment]


class TimingLLM(_LLM if _LLM is not None else object):  # type: ignore[misc]
    """LLM subclass that records TTFT via the tokenizer -> driver callback.

    The parent ``LLM.offline_send_result`` is called every scheduler
    tick that produced a token, with a list of ``DetokenizeMsg`` items
    (one per uid that produced a token that tick). Each such call, if
    the timer is armed, is a wall-clock event.

      * ``_first_token_time[uid]`` records the very first tick that
        contributed a token for that uid. ``TTFT[uid] = _first_token_time
        - _t_start``.
      * ``_last_msg_time`` records the most recent token-producing tick;
        together with ``_t_start`` and ``_t_end`` (sampled by the caller
        around ``generate()``) it lets the caller reconstruct decode
        throughput without piercing the scheduler.

    ``_arm_timing()`` MUST be called immediately before each measured
    ``generate()`` invocation. Warmups also arm the timer so the sink
    stays consistent, but the caller drops warmup rows.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._t_start: float | None = None
        self._first_token_time: Dict[int, float] = {}
        self._last_msg_time: float | None = None

    def _arm_timing(self, t_start: float) -> None:
        self._t_start = t_start
        self._first_token_time = {}
        self._last_msg_time = None

    def offline_send_result(self, reply):  # type: ignore[override]
        # Delegate to base first so status.output_ids is up-to-date.
        super().offline_send_result(reply)
        if self._t_start is None:
            return
        if not reply:
            return
        now = time.perf_counter()
        self._last_msg_time = now
        for msg in reply:
            uid = getattr(msg, "uid", None)
            if uid is None:
                continue
            if uid not in self._first_token_time:
                self._first_token_time[uid] = now


def _build_llm(args):
    """Boot the TimingLLM under the Gate 3.1/3.2/3.3/3.4 frozen envelope."""
    import torch

    t0 = time.time()
    llm = TimingLLM(
        model_path=args.model_path,
        dtype=torch.bfloat16,
        attention_backend=args.attention_backend,
        max_running_req=8,
        memory_ratio=0.85,
        cuda_graph_bs=[],
        page_size=16,
    )
    _log("boot", f"LLM initialised in {time.time() - t0:.2f}s")
    return llm


# ---------------------------------------------------------------- one repeat

def _run_repeat(
    llm,
    model_name: str,
    model_path: str,
    case_name: str,
    description: str,
    prompts: List[str],
    requested_max_new_tokens: List[int],
    warmup_count: int,
    repeat_id: int,   # -1 = warmup; 0..N-1 = measured
    jsonl_rows: List[dict],
) -> str:
    """Run one repeat of a case. Returns PASS/FAIL. Warmup rows are still
    emitted with ``status=WARMUP`` for evidence but do not count toward
    the median summary.
    """
    from minisgl.core import SamplingParams

    is_warmup = repeat_id < 0
    tag_id = "warmup" if is_warmup else f"r{repeat_id}"
    tag = f"{case_name}[{model_name}][{tag_id}]"
    baseline = _check_alloc_and_integrity(llm, f"{tag}_baseline")

    sps = [
        SamplingParams(
            temperature=0.0, top_k=1, top_p=1.0,
            max_tokens=n, ignore_eos=True,
        )
        for n in requested_max_new_tokens
    ]

    failure_reason: str | None = None
    lens: List[int] = []
    ttfts_ms: Dict[int, float] = {}
    t_start = 0.0
    e2e_ms = 0.0
    output_tokens_total = 0

    try:
        # Arm the timer and snapshot t_start immediately before generate().
        t_start = time.perf_counter()
        llm._arm_timing(t_start)
        out = llm.generate(prompts, sps)
        t_end = time.perf_counter()
        e2e_ms = (t_end - t_start) * 1000.0

        for uid, first_t in llm._first_token_time.items():
            ttfts_ms[uid] = (first_t - t_start) * 1000.0

        lens = [len(o["token_ids"]) for o in out]
        output_tokens_total = sum(lens)
        output_len = len(out)
        for i, o in enumerate(out):
            _log(tag, f"req{i} n_tokens={len(o['token_ids'])} text={o['text']!r}")
    except Exception as e:  # noqa: BLE001
        failure_reason = f"{type(e).__name__}: {e}"
        _log(tag, f"generate raised: {failure_reason}")
        traceback.print_exc()
        output_len = 0
    finally:
        # Disarm to avoid stale sampling if the next call forgets to arm.
        llm._t_start = None

    after = _check_alloc_and_integrity(llm, f"{tag}_after")

    # Metrics
    ttft_ms = max(ttfts_ms.values()) if ttfts_ms else 0.0
    tokens_per_second = (output_tokens_total / (e2e_ms / 1000.0)) if e2e_ms > 0 else 0.0
    ms_per_output_token = (e2e_ms / output_tokens_total) if output_tokens_total > 0 else 0.0

    pass_ok = (
        failure_reason is None
        and output_len == len(requested_max_new_tokens)
        and lens == requested_max_new_tokens
        and after["available_tokens"] == baseline["available_tokens"]
        and after["deferred_abort_uids"] == 0
        and after.get("check_integrity_ok", False)
    )
    verdict = ("WARMUP" if is_warmup and pass_ok
               else "PASS" if pass_ok
               else "FAIL")

    if not pass_ok and failure_reason is None:
        reasons: List[str] = []
        if output_len != len(requested_max_new_tokens):
            reasons.append(
                f"output_count={output_len} expected={len(requested_max_new_tokens)}"
            )
        if lens != requested_max_new_tokens:
            reasons.append(f"lens={lens} expected={requested_max_new_tokens}")
        if after["available_tokens"] != baseline["available_tokens"]:
            reasons.append(
                f"available_tokens {baseline['available_tokens']} -> "
                f"{after['available_tokens']}"
            )
        if after["deferred_abort_uids"] != 0:
            reasons.append(f"deferred_abort_uids={after['deferred_abort_uids']}")
        if not after.get("check_integrity_ok", False):
            reasons.append("check_integrity_failed")
        failure_reason = "; ".join(reasons) or None

    row = {
        "model_name": model_name,
        "model_path": model_path,
        "case_name": case_name,
        "description": description,
        "prompt_lengths_chars": [len(p) for p in prompts],
        "batch_size": len(requested_max_new_tokens),
        "requested_max_new_tokens": requested_max_new_tokens,
        "actual_output_tokens_per_request": lens,
        "warmup_count": warmup_count,
        "repeat_id": repeat_id,
        "is_warmup": is_warmup,
        "ttft_ms": ttft_ms,
        "ttft_ms_per_uid": {str(k): v for k, v in ttfts_ms.items()},
        "e2e_latency_ms": e2e_ms,
        "output_tokens_total": output_tokens_total,
        "tokens_per_second": tokens_per_second,
        "ms_per_output_token": ms_per_output_token,
        "baseline_available_tokens": baseline["available_tokens"],
        "available_tokens_after_case": after["available_tokens"],
        "baseline_free_pages": baseline["free_pages"],
        "free_pages_after_case": after["free_pages"],
        "total_pages": baseline["total_pages"],
        "deferred_abort_uids": after["deferred_abort_uids"],
        "cache_integrity_ok": after.get("check_integrity_ok", False),
        "status": verdict,
        "failure_reason": failure_reason,
    }
    jsonl_rows.append(row)
    _log(
        tag,
        f"verdict={verdict} ttft_ms={ttft_ms:.2f} "
        f"e2e_ms={e2e_ms:.2f} tps={tokens_per_second:.2f} "
        f"ms_per_tok={ms_per_output_token:.2f} tokens_total={output_tokens_total}",
    )
    return verdict


# ---------------------------------------------------------------- cases
#
# Cases A / B / C reuse the same short prompt across warmup + measured
# repeats — the resulting radix-cache retention is fully cached
# (cached_len == prompt_len), so the FIA path stays on the supported
# extend_len ∈ {0, 1} branches at every repeat.
#
# Case D (B=2 ragged prefill) requires per-repeat prompt content to
# stay on the supported FIA branch. If the SAME ragged prompt pair is
# reused across repeats, the long prompt's first page (16 tokens) is
# retained in the radix cache; the next repeat sees cached_len=16 AND
# extend_len>1 for the same request, which triggers Gate 2.2f's
# explicit ``NotImplementedError`` for "ragged batches with a non-zero
# cached_len and extend_len>1". That combination is a documented Ascend
# FIA port limitation, NOT a scheduler / allocator bug.
#
# Standard benchmark methodology for timing baselines uses distinct
# input content per repeat to eliminate cache-hit artifacts. Case D
# therefore supplies a small pool of (short, long) prompt variants
# indexed by repeat id. All variants share the same shape (very short
# prompt + long English paragraph, N=[8, 8]) — only the words differ.


def _case_D_prompts_for(variant_id: int) -> List[str]:
    """Return the (short, long) prompt pair for a given repeat/variant id.

    ``variant_id`` is monotonically increasing across warmup + repeats
    (warmup uses id=0, r0 uses id=1, ...). This ensures each measurement
    tick sees fresh content so the FIA ragged-prefill path stays on its
    supported cached_len==0 branch (see Gate 2.2f). The shape is
    preserved: a very short prompt + a long English paragraph, both bf16
    tokenized, N=[8, 8].
    """
    short_pool = ["Hi.", "Hello.", "Hey there.", "Greetings."]
    long_pool = [
        "The Ascend 910B1 accelerator has 64 gigabytes of high bandwidth "
        "memory and is used for large language model inference.",
        "Modern paged attention manages the key-value cache as fixed-size "
        "pages, letting the scheduler batch requests with unequal prompt "
        "lengths without wasting device memory on padding.",
        "A transformer decoder layer applies self-attention followed by a "
        "feed-forward network, both wrapped in residual connections and "
        "layer normalization, with rotary position encoding for the queries.",
        "Bfloat16 is preferred over float16 for large transformer inference "
        "because its wider dynamic range prevents overflow in softmax and "
        "gelu without needing loss-scaling tricks at every attention layer.",
    ]
    idx = variant_id % len(short_pool)
    return [short_pool[idx], long_pool[idx]]


_CASES = [
    dict(
        name="A",
        description="B=1 single request, N=8",
        prompts=["The capital of France is"],
        max_new_tokens=[8],
        per_repeat_prompts=False,
    ),
    dict(
        name="B",
        description="B=1 single request, N=16",
        prompts=["The capital of France is"],
        max_new_tokens=[16],
        per_repeat_prompts=False,
    ),
    dict(
        name="C",
        description="B=2 equal-length batching, N=8",
        prompts=["The capital of France is", "The capital of France is"],
        max_new_tokens=[8, 8],
        per_repeat_prompts=False,
    ),
    dict(
        name="D",
        description=(
            "B=2 ragged prefill (short + long), N=8 — per-repeat prompt "
            "variants to keep the FIA path on its supported "
            "cached_len==0 ragged-prefill branch (Gate 2.2f)."
        ),
        prompts=None,  # supplied per repeat via _case_D_prompts_for()
        max_new_tokens=[8, 8],
        per_repeat_prompts=True,
    ),
]


# ---------------------------------------------------------------- summary

def _summarise(jsonl_rows: List[dict]) -> dict:
    """Group measured rows by (model, case) and compute median/min/max."""
    from collections import defaultdict

    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for row in jsonl_rows:
        if row.get("is_warmup"):
            continue
        if row.get("status") != "PASS":
            continue
        groups[(row["model_name"], row["case_name"])].append(row)

    summary: Dict[str, dict] = {}
    for (model_name, case_name), rows in groups.items():
        ttfts = [r["ttft_ms"] for r in rows]
        e2es = [r["e2e_latency_ms"] for r in rows]
        tps = [r["tokens_per_second"] for r in rows]
        mspt = [r["ms_per_output_token"] for r in rows]
        summary[f"{model_name}::{case_name}"] = {
            "model_name": model_name,
            "case_name": case_name,
            "n_measured": len(rows),
            "ttft_ms":            {"median": statistics.median(ttfts),
                                   "min": min(ttfts), "max": max(ttfts)},
            "e2e_latency_ms":     {"median": statistics.median(e2es),
                                   "min": min(e2es), "max": max(e2es)},
            "tokens_per_second":  {"median": statistics.median(tps),
                                   "min": min(tps), "max": max(tps)},
            "ms_per_output_token":{"median": statistics.median(mspt),
                                   "min": min(mspt), "max": max(mspt)},
        }
    return summary


# ---------------------------------------------------------------- single-model driver

def run_single_model(args) -> int:
    model_name = _model_name_from_path(args.model_path)
    _log("boot", f"model_name={model_name}")
    _log("boot", f"model_path={args.model_path}")
    _log("boot", f"attention_backend={args.attention_backend}")
    _log("boot",
         f"warmup_count={args.warmup_count} measured_repeats={args.measured_repeats}")

    jsonl_rows: List[dict] = []
    case_verdicts: Dict[str, str] = {}

    llm = None
    try:
        llm = _build_llm(args)

        for case in _CASES:
            case_name = case["name"]
            _log(f"case-{case_name}", f"start ({case['description']})")

            # variant_id counts monotonically across warmup + measured
            # repeats for cases that need per-repeat prompt variants.
            variant_id = 0

            def _prompts_for(case_dict, vid: int) -> List[str]:
                if not case_dict.get("per_repeat_prompts"):
                    return list(case_dict["prompts"])
                if case_dict["name"] == "D":
                    return _case_D_prompts_for(vid)
                return list(case_dict["prompts"])

            # ----- warmup(s) — arms timer but rows are marked is_warmup=True
            for w in range(args.warmup_count):
                _run_repeat(
                    llm,
                    model_name=model_name,
                    model_path=args.model_path,
                    case_name=case_name,
                    description=case["description"],
                    prompts=_prompts_for(case, variant_id),
                    requested_max_new_tokens=list(case["max_new_tokens"]),
                    warmup_count=args.warmup_count,
                    repeat_id=-1 - w,
                    jsonl_rows=jsonl_rows,
                )
                variant_id += 1

            # ----- measured repeats
            all_pass = True
            for r in range(args.measured_repeats):
                v = _run_repeat(
                    llm,
                    model_name=model_name,
                    model_path=args.model_path,
                    case_name=case_name,
                    description=case["description"],
                    prompts=_prompts_for(case, variant_id),
                    requested_max_new_tokens=list(case["max_new_tokens"]),
                    warmup_count=args.warmup_count,
                    repeat_id=r,
                    jsonl_rows=jsonl_rows,
                )
                if v != "PASS":
                    all_pass = False
                variant_id += 1
            case_verdicts[case_name] = "PASS" if all_pass else "FAIL"
            _log(f"case-{case_name}", f"case verdict={case_verdicts[case_name]}")
    finally:
        if llm is not None:
            del llm
        gc.collect()

    # ---------- summary
    summary = _summarise(jsonl_rows)
    for key, s in summary.items():
        _log(
            "median",
            f"{key} n={s['n_measured']} "
            f"ttft_ms={s['ttft_ms']['median']:.2f} "
            f"e2e_ms={s['e2e_latency_ms']['median']:.2f} "
            f"tps={s['tokens_per_second']['median']:.2f} "
            f"ms_per_tok={s['ms_per_output_token']['median']:.2f}",
        )

    model_verdict = "PASS" if all(v == "PASS" for v in case_verdicts.values()) else "FAIL"

    # Emit JSONL rows and summary via scrape-friendly prefixes.
    for row in jsonl_rows:
        print("GATE3.4_JSONL " + json.dumps(row, ensure_ascii=False), flush=True)
    for key, s in summary.items():
        print("GATE3.4_SUMMARY " + json.dumps({key: s}, ensure_ascii=False), flush=True)

    if args.jsonl_out:
        with open(args.jsonl_out, "w", encoding="utf-8") as f:
            for row in jsonl_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for case_name, verdict in case_verdicts.items():
        print(f"GATE3.4_MODEL_{model_name}_CASE_{case_name}={verdict}", flush=True)
    print(f"GATE3.4_MODEL_{model_name}={model_verdict}", flush=True)
    return 0 if model_verdict == "PASS" else 1


# ---------------------------------------------------------------- parent orchestrator

def run_matrix(args) -> int:
    model_paths = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_paths:
        _log("boot", "no --models paths provided; nothing to do")
        print("GATE3.4_TIMING_RESULT=BLOCKED", flush=True)
        return 2

    _log("boot", f"model_paths={model_paths}")
    _log("boot", f"warmup_count={args.warmup_count} "
                 f"measured_repeats={args.measured_repeats}")
    _log("boot", f"jsonl_out={args.jsonl_out or '<stdout only>'}")

    aggregated_rows: List[dict] = []
    aggregated_summary: Dict[str, dict] = {}
    model_verdicts: Dict[str, str] = {}
    model_case_verdicts: Dict[str, Dict[str, str]] = {}

    env = os.environ.copy()

    for model_path in model_paths:
        model_name = _model_name_from_path(model_path)
        _log("matrix", f"--- START model={model_name} ---")

        child_cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--single",
            "--model-path", model_path,
            "--attention-backend", args.attention_backend,
            "--warmup-count", str(args.warmup_count),
            "--measured-repeats", str(args.measured_repeats),
        ]
        _log("matrix", f"child_cmd={' '.join(child_cmd)}")

        t0 = time.time()
        proc = subprocess.run(
            child_cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        elapsed = time.time() - t0
        _log("matrix", f"child model={model_name} rc={proc.returncode} "
                       f"elapsed={elapsed:.2f}s")

        for line in proc.stdout.splitlines():
            print(line, flush=True)

        cases_for_this_model: Dict[str, str] = {}
        for line in proc.stdout.splitlines():
            if line.startswith("GATE3.4_JSONL "):
                try:
                    row = json.loads(line[len("GATE3.4_JSONL "):])
                except json.JSONDecodeError:
                    continue
                aggregated_rows.append(row)
            elif line.startswith("GATE3.4_SUMMARY "):
                try:
                    payload = json.loads(line[len("GATE3.4_SUMMARY "):])
                except json.JSONDecodeError:
                    continue
                aggregated_summary.update(payload)
            elif line.startswith(f"GATE3.4_MODEL_{model_name}_CASE_"):
                key = line.split("=", 1)[0]
                letter = key.rsplit("_CASE_", 1)[1]
                cases_for_this_model[letter] = line.split("=", 1)[1]
            elif line.startswith(f"GATE3.4_MODEL_{model_name}="):
                model_verdicts[model_name] = line.split("=", 1)[1]

        model_case_verdicts[model_name] = cases_for_this_model
        _log("matrix", f"--- END model={model_name} ---")

    if args.jsonl_out:
        with open(args.jsonl_out, "w", encoding="utf-8") as f:
            for row in aggregated_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        _log("matrix", f"wrote {len(aggregated_rows)} JSONL rows -> {args.jsonl_out}")

    # Verdict:
    #   PASS     — every model completes A/B/C/D with 3 measured repeats,
    #              every measured repeat produced timing metrics, and
    #              allocator returned to baseline after every repeat.
    #   PARTIAL  — one model passes A-D, the other has a documented
    #              case-level failure recorded in the row.
    #   BLOCKED  — neither model completed even the B=1 baseline.
    passing_models = [m for m, v in model_verdicts.items() if v == "PASS"]
    if len(passing_models) == len(model_paths) and passing_models:
        overall = "PASS"
    elif not model_verdicts:
        overall = "BLOCKED"
    elif len(passing_models) >= 1:
        overall = "PARTIAL"
    else:
        any_baseline_pass = any(
            cases.get("A") == "PASS" or cases.get("B") == "PASS"
            for cases in model_case_verdicts.values()
        )
        overall = "PARTIAL" if any_baseline_pass else "BLOCKED"

    for model_name, verdict in model_verdicts.items():
        cases = model_case_verdicts.get(model_name, {})
        _log(
            "summary",
            f"model={model_name} verdict={verdict} cases={cases}",
        )
        for letter in ("A", "B", "C", "D"):
            print(
                f"GATE3.4_MODEL_{model_name}_CASE_{letter}={cases.get(letter, 'MISSING')}",
                flush=True,
            )
        print(f"GATE3.4_MODEL_{model_name}={verdict}", flush=True)

    for key, s in aggregated_summary.items():
        print("GATE3.4_SUMMARY " + json.dumps({key: s}, ensure_ascii=False), flush=True)

    print(f"GATE3.4_TIMING_RESULT={overall}", flush=True)
    return 0 if overall == "PASS" else (1 if overall == "PARTIAL" else 2)


# ------------------------------------------------------------------ main

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single",
        action="store_true",
        help="Child mode: run cases A-D against a single --model-path.",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="(child mode) Absolute path to one local model directory.",
    )
    parser.add_argument(
        "--models",
        default="",
        help=(
            "(parent mode) Comma-separated list of model paths. Spawns one "
            "child subprocess per model because "
            "``minisgl.distributed.info.set_tp_info`` is a process-global "
            "singleton."
        ),
    )
    parser.add_argument("--attention-backend", default="npu_fia")
    parser.add_argument("--warmup-count", type=int, default=1)
    parser.add_argument("--measured-repeats", type=int, default=3)
    parser.add_argument(
        "--jsonl-out",
        default="",
        help="Optional path to write aggregated JSONL rows.",
    )
    args = parser.parse_args()

    if args.single:
        if not args.model_path:
            print("--single requires --model-path", file=sys.stderr, flush=True)
            return 2
        return run_single_model(args)

    if not args.models:
        print("--models is required in parent mode", file=sys.stderr, flush=True)
        return 2
    return run_matrix(args)


if __name__ == "__main__":
    sys.exit(main())
