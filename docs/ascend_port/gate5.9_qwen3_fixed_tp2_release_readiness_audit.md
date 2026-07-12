# Gate 5.9 Verdict — Qwen3 fixed-TP2 release-readiness audit

**Gate ID:** 5.9 (Qwen3 fixed-TP2 release-readiness audit for the
prospective `v0.2.0a1` technical-preview candidate)
**Verdict:** PASS *(conditional on external credential-rotation /
git-history-hygiene decision, see §9)*
**Branch:** `gate5.9-qwen3-fixed-tp2-release-readiness-audit`
**Base commit:** `3ceff55` (tip of `ascend-port`, Gate 5.8 merge)
**Audited HEAD:** `3ceff55` (identical — audit branch only adds
this doc)
**Date:** 2026-07-12
**Kind:** Documentation-only release-readiness audit. No new
experiments, no runtime edits, no tests edits, no scripts edits,
no release-tag mutations, no GitHub Release mutations, no
history rewrite, no force-push. Answers a single yes/no
question: is the current `ascend-port` HEAD ready for a
`v0.2.0a1` technical-preview release, subject to whatever
external credential-rotation / history-hygiene decision the
owner chooses to attach to it?

> **This is a Qwen3 fixed-TP2 release-readiness audit.**
> **It is not a release.**
> **It does not create `v0.2.0a1`.**
> **It does not push a GitHub Release.**
> **It does not modify `CHANGELOG.md` or add a release entry.**
> **It does not modify the existing `v0.1.0a1` tag or its
>  GitHub Release.**
> The audit only produces this verdict file; the release-cut
> decision remains with the human owner.

---

## 1. Verdict summary

**Release-readiness decision: PASS (conditional).**

The audited HEAD (`3ceff55`) is documentation-consistent, has
zero tracked-HEAD credential leaks, links every capability claim
to a frozen evidence file, and makes no benchmark / TP
elasticity / TP switching / cross-stack / performance-
superiority claims. A `v0.2.0a1` technical-preview release from
this HEAD is technically viable.

The **conditional** qualifier is one item that this gate has no
authority to resolve, and that the release cutter must
acknowledge before pushing a new tag / GitHub Release:

> Whether the repository's Git *history* (as distinct from the
> tracked HEAD) still contains historical raw credential leaks
> from before the Gate 4.13a / 4.13b redaction sweep, and, if
> so, whether the exposed credentials have been rotated
> externally.

This audit does **not** attempt to enumerate, quote, or verify
those historical substrings. See §9.

## 2. Release candidate scope

If cut, the prospective `v0.2.0a1` would package (at
`3ceff55`):

* Runtime source under `python/minisgl/` at HEAD (delta from
  `v0.1.0a1` is 3 files, +85 / −24 lines — the `python/minisgl/
  engine/config.py` + `llm/llm.py` + `server/api_server.py`
  changes introduced by the Gate 4.1 TP=2 bring-up commit
  `8431743`; no runtime source was touched anywhere in Gate 4.2
  – 4.15 or Gate 5.1 – 5.8)
* Two Gate-freeze bring-up drivers under `scripts/`
  (`gate5_6_qwen4b_tp2_timing_baseline.py`,
  `gate5_7_fixed_tp2_qwen3_three_model_matrix.py`) — script-only
  reproducers, not runtime
* Two hermetic tests under `tests/misc/`
  (`test_exposed_path_abort_ack.py`,
  `test_shell_cancel_cleanup.py`)
* The Gate 3.4 / 4.1 – 4.14 / 4.15 / 4.15a / 5.1 – 5.7 / 5.7a /
  5.8 verdict corpus under `docs/ascend_port/`
* Milestone summary `docs/ascend_port/fixed_tp2_adaptation
  _milestone.md`
* README fixed-TP2 status paragraph and reference table

Commits between `v0.1.0a1` (`a09efd3`, dated 2026-07-06) and
HEAD: 87 (verified via `git rev-list --count v0.1.0a1..HEAD`).
Files touched between `v0.1.0a1` and HEAD: 65 files, ~+27 267
insertions / −24 deletions. Documentation and Gate verdicts
dominate the diff.

## 3. Evidence inventory

The claim "Qwen3-0.6B, Qwen3-1.7B, and Qwen3-4B all pass the
fixed-TP2 functional matrix (A / B / C / D / E / F)" is
supported by the following frozen verdicts and per-capability
gates:

**Qwen3-0.6B (Gate 4.1 – 4.7, plus Gate 5.7 as unified matrix)**
* Gate 4.1 TP=2 init + B=1 single-request
* Gate 4.2 B=2 equal-length
* Gate 4.3 B=2 ragged prefill
* Gate 4.4 B=2 mixed-KV decode
* Gate 4.5 dynamic admission B: 1 → 2 → 1
* Gate 4.6 dynamic grow-shrink B: 1 → 2 → 3 → 2 → 1
* Gate 4.7 fixed-TP2 timing snapshot (not a benchmark)
* Gate 5.7 §6 three-model unified matrix — Qwen3-0.6B column

**Qwen3-1.7B (Gate 4.8 – 4.13, plus Gate 5.7 as unified matrix)**
* Gate 4.8 TP=2 init + B=1 single-request
* Gate 4.9 B=2 equal-length
* Gate 4.10 B=2 ragged prefill
* Gate 4.11 B=2 mixed-KV decode
* Gate 4.12 dynamic admission B: 1 → 2 → 1
* Gate 4.13 fixed-TP2 timing snapshot (not a benchmark)
* Gate 5.7 §6 three-model unified matrix — Qwen3-1.7B column

**Qwen3-4B (Gate 5.1 – 5.6, plus Gate 5.7 as unified matrix)**
* Gate 5.1 TP=2 init + B=1 single-request
* Gate 5.2 B=2 equal-length
* Gate 5.3 B=2 ragged prefill
* Gate 5.4 B=2 mixed-KV decode
* Gate 5.5 dynamic admission B: 1 → 2 → 1
* Gate 5.6 fixed-TP2 timing snapshot (not a benchmark)
* Gate 5.7 §6 three-model unified matrix — Qwen3-4B column

**Cross-model unified matrix (single driver, per-model process
isolation):**
* Gate 5.7 verdict — 36 records total (3 models × 6 cases ×
  2 ranks), all `status == "PASS"`, all
  `available_tokens_after_case == baseline`, all
  `deferred_abort_uids == 0`, all
  `cache_integrity_ok == true`, all rank-0/rank-1 outputs
  byte-identical

**Two-model capability matrix (Qwen3-0.6B + Qwen3-1.7B):**
* Gate 4.14 verdict

**Redaction / hygiene:**
* Gate 4.13a — targeted redaction (Gate 4.13 verdict only)
* Gate 4.13b — repo-wide tracked-HEAD redaction sweep
* Gate 4.15a — self-audit redaction of grep patterns

**Milestone consolidation:**
* Gate 4.15 — initial fixed-TP2 milestone doc (two-model scope)
* Gate 5.8 — milestone extended to Qwen3-4B, links Gate 5.7 as
  three-model unified evidence

Every capability claim in README §"Fixed-TP2 adaptation status"
and in `fixed_tp2_adaptation_milestone.md` links to one or more
of the above frozen verdicts. No orphan claim was found.

## 4. Public README / milestone consistency audit

| Check | Result |
|---|---|
| README fixed-TP2 status paragraph names all three models (Qwen3-0.6B / 1.7B / 4B) | PASS |
| README fixed-TP2 status paragraph disclaims TP elasticity / TP switching / benchmark / cross-stack comparison | PASS |
| README reference table links `fixed_tp2_adaptation_milestone.md` | PASS |
| README reference table links Gate 4.14 (two-model matrix) verdict | PASS |
| README reference table links Gate 5.7 (three-model matrix) verdict | PASS |
| Milestone doc header status names Gate 4.1 – 4.14 + Gate 5.1 – 5.7 corpus | PASS |
| Milestone doc Models table lists all three models with paths | PASS |
| Milestone doc Covered-capabilities table has Qwen3-4B column | PASS |
| Milestone doc explicit A / B / C / D / E / F case listing present | PASS |
| Milestone doc Evidence-source list links Gate 5.1 – Gate 5.7 verdicts | PASS |
| Milestone doc Non-goals excludes Qwen3-14B / 32B / quantized / MoE, TP > 2, TP elasticity, benchmark, cross-stack, upstream-merge | PASS |
| Milestone doc Closure paragraph names all three models and combined Gate 4 + Gate 5 evidence | PASS |
| Milestone doc removes Qwen3-4B from Non-goals now that it is in scope | PASS |
| Gate 5.7 verdict header's `Freeze commit` field points at final freeze SHA `d5d62e4` (fixed at Gate 5.7a) | PASS |

## 5. Non-claim audit

Explicit disclaimers present in the release-facing docs:

| Disclaimer | README | Milestone doc | Gate 5.7 verdict |
|---|:---:|:---:|:---:|
| "not TP elasticity" | ✓ (L57) | ✓ (§Scope) | ✓ (§ header block) |
| "not TP switching" | ✓ (L57) | ✓ (§Scope) | ✓ (§ header block) |
| "not a benchmark" | ✓ (L57) | ✓ (§Scope) | ✓ (§ header block) |
| "not a cross-stack comparison" | ✓ (L57–58) | ✓ (§Scope) | ✓ (§ header block) |
| "not a performance-superiority claim" | ✓ ("No performance-leadership claim" — §Limitations L80) | ✓ (§Scope) | ✓ (§ header block) |

Grep audit of `README.md` and
`docs/ascend_port/fixed_tp2_adaptation_milestone.md` for the
positive forms of the forbidden claims
(`benchmark|superior|faster|beats|outperform|TP elasticity|
TP switching|SGLang|vLLM|TGI|TensorRT`) finds only:

* the disclaimer strings themselves (all six above)
* upstream-inherited README sections `## Upstream: ...` and
  `### Upstream: Benchmark (CUDA / H200)` that are unambiguously
  labelled as **upstream Mini-SGLang / CUDA** content (not
  Ascend claims), preserved verbatim from
  `sgl-project/mini-sglang`

No text on the Ascend fork side claims performance superiority,
cross-stack comparability, TP elasticity, or runtime TP
switching.

## 6. Public-hygiene audit (tracked HEAD)

File-name-level grep at the audited HEAD against the six
leaked-substring classes documented at Gate 4.13a / 4.13b:

```
$ git grep -l -E "sshpass|root@|docker exec [0-9a-f]{12}|[0-9]{1,3}(\\.[0-9]{1,3}){3}" \
    -- README.md docs scripts .github CHANGELOG.md
```

Returned 18 files. Line-level spot-check of a representative
subset (README.md, `scripts/ascend/run_hccl_smoke.sh`,
`docs/ascend_port/gate4.13b_repo_head_redaction_verdict.md`,
`docs/ascend_port/gate5.1_qwen4b_tp2_single_request_verdict.md`,
`docs/ascend_port/gate5.7_fixed_tp2_qwen3_three_model_matrix
_verdict.md`) shows every match falls into one of the following
benign categories:

* Loopback address `127.0.0.1` (rendezvous localhost, HCCL smoke
  script)
* Bind address `0.0.0.0` (documentation of `--host 0.0.0.0` for
  the Ascend deployment example)
* Placeholder tokens (`<HOST>`, `<PORT>`, `<USER>`,
  `<CONTAINER>`, `<REMOTE_PATH>`, `<IP>`, `<pw>`) that are used
  by the Gate 4.13b redaction verdict to describe **what was
  redacted** — the redaction verdict does not itself contain
  real credentials
* Torch / CANN / torch_npu version numbers such as `2.4.0`,
  `2.9.0.post1`, `8.5.1` which incidentally match the broad
  `\d{1,3}(\.\d{1,3}){3}` regex

Zero real public IPs, zero real usernames, zero real passwords,
and zero real container IDs were found in the tracked HEAD.

**HEAD public-hygiene: PASS.**

## 7. Git history / credential-rotation risk note

The Gate 4.13a and 4.13b redaction sweeps were applied to the
tracked **HEAD** at the time of each sweep. They did **not**
rewrite history. Consequently, any commit that predates those
sweeps and that originally contained one of the six
leaked-substring classes still contains that substring in Git
history. Enumeration or verification of those historical
substrings is explicitly out of scope for this audit (spec §4
"不得打印任何真实历史敏感 substring", spec §7 "不做 history
rewrite / force-push / credential printing").

Concrete audit statement:

```
HEAD public hygiene:         PASS
Git history hygiene:         NOT CLOSED
Credential rotation:         UNKNOWN / EXTERNAL
Release readiness:           conditional
```

The release cutter — before pushing a `v0.2.0a1` tag or a new
GitHub Release — must independently confirm that any
credentials that were once printed in tracked history have been
rotated at their source (SSH password, container access token,
etc.). This audit takes no position on and no responsibility
for that decision; it is external to `mini-sglang-ascend`.

If the release cutter's answer is:

* **Rotated / accepted risk** → proceed with `v0.2.0a1` cut
  from `3ceff55`.
* **Not rotated / risk unacceptable** → delay release; either
  (a) rotate credentials externally then proceed, or (b) plan a
  history-rewrite / force-push effort under a **separate**
  future gate — deliberately out of scope for Gate 5.9.

## 8. `v0.1.0a1` and existing GitHub Release status

| Item | Status |
|---|---|
| Local tag `v0.1.0a1` (`a09efd3`, 2026-07-06) | UNCHANGED |
| Upstream `origin` tag `v0.1.0a1` | UNCHANGED |
| GitHub Release for `v0.1.0a1` | UNCHANGED |
| CHANGELOG entry for `v0.1.0a1` | UNCHANGED |
| Release-notes body for `v0.1.0a1` | UNCHANGED |

Gate 5.9 makes zero mutations to release metadata. This is the
same non-mutation posture enforced at Gates 5.1 – 5.8.

## 9. Delta from `v0.1.0a1` (user-visible summary)

Between tag `v0.1.0a1` (`a09efd3`) and audited HEAD
(`3ceff55`):

| Area | Delta |
|---|---|
| Commits | 87 |
| Files touched | 65 |
| Lines added | ~27 267 |
| Lines removed | 24 |
| `python/minisgl/` runtime files touched | 3 (`engine/config.py`, `llm/llm.py`, `server/api_server.py` — Gate 4.1 TP=2 bring-up commit `8431743`) |
| New tests | 2 (`tests/misc/test_exposed_path_abort_ack.py`, `tests/misc/test_shell_cancel_cleanup.py` — pre-Gate 4.14) |
| New scripts | 2 (`scripts/gate5_6_qwen4b_tp2_timing_baseline.py`, `scripts/gate5_7_fixed_tp2_qwen3_three_model_matrix.py`) |
| New Gate verdicts under `docs/ascend_port/` | ≥ 25 (Gate 3.4, 4.1 – 4.14, 4.13b, 4.15, 4.15a, 5.1 – 5.7, 5.7a, 5.8, plus 5.9 in-flight) |
| README fixed-TP2 status paragraph | Rewritten (Gate 4.15 first-pass, Gate 5.8 extended to three models) |
| CHANGELOG.md | Not modified since `v0.1.0a1` (deliberate — Gate 5.9 makes no CHANGELOG entry) |
| Release tag / GitHub Release | Not modified |

User-visible new capability vs `v0.1.0a1`:

* Explicit fixed-TP2 capability record for three dense Qwen3
  models (Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B) under the frozen
  eager / `npu_fia` / bf16 / greedy envelope
* Six-case functional matrix (A / B / C / D / E / F) unified
  across all three models under a single driver (Gate 5.7)
* Milestone summary document consolidating Gate 4.1 – 4.14 +
  Gate 5.1 – 5.7 evidence

Runtime behaviour vs `v0.1.0a1` is preserved except for the
Gate 4.1 TP=2 bring-up changes (config, LLM constructor, API
server) that were already tracked at the point of the earlier
Gate 4.1 verdict.

## 10. Decision matrix

| Question | Answer |
|---|---|
| Does the audited HEAD match the recorded `ascend-port` tip? | Yes (`3ceff55`) |
| Are all fixed-TP2 capability claims in README linked to frozen verdicts? | Yes |
| Are all fixed-TP2 capability claims in the milestone doc linked to frozen verdicts? | Yes |
| Does the milestone doc cover Qwen3-0.6B **and** Qwen3-1.7B **and** Qwen3-4B? | Yes |
| Is Gate 5.7 linked as the three-model unified matrix evidence in README and milestone? | Yes |
| Are the "not TP elasticity / not TP switching / not a benchmark / not a cross-stack comparison / not a performance-superiority claim" disclaimers all present in release-facing docs? | Yes |
| Is any misleading benchmark / superiority / cross-stack / elasticity / switching claim present on the Ascend fork side? | No |
| Does tracked HEAD contain real public IPs / usernames / passwords / container IDs? | No (only benign loopback / bind / placeholders / version numbers) |
| Is the release tag `v0.1.0a1` unchanged? | Yes |
| Is the `v0.1.0a1` GitHub Release unchanged? | Yes |
| Is `CHANGELOG.md` unchanged since `v0.1.0a1`? | Yes |
| Did the audit touch `python/minisgl/` / `tests/` / `scripts/`? | No |
| Did the audit create a `v0.2.0a1` tag? | No |
| Did the audit push a new GitHub Release? | No |
| Did the audit rewrite history or force-push? | No |
| Does Git history still (potentially) contain historical leaked substrings from before Gate 4.13b? | UNKNOWN / EXTERNAL — deliberately not verified here |
| Is a `v0.2.0a1` technical-preview release technically viable from this HEAD? | Yes, conditional on §9 external decision |

**Release readiness decision: PASS (conditional).**

## 11. Known limitations of this audit

* **Documentation-only, no runtime re-verification.** This
  audit does not re-execute any Gate 4 or Gate 5 bring-up
  script and does not re-open any log file. It relies on the
  frozen verdict corpus to be truthful; each verdict already
  carries its own freeze SHA and rank-0/rank-1 evidence.
* **HEAD hygiene, not history hygiene.** §7 states this
  explicitly. Gate 5.9 does **not** grep or read pre-4.13b
  historical blobs.
* **No external credential-rotation verification.** The audit
  cannot confirm what credentials exist at the SSH host or
  Docker container endpoint referenced in the historical
  substring classes. That decision is external.
* **No release-cut authority.** This gate audits readiness; it
  does not cut `v0.2.0a1`. Only a human owner may create the
  tag and the GitHub Release.
* **No `CHANGELOG.md` release entry added.** Deliberate — the
  spec forbids CHANGELOG release updates at this gate. If the
  release is cut later, the release-cut gate must add its own
  CHANGELOG entry.

## 12. Freeze boundary

The frozen artefacts for Gate 5.9 are:

* `docs/ascend_port/gate5.9_qwen3_fixed_tp2_release_readiness_audit.md`

No files under `python/minisgl/` were modified at this gate. No
tests were modified at this gate. No scripts were modified at
this gate. No existing verdict document was modified at this
gate. The freeze commit SHA is recorded on the audit branch tip
after commit; it is recorded on the `ascend-port` tip after
`--no-ff` merge.
