# Gate 4.13b Verdict — Repo HEAD tracked-file credential redaction

**Gate ID:** 4.13b (Repo HEAD full tracked-file credential redaction)
**Verdict:** PASS
**Branch:** `gate4.13b-repo-head-redaction`
**Base commit:** `76f1cac` (tip of `ascend-port`, Gate 4.13a merge)
**Redaction commit:** `7d20521`
**Date:** 2026-07-11
**Kind:** Documentation-only sweep. Substitutes remote-host,
username, password, container-id, and public-IP substrings with
neutral placeholders across every tracked docs/scripts file at
repo HEAD. No runtime source under `python/minisgl/` is modified;
no test file is modified; no release tag or CHANGELOG is touched;
Git history is **not** rewritten and there is **no** force-push.

> **Public hygiene note:** This verdict never quotes the original
> secret substrings. Only placeholder tokens and neutral file
> counts appear below.

---

## 1. Verdict summary

**PASS.** At repo HEAD on branch `gate4.13b-repo-head-redaction`,
no tracked file matches any of the six real-secret substring
patterns targeted by Gate 4.13b:

* `sshpass` invocations (with the associated password token)
* the plaintext password token
* the full public IPv4 address
* the `<IP>:<PORT>` composite
* the twelve-hex-digit container id
* the `root@<IP>` composite

`git grep -l -E "<OLD_SSHPASS_PATTERN>|<OLD_PASSWORD_PATTERN>|<OLD_HOST_PATTERN>|<OLD_CONTAINER_PATTERN>"`
returned **zero matches** across the entire tracked working tree
(the grep expression is quoted only for reproducibility; it maps
onto the six secret substrings). Post-sweep `git diff --check`
returned clean.

## 2. Placeholder scheme

The redactor substitutes the six real-secret patterns with the
following neutral placeholders (order matters — longer patterns
first so they win against generic substrings):

| Pattern class | Placeholder |
|---|---|
| Full `sshpass -p '<pw>' ssh -p <port> <user>@<ip>` prefix | `ssh -p <PORT> <USER>@<HOST>` |
| Full `sshpass ... -o StrictHostKeyChecking=no ...` variant | `ssh -p <PORT> <USER>@<HOST>` |
| Password token (bare) | `<REDACTED>` |
| `<ip>:<port>` composite | `<HOST>:<PORT>` |
| Bare public IPv4 | `<HOST>` |
| `root@<HOST>` residue after IP substitution | `<USER>@<HOST>` |
| Twelve-hex container id | `<CONTAINER>` |

## 3. Files changed by this sweep

**18 tracked verdict/audit files** under `docs/ascend_port/`
were modified. No scripts, runtime source, tests, CHANGELOG,
README, or `.github/` files were modified — those either did not
contain any real-secret substrings or contained only false
positives (loopback `127.0.0.1`, bind address `0.0.0.0`) that are
explicitly preserved.

Modified files (paths only, no secret contents):

* `docs/ascend_port/gate2.5_shell_cancel_verdict.md`
* `docs/ascend_port/gate3.1_qwen_model_size_audit.md`
* `docs/ascend_port/gate3.1_qwen_model_size_verdict.md`
* `docs/ascend_port/gate3.2_qwen1.7b_request_shapes_verdict.md`
* `docs/ascend_port/gate3.3_tp1_capability_matrix_verdict.md`
* `docs/ascend_port/gate3.4_tp1_timing_baseline_verdict.md`
* `docs/ascend_port/gate4.1_tp2_single_request_verdict.md`
* `docs/ascend_port/gate4.2_tp2_b2_equal_length_verdict.md`
* `docs/ascend_port/gate4.3_tp2_b2_ragged_prefill_verdict.md`
* `docs/ascend_port/gate4.4_tp2_b2_mixed_kv_decode_verdict.md`
* `docs/ascend_port/gate4.5_tp2_dynamic_admission_b1_b2_b1_verdict.md`
* `docs/ascend_port/gate4.6_tp2_dynamic_grow_shrink_b1_b2_b3_b2_b1_verdict.md`
* `docs/ascend_port/gate4.7_tp2_timing_baseline_verdict.md`
* `docs/ascend_port/gate4.8_qwen1.7b_tp2_single_request_verdict.md`
* `docs/ascend_port/gate4.9_qwen1.7b_tp2_b2_equal_length_verdict.md`
* `docs/ascend_port/gate4.10_qwen1.7b_tp2_b2_ragged_prefill_verdict.md`
* `docs/ascend_port/gate4.11_qwen1.7b_tp2_b2_mixed_kv_decode_verdict.md`
* `docs/ascend_port/gate4.12_qwen1.7b_tp2_dynamic_admission_b1_b2_b1_verdict.md`

The Gate 4.13 verdict (`gate4.13_qwen1.7b_tp2_timing_baseline_verdict.md`)
was already redacted at Gate 4.13a (`f556468`) and does not appear
in this sweep. The Gate 4.13a verdict itself was authored
placeholder-only and does not need sweeping.

## 4. Post-redaction grep summary

Post-redaction verification checked the known leaked
host/user/container/password substrings using local-only
patterns. This document intentionally does not print those
substrings — the placeholders below stand in for the real
patterns actually executed against the tracked tree.

Real-secret substring pattern — `<OLD_SSHPASS_PATTERN>|<OLD_PASSWORD_PATTERN>|<OLD_HOST_PATTERN>|<OLD_CONTAINER_PATTERN>`:

```
$ git grep -l -E "<OLD_SSHPASS_PATTERN>|<OLD_PASSWORD_PATTERN>|<OLD_HOST_PATTERN>|<OLD_CONTAINER_PATTERN>"
(no output — zero files)
```

Placeholder-string substring pattern —
`ssh -p <PORT> <USER>@<HOST>` (intentional redaction text left by
this sweep and Gate 4.13a): matches only under
`docs/ascend_port/` in verdict files where the launch command is
documented. This is the expected post-sweep state.

Loopback / bind-address pattern — `127\.0\.0\.1|0\.0\.0\.0`:
retained in `README.md`, `scripts/ascend/run_hccl_smoke.sh`,
`docs/ascend_port/gate2.4_exposed_path_abortack_audit.md`,
`docs/ascend_port/gate4.1_tp2_readiness_audit.md`, and a handful
of gate verdicts. These are **not** credentials — they are
`--host 0.0.0.0` bind addresses and `tcp://127.0.0.1:2333`
rendezvous URIs. Preserved intentionally.

## 5. Runtime / test / release integrity

| Question | Answer |
|---|---|
| Any file under `python/minisgl/` modified? | No |
| Any file under `tests/` modified? | No |
| `CHANGELOG.md` modified? | No |
| `README.md` modified? | No |
| Release tag `v0.1.0a1` mutated? | No |
| GitHub Release notes edited? | No |
| Git history rewritten (rebase / filter-repo)? | No |
| Any force-push executed? | No |
| Working-tree `git diff --check` result | clean |

## 6. Git history status

Gate 4.13b explicitly does **not** rewrite Git history. Every
commit before `76f1cac` still contains the original secret
substrings in its blob content. Anyone with `git log -p`, GitHub
web history, or a fork/mirror can still recover the original
plaintext credentials from previous commits (Gate 4.13a §7 and
Gate 4.13b spec §0 flagged this explicitly).

The only durable mitigation for the exposed credentials is
external to this repo: **rotate the Ascend host SSH password and
regenerate the container id**. Status of that rotation is
tracked outside the repo and is **unknown** to this gate.

## 7. Regression evidence

No runtime code, no test code, and no CI configuration was
modified at this gate. Test-suite headers were not re-executed
because the change surface is documentation-only and byte-scoped
to fixed substring substitutions. Gate 4.13 (immediately prior)
recorded `51 / 51 PASS` per-file pytest on `tests/misc/`; that
tally is unchanged by construction.

## 8. Known limitations

* **Documentation-only.** Runtime behaviour of `python/minisgl/`
  is unchanged by this gate.
* **HEAD only.** Every commit reachable from `76f1cac` and
  earlier still contains the pre-redaction secrets. Public
  hygiene requires host-side credential rotation.
* **Substring-based.** The sweep matches exact byte substrings
  from the SUBS list; near-misses (e.g. deliberate typos or
  base64-encoded copies) would not be caught. None have been
  observed in the tracked tree.
* **False-positive preservation.** Loopback `127.0.0.1` and bind
  `0.0.0.0` are intentionally not redacted. The word `token`
  appears widely as a domain term (`max_new_tokens`,
  `output_tokens`, `KV tokens`) and is not touched.
* **No credential rotation performed.** This gate cannot rotate
  the exposed SSH password or container id; that must be done on
  the Ascend host outside this repo.

## 9. Decision matrix

| Question | Answer |
|---|---|
| Does `git grep -l -E "<OLD_SSHPASS_PATTERN>\|<OLD_PASSWORD_PATTERN>\|<OLD_HOST_PATTERN>\|<OLD_CONTAINER_PATTERN>"` return zero files? | Yes |
| Any tracked HEAD file still contains the plaintext SSH password? | No |
| Any tracked HEAD file still contains the plaintext public IPv4? | No |
| Any tracked HEAD file still contains the plaintext container id? | No |
| Any tracked HEAD file still contains a `root@<ip>` composite? | No |
| Did this gate rewrite Git history? | No |
| Did this gate force-push? | No |
| Did this gate modify `python/minisgl/`? | No |
| Did this gate modify tests? | No |
| Did this gate touch release tag / CHANGELOG / GitHub Release? | No |
| Is credential rotation on the Ascend host handled by this gate? | No (external, unknown) |

**Verdict: PASS.** Repo HEAD is clean of the six targeted
real-secret substrings. History and host-side credential rotation
remain out of scope and are separately tracked.

## 10. Freeze boundary

The frozen artefacts for Gate 4.13b are:

* the 18 documentation files listed in §3 (substring redactions
  applied)
* this verdict document (`docs/ascend_port/gate4.13b_repo_head_redaction_verdict.md`)

No files under `python/minisgl/` were modified at this gate. No
tests were modified. The redaction commit SHA is recorded on the
branch tip once committed with message
`Redact tracked public command hygiene`; the merge commit SHA is
recorded on `ascend-port` once merged `--no-ff` with message
`Merge Gate 4.13b repo HEAD redaction`.
