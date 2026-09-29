# Git history of the research repository

This release was exported from the research repository at commit `66f551c8651bdc34bca1dabb2347d5cb4d31b914`. The commit order is part of the study's integrity record: the protocol was committed before the development sweep, each test lexicon was frozen by hash before its evaluation, and the final evaluation ran once from a clean commit. Commit hashes cited in `PROTOCOL.md`, `RELEASE_NOTES.md` and `results/rebuild/FINAL_PAPER_DISCLOSURES.md` refer to this list.

| Commit | Date | Message |
|---|---|---|
| `ad2f0ef` | 2026-09-29 00:55:20 +0530 | Pre-rebuild checkpoint: full project state after research audit |
| `2c1b9de` | 2026-09-29 01:14:38 +0530 | Add v4 measurement pipeline, legacy regression and draft protocol |
| `e265204` | 2026-09-29 01:39:45 +0530 | Freeze experimental protocol v1.0 |
| `85a730a` | 2026-09-29 18:47:31 +0530 | Development sweep under frozen protocol v1.0 |
| `605a4b9` | 2026-09-29 18:57:51 +0530 | Amend protocol to v1.1: interior beta grid for the proposed system |
| `c2f885f` | 2026-09-29 19:00:25 +0530 | Development sweep under protocol v1.1 (from scratch) |
| `ce96f94` | 2026-09-29 19:29:17 +0530 | chore: save historical test overlap audit |
| `c92e434` | 2026-09-29 19:32:10 +0530 | Freeze lexicon B (LLM fallback) |
| `8d0a7c0` | 2026-09-29 19:32:51 +0530 | Write final evaluation outputs to results/rebuild/test |
| `33bd63d` | 2026-09-29 19:35:13 +0530 | Final evaluation under protocol v1.1 (run once) |
| `dc57f6e` | 2026-09-29 19:51:05 +0530 | feat: phase 5 final diagnostics and disclosures |
| `8a33114` | 2026-09-29 22:46:59 +0530 | Freeze co-author lexicon B-human as a second test lexicon |
| `e8f7042` | 2026-09-29 22:50:15 +0530 | Second test lexicon (co-author, hand-written): results and disclosure |
| `9c349f4` | 2026-09-29 23:18:07 +0530 | Post-hoc reviewer checks: untouched records, two-way bootstrap, tie bounds, pool CIs, truncation |
| `66f551c` | 2026-09-29 23:54:27 +0530 | chore: final release packaging |

## Tags

- `dev-sweep-v1.0`: Development sweep under protocol v1.0 (superseded by v1.1; beta = 1 selected)
- `pre-rebuild-audit`: State of the project after the research audit, before the experimental rebuild

Omitted from this public export (present in the research repository): `reference_papers/` (third-party copyrighted papers) and `data/safety_testset*.csv` (out-of-scope, contain Reddit posts).
