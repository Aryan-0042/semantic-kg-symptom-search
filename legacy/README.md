# legacy/

Superseded material, kept so that the project's history stays complete and recoverable. No current analysis reads from this folder, except that the scripts in `scripts/` can still be run from the repository root.

| Path | Contents | Origin |
|---|---|---|
| `results/rebuild_dev_v1.0/` | complete development sweep under protocol v1.0, which selected β = 1 | superseded by protocol v1.1 (`PROTOCOL.md` §12); commit `85a730a`, tag `dev-sweep-v1.0` |
| `old_configs/selected_params_v1.0.json` | parameters selected by that sweep | same |
| `old_configs/requirements_v1_v3.txt` | the original v1–v3 requirements (torch 2.2.2, no statsmodels or PyYAML) | was `requirements.txt` at the project root |
| `scripts/master_results_verification_v3.py` | v3 "master verification", which used a different index pool (see `RESEARCH_AUDIT.md`) | project root; `ROOT` path updated after the move |
| `scripts/rrf_rank_diagnostic_v3.py`, `scripts/rrf_shortlist_diagnostic_v3.py` | v3 diagnostics of the RRF shortlist | project root; `ROOT` path updated after the move |
| `scripts/pre_phase3_audit.py` | an earlier ad-hoc overlap check of the v2 test set | project root |
| `old_papers/cureent_paper.tex`, `old_papers/figs/` | the v3-era paper draft and its figures (its claims did not replicate; see the audit) | project root and `figs/` |
| `misc/temp.py`, `misc/files_structure.py`, `misc/combined_results.txt`, `misc/arthritis_uti_subset.csv` | scratch utilities and outputs from the v3 period | project root |
| `pipeline_common.py`, `src2/` | older copies of v1 pipeline code | pre-rebuild |

`RESEARCH_AUDIT.md` describes these files at their original root-level locations; the table above maps them to their current place.
