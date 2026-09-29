# Development sweep under protocol v1.0 (superseded)

These are the complete outputs of the development sweep run under protocol v1.0 (commit `85a730a`, git tag `dev-sweep-v1.0`). The selection file is in `legacy/old_configs/selected_params_v1.0.json`.

That sweep selected β = 1.0 for the proposed Hybrid+KG-linked system. β = 1 is the top of the v1.0 grid, and at that value the system is a pure KG model. Protocol v1.1 (PROTOCOL.md §12) therefore restricts the proposed system's β to {0.05, …, 0.95}, and the sweep was rerun from scratch. The current development results are in `results/rebuild/dev/`.

These files are kept for the record only. No current analysis reads them.
