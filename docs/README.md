# ROGII Development Docs

This folder keeps compact project memory: strategy notes, experiment logs, and
small summary tables. Large row-level diagnostics should stay out of git.

## Read First

- `project_state.md` - current project orientation and file layout.
- `next_strategy.md` - active strategy, rejected directions, and next moves.
- `experiment_log.md` - chronological experiment decisions and LB results.
- `artifact_hidden_mask_cv_findings.md` - latest artifact-CV rejection and exact-overlap interpretation.
- `best_versions_reassessment_2026_06_27.md` - current best-version interpretation.
- `score_review_protocol.md` - required reflection and re-planning steps after each new Kaggle score.

## Keep In Git

- Findings and strategy markdown.
- Small summary CSV files that support a decision.
- Compact per-well diagnostic tables.

## Keep Out Of Git

- Row-level `*_details.csv` files.
- Temporary Kaggle outputs and downloaded datasets.
- Local submissions, model artifacts, caches, and smoke-test scratch data.
