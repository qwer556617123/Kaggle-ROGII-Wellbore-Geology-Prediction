# Archived Scripts

This folder contains scripts that are not part of the active experiment loop.

Reasons for archiving include:

- superseded by current baselines;
- known weak CV or leaderboard behavior;
- useful only as historical reference;
- diagnostic/submission helpers that should not be run by default.

Subfolders:

- `experiments/`: old model and feature variants.
- `diagnostics/`: old analysis and submission helper scripts.
- `helpers/`: reserved for archived utility scripts.

Move a script back out of archive only when it has a concrete hypothesis, a target validation gate, and an expected impact on Public LB.
