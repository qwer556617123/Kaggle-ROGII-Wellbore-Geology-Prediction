# Score Review Protocol

When a new Kaggle score arrives, do not only record it. Treat it as a strategy
checkpoint.

For each score update:

1. Compare against the intended baseline and the paired/related submissions.
2. State which hypothesis was supported, weakened, or rejected.
3. Check whether the result passes the pre-defined action gate.
4. Decide one of:
   - submit calibrated follow-up;
   - run a required audit before more submissions;
   - move to a different basis or method family;
   - stop the branch.
5. Update `experiment_log.md`, `lb_history.csv`, and the active strategy note.
6. Commit the revised strategy before starting the next submission.

For symmetric residual probes, small directional differences are not enough.
If the paired score gap is below the signal gate, prefer a new basis or an
audit over more fine-tuning on the same basis.
