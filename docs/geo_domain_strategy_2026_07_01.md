# Geo-Datum Strategy - 2026-07-01

## Hypothesis

Treat TVT as stratigraphic position relative to formation/contact surfaces, not
as a generic row-level regression target. In train wells, each contact column
nearly satisfies:

```text
TVT ~= contact_surface - Z + per-well_offset
```

The submit-safe problem is therefore contact-surface reconstruction plus
prefix-only offset calibration.

## Implemented Audit Path

- `rogii-pf-artifact-blend.py` now supports a geo contact-shape basis using all
  six contacts: `ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA`.
- The surface estimator is `local_plane_residual_knn_v1`:
  local affine plane from nearby train contact samples, then weighted-median
  residual correction.
- Contact TVT uses `contact_pred - Z + offset`, where `offset` is a robust
  trimmed tail estimate from known `TVT_input` rows.
- Candidate confidence combines prefix fit, tail stability, plane/KNN
  disagreement, contact-order consistency, surface roughness, and a downweighted
  GR/typewell scorer.
- Default notebook behavior remains unchanged while
  `ROGII_CONTACT_BASIS_RULE=off`.

## Local Smoke Results

- `geo_datum_audit_smoke`: 3 lb-like train wells, known fraction `0.60`,
  stride `60`; best single contacts are about `7.12` row RMSE.
- `contact_shape_basis_smoke`: 2 lb-like train wells, known fraction `0.60`,
  PF smoke base; `+0.25` improved row RMSE from `6.805` to `6.378`, while
  `-0.25` worsened to `7.310`.

These are implementation smokes, not submission gates.

## Submission Gate

Before any Kaggle probe, run the full contact-shape audit:

```powershell
python scripts\diagnostics\evaluate_contact_shape_basis.py --data-dir . --selection lb_like --well-limit 100 --known-fracs 0.45,0.60,0.75 --summary-output docs\contact_shape_basis_summary.csv --detail-output docs\contact_shape_basis_details.csv --gate-output docs\contact_shape_basis_gate.json
```

Submit the `+0.25/-0.25` pair only if `contact_shape_basis_gate.json` reports
`PASS`.
