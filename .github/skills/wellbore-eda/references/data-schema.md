# Data Schema Reference

## horizontal_well.csv

| Column | Present In | Description |
|--------|-----------|-------------|
| MD | train + test | Measured depth (feet) — monotonically increasing |
| X, Y, Z | train + test | 3D coordinates of wellbore point |
| GR | train + test | Gamma ray value (may be NaN) |
| TVT_input | train + test | Known TVT (empty after Prediction Start point) |
| TVT | train only | Ground truth TVT (target variable) |
| ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA | train only | Formation top TVT depths at this MD |

## typewell.csv

| Column | Present In | Description |
|--------|-----------|-------------|
| TVT | train + test | True vertical thickness in the typewell |
| GR | train + test | Gamma ray value in the typewell |
| Geology | train only | Geological formation name at this TVT |

## Geology Formation Labels
- ANCC, ASTNL, ASTNU, BUDA, EGFDL, EGFDU, LBHL, LTGT, LTHL, MNSS

## Submission Format
```
id,tvt
{well_id}_{row_index},<float>
```
- `row_index` is 0-based row index in the test horizontal_well.csv
- Only rows where `TVT_input` is empty are included

## Key Constants
- Typewell TVT step: 0.5 ft (uniform)
- Horizontal well MD step: 1.0 ft (uniform)
- 3 test wells: 000d7d20, 00bbac68, 00e12e8b
- ~773 training wells
