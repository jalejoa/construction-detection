# flagged_macro.csv

## What is it?

`flagged_macro.csv` is a manual exclusion list. Each row marks a specific
`(order, year_month)` combination — i.e. one macro patch for one construction
site in one month — that should be **excluded from training and evaluation**.

The file must always exist, even if empty (only the header row). When empty,
`filter_macro_catalog` skips the filtering step and uses all available patches.

## When to add entries

Add a row when visual inspection reveals that a patch is unusable, for example:
- Heavy cloud cover or haze
- Corrupted or missing tile data (nodata stripes, all-zero regions)
- Incorrect label alignment
- Patch covers the wrong area due to a quad download error

## Format

```
order,year_month
12,2022_03
47,2021_11
```

| Column | Type | Description |
|---|---|---|
| `order` | integer | Unique site ID (matches `order` column in macro catalog) |
| `year_month` | string | Month of the patch in `YYYY_MM` format |

## How it is used

`filter_macro_catalog()` in `src/patches.py` performs an anti-join: any row in
the macro catalog whose `(order, year_month)` matches an entry here is dropped
before splits and training.

Called in notebook `02_image_cropping.ipynb`, cell *Filter flagged patches*.
