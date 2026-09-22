# Data

This folder intentionally does not contain the raw or cleaned crime
data -- see the root `.gitignore`. This is what's needed to regenerate
everything from scratch.

## Raw data

The raw per-year Boston Police Department incident-report CSVs are
**not** included in this repository. Point `01_clean_data.py` at
wherever you keep a local copy:

```bash
python scripts/01_clean_data.py \
    --raw-data-dir /path/to/raw/csvs \
    --output-dir results/train_test
```

TODO (fill in once decided): link / instructions for downloading the
raw BPD crime incident reports this project uses, and which date
range of files to include.

## District boundaries

`04_make_figures.py` optionally takes `--geojson-path` for a Boston
Police District boundaries GeoJSON (used for the self-excitation,
background, and event-count maps). Not included here either --
point it at your own copy, or omit `--geojson-path` to skip the maps.

## What `01_clean_data.py` produces

Running the command above creates, under `--output-dir`:

```
results/train_test/
├── full_city/
│   ├── full_city_all_2020_end.csv
│   ├── full_city_train_2020_2024.csv
│   ├── full_city_test_2025_end.csv
│   └── full_city_removed_timestamp_ties.csv
├── district_A1/  (same four files, per district)
├── ...
└── train_test_dataset_summary.csv
```

This whole folder is gitignored (`results/train_test/`) -- it's
regenerated from the raw data by `01_clean_data.py`, not versioned.
