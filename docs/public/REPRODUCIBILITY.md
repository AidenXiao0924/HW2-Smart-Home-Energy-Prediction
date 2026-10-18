# Installation and Reproduction

## Environment

- Python 3.11
- Node.js 22 or newer for browser-side unit tests and static release verification
- Python dependencies: `requirements.txt`

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
node --test tests/javascript/*.test.mjs
node tools/build-rc.mjs --verify
python -m http.server 8000 --directory releases/web-rc3/site
```

## Dataset

The raw dataset is intentionally excluded. Download it from the UCI source in `docs/public/DATASET_ATTRIBUTION.md` and verify the CSV SHA-256:

`2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d`

Place it at `data/raw/energydata_complete.csv`. Do not commit raw or processed data.

## Research boundaries

The frozen production model is `mlr-1.0.0`. The formal Test evaluation has already been completed. Reproduction must not use Test results to change feature selection, coefficients, preprocessing, Conformal `q`, or model choice. The public static release is `releases/web-rc3/site`; it requires no Python backend and does not load `.joblib` files.

The public Repository contains source and selected reproducibility artifacts. Internal checkpoints, per-row research outputs, private evidence, original conversations, virtual environments and the frozen academic PDF are excluded from the first public commit.
