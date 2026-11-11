# What Makes a Steam Game Well-Reviewed?

STA 4724 Final Project, Fall 2026.
Predicts whether a Steam game is well-reviewed (>= 80% positive reviews)
from pre-launch traits: price, genre, release year, indie vs. publisher,
multiplayer, and number of supported languages.

## Setup (manual)

1. Create a virtual environment in the repo root:

       python -m venv .venv

2. Activate it:

       source .venv/bin/activate      # Mac / Linux
       .venv\Scripts\activate         # Windows

3. Install requirements:

       pip install -r requirements.txt

## Setup (script)

    cd scripts/
    chmod +x unix_env.sh && ./unix_env.sh      # Mac / Linux
    # Windows: double-click dos_env.bat

## Data

Download the Steam Games Dataset (FronkonGames) from Kaggle and put the raw
CSV in `data/` as `games.csv`. See `data/README.md`.
MAKE SURE `data/games.csv` EXISTS BEFORE RUNNING ANYTHING.

## Run order

1. `python src/clean_steam.py`   -> writes `data/steam_clean.csv`
2. `python src/models.py`        -> fits models, writes tables/figures
3. Open `notebooks/analysis.ipynb` to review results end to end

## Layout

    data/        raw + cleaned data (gitignored)
    src/         cleaning and modeling code
    notebooks/   analysis notebook
    figures/     saved plots
    report/      report and slides
    scripts/     environment setup helpers

## Team

| Member | Area |
|--------|------|
| A | Question, dataset, cleaning (`src/clean_steam.py`) |
| B | Literature, methods, CV setup |
| C | Models and evaluation (`src/models.py`) |
| D | Findings, contributions, AI statement, final export |
