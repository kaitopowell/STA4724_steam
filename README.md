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
2. `python src/models.py`        -> fits models, writes `results/`, `figures/`, `models/`
3. `streamlit run app.py`        -> launches the predictor app
4. Open `notebooks/analysis.ipynb` to review everything end to end

The cleaned data, results, figures and trained model are committed, so you can run
step 3 without the raw `games.csv`.

## Deploy the app

1. Push the repo to GitHub.
2. Go to share.streamlit.io, sign in with GitHub, click **Create app**.
3. Pick this repo, branch `main`, main file `app.py`, then **Deploy**.

## Layout

    data/        raw + cleaned data (gitignored)
    app.py       Streamlit predictor app
    src/         cleaning and modeling code
    results/     metrics, odds ratios, importances (CSV)
    models/      trained model used by the app
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
