"""Clean the raw Steam data.

Input:  data/games.csv        (raw Kaggle file)
Output: data/steam_clean.csv  (features + response + train/test split column)

Run from the repo root:  python src/clean_steam.py [path/to/games.csv]
"""
import ast
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/games.csv"
CLEAN_PATH = "data/steam_clean.csv"
MIN_REVIEWS = 50
THRESHOLD = 0.80
SEED = 42


def load_raw(path):
    # The header has one fewer name than the rows have values: the column
    # "DiscountDLC count" is really "Discount" + "DLC count". Read the header
    # first, split that name, then re-read with the corrected names.
    cols = list(pd.read_csv(path, nrows=0).columns)
    i = cols.index("DiscountDLC count")
    cols[i:i + 1] = ["Discount", "DLC count"]
    return pd.read_csv(path, header=0, names=cols, index_col=False)


def filter_games(df):
    # Keep games with >= MIN_REVIEWS total reviews (drops playtests, unreleased)
    n = df["Positive"] + df["Negative"]
    out = df[n >= MIN_REVIEWS].copy()
    out["n_reviews"] = n[n >= MIN_REVIEWS]  # only used to build the response
    return out


def build_response(df):
    # Keep both the raw percentage (for RMSE / regression) and the 0/1 label
    df["pct_positive"] = df["Positive"] / df["n_reviews"]
    df["well_reviewed"] = (df["pct_positive"] >= THRESHOLD).astype(int)
    return df


def recover_list_price(df):
    # Price in the file is the sale price. Undo the discount.
    # Guard against a 100% discount (divide by zero).
    d = df["Discount"].clip(upper=99)
    df["list_price"] = (df["Price"] / (1 - d / 100)).round(2)
    return df


def _parse_list(s):
    # "['English', 'French']" -> list; blank or malformed -> []
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else []
    except (ValueError, SyntaxError):
        return []


def build_features(df):
    out = pd.DataFrame(index=df.index)
    out["app_id"] = df["AppID"]
    out["name"] = df["Name"]
    out["list_price"] = df["list_price"]
    out["free"] = (df["list_price"] == 0).astype(int)

    # Release year
    out["release_year"] = pd.to_datetime(
        df["Release date"], format="mixed", errors="coerce").dt.year

    # Number of supported languages
    out["n_languages"] = df["Supported languages"].fillna("").map(
        lambda s: len(_parse_list(s)))

    # Multiplayer: any category mentioning multi-player, PvP or co-op
    cats = df["Categories"].fillna("").str.lower()
    out["multiplayer"] = cats.str.contains(
        "multi-player|pvp|co-op", regex=True).astype(int)

    # Genres: one 0/1 column per genre
    genres = df["Genres"].fillna("").str.split(",")
    dummies = (genres.explode().str.strip()
               .replace("", np.nan).dropna()
               .pipe(pd.get_dummies).groupby(level=0).max().astype(int))
    dummies = dummies.reindex(df.index, fill_value=0)
    dummies.columns = ["genre_" + c.lower().replace(" ", "_").replace("&", "and")
                       for c in dummies.columns]

    # Indie: Indie genre tag, or developer == publisher
    same = (df["Developers"].fillna("").str.strip().str.lower()
            == df["Publishers"].fillna("").str.strip().str.lower())
    has_tag = genres.map(lambda g: "Indie" in [x.strip() for x in g])
    out["indie"] = ((has_tag | (same & (df["Developers"].fillna("") != "")))
                    .astype(int))

    # The Indie genre is already captured by the `indie` flag above
    dummies = dummies.drop(columns="genre_indie")

    out = pd.concat([out, dummies], axis=1)
    out["pct_positive"] = df["pct_positive"]
    out["well_reviewed"] = df["well_reviewed"]
    return out.dropna(subset=["release_year"]).assign(
        release_year=lambda x: x["release_year"].astype(int))


def split_and_save(df, path=CLEAN_PATH):
    # 80/20 split stratified on the response; stored as a column so every
    # downstream script uses the exact same split.
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.20, stratify=df["well_reviewed"],
        random_state=SEED)
    df = df.copy()
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    raw_path = sys.argv[1] if len(sys.argv) > 1 else RAW_PATH
    df = load_raw(raw_path)
    df = filter_games(df)
    df = build_response(df)
    df = recover_list_price(df)
    df = build_features(df)
    df = split_and_save(df)
    print(f"saved {CLEAN_PATH}: {len(df):,} games, "
          f"{(df.split == 'train').sum():,} train / "
          f"{(df.split == 'test').sum():,} test")
