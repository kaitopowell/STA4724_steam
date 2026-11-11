"""Clean the raw Steam data. Input: data/games.csv, output: data/steam_clean.csv."""
import pandas as pd

RAW_PATH = "data/games.csv"
CLEAN_PATH = "data/steam_clean.csv"
MIN_REVIEWS = 50
SEED = 42


def load_raw(path):
    # TODO: fix the CSV header (a "DiscountDLC count" column is really
    # two columns: Discount and DLC count). Check for shifted columns.
    pass


def filter_games(df):
    # TODO: keep only games with at least MIN_REVIEWS reviews
    pass


def build_response(df):
    # TODO: positive / (positive + negative) >= 0.80 -> well_reviewed (0/1)
    pass


def recover_list_price(df):
    # TODO: undo the sale discount: price / (1 - discount/100)
    pass


def build_features(df):
    # TODO: one-hot genres, language count, multiplayer flag, indie flag,
    # release year. Leave out anything that only exists after launch.
    pass


def split_and_save(df):
    # TODO: 80/20 stratified split on the response (seed = SEED),
    # then save the clean file to CLEAN_PATH
    pass


if __name__ == "__main__":
    df = load_raw(RAW_PATH)
    df = filter_games(df)
    df = build_response(df)
    df = recover_list_price(df)
    df = build_features(df)
    split_and_save(df)
