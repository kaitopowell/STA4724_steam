"""Fit and evaluate models on data/steam_clean.csv."""
import pandas as pd

CLEAN_PATH = "data/steam_clean.csv"
SEED = 42


def load_data():
    # TODO: load CLEAN_PATH, separate X and y, recreate the same
    # train/test split (or read a split column if you saved one)
    pass


def fit_logistic(X_train, y_train):
    # TODO: logistic regression (scale numeric features first)
    pass


def fit_tree(X_train, y_train):
    # TODO: decision tree, tune depth with cross-validation
    pass


def fit_forest(X_train, y_train):
    # TODO: random forest, tune with cross-validation
    pass


def evaluate(model, X_test, y_test):
    # TODO: accuracy, F1, AUC on the held-out test set
    pass


def feature_importance(model, feature_names):
    # TODO: forest importances and logistic odds ratios
    pass


if __name__ == "__main__":
    # TODO: wire it together, save tables/figures to figures/
    pass
