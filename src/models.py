"""Fit and evaluate models on data/steam_clean.csv.

Plan (matches slide 5):
  - use the saved 80/20 split (the `split` column)
  - tune each model with 5-fold stratified CV on the training data, scoring ROC AUC
  - evaluate ONCE on the held-out test set (AUC, F1, accuracy)
  - compare against a majority-class baseline
  - the app model is picked by CV AUC (not by test score)

Run from the repo root:  python src/models.py
Outputs: results/*.csv, models/best_model.joblib, models/meta.json, figures/*.png
"""
import json
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLEAN_PATH = "data/steam_clean.csv"
SEED = 42

NUMERIC = ["list_price", "release_year", "n_languages"]
BINARY = ["multiplayer", "indie"]
# NOTE: `free` is left out on purpose: it is just list_price == 0.
# `pct_positive`, `well_reviewed`, `split`, `app_id`, `name` are NOT predictors.


def load_data():
    df = pd.read_csv(CLEAN_PATH)
    genres = [c for c in df.columns if c.startswith("genre_")]
    features = NUMERIC + BINARY + genres
    train, test = df[df.split == "train"], df[df.split == "test"]
    return (train[features], train["well_reviewed"],
            test[features], test["well_reviewed"], features, genres)


def make_pipeline(model, features, scale):
    # Scaling happens inside the pipeline, so it is fit within each CV fold
    if scale:
        pre = ColumnTransformer(
            [("num", StandardScaler(), NUMERIC)], remainder="passthrough")
        pre.set_output(transform="pandas")
        return Pipeline([("pre", pre), ("model", model)])
    return Pipeline([("model", model)])


def candidates():
    # name -> (estimator, param grid, scale?)
    return {
        "Majority baseline": (DummyClassifier(strategy="prior"), {}, False),
        "Logistic regression": (
            LogisticRegression(max_iter=2000),
            {"model__C": [0.01, 0.1, 1, 10]}, True),
        "Decision tree": (
            DecisionTreeClassifier(random_state=SEED),
            {"model__max_depth": [3, 5, 8, 12],
             "model__min_samples_leaf": [20, 50, 100]}, False),
        "Random forest": (
            RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
            {"model__max_depth": [8, 12, 16],
             "model__min_samples_leaf": [10, 30]}, False),
    }


def fit_all(X_train, y_train, features):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fitted = {}
    for name, (est, grid, scale) in candidates().items():
        pipe = make_pipeline(est, features, scale)
        search = GridSearchCV(pipe, grid or {"model": [est]}, cv=cv,
                              scoring="roc_auc", n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        fitted[name] = search
        print(f"{name:22s} CV AUC {search.best_score_:.4f}  {search.best_params_}")
    return fitted


def evaluate(fitted, X_test, y_test):
    rows, rocs = [], []
    for name, s in fitted.items():
        p = s.predict_proba(X_test)[:, 1]
        pred = (p >= 0.5).astype(int)
        rows.append({
            "model": name,
            "cv_auc": round(s.best_score_, 4),
            "test_auc": round(roc_auc_score(y_test, p), 4),
            "test_f1": round(f1_score(y_test, pred), 4),
            "test_accuracy": round(accuracy_score(y_test, pred), 4),
            "best_params": json.dumps(
                {k.replace("model__", ""): v for k, v in s.best_params_.items()
                 if k != "model"}),
        })
        fpr, tpr, _ = roc_curve(y_test, p)
        step = max(1, len(fpr) // 300)  # thin the curve for plotting
        rocs.append(pd.DataFrame({"model": name, "fpr": fpr[::step], "tpr": tpr[::step]}))
    return pd.DataFrame(rows), pd.concat(rocs)


def odds_ratios(fitted, features):
    lr = fitted["Logistic regression"].best_estimator_
    coefs = lr.named_steps["model"].coef_[0]
    names = list(lr.named_steps["pre"].get_feature_names_out())
    names = [n.split("__", 1)[1] for n in names]
    out = pd.DataFrame({"feature": names, "odds_ratio": np.exp(coefs)})
    out["note"] = np.where(out.feature.isin(NUMERIC), "per 1 SD increase", "yes vs. no")
    return out.sort_values("odds_ratio", ascending=False).reset_index(drop=True)


def feature_importance(fitted, X_test, y_test):
    # Permutation importance on the test set: drop in AUC when a feature is shuffled
    rf = fitted["Random forest"].best_estimator_
    r = permutation_importance(rf, X_test, y_test, scoring="roc_auc",
                               n_repeats=5, random_state=SEED, n_jobs=-1)
    out = pd.DataFrame({"feature": X_test.columns, "auc_drop": r.importances_mean})
    return out.sort_values("auc_drop", ascending=False).reset_index(drop=True)


# ---------- figures (single hue for magnitude; first 3 categorical slots for lines)
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85", "#0b0b0b", "#52514e")


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_roc(roc, metrics, path):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    colors = {"Logistic regression": BLUE, "Decision tree": ORANGE,
              "Random forest": AQUA}
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle=(0, (4, 3)), linewidth=1.2)
    for name, c in colors.items():
        d = roc[roc.model == name]
        auc = metrics.set_index("model").loc[name, "test_auc"]
        ax.plot(d.fpr, d.tpr, color=c, linewidth=2, label=f"{name} (AUC {auc:.3f})")
    ax.set_xlabel("False positive rate", color=MUTED)
    ax.set_ylabel("True positive rate", color=MUTED)
    ax.set_title("ROC curves on the held-out test set", color=INK, loc="left")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK)
    ax.set_aspect("equal")
    style(ax)
    ax.grid(axis="both", color="#e6e5e0", linewidth=0.8)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_bars(df, col, label_col, title, xlabel, path, top=12, ref=None):
    d = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=200)
    ax.barh(d[label_col].str.replace("genre_", "").str.replace("_", " "),
            d[col], color=BLUE, height=0.55)
    if ref is not None:
        ax.axvline(ref, color=GRAY, linewidth=1.2)
    ax.set_title(title, color=INK, loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_app_model(fitted, features, genres, df_all):
    # Pick the app model by CV AUC among the real models (not the baseline)
    real = {k: v for k, v in fitted.items() if k != "Majority baseline"}
    best_name = max(real, key=lambda k: real[k].best_score_)
    joblib.dump(real[best_name].best_estimator_, "models/best_model.joblib", compress=3)
    meta = {
        "model_name": best_name,
        "features": features,
        "genres": genres,
        "numeric": NUMERIC,
        "base_rate": float(df_all["well_reviewed"].mean()),
        "n_games": int(len(df_all)),
        "price_median": float(df_all["list_price"].median()),
        "year_min": int(df_all["release_year"].min()),
        "year_max": int(df_all["release_year"].max()),
        "lang_max": int(df_all["n_languages"].max()),
    }
    with open("models/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("app model:", best_name)


if __name__ == "__main__":
    for d in ("results", "models", "figures"):
        os.makedirs(d, exist_ok=True)

    X_train, y_train, X_test, y_test, features, genres = load_data()
    print(f"train {len(X_train):,} | test {len(X_test):,} | features {len(features)}")

    fitted = fit_all(X_train, y_train, features)
    metrics, roc = evaluate(fitted, X_test, y_test)
    odds = odds_ratios(fitted, features)
    imp = feature_importance(fitted, X_test, y_test)

    metrics.to_csv("results/metrics.csv", index=False)
    roc.to_csv("results/roc.csv", index=False)
    odds.to_csv("results/odds_ratios.csv", index=False)
    imp.to_csv("results/importance.csv", index=False)
    print(metrics.drop(columns="best_params").to_string(index=False))

    plot_roc(roc, metrics, "figures/roc.png")
    plot_bars(imp, "auc_drop", "feature",
              "What the random forest relies on most",
              "Drop in test AUC when the feature is shuffled", "figures/importance.png")
    plot_bars(odds, "odds_ratio", "feature",
              "Largest odds ratios (logistic regression)",
              "Odds ratio (1 = no effect)", "figures/odds_ratios.png", ref=1)

    df_all = pd.read_csv(CLEAN_PATH)
    save_app_model(fitted, features, genres, df_all)
