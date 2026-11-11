"""Streamlit app: enter a game's pre-launch traits, see how likely it is to be well-reviewed.

Run locally:  streamlit run app.py
Needs the files produced by `python src/models.py` (models/, results/, figures/).
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent

st.set_page_config(page_title="Steam Review Predictor", layout="wide")


@st.cache_resource
def load_model():
    try:
        model = joblib.load(ROOT / "models" / "best_model.joblib")
        meta = json.loads((ROOT / "models" / "meta.json").read_text())
    except Exception as e:  # missing file, or saved with a different scikit-learn version
        st.error("Could not load the saved model. Run `python src/models.py` to rebuild it, "
                 f"then restart the app. ({type(e).__name__})")
        st.stop()
    return model, meta


@st.cache_data
def load_results():
    return (pd.read_csv(ROOT / "results" / "metrics.csv"),
            pd.read_csv(ROOT / "results" / "odds_ratios.csv"),
            pd.read_csv(ROOT / "results" / "importance.csv"))


def pretty(genre_col):
    name = genre_col.replace("genre_", "").replace("_", " ").title()
    return name.replace("Rpg", "RPG").replace(" And ", " and ")


def build_row(meta, price, year, langs, indie, multi, genres):
    row = {f: 0 for f in meta["features"]}
    row.update({"list_price": price, "release_year": year, "n_languages": langs,
                "indie": int(indie), "multiplayer": int(multi)})
    for g in genres:
        row[g] = 1
    return pd.DataFrame([row])[meta["features"]]


def predict(model, meta, **kw):
    return float(model.predict_proba(build_row(meta, **kw))[0, 1])


model, meta = load_model()
metrics, odds, importance = load_results()
genre_cols = meta["genres"]
base = meta["base_rate"]

st.title("What makes a Steam game well-reviewed?")
st.caption(
    f"STA 4724 Final Project. Trained on {meta['n_games']:,} Steam games with 50+ reviews. "
    "Well-reviewed means at least 80% positive reviews."
)

tab_predict, tab_perf, tab_why = st.tabs(
    ["Predict a game", "How well does it work?", "What matters most?"])

# ------------------------------------------------------------------ predict
with tab_predict:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Describe your game")
        price = st.number_input("List price (USD, 0 for free)", 0.0, 270.0,
                                float(round(meta["price_median"], 2)), step=1.0)
        year = st.slider("Release year", meta["year_min"], meta["year_max"], meta["year_max"])
        langs = st.slider("Supported languages", 1, 30, 3)
        c1, c2 = st.columns(2)
        indie = c1.checkbox("Indie (self-published)", value=True)
        multi = c2.checkbox("Multiplayer or co-op", value=False)
        chosen = st.multiselect("Genres", genre_cols, default=["genre_action"],
                                format_func=pretty)

    args = dict(price=price, year=year, langs=langs, indie=indie, multi=multi, genres=chosen)
    p = predict(model, meta, **args)

    with right:
        st.subheader("Estimated chance of being well-reviewed")
        st.metric("Probability of 80%+ positive reviews", f"{p:.0%}",
                  delta=f"{(p - base) * 100:+.0f} points vs. the average game ({base:.0%})",
                  delta_color="normal")
        st.progress(min(max(p, 0.0), 1.0))

        st.markdown("**What if you changed one thing?**")
        variants = {
            "Add multiplayer" if not multi else "Remove multiplayer":
                dict(args, multi=not multi),
            "Support 5 more languages": dict(args, langs=min(langs + 5, 30)),
            "Make it free" if price > 0 else "Charge $9.99":
                dict(args, price=0.0 if price > 0 else 9.99),
            "Publish as indie" if not indie else "Publish with a publisher":
                dict(args, indie=not indie),
        }
        table = pd.DataFrame({
            "Change": list(variants),
            "New chance": [f"{predict(model, meta, **v):.0%}" for v in variants.values()],
            "Difference": [f"{(predict(model, meta, **v) - p) * 100:+.1f} pts"
                           for v in variants.values()],
        })
        st.dataframe(table, hide_index=True, width="stretch")

        st.info(
            "Treat this as a rough guide. The model uses only traits known before launch, "
            "so it cannot see game quality. It finds associations, not causes, and "
            "scraped prices and tags may differ from launch-day values.")

# --------------------------------------------------------------------- perf
with tab_perf:
    st.subheader("Results on 6,125 held-out games")
    show = metrics.rename(columns={
        "model": "Model", "cv_auc": "CV AUC (train)", "test_auc": "Test AUC",
        "test_f1": "Test F1", "test_accuracy": "Test accuracy"}).drop(columns="best_params")
    st.dataframe(show, hide_index=True, width="stretch")

    best = metrics[metrics.model == meta["model_name"]].iloc[0]
    st.markdown(
        f"The **{meta['model_name']}** was the best model in cross-validation, so it powers "
        f"the predictions. On games it had never seen it reached an AUC of "
        f"**{best.test_auc:.2f}** (0.50 is a coin flip) and **{best.test_accuracy:.0%}** accuracy, "
        f"versus **{base:.0%}** for always guessing \"well-reviewed\".")
    st.caption(
        "F1 looks high for the majority baseline because always answering \"yes\" never misses a "
        "well-reviewed game. AUC and accuracy are the fairer comparison here.")
    st.image(str(ROOT / "figures" / "roc.png"), width=520)

# ---------------------------------------------------------------------- why
with tab_why:
    st.subheader("Which traits carry the most signal?")
    st.markdown(
        "Release year and price matter most. Genre and language count come next. "
        "Longer bars mean the model loses more accuracy when that trait is scrambled.")
    st.image(str(ROOT / "figures" / "importance.png"), width=640)

    st.subheader("Odds ratios from logistic regression")
    st.markdown(
        "Above 1 means higher odds of being well-reviewed, below 1 means lower, holding the "
        "other traits fixed. Numeric traits are per one standard deviation. Rare genres have "
        "few games, so treat their ratios loosely.")
    st.image(str(ROOT / "figures" / "odds_ratios.png"), width=640)
    with st.expander("Full odds ratio table"):
        t = odds.copy()
        t["feature"] = t["feature"].map(lambda f: pretty(f) if f.startswith("genre_")
                                        else f.replace("_", " ").title())
        st.dataframe(t.round(3), hide_index=True, width="stretch")
