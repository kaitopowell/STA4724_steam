"""Streamlit app: describe a game, see how likely it is to be well-reviewed.

Store-style layout: top navigation, a featured "game capsule", and a side panel
with the model's estimate.

Run locally:  streamlit run app.py
Needs the files produced by `python src/models.py` (models/, results/, figures/).
"""
import html
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent

st.set_page_config(page_title="Review Predictor", layout="wide")

# ----------------------------------------------------------------- styling
st.markdown(
    """
<style>
  .stApp { background: linear-gradient(180deg,#1d3048 0px,#14202f 420px,#101826 900px); }
  header[data-testid="stHeader"] { background:#0d1520; }
  .block-container { padding-top: 3.8rem; max-width: 1180px; }
  h1,h2,h3 { letter-spacing:-0.01em; }

  .section-title { font-size:1.35rem; font-weight:700; color:#f2f5f9; margin:4px 0 12px 0; }
  .muted { color:#8da2ba; }

  /* featured capsule */
  .banner { position:relative; height:330px; border-radius:6px; overflow:hidden;
            background:
              radial-gradient(circle at 78% 18%, rgba(102,192,244,.45), transparent 42%),
              radial-gradient(circle at 12% 92%, rgba(167,92,214,.40), transparent 46%),
              linear-gradient(135deg,#231a45 0%,#1c3560 55%,#0e5f86 100%);
            box-shadow: 0 8px 28px rgba(0,0,0,.45); }
  .banner::after { content:""; position:absolute; inset:0;
            background:linear-gradient(180deg,transparent 40%,rgba(8,12,20,.65) 100%); }
  .banner .inner { position:absolute; left:28px; right:28px; bottom:24px; z-index:2; }
  .banner .gtitle { font-size:2.6rem; font-weight:800; color:#fff; line-height:1.08;
            text-shadow:0 2px 14px rgba(0,0,0,.55); word-break:break-word; }
  .banner .facts { margin-top:10px; color:#d6e4f2; font-size:.95rem; }
  .banner .pricetag { position:absolute; right:22px; top:20px; z-index:2;
            background:rgba(10,16,26,.82); color:#fff; padding:7px 14px; border-radius:3px;
            font-weight:700; }

  /* side panel */
  .side { background:linear-gradient(180deg,#27425e 0%,#1b2f45 100%); border-radius:6px;
          padding:22px 22px 20px 22px; height:330px; box-shadow:0 8px 28px rgba(0,0,0,.35);
          display:flex; flex-direction:column; }
  .side .stitle { font-size:1.55rem; font-weight:700; color:#f2f5f9; line-height:1.15;
          word-break:break-word; }
  .side .rating { margin-top:6px; font-size:1.05rem; }
  .rate-good { color:#66c0f4; font-weight:700; }
  .rate-mid  { color:#e6c35c; font-weight:700; }
  .rate-bad  { color:#f0907f; font-weight:700; }
  .side .ssub { color:#9fb3c8; font-size:.82rem; margin:2px 0 12px 0; }
  .meter { position:relative; height:10px; background:#142234; border-radius:999px; }
  .meter .fill { height:10px; border-radius:999px; background:#66c0f4; }
  .meter .avg { position:absolute; top:-5px; width:2px; height:20px; background:#e8ecf2; }
  .meter-labels { display:flex; justify-content:space-between; color:#8da2ba;
          font-size:.75rem; margin-top:7px; }
  .side .bigpct { font-size:2.5rem; font-weight:800; color:#fff; line-height:1; margin-top:10px; }
  .chips { margin-top:auto; }
  .chip { display:inline-block; background:rgba(255,255,255,.10); color:#dce8f5;
          padding:5px 11px; border-radius:3px; font-size:.85rem; margin:0 6px 6px 0; }

  /* cards */
  .card { background:#16273b; border:1px solid #243a54; border-radius:6px; padding:18px 20px; }
  .card .label { color:#9fb3c8; font-size:.9rem; margin-bottom:6px; }
  .whatif { width:100%; border-collapse:collapse; font-size:.95rem; }
  .whatif, .whatif tr, .whatif td { border:none !important; background:transparent; }
  .whatif td { padding:9px 4px; border-bottom:1px solid #243a54 !important; color:#dfe8f2; }
  .whatif tr:last-child td { border-bottom:none !important; }
  .whatif td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
  .whatif td.up { color:#66c0f4; font-weight:600; }
  .whatif td.down { color:#f0907f; font-weight:600; }

  [data-testid='stImage'] img { border-radius:8px; }
  div[data-testid="stButton"] button { border-radius:3px; }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------- helpers
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


def pretty(col):
    name = col.replace("genre_", "").replace("_", " ").title()
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
best = metrics[metrics.model == meta["model_name"]].iloc[0]
baseline = metrics[metrics.model == "Majority baseline"].iloc[0]

# ------------------------------------------------- inputs that survive page changes
DEFAULTS = dict(title="Untitled Game", price=float(round(meta["price_median"], 2)),
                year=meta["year_max"], langs=3, indie=True, multi=False,
                genres=["genre_action"])
PRESETS = {
    "Cozy indie": dict(title="Cozy Harbor", price=9.99, year=2025, langs=5, indie=True,
                       multi=False, genres=["genre_casual", "genre_simulation"]),
    "Co-op shooter": dict(title="Last Squad", price=19.99, year=2024, langs=8, indie=False,
                          multi=True, genres=["genre_action"]),
    "Big-budget RPG": dict(title="Realm of Ash", price=59.99, year=2024, langs=15,
                           indie=False, multi=False, genres=["genre_rpg", "genre_adventure"]),
    "Free multiplayer": dict(title="Arena Rush", price=0.0, year=2023, langs=12, indie=False,
                             multi=True, genres=["genre_action", "genre_free_to_play"]),
}
for k, v in DEFAULTS.items():
    # Streamlit drops widget state when a widget is not on screen, so keep a copy
    if k in st.session_state:
        st.session_state["_saved_" + k] = st.session_state[k]
    else:
        st.session_state[k] = st.session_state.get("_saved_" + k, v)


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


# --------------------------------------------------------------------- pages
def page_predict():
    st.markdown("<div class='section-title'>Featured game</div>", unsafe_allow_html=True)
    hero = st.container()
    st.write("")
    controls = st.container()

    with controls:
        left, right = st.columns([2.1, 1], gap="large")
        with left:
            st.markdown("<div class='section-title'>Build your game</div>",
                        unsafe_allow_html=True)
            st.caption("Try an example, or set every trait yourself.")
            ex = st.columns(len(PRESETS))
            for c, name in zip(ex, PRESETS):
                c.button(name, on_click=apply_preset, args=(name,), width="stretch")
            st.write("")
            a, b = st.columns(2, gap="large")
            title = a.text_input("Game title", key="title", max_chars=40)
            price = a.number_input("List price (USD, 0 for free)", 0.0, 270.0, step=1.0,
                                   key="price")
            year = a.slider("Release year", meta["year_min"], meta["year_max"], key="year")
            langs = b.slider("Supported languages", 1, 30, key="langs")
            c1, c2 = b.columns(2)
            indie = c1.checkbox("Indie", key="indie")
            multi = c2.checkbox("Multiplayer", key="multi")
            chosen = b.multiselect("Genres", genre_cols, format_func=pretty, key="genres")

    args = dict(price=price, year=year, langs=langs, indie=indie, multi=multi, genres=chosen)
    p = predict(model, meta, **args)
    diff = (p - base) * 100
    if diff >= 5:
        cls, label, arrow = "rate-good", "Above Average", "&#9650;"
    elif diff <= -5:
        cls, label, arrow = "rate-bad", "Below Average", "&#9660;"
    else:
        cls, label, arrow = "rate-mid", "About Average", "&#9644;"

    name = html.escape(title.strip() or "Untitled Game")
    price_label = "Free To Play" if price == 0 else f"${price:,.2f}"
    facts = [str(year), f"{langs} language{'s' if langs != 1 else ''}",
             "Indie" if indie else "Publisher-backed",
             "Multiplayer" if multi else "Single-player"]
    chips = "".join(f"<span class='chip'>{pretty(g)}</span>" for g in chosen) \
        or "<span class='chip'>No genre selected</span>"

    with hero:
        h1, h2 = st.columns([2.1, 1], gap="large")
        h1.markdown(
            f"""
<div class="banner">
  <div class="pricetag">{price_label}</div>
  <div class="inner">
    <div class="gtitle">{name}</div>
    <div class="facts">{' &middot; '.join(facts)}</div>
  </div>
</div>""",
            unsafe_allow_html=True)
        h2.markdown(
            f"""
<div class="side">
  <div class="stitle">{name}</div>
  <div class="rating"><span class="{cls}">{arrow} {label}</span></div>
  <div class="ssub">{abs(diff):.0f} pts {'above' if diff >= 0 else 'below'} the average game ({base:.0%})</div>
  <div class="meter"><div class="fill" style="width:{p*100:.1f}%"></div>
    <div class="avg" style="left:{base*100:.1f}%"></div></div>
  <div class="meter-labels"><span>0%</span><span>white line = average</span><span>100%</span></div>
  <div class="bigpct">{p:.0%}</div>
  <div class="ssub">estimated chance of 80%+ positive reviews</div>
  <div class="chips">{chips}</div>
</div>""",
            unsafe_allow_html=True)

    with right:
        variants = {
            "Add multiplayer" if not multi else "Remove multiplayer":
                dict(args, multi=not multi),
            "Support 5 more languages": dict(args, langs=min(langs + 5, 30)),
            "Make it free" if price > 0 else "Charge $9.99":
                dict(args, price=0.0 if price > 0 else 9.99),
            "Publish as indie" if not indie else "Publish with a publisher":
                dict(args, indie=not indie),
        }
        rows = ""
        for lab, v in variants.items():
            q = predict(model, meta, **v)
            d = (q - p) * 100
            c, ar = ("up", "&#9650;") if d >= 0 else ("down", "&#9660;")
            rows += (f"<tr><td>{lab}</td><td class='num'>{q:.0%}</td>"
                     f"<td class='num {c}'>{ar} {abs(d):.1f} pts</td></tr>")
        st.markdown("<div class='section-title'>What if you changed one thing?</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='card'><table class='whatif'>{rows}</table></div>",
                    unsafe_allow_html=True)

    st.caption(
        "A rough guide. The model sees only traits known before launch, so it cannot see game "
        "quality. It finds associations, not causes, and scraped prices and tags may differ "
        "from launch-day values.")


def page_results():
    st.markdown(f"<div class='section-title'>Results on 6,125 held-out games: "
                f"{meta['model_name']}</div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Test AUC (0.50 = coin flip)", f"{best.test_auc:.2f}")
    m2.metric("Test accuracy", f"{best.test_accuracy:.1%}",
              delta=f"{(round(best.test_accuracy, 3) - round(baseline.test_accuracy, 3)) * 100:+.1f}"
                    " pts vs. always guessing")
    m3.metric("Always-guess baseline", f"{baseline.test_accuracy:.1%}")

    show = metrics.rename(columns={
        "model": "Model", "cv_auc": "CV AUC (train)", "test_auc": "Test AUC",
        "test_f1": "Test F1", "test_accuracy": "Test accuracy"}).drop(columns="best_params")
    for c in ["CV AUC (train)", "Test AUC", "Test F1"]:
        show[c] = show[c].map(lambda x: f"{x:.3f}")
    show["Test accuracy"] = show["Test accuracy"].map(lambda x: f"{x:.1%}")
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption(
        "F1 looks high for the majority baseline because always answering \"yes\" never misses a "
        "well-reviewed game. AUC and accuracy are the fairer comparison here.")
    st.image(str(ROOT / "figures" / "roc.png"), width=520)


def page_insights():
    st.markdown("<div class='section-title'>Which traits carry the most signal?</div>",
                unsafe_allow_html=True)
    st.markdown(
        "How much the random forest's AUC drops when a trait is shuffled. Longer bars mean the "
        "model relies on that trait more.")
    top = importance.head(12).copy()
    top["Trait"] = top["feature"].map(
        lambda f: pretty(f) if f.startswith("genre_") else f.replace("_", " ").capitalize())
    top = top.rename(columns={"auc_drop": "Drop in AUC"})[["Trait", "Drop in AUC"]]
    st.bar_chart(top, x="Trait", y="Drop in AUC", horizontal=True, sort="-Drop in AUC",
                 color="#66c0f4", height=380)

    st.markdown("<div class='section-title'>Odds ratios from logistic regression</div>",
                unsafe_allow_html=True)
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


def page_about():
    st.markdown("<div class='section-title'>About this project</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
**Question.** Which traits a developer controls before launch predict whether a Steam game is
well-reviewed (at least 80% positive reviews)?

**Data.** The Steam Games Dataset by FronkonGames on Kaggle. After keeping games with 50 or more
reviews, **{meta['n_games']:,} games** remain, and **{base:.1%}** of them are well-reviewed.

**Predictors.** List price, release year, number of supported languages, indie vs. publisher,
multiplayer support, and {len(genre_cols)} genre indicators. Review counts, owners, and playtime
are left out because they only exist after launch.

**Models.** Logistic regression, a decision tree, and a random forest, tuned with five-fold
stratified cross-validation and tested once on a held-out 20% of games. The app uses the
{meta['model_name'].lower()}.

**Limits.** Pre-launch traits carry real but modest signal. The data cannot see gameplay
quality, scraped prices and tags may differ from launch-day values, and everything here is an
association, not a cause.

*STA 4724 Final Project, Fall 2026.*
""")


# ---------------------------------------------------------------- navigation
nav = st.navigation(
    [
        st.Page(page_predict, title="Predict", icon=":material/sports_esports:",
                url_path="predict", default=True),
        st.Page(page_results, title="Results", icon=":material/bar_chart:",
                url_path="results"),
        st.Page(page_insights, title="Insights", icon=":material/insights:",
                url_path="insights"),
        st.Page(page_about, title="About", icon=":material/info:", url_path="about"),
    ],
    position="top",
)
nav.run()
