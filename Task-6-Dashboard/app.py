"""
Netflix Content Intelligence Dashboard
======================================

Auspify Technologies — Data Science Internship, Task 6
Intern: Wandiya James

An end-to-end dashboard over the Netflix catalogue: exploratory analysis,
a trained classifier served live, and the business recommendations that
follow from both.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Netflix Content Intelligence",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A single palette used by every chart, so the dashboard reads as one piece.
# Tuned for the dark theme in .streamlit/config.toml — the print-oriented
# hues used in the notebooks sit too close to the background here, so each
# is lifted in luminance while keeping its hue.
NAVY = "#4A7BB7"
BLUE = "#3BA3E0"
PURPLE = "#A95FD6"
RED = "#E74C3C"
GREEN = "#2ECC71"
ORANGE = "#F39C12"
GREY = "#7F8C9B"

TYPE_COLORS = {"Movie": BLUE, "TV Show": PURPLE}

# Charts inherit the app background rather than painting their own panel,
# so a Plotly figure reads as part of the page instead of a pasted-in image.
CHART_TEMPLATE = go.layout.Template(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EAF2", size=13),
        title=dict(font=dict(color="#E6EAF2")),
        xaxis=dict(gridcolor="#263449", zerolinecolor="#263449",
                   linecolor="#334155", tickfont=dict(color="#A8B4C7")),
        yaxis=dict(gridcolor="#263449", zerolinecolor="#263449",
                   linecolor="#334155", tickfont=dict(color="#A8B4C7")),
        legend=dict(font=dict(color="#E6EAF2")),
        hoverlabel=dict(bgcolor="#182538", font=dict(color="#E6EAF2")),
        colorway=[BLUE, PURPLE, ORANGE, GREEN, RED, GREY],
    )
)
pio.templates["netflix_dash"] = CHART_TEMPLATE
pio.templates.default = "netflix_dash"

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
      h1 {font-size: 2.1rem !important;}

      /* Metrics as raised cards, so the headline numbers read as a
         distinct band rather than floating text on the background. */
      [data-testid="stMetric"] {
          background: #182538;
          border: 1px solid #263449;
          border-radius: 10px;
          padding: 14px 16px;
      }
      [data-testid="stMetricValue"] {font-size: 1.8rem; color: #E6EAF2;}
      [data-testid="stMetricLabel"] {font-size: 0.82rem; color: #A8B4C7;}

      /* Give the tab row a visible baseline against the dark background. */
      .stTabs [data-baseweb="tab-list"] {
          gap: 6px;
          border-bottom: 1px solid #263449;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Data and model loading
# --------------------------------------------------------------------------

DATA_NAMES = ["netflix_cleaned.csv", "cleaned_netflix_data.csv"]
MODEL_NAMES = ["netflix_type_classifier.pkl"]
# Searched in order. The repo-root-relative entries matter for Streamlit
# Community Cloud, where the working directory is the repository root
# rather than the folder holding app.py.
SEARCH_DIRS = [
    Path("."), Path("output"), Path("data"),
    Path("Task-6-Dashboard"), Path("Task-6-Dashboard/output"),
    Path("Task-1-Data-Cleaning/output"),
    Path("Task-5-ML-Classification/output"),
    Path("../Task-1-Data-Cleaning/output"),
    Path("../Task-5-ML-Classification/output"),
    Path(".."),
]


def _locate(filenames):
    """Find the first matching file across the likely project locations."""
    for folder in SEARCH_DIRS:
        for name in filenames:
            candidate = folder / name
            if candidate.exists():
                return candidate
    return None


@st.cache_data(show_spinner=False)
def load_data():
    """Load the cleaned dataset produced by Task 1."""
    path = _locate(DATA_NAMES)
    if path is None:
        return None, None

    df = pd.read_csv(path, parse_dates=["date_added"])
    df["genres"] = df["genres"].fillna("").str.split(", ")
    return df, path


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the classifier pipeline saved by Task 5."""
    path = _locate(MODEL_NAMES)
    if path is None:
        return None, None
    return joblib.load(path), path


df_all, data_path = load_data()

if df_all is None:
    st.error(
        "**Cleaned dataset not found.**\n\n"
        "This dashboard reads `netflix_cleaned.csv`, produced by the Task 1 notebook. "
        "Run that notebook first, then place the file beside `app.py` or in "
        "`Task-1-Data-Cleaning/output/`."
    )
    st.stop()

pipeline, model_path = load_model()


# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------

st.sidebar.title("Filters")
st.sidebar.caption("Every chart below responds to these.")

year_min = int(df_all["year_added"].min())
year_max = int(df_all["year_added"].max())

year_range = st.sidebar.slider(
    "Year added to Netflix",
    min_value=year_min, max_value=year_max,
    value=(2015, year_max),
)

type_choice = st.sidebar.multiselect(
    "Content type",
    options=["Movie", "TV Show"],
    default=["Movie", "TV Show"],
)

country_options = df_all["primary_country"].value_counts().head(20).index.tolist()
country_choice = st.sidebar.multiselect(
    "Country (blank = all)",
    options=country_options,
    default=[],
)

audience_choice = st.sidebar.multiselect(
    "Audience",
    options=sorted(df_all["audience_category"].dropna().unique()),
    default=sorted(df_all["audience_category"].dropna().unique()),
)

# Genre is multi-valued — a title tagged "Dramas, International Movies" must
# match a filter on either. Streamlit's multiselect is type-to-search, which
# matters with 42 options.
ALL_GENRES = sorted({g for lst in df_all["genres"] for g in lst if g})
genre_choice = st.sidebar.multiselect(
    "Genre (blank = all)",
    options=ALL_GENRES,
    default=[],
)

# Apply the filters
mask = (
    df_all["year_added"].between(*year_range)
    & df_all["type"].isin(type_choice)
    & df_all["audience_category"].isin(audience_choice)
)
if country_choice:
    mask &= df_all["primary_country"].isin(country_choice)

if genre_choice:
    wanted = set(genre_choice)
    mask &= df_all["genres"].apply(lambda lst: bool(wanted & set(lst)))

df = df_all[mask]

st.sidebar.markdown("---")
st.sidebar.metric("Titles in selection", f"{len(df):,}")
st.sidebar.caption(f"out of {len(df_all):,} in the catalogue")

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Data** · `{data_path.name}` — the cleaned output of Task 1 "
    f"({len(df_all):,} titles, 28 columns)."
)
if pipeline is not None:
    st.sidebar.caption(
        f"**Model** · {pipeline['model_name']} from Task 5 "
        f"(F1 {pipeline['test_f1']}, baseline {pipeline['majority_baseline']})."
    )

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("Netflix Content Intelligence")
st.caption(
    "Auspify Technologies — Data Science Internship, Task 6 · Wandiya James  \n"
    "Catalogue snapshot to September 2021. Analysis, live model, and the "
    "recommendations that follow."
)

if df.empty:
    st.warning("No titles match the current filters. Widen the selection in the sidebar.")
    st.stop()

# A headline strip above the tabs, so the key numbers stay on screen whichever
# tab is open — and so a single screenshot carries the summary.
k1, k2, k3, k4, k5 = st.columns(5)

n_movies = int((df["type"] == "Movie").sum())
n_shows = int((df["type"] == "TV Show").sum())
n_countries = df.loc[df["country_known"], "primary_country"].nunique()
n_genres = len({g for lst in df["genres"] for g in lst if g})

k1.metric("Titles", f"{len(df):,}")
k2.metric("Movies", f"{n_movies:,}", f"{n_movies / len(df) * 100:.0f}%")
k3.metric("TV Shows", f"{n_shows:,}", f"{n_shows / len(df) * 100:.0f}%")
k4.metric("Countries", f"{n_countries:,}")
k5.metric("Genres", f"{n_genres:,}")

st.markdown("")

tab_overview, tab_trends, tab_geo, tab_predict, tab_insights = st.tabs(
    ["Overview", "Trends", "Geography", "Predict", "Business insights"]
)


# --------------------------------------------------------------------------
# Tab 1 — Overview
# --------------------------------------------------------------------------

with tab_overview:
    # The headline counts live above the tabs; these are the shape-of-the-data
    # numbers that would otherwise need a chart to read off.
    c1, c2, c3, c4 = st.columns(4)

    median_runtime = df["movie_minutes"].median()
    one_season = (df["tv_seasons"] == 1).sum()
    total_shows = df["tv_seasons"].notna().sum()
    median_lag = df["years_to_platform"].median()
    recent_share = (df["release_year"] >= 2010).mean() * 100

    c1.metric("Median movie runtime",
              f"{median_runtime:.0f} min" if pd.notna(median_runtime) else "—")
    c2.metric("Shows lasting one season",
              f"{one_season / total_shows * 100:.0f}%" if total_shows else "—",
              f"{one_season:,} of {total_shows:,}" if total_shows else None)
    c3.metric("Released 2010 or later", f"{recent_share:.0f}%")
    c4.metric("Median years to platform",
              f"{median_lag:.0f}" if pd.notna(median_lag) else "—")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("Content type")
        counts = df["type"].value_counts()
        fig = px.pie(
            values=counts.values, names=counts.index, hole=0.45,
            color=counts.index, color_discrete_map=TYPE_COLORS,
        )
        fig.update_traces(textinfo="percent+label", textfont_size=14)
        fig.update_layout(showlegend=False, height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Audience")
        order = ["Kids", "Older Kids", "Teens", "Adults", "Unrated"]
        aud = df["audience_category"].value_counts()
        aud = aud.reindex([o for o in order if o in aud.index])
        fig = px.bar(
            x=aud.index, y=aud.values,
            color=aud.index,
            color_discrete_sequence=[GREEN, BLUE, ORANGE, RED, GREY],
        )
        fig.update_layout(
            showlegend=False, height=340, xaxis_title="", yaxis_title="Titles",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Top genres")
    genre_counts = df.explode("genres")["genres"].value_counts().head(12)
    fig = px.bar(
        x=genre_counts.values, y=genre_counts.index, orientation="h",
        color_discrete_sequence=[ORANGE],
    )
    fig.update_layout(
        height=420, xaxis_title="Titles", yaxis_title="",
        yaxis={"categoryorder": "total ascending"}, margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    if len(df) >= 50 and total_shows > 0:
        st.info(
            f"**{one_season / total_shows * 100:.0f}% of the TV shows in this selection "
            f"run for exactly one season.** Across the full catalogue the figure is 67% — "
            "a mix of limited-series formats and early cancellation."
        )


# --------------------------------------------------------------------------
# Tab 2 — Trends
# --------------------------------------------------------------------------

with tab_trends:
    st.subheader("Titles added per year")

    per_year = df["year_added"].value_counts().sort_index()
    colors = [GREY if yr == 2021 else BLUE for yr in per_year.index]

    fig = go.Figure(go.Bar(
        x=per_year.index.astype(int), y=per_year.values,
        marker_color=colors,
        hovertemplate="%{x}<br>%{y:,} titles<extra></extra>",
    ))
    fig.update_layout(
        height=360, xaxis_title="Year added", yaxis_title="Titles added",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "⚠️ **2021 is a partial year** (greyed out) — the dataset ends in September, "
        "so it covers nine months against twelve for every other year. Reading it as a "
        "decline would be a mistake."
    )

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("Movie / TV balance over time")
        mix = pd.crosstab(df["year_added"], df["type"])
        mix = mix[mix.sum(axis=1) >= 20]

        if not mix.empty and mix.shape[1] == 2:
            share = mix.div(mix.sum(axis=1), axis=0).mul(100)
            fig = go.Figure()
            for col in share.columns:
                fig.add_trace(go.Scatter(
                    x=share.index.astype(int), y=share[col],
                    name=col, mode="lines+markers",
                    line=dict(color=TYPE_COLORS[col], width=3),
                ))
            fig.update_layout(
                height=340, yaxis_title="Share of that year (%)",
                xaxis_title="Year added", yaxis_range=[0, 100],
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("Not enough data in this selection to show the balance.")

    with right:
        st.subheader("How quickly content reaches Netflix")
        lag = df[df["year_added"] >= 2015].groupby(["year_added", "type"])[
            "years_to_platform"].median().unstack()

        if not lag.empty:
            fig = go.Figure()
            for col in lag.columns:
                fig.add_trace(go.Scatter(
                    x=lag.index.astype(int), y=lag[col],
                    name=col, mode="lines+markers",
                    line=dict(color=TYPE_COLORS.get(col, GREY), width=3),
                ))
            fig.update_layout(
                height=340, yaxis_title="Median years from release",
                xaxis_title="Year added", margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("Not enough data in this selection.")

    st.subheader("Are movies getting shorter?")
    runtime = (df[df["release_year"] >= 1990]
               .groupby(df["release_year"] // 5 * 5)["movie_minutes"]
               .median().dropna())

    if len(runtime) >= 2:
        fig = px.line(x=runtime.index.astype(int), y=runtime.values, markers=True)
        fig.update_traces(line_color=RED, line_width=3)
        fig.update_layout(
            height=320, xaxis_title="Release period (5-year buckets)",
            yaxis_title="Median runtime (minutes)", margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("Not enough data in this selection.")


# --------------------------------------------------------------------------
# Tab 3 — Geography
# --------------------------------------------------------------------------

with tab_geo:
    st.subheader("Where the catalogue comes from")

    known = df[df["country_known"]]
    country_counts = known["primary_country"].value_counts().head(15)

    view = st.radio("View as", ["Bar chart", "Treemap"],
                    horizontal=True, label_visibility="collapsed")

    if view == "Bar chart":
        fig = px.bar(
            x=country_counts.values, y=country_counts.index, orientation="h",
            color_discrete_sequence=[NAVY],
        )
        fig.update_layout(
            height=460, xaxis_title="Titles", yaxis_title="",
            yaxis={"categoryorder": "total ascending"}, margin=dict(t=10, b=10),
        )
    else:
        # Area encodes volume, which makes the scale gap obvious at a glance —
        # the United States block dwarfs everything else in a way a bar axis
        # lets you read past.
        fig = px.treemap(
            names=country_counts.index, parents=[""] * len(country_counts),
            values=country_counts.values,
            color=country_counts.values, color_continuous_scale="Blues",
        )
        fig.update_traces(textinfo="label+value",
                          hovertemplate="%{label}<br>%{value:,} titles<extra></extra>")
        fig.update_layout(height=460, margin=dict(t=10, b=10),
                          coloraxis_showscale=False)

    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Movie / TV split differs sharply by country")

    top8 = country_counts.head(8).index
    if len(top8) >= 2:
        split = pd.crosstab(
            known.loc[known["primary_country"].isin(top8), "primary_country"],
            known["type"], normalize="index",
        ).mul(100)
        split = split.reindex([c for c in top8 if c in split.index])

        fig = go.Figure()
        for col in split.columns:
            fig.add_trace(go.Bar(
                x=split.index, y=split[col], name=col,
                marker_color=TYPE_COLORS.get(col, GREY),
                text=[f"{v:.0f}%" for v in split[col]], textposition="outside",
            ))
        fig.update_layout(
            barmode="group", height=400, yaxis_title="Share of that country (%)",
            xaxis_title="", yaxis_range=[0, 105], margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")

        st.info(
            "**There is no single answer to what Netflix's catalogue looks like — "
            "it depends which country you ask about.** India is around 92% movies; "
            "Pakistan is around 83% TV shows. Any model using country as a feature "
            "is picking up this structure."
        )

    st.markdown("---")
    st.subheader("Median movie runtime by country and audience")

    pivot = pd.pivot_table(
        df[df["type"] == "Movie"], values="movie_minutes",
        index="primary_country", columns="audience_category", aggfunc="median",
    )
    pivot = pivot.reindex([c for c in country_counts.head(8).index if c in pivot.index])
    cols = [c for c in ["Kids", "Older Kids", "Teens", "Adults"] if c in pivot.columns]

    if not pivot.empty and cols:
        fig = px.imshow(
            pivot[cols], text_auto=".0f", aspect="auto",
            color_continuous_scale="Viridis", labels=dict(color="Minutes"),
        )
        fig.update_layout(height=380, xaxis_title="", yaxis_title="",
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Indian films run over half an hour longer than American ones at every "
            "audience level — the Bollywood convention, visible directly in the data."
        )


# --------------------------------------------------------------------------
# Tab 4 — Predict
# --------------------------------------------------------------------------

with tab_predict:
    st.subheader("Is it a Movie or a TV Show?")

    if pipeline is None:
        st.warning(
            "**Model file not found.** This tab serves `netflix_type_classifier.pkl`, "
            "saved by the Task 5 notebook. Run that notebook, then place the file "
            "beside `app.py` or in `Task-5-ML-Classification/output/`."
        )
    else:
        st.caption(
            f"Served by the {pipeline['model_name']} trained in Task 5 — "
            f"test F1 {pipeline['test_f1']} against a majority baseline of "
            f"{pipeline['majority_baseline']}. It uses only release metadata: "
            "duration and genre were excluded because they leak the answer."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            in_release = st.number_input("Release year", 1950, 2025, 2020)
            in_added = st.number_input("Year added to Netflix", 2008, 2025, 2021)
        with c2:
            in_month = st.selectbox(
                "Month added", list(range(1, 13)), index=5,
                format_func=lambda m: pd.Timestamp(2020, m, 1).strftime("%B"),
            )
            in_country = st.selectbox(
                "Country", ["Unknown"] + country_options, index=1,
            )
        with c3:
            in_audience = st.selectbox(
                "Audience", sorted(df_all["audience_category"].dropna().unique()),
                index=0,
            )

        if st.button("Predict", type="primary"):
            row = pd.Series(0.0, index=pipeline["features"])
            row["release_year"] = in_release
            row["year_added"] = in_added
            row["month_added"] = in_month
            row["country_known"] = 0 if in_country == "Unknown" else 1

            for prefix, value in [("country_", in_country), ("audience_", in_audience)]:
                column = f"{prefix}{value}"
                if column in row.index:
                    row[column] = 1

            frame = row.to_frame().T[pipeline["features"]]
            model_input = (pipeline["scaler"].transform(frame)
                           if pipeline["needs_scaling"] else frame)

            proba = pipeline["model"].predict_proba(model_input)[0]
            predicted = pipeline["classes"][int(np.argmax(proba))]
            confidence = float(np.max(proba))

            st.markdown("---")
            r1, r2 = st.columns([1, 2])
            r1.metric("Prediction", predicted, f"{confidence * 100:.1f}% confident")

            with r2:
                fig = go.Figure(go.Bar(
                    x=[proba[0] * 100, proba[1] * 100],
                    y=["Movie", "TV Show"], orientation="h",
                    marker_color=[BLUE, PURPLE],
                    text=[f"{proba[0] * 100:.1f}%", f"{proba[1] * 100:.1f}%"],
                    textposition="outside",
                ))
                fig.update_layout(
                    height=180, xaxis_title="Probability (%)", xaxis_range=[0, 115],
                    margin=dict(t=10, b=10), showlegend=False,
                )
                st.plotly_chart(fig, width="stretch")

            if confidence < 0.65:
                st.warning(
                    "**Low confidence.** With only release metadata, a recent "
                    "American adult title genuinely could be either. The model is "
                    "hedging rather than being confidently wrong — which is the "
                    "honest failure mode."
                )


# --------------------------------------------------------------------------
# Tab 5 — Business insights
# --------------------------------------------------------------------------

with tab_insights:
    st.subheader("What the analysis supports")

    c1, c2, c3 = st.columns(3)
    c1.metric("Catalogue released 2010+", "84.8%")
    c2.metric("TV shows lasting one season", "67.2%")
    c3.metric("Largest genre tag", "International Movies")

    st.markdown("---")

    st.markdown("""
#### 1. Localise acquisition rather than treating the catalogue as one market

Country mixes differ more than any other variable in this dataset. India is about
92% movies; Pakistan is about 83% TV shows; Japan and South Korea lean heavily to
series. A single global acquisition strategy is being applied to markets that
behave nothing alike.

**Action:** set acquisition targets per market by format, not globally.

#### 2. The one-season pattern is a retention risk worth measuring

Two thirds of TV shows run for exactly one season and only 6% reach five. Some of
that is deliberate limited-series commissioning, but subscribers who invest in a
series and see it end abruptly have a reason to churn.

**Action:** this dataset cannot separate "designed as a limited series" from
"cancelled early". Joining commissioning intent to this catalogue would make the
distinction measurable — and that is the question worth answering.

#### 3. Production speed is now the competitive advantage

TV shows reach the platform the same year they are released; movies still lag by
about a year. That gap is the clearest signal in the data of the shift from
licensing other studios' back catalogues to commissioning original content.

**Action:** treat the licensing-window lag as the measurable cost of licensed
film content when comparing it against originals.

#### 4. Metadata quality is limiting what can be modelled

Director is unrecorded for roughly 30% of titles, country for 3%, and no plot
description or cast list exists in this extract. Task 5 showed the practical
consequence: a classifier using only legitimate features reaches 76% accuracy
against a 69.7% baseline, and eight separate improvement attempts moved none of it.

**Action:** the constraint is the data, not the modelling. Capturing descriptions
and cast would do more for predictive work than any amount of tuning.

#### 5. Children's content is a small but distinctly different segment

Kids and Older Kids together are under 20% of the catalogue, but US children's
content is 43% TV shows against 22% for adult content — the most TV-weighted
segment there is. The Task 5 model found this pattern independently.

**Action:** treat children's content as a separate planning category with its own
format mix, not a slice of the general catalogue.
    """)

    st.markdown("---")
    st.subheader("What this dashboard cannot tell you")
    st.warning("""
**No viewership, ratings or revenue data.** Nothing here indicates how popular or
profitable anything was, so "what is Netflix's most successful content" is
unanswerable from this source.

**It is a snapshot, not a history.** Only titles present in September 2021 appear.
Anything removed before then is invisible, which biases the picture toward content
Netflix chose to keep.

**`release_year` means different things by type** — latest season for TV shows,
original release for movies. Comparisons across types on that field need care.
    """)

    st.markdown("---")
    st.caption(
        "Built from `netflix_cleaned.csv` (Task 1) and `netflix_type_classifier.pkl` "
        "(Task 5). Full methodology in the Task 1, 2, 3 and 5 notebooks."
    )