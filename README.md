# Netflix Content Analysis — Auspify Data Science Internship

**Intern:** Wandiya James
**Program:** Auspify Technologies — 4-Week Data Science Internship
**Dataset:** Netflix catalogue, 8,790 titles, snapshot to September 2021

> ### 🔗 [**Live dashboard →**](https://auspify-internship-dashboard.streamlit.app/)
> `https://auspify-internship-dashboard.streamlit.app/`

Five of the six available tasks are completed, against a requirement of any four.

| Task | Level | Deliverable | Write-up |
|---|---|---|---|
| 1 — Data Cleaning & Preprocessing | Easy | `data_cleaning.ipynb` | `cleaning_report.txt` |
| 2 — Exploratory Data Analysis | Easy | `eda_analysis.ipynb` | [`summary.md`](Task-2-EDA/summary.md) |
| 3 — Recommendation System | Medium | `recommendation_system.ipynb` | [`report.md`](Task-3-Recommendation-System/report.md) |
| 5 — ML Classification Model | Advanced | `ml_classification.ipynb` | [`report.md`](Task-5-ML-Classification/report.md) |
| 6 — Business Insights Dashboard | Advanced | `app.py` | [`report.md`](Task-6-Dashboard/report.md) |

---

## Quick start

```bash
conda env create -f environment.yml
conda activate austech
```

Run the notebooks in order — Task 1 produces the cleaned dataset every later task
depends on. Each runs end to end with **Restart Kernel and Run All**.

To launch the dashboard:

```bash
cd Task-6-Dashboard
streamlit run app.py
```

> **Note on pip:** if you install packages manually, use `python -m pip install ...`
> rather than bare `pip`. On some systems bare `pip` resolves to the system Python
> and refuses to install (PEP 668).

---

## Repository structure

```
Auspify-Internship/
│
├── Task-1-Data-Cleaning/
│   ├── Dataset.csv                          ← raw input
│   ├── data_cleaning.ipynb
│   ├── output/
│   │   ├── netflix_cleaned.csv              ← input for every later task
│   │   ├── netflix_cleaned.xlsx
│   │   └── cleaning_report.txt
│   └── screenshots/
│
├── Task-2-EDA/
│   ├── eda_analysis.ipynb
│   ├── summary.md
│   └── screenshots/
│
├── Task-3-Recommendation-System/
│   ├── recommendation_system.ipynb
│   ├── report.md
│   └── screenshots/
│
├── Task-5-ML-Classification/
│   ├── ml_classification.ipynb
│   ├── report.md
│   ├── output/
│   │   └── netflix_type_classifier.pkl      ← saved model pipeline
│   └── screenshots/
│
├── Task-6-Dashboard/
│   ├── app.py                               ← Streamlit dashboard
│   ├── requirements.txt                     ← lean deps for cloud deployment
│   ├── report.md
│   └── screenshots/
│
├── environment.yml
├── requirements.txt
└── README.md
```

Only Task 1 holds a copy of the dataset. Every later task locates
`Task-1-Data-Cleaning/output/netflix_cleaned.csv` automatically, so nothing is
duplicated across folders.

---

## Task 1 — Data Cleaning & Preprocessing

Prepared the raw catalogue for analysis: 8,790 → 8,786 rows, 10 → 28 columns,
15 of 15 validation checks passing.

**What the audit found**

- `isna()` reported zero nulls, but 2,588 directors and 287 countries were missing
  behind the placeholder string `"Not Given"`.
- One row has the legitimate title *Unknown* — a blanket placeholder replacement
  would have destroyed it, so replacement was scoped to the two affected columns.
- Four duplicate records sat under different `show_id` values. The fourth only became
  detectable after whitespace was stripped, which is why cleaning order matters.
- `duration` mixed minutes and seasons in one text column, making it unusable for
  comparison. Split into a numeric value and a unit.
- 703 runtime outliers were flagged but **retained**. Inspection showed them to be
  genuine short films and epics; removal would have moved the median by 0.3 minutes
  while deleting 7.3% of all movies.

## Task 2 — Exploratory Data Analysis

Univariate, bivariate and multivariate analysis across 20 visualisations.

**What the data shows**

- 67.2% of TV shows run for exactly one season; only 6% reach five.
- Country mixes differ sharply — India is 92% movies, Pakistan 83% TV shows.
- Indian films run over half an hour longer than American ones at every audience level.
- TV shows now reach Netflix the same year they are released; movies still lag by
  about a year.
- 84.8% of the catalogue was released in 2010 or later.
- `release_year` and `years_to_platform` correlate at −0.98 **by construction** — a
  leakage warning carried forward into Task 5.

## Task 3 — Recommendation System

Content-based recommender using TF-IDF and cosine similarity over genres, director,
country, type and audience category.

| Metric | Result | Random baseline |
|---|---|---|
| Genre overlap (Jaccard) | **0.945** | 0.093 |
| Content type match | **1.000** | 0.582 |
| Catalogue coverage | 19.9% | 28.7% |

Feature weighting was tested rather than assumed. Tripling genre weight raised overlap
from 0.727 to 0.945; quintupling it reduced performance to 0.924 — a non-monotonic
result that could not have been reasoned out in advance.

Coverage below 20% is reported as the system's principal weakness. Measured on
relevance alone it would look flawless.

## Task 5 — Machine Learning Classification

Four models compared on predicting Movie vs TV Show, against a 69.7% majority baseline.

**The main work was a leakage audit.** Three feature groups encode the target:

| Feature set | Accuracy | Verdict |
|---|---|---|
| Everything | 1.0000 | Meaningless — the label in disguise |
| No duration | 0.9983 | Still leaking through genre |
| No duration or genres | 0.9471 | Mostly a metadata artifact |
| **Content features only** | **0.7622** | **The real figure** |

`duration_unit` separates the classes perfectly, and Netflix's genre taxonomy restates
content type in both directions — including by *absence*, since 2,043 TV shows carry
zero non-TV genres while no movie does.

**Model selection went against the headline metric.** Gradient Boosting scored highest
on accuracy (0.762) but misses 62% of TV shows, varies most across cross-validation
folds, and is the only model overfitting (+0.064 train/test F1 gap). Random Forest is
reported instead (F1 0.637) and saved as a reusable pipeline.

**Eight improvement attempts were measured rather than estimated** — threshold tuning,
hyperparameter search, resampling, ensembling, feature interactions and two country
encodings. None beat the baseline configuration, which is itself the evidence that the
features, not the modelling, are the constraint.

## Task 6 — Business Insights Dashboard

An interactive Streamlit application combining the analysis, the trained model and the
recommendations that follow from both.

**Five tabs** — Overview, Trends, Geography, Predict and Business insights — driven by
four sidebar filters that every chart responds to.

The **Predict** tab serves the Random Forest saved in Task 5, loaded from its pipeline
rather than retrained. It reports the model's F1 alongside the majority baseline, and
warns explicitly when confidence falls below 65% rather than presenting an uncertain
prediction with the same weight as a confident one.

The **Trends** tab greys out the 2021 bar and labels it, because the dataset ends in
September — nine months against twelve for every other year.

---

## What this dataset cannot support

Stated explicitly, because knowing the limits is part of the analysis:

- **No viewership, ratings or revenue data.** Questions about Netflix's most successful
  content are unanswerable from this source.
- **A snapshot, not a history.** Only titles present in September 2021 appear, biasing
  the picture toward content Netflix chose to keep.
- **`release_year` means different things by content type** — latest season for TV
  shows, original release for movies.
- **Director is unrecorded for roughly 30% of titles**, and no plot description or cast
  list exists in this extract.

---

## Environment

Python 3.11 with pandas, numpy, matplotlib, seaborn, scikit-learn, openpyxl, joblib,
streamlit and plotly. Exact versions in `environment.yml`.
