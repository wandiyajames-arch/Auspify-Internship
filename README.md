# Auspify Technologies — Data Science Internship

**Intern:** Wandiya James
**Program:** 4-Week Data Science Internship Program
**Dataset:** Netflix Titles (8,790 records)

Four of the six available tasks are completed here, meeting the programme's
"any 4 out of 6" requirement.

---

## Repository structure

```
Auspify-Internship/
│
├── Task-1-Data-Cleaning/
│   ├── Dataset.csv                     <- raw input
│   ├── data_cleaning.ipynb
│   ├── output/
│   │   ├── netflix_cleaned.csv         <- input for all later tasks
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
│   │   └── netflix_type_classifier.pkl  <- saved model pipeline
│   └── screenshots/
│
├── environment.yml
├── requirements.txt
└── README.md
```

**Note on data files:** only Task 1 holds a copy of the dataset. Tasks 2, 3 and 5
locate the cleaned output from `Task-1-Data-Cleaning/output/` automatically, so the
CSV is not duplicated across folders.

---

## Setup

```bash
conda env create -f environment.yml
conda activate austech
```

Or into an existing environment:

```bash
pip install -r requirements.txt
```

Run the notebooks in order — Task 1 produces the cleaned dataset the others depend on.
Each notebook runs end to end with "Restart Kernel and Run All".

---

## Tasks

### Task 1 — Data Cleaning & Preprocessing

Prepared the raw Netflix dataset for analysis.

**Key findings**
- `isna()` reported zero nulls, but 2,588 directors and 287 countries were missing
  behind the placeholder string `"Not Given"`
- One row has the legitimate title *Unknown* — a blanket placeholder replacement
  would have destroyed it, so replacement was scoped by column
- 4 duplicate records under different `show_id` values; the fourth only became
  detectable after whitespace was stripped, which is why cleaning order matters
- `duration` mixed minutes and seasons in one text column; split into a numeric
  value and a unit
- 703 runtime outliers flagged but **retained** — inspection showed them to be
  genuine short films and epics, and removal would have changed the median by
  0.3 minutes while deleting 7.3% of all movies

**Result:** 8,790 → 8,786 rows, 10 → 28 columns, 15/15 validation checks passing.

### Task 2 — Exploratory Data Analysis

Univariate, bivariate and multivariate analysis across 20 visualisations.

**Key findings**
- 67.2% of TV shows run for exactly one season; only 6% reach five
- Country mixes differ sharply: India is 92% movies, Pakistan is 83% TV shows
- Indian films run over half an hour longer than American ones at every audience level
- TV shows now arrive on Netflix the same year they are released; movies still lag
  by about a year
- 84.8% of the catalogue was released in 2010 or later
- `release_year` and `years_to_platform` correlate at −0.98 **by construction** —
  a leakage warning carried forward to Task 5

Full write-up in `Task-2-EDA/summary.md`.

### Task 3 — Recommendation System

Content-based recommender using TF-IDF and cosine similarity over genres, director,
country, type and audience category.

**Results**
- Genre overlap **0.945** against a random baseline of **0.093**
- Type match 1.000
- Catalogue coverage ~19.9% — reported as the system's main weakness

Feature weighting was tested rather than assumed: tripling genre weight raised overlap
from 0.727 to 0.945, while quintupling it reduced performance to 0.924.

Full write-up in `Task-3-Recommendation-System/report.md`.

### Task 5 — Machine Learning Classification

Four models compared on predicting Movie vs TV Show, against a 69.7% majority baseline.

**The main finding was label leakage.** Three feature groups encode the target:

| Feature set | Accuracy |
|---|---|
| Everything | 1.0000 |
| No duration | 0.9983 |
| No duration or genres | 0.9471 |
| **Content features only** | **0.7622** |

`duration_unit` separates the classes perfectly, and Netflix's genre taxonomy restates
content type in both directions — including by *absence*, since 2,043 TV shows carry
zero non-TV genres while no movie does.

**Model selection went against the headline metric.** Gradient Boosting scored highest
on accuracy (0.762) but misses 62% of TV shows, varies most across CV folds, and is the
only model overfitting (+0.064 train/test F1 gap). Random Forest is reported instead
(F1 0.637), and saved as a reusable pipeline.

**Eight improvement attempts were measured, not estimated** — threshold tuning,
hyperparameter search, resampling, ensembling, interactions and two country encodings.
None beat the baseline configuration, which is itself the evidence that the features,
not the modelling, are the constraint.

Full write-up in `Task-5-ML-Classification/report.md`.

---

## Environment

Python 3.11 with pandas, numpy, matplotlib, seaborn, scikit-learn, openpyxl and joblib.
See `environment.yml` for exact versions.
