# Task 6 — Business Insights Dashboard: Report

**Auspify Technologies — Data Science Internship Program**

| | |
|---|---|
| **Intern** | Wandiya James |
| **Task** | Task 6 (Advanced) — Data Science Business Insights Dashboard |
| **Deliverable** | `app.py` — an interactive Streamlit dashboard |
| **Inputs** | `netflix_cleaned.csv` (Task 1), `netflix_type_classifier.pkl` (Task 5) |

---

## Objective

Create an end-to-end data science project combining analytics, machine learning and business
insights into a single interactive application.

## Running it

```bash
conda activate austech
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. It locates its two input files automatically,
searching the current folder, `output/`, and the Task 1 and Task 5 folders — so it runs from
anywhere in the project tree without configuration.

If either file is missing the dashboard says which one and how to produce it, rather than
crashing.

---

## How this maps to the task workflow

| Required step | Where it happens |
|---|---|
| **1. Perform complete data analysis** | Overview, Trends and Geography tabs — descriptive statistics, distributions, time series and the country/audience cross-analysis |
| **2. Develop predictive models** | Predict tab — serves the Random Forest trained in Task 5, loaded from its saved pipeline |
| **3. Create interactive visualizations** | Plotly charts throughout, driven by four sidebar filters that every chart responds to |
| **4. Generate business recommendations** | Business insights tab — five recommendations, each tied to a measured finding |
| **5. Present findings in a professional report** | This document |

---

## Architecture

**Four sidebar filters** — year added, content type, country, audience — feed a single filtered
DataFrame that every chart reads from. Changing any filter updates the whole dashboard, and the
sidebar shows how many titles the current selection holds.

**Five tabs**, each answering a different question:

| Tab | Question it answers |
|---|---|
| Overview | What is in the catalogue? |
| Trends | How has it changed over time? |
| Geography | Where does it come from, and does that matter? |
| Predict | Can we classify an unseen title? |
| Business insights | What should be done about any of it? |

**Caching.** The dataset loads through `@st.cache_data` and the model through
`@st.cache_resource`, so neither is re-read when a filter changes. Without this, every slider
move would re-parse a 2 MB CSV and unpickle a 7.9 MB model.

**One palette** is defined at the top and used by every chart, so the dashboard reads as one
piece rather than five separately-styled pages.

---

## Machine learning integration

The Predict tab does not retrain anything. It loads the pipeline saved by Task 5 — model,
scaler and feature order together — and serves it.

The input form takes human-readable values (release year, year added, month, country, audience)
and builds the 21-feature vector internally. That matters: a model takes positional input, so
passing columns in a different order produces a confidently wrong answer with no error. Building
the vector from the saved feature list keeps that trap shut.

**The dashboard reports the model honestly.** The tab header states the test F1 (0.637) alongside
the majority baseline (0.697), so nobody reads the output as more authoritative than it is. It
also explains why duration and genre are absent — they leak the target, as Task 5 established.

**When the model is unsure, the dashboard says so.** Any prediction below 65% confidence triggers
a warning explaining that a recent American adult title genuinely could be either, and that the
model is hedging rather than being confidently wrong. A dashboard that presented a 55% prediction
with the same visual weight as a 95% one would be misleading its user.

---

## Business recommendations

Each one is tied to a measured finding rather than an impression.

### 1. Localise acquisition rather than treating the catalogue as one market

Country mix varies more than any other variable in this dataset. India is about 92% movies;
Pakistan is about 83% TV shows; Japan and South Korea lean heavily toward series.

**Action:** set acquisition targets per market by format, not globally.

### 2. The one-season pattern is a retention risk worth measuring

Two thirds of TV shows run exactly one season and only 6% reach five. Subscribers who invest in
a series and see it end abruptly have a reason to churn.

**Action:** this dataset cannot separate "designed as a limited series" from "cancelled early".
Joining commissioning intent to the catalogue would make that distinction measurable — and it is
the question worth answering before acting.

### 3. Production speed is now the competitive advantage

TV shows reach the platform the same year they are released; movies still lag by about a year.
That gap is the clearest signal in the data of the shift from licensing back catalogues to
commissioning originals.

**Action:** treat the licensing-window lag as the measurable cost of licensed film content when
comparing it against originals.

### 4. Metadata quality is limiting what can be modelled

Director is unrecorded for roughly 30% of titles, country for 3%, and no plot description or cast
list exists in this extract. Task 5 showed the consequence: a classifier using only legitimate
features reaches 76% accuracy against a 69.7% baseline, and eight separate improvement attempts
moved none of it.

**Action:** the constraint is the data, not the modelling. Capturing descriptions and cast would
do more for predictive work than any amount of tuning.

### 5. Children's content is a small but distinctly different segment

Kids and Older Kids together are under 20% of the catalogue, but US children's content is 43% TV
shows against 22% for adult content — the most TV-weighted segment there is. The Task 5 model
found this pattern independently, and Task 2's analysis had not reported it.

**Action:** treat children's content as a separate planning category with its own format mix.

---

## Limitations stated in the dashboard itself

The Business insights tab ends with what the data cannot support, which is as important as what
it can:

- **No viewership, ratings or revenue.** "What is Netflix's most successful content" is
  unanswerable from this source.
- **A snapshot, not a history.** Only titles present in September 2021 appear, biasing the picture
  toward content Netflix chose to keep.
- **`release_year` means different things by type** — latest season for TV, original release for
  movies.

The Trends tab also greys out the 2021 bar and labels it, because the dataset ends in September:
nine months against twelve for every other year. A viewer who read that bar as a decline would
draw the wrong conclusion, and the dashboard prevents it rather than relying on them knowing.

---

## Design decisions worth noting

**Plotly rather than static images.** Every chart supports hover, zoom and pan. A static PNG in a
dashboard is a report with extra steps.

**Metrics before charts.** Each tab opens with the headline numbers, so a viewer gets the answer
before deciding whether to study the detail.

**Graceful degradation.** Narrow filter selections can leave too few rows for a meaningful chart.
Rather than erroring or drawing a misleading two-point line, those charts print a short note and
the rest of the dashboard continues working.

**Responsive layout.** `st.columns` and `width="stretch"` mean the charts reflow to the browser
width rather than being fixed-size.

---

## Files

| File | Description |
|---|---|
| `app.py` | The dashboard — about 600 lines |
| `report.md` | This document |
| `screenshots/` | Captured views of the running dashboard |

**Requires** `netflix_cleaned.csv` from Task 1 and `netflix_type_classifier.pkl` from Task 5.
Both are located automatically.

## Dependencies

Beyond the existing environment, this task adds `streamlit` and `plotly` — both are in the
updated `environment.yml` and `requirements.txt`.
