# Task 2 — Exploratory Data Analysis: Summary of Findings

**Auspify Technologies — Data Science Internship Program**

| | |
|---|---|
| **Intern** | Wandiya James |
| **Task** | Task 2 (Easy) — Exploratory Data Analysis |
| **Dataset** | `netflix_cleaned.csv` — 8,786 titles, 28 columns (output of Task 1) |
| **Notebook** | `Task2_Exploratory_Data_Analysis.ipynb` |
| **Coverage** | Netflix catalogue as of September 2021 |

---

## Objective

Explore the cleaned Netflix dataset to uncover trends, patterns and insights, and establish what the
data can and cannot support before modelling begins in Tasks 3 to 6.

## Approach

The analysis follows the standard EDA progression:

| Level | Question | Sections |
|---|---|---|
| Univariate | What does a single variable look like on its own? | 2, 3, 4 |
| Bivariate | How do two variables relate? | 5, 6 |
| Multivariate | What happens when three or more interact? | 7 |

20 visualisations were produced, each with a written interpretation. All analysis runs on the cleaned
dataset from Task 1, so derived fields (`year_added`, `primary_genre`, `audience_category`,
`years_to_platform`) were available from the start.

---

## Key findings

### 1. Movies dominate, but the balance is shifting

Movies outnumber TV shows roughly **70 to 30** (6,123 against 2,663). That figure is *cumulative*
though, and the recent flow looks different: TV shows fell to 25% of yearly additions in 2018, then
climbed steadily to **33.7% by 2021**. The stock and the flow tell different stories.

### 2. Two thirds of Netflix shows run for exactly one season

**1,790 of 2,663 shows (67.2%)** have a single season, and only **160 (6.0%)** reach five or more.
Some of this is limited-series and documentary formats designed to end after one run; some is Netflix's
documented pattern of early cancellation. Either way, a viewer picking a random Netflix series is far
more likely to be starting something self-contained than a long-running saga.

### 3. The catalogue is overwhelmingly recent

**84.8%** of titles were released in 2010 or later; only **2.9%** predate 1990. Netflix is a
current-content library, not a film archive.

### 4. Genuinely international, but with a US-centric vocabulary

The United States supplies about 38% of titles, but the top ten producing countries include India
(1,056), the United Kingdom (638), Pakistan (420), Canada, Japan, South Korea, France and Spain.

Notably, **"International Movies" is the single largest genre tag** (2,751 titles) — larger than Dramas
or Comedies. That label is defined relative to a US viewer rather than describing the content, which
says as much about Netflix's metadata conventions as about the catalogue itself.

### 5. Country mixes differ enormously

| Country | Movies | TV Shows |
|---|---|---|
| India | 92.3% | 7.7% |
| United States | 73.9% | 26.1% |
| United Kingdom | 60.7% | 39.3% |
| Pakistan | 16.9% | **83.1%** |

India and Pakistan are near-mirror images. There is no single answer to what Netflix's catalogue looks
like — it depends entirely which country you ask about.

### 6. Growth peaked in 2019

Additions climbed steeply from 2015, peaked at **2,013 titles in 2019**, then fell through 2020 as
COVID halted production worldwide.

**Important caveat:** the 2021 figure (1,498) covers nine months only — the dataset ends in September.
The year-month heatmap in Section 7 shows October to December 2021 as empty cells, not low ones.

### 7. TV shows now arrive the same year they are released

Median lag between release and arriving on Netflix:

- **TV shows: 0 years** — effectively simultaneous
- **Movies: 1 year** — reflecting theatrical and home-release windows

This is the clearest signal in the dataset of Netflix's shift from licensing other studios' back
catalogues to commissioning its own content.

### 8. Movies have got shorter

Median runtime by release period fell from **109 minutes (mid-1990s)** to **95 minutes (mid-2010s)** —
roughly a quarter of an hour in twenty years. The scatter plot also shows the *spread* narrowing, with
recent films clustering tightly around 90 to 100 minutes.

This reflects the films Netflix carries, not cinema as a whole; separating the two would need data this
dataset does not contain.

### 9. Indian films run over half an hour longer than American ones

The strongest multivariate finding. Median movie runtime by country and audience:

| Country | Teens | Adults |
|---|---|---|
| India | **130 min** | **118 min** |
| United States | 95 min | 94 min |
| United Kingdom | 96 min | 97 min |
| Pakistan | 117 min | 65 min |

The pattern holds across every audience level, so it is not one outlier dragging an average — it is the
Bollywood convention of longer, often musical features, visible directly in the data.

This only appears at the multivariate level. Runtime alone gives a single global median of 98 minutes;
country alone gives counts; neither reveals that "typical movie length" means something different
depending on where a film was made.

### 10. Longer films carry more genre tags

Correlation of **0.41** between runtime and number of genres assigned — modest but real. Whether that
reflects genuinely richer content or simply more thorough metadata on bigger productions cannot be
settled with this data.

### 11. The audience mix has barely moved

Adults hold steady around 45% of yearly additions and teens around 40%, year after year. Together they
account for over 80% of the catalogue — Netflix markets itself as a family service, but the library is
built for grown-ups.

The one visible movement, *Unrated* content shrinking to nothing after 2019, is **Netflix's metadata
getting more complete, not a content trend**. Changes in how data is recorded are easily mistaken for
changes in what is happening.

### 12. Seasonality is weak

July (827) and December (812) are the busiest months, February (562) the quietest. The spread is modest
and inconsistent year to year. A useful negative result: worth checking, worth reporting, and worth not
over-interpreting.

---

## A methodological finding: derived-column correlation

The correlation matrix's strongest relationship is **meaningless**, and recognising that matters more
than the chart itself.

`release_year` and `years_to_platform` correlate at **-0.98**. This is not a discovery about Netflix —
`years_to_platform` was *calculated* as `year_added - release_year` during Task 1, so the two columns
are algebraically linked. The same applies to `duration_value` against `movie_minutes` and
`tv_seasons`.

**Consequence for Task 5:** feeding a derived feature and its parent into the same model is textbook
data leakage. The model would appear to perform brilliantly because it had been handed the answer in
disguise. `release_year` and `years_to_platform` must not both be used as features.

---

## Limitations of the dataset

Stating these explicitly is part of the analysis:

- **No viewership, ratings or revenue data.** Nothing indicates how popular anything was, so questions
  about Netflix's most successful content are unanswerable here.
- **It is a snapshot, not a history.** Only titles present in September 2021 appear. Anything removed
  before then is invisible, biasing the picture toward content Netflix chose to keep.
- **`release_year` means different things by content type** — latest season for TV shows, original
  release for movies (established in Task 1). Comparisons across types on this field need care.
- **Director is unknown for about 30% of titles**, so director-level findings cover roughly 70% of the
  catalogue.
- **Several columns are derived from others** and their correlations describe arithmetic, not Netflix.

---

## Handover to later tasks

| Task | What this analysis provides |
|---|---|
| **Task 3** — Recommendation system | Genre tags are multi-valued and US-centric. `genres` and `primary_genre` are ready to use; expect "International Movies" to dominate similarity scores unless handled |
| **Task 4** — Trend prediction | 2021 is a partial year (9 of 12 months) and must be excluded or annualised. The 2015–2019 growth curve is the real signal |
| **Task 5** — Classification | Country, duration and genre separate Movies from TV Shows strongly. Country and runtime **interact** — Indian films are systematically longer. Do not use `release_year` and `years_to_platform` together |
| **Task 6** — Dashboard | Strongest stories to lead with: the growth curve and its 2019 peak, the country mix contrast, and the one-season finding |

---

## Files produced

| File | Description |
|---|---|
| `Task2_Exploratory_Data_Analysis.ipynb` | Full analysis, 85 cells, 20 visualisations |
| `summary.md` | This document |

**Input:** `output/netflix_cleaned.csv` from Task 1.
