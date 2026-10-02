# Task 3 — Recommendation System Analysis: Report

**Auspify Technologies — Data Science Internship Program**

| | |
|---|---|
| **Intern** | Wandiya James |
| **Task** | Task 3 (Medium) — Recommendation System Analysis |
| **Dataset** | `netflix_cleaned.csv` — 8,786 titles, 28 columns (output of Task 1) |
| **Notebook** | `recommendation_system.ipynb` |
| **Approach** | Content-based filtering, TF-IDF + cosine similarity |

---

## Objective

Develop a content-based recommendation model using Netflix titles and categories, then evaluate whether
its recommendations are actually relevant rather than merely plausible-looking.

## Why content-based, and not collaborative filtering

There are two broad families of recommender. **Collaborative filtering** learns from user behaviour —
people who watched X also watched Y. **Content-based filtering** learns from the items themselves — Y
resembles X, so someone who liked X may like Y.

This dataset contains no user data at all: no ratings, no watch history, no viewing figures. That rules
out collaborative filtering entirely, making this content-based by necessity rather than by choice.

Stating that plainly matters, because it also bounds what the system can do. It can find *similar*
content. It can never know what is *popular* or *good*.

---

## Method

### Feature selection

Five columns were used, chosen because each carries signal about what a title actually is:

| Feature | Why included |
|---|---|
| `genres` | The strongest descriptive signal available |
| `director` | Directors have recognisable styles; known for ~70% of titles |
| `primary_country` | Captures regional production style and language |
| `type` | A movie and a TV show are rarely interchangeable recommendations |
| `audience_category` | A children's title should not be recommended for an adult one |

Deliberately excluded: `title` (names rarely indicate content), `release_year` (recency is not
similarity), `date_added` (describes Netflix's schedule, not the content) and `duration` (a 90-minute
comedy is not similar to a 90-minute horror).

**The important absence is a plot description.** Most published Netflix recommenders use a `description`
field, which carries far more signal than genre tags alone. This dataset has none, and that ceiling is
visible in the results.

### Text preprocessing — two problems solved

**Problem 1: multi-word values split into meaningless tokens.** A vectoriser splits on whitespace, so
`"United States"` becomes `united` and `states`. *United States* and *United Kingdom* would then share
the token `united` and be scored as partly similar — they share a word, not a property. Each categorical
value was therefore collapsed to a single indivisible token: `unitedstates`.

**Problem 2: every feature counting equally.** With one token each, a shared country would count as much
as a shared genre. Two American titles are not similar simply because both are American, but two horror
films genuinely are. Weighting was applied by **token repetition** — and then tested rather than assumed.

### Why TF-IDF rather than raw counts

A plain count vectoriser treats every token as equally informative. TF-IDF down-weights tokens that
appear everywhere and up-weights rare ones.

That matters directly here. Task 2 found *International Movies* to be the single largest genre tag, on
2,751 titles. Under raw counts, two titles sharing only that tag would look strongly similar — but a tag
on nearly a third of the catalogue tells you almost nothing.

The IDF scores confirm the vectoriser worked this out on its own:

| Token | IDF | Reading |
|---|---|---|
| `movie` | 1.36 | Lowest — nearly universal, heavily discounted |
| `adults` | 1.79 | Very common |
| `unitedstates` | 2.00 | Common |
| `internationalmovies` | 2.16 | Common — discounted exactly as intended |
| `aamirkhan` | 9.39 | Rare — weighted heavily |

The resulting matrix is 8,786 × 4,660 tokens and **99.87% empty**, which is why a sparse representation
matters at this scale.

### Computing similarity without a 309 MB matrix

The textbook approach precomputes the full pairwise cosine similarity matrix: 8,786 × 8,786 values,
roughly **309 MB** in float32, and slow to build.

It is also unnecessary. A recommendation needs one row — the query against everything else. Using
`linear_kernel` to compute that single row on demand returns results in milliseconds with no
precomputation at all.

(Cosine similarity and `linear_kernel` are identical here because TF-IDF vectors are already
L2-normalised, which makes the dot product equal to the cosine.)

---

## A bug worth naming

Many tutorials drop the query title from its own recommendations by taking `argsort()[1:11]` — skipping
whatever lands in first position.

**That is not reliable.** Several titles in this dataset share an identical feature soup and therefore
score exactly 1.0 against each other. When ties occur, the query title may not sort into first place, so
slicing from position 1 can leave the query in its own results while silently dropping a genuine
recommendation.

This implementation excludes the query **by its index**, not by its sort position.

---

## Results

### Evaluation methodology

There is no ground truth here. Nobody has labelled which recommendations are correct, and with no user
data there is no click-through rate to measure against.

What can be measured is whether recommendations are **more relevant than chance**, using the catalogue's
own labels:

| Metric | What it measures |
|---|---|
| Genre overlap (Jaccard) | Share of genres common to query and recommendation |
| Type match rate | How often a Movie query returns Movies |
| Catalogue coverage | Share of the catalogue that ever gets recommended |
| Random baseline | The same metrics for randomly picked titles |

**The baseline is the part that matters.** A Jaccard score of 0.7 means nothing in isolation — it only
becomes evidence once you know what random picking scores.

### Feature weighting was tested, not assumed

The hypothesis was that genres should count for more than country. Four schemes were measured over 300
queries returning 10 titles each:

| Weighting | Genre overlap | Type match | Coverage |
|---|---|---|---|
| Random baseline | 0.093 | 0.582 | 28.7% |
| Equal (1,1,1) | 0.727 | 0.950 | 20.0% |
| Genre ×2 (2,1,1) | 0.904 | 1.000 | 20.6% |
| **Genre ×3 (3,1,1)** | **0.945** | **1.000** | **19.9%** |
| Genre ×5, director ×2 | 0.924 | 1.000 | 17.1% |

**The hypothesis was correct, but only up to a point.** Tripling genre weight lifted overlap from 0.727
to 0.945. Quintupling it dropped performance back to 0.924, because pushing genre weight too high
effectively erases director and country from the vector — the model starts matching titles that share
tags but nothing else.

That non-monotonic result could not have been reasoned out in advance. It is a small demonstration of a
general point: more weight on the obvious feature is not automatically better, and measuring is the only
way to find the turn.

**The 3× scheme was adopted for the final model.**

### Final performance

| Metric | Result | Baseline | Interpretation |
|---|---|---|---|
| Genre overlap | **0.945** | 0.093 | Roughly ten times better than chance |
| Type match | **1.000** | 0.582 | Content type respected perfectly |
| Catalogue coverage | **19.9%** | 28.7% | Below random — the principal weakness |

---

## Key findings

**1. The system works, by a wide margin.** Genre overlap of 0.945 against a random baseline of 0.093 is
not a marginal result. The similarity matrix over a sample of well-known titles partitions them exactly
as a person would: three American crime dramas score highly against each other and near zero against an
Indian film, a documentary and a children's title.

**2. Coverage is the honest weakness.** Across 300 queries, under a fifth of the catalogue was ever
recommended — lower than random picking achieves. This is the **popularity-cluster problem**: titles in
dense regions of feature space (American adult dramas, of which there are hundreds) get recommended
constantly, while titles in sparse regions are almost never surfaced. It is the standard failure mode of
content-based systems, not a bug to be patched away.

**3. Similarity scores are close to binary.** 22.6% of returned recommendations score above 0.95 —
near-identical feature profiles the model genuinely cannot distinguish. With only five categorical
features there are a limited number of distinct combinations, so many titles are indistinguishable to
it. Scores sit near 1 or near 0 with little in between.

**4. Ranking within a cluster is close to arbitrary.** The top ten scores for any query are tightly
bunched. The model is confident about *which cluster* to draw from but has little basis for ordering
within it — the difference between the first and tenth recommendation is small enough to be noise.

**5. The recommendations are coherent but narrow.** Indian queries return Indian titles; children's
queries return children's titles; nothing crosses between clusters. Relevance and diversity pull against
each other, and this model is tuned entirely toward relevance.

---

## Limitations

**No plot or cast data.** The single biggest constraint. Two crime dramas with completely different
tones are identical to this model. The 22.6% figure above quantifies the cost.

**No notion of quality or popularity.** Without viewing data the system cannot tell a beloved film from
an ignored one. It finds similar content, never good content.

**Coverage under 20%.** Sparse regions of feature space are effectively invisible.

**Cold start.** A new title with no genre tags cannot be positioned at all.

**Inherited metadata bias.** As Task 2 found, *International Movies* is defined relative to a US viewer.
The model treats that framing as a content property.

**Director sparsity.** Director is unknown for roughly 30% of titles, so that feature contributes
nothing for those rows and the model leans harder on the remaining four.

## On the evaluation itself

The metrics used here are **proxies, not ground truth**. Genre overlap measures whether recommendations
share tags with the query — but sharing tags is not the same as being a good recommendation. A perfect
score on this metric could be achieved by a system returning near-duplicates, which would be useless in
practice.

That is precisely why coverage was reported alongside relevance. Measured on relevance alone this system
looks flawless; measured on both, its actual shape becomes visible. Any single-metric evaluation of a
recommender should be treated with suspicion, including this one.

---

## Future improvements

| Improvement | Expected effect | Effort |
|---|---|---|
| Add plot descriptions and cast | Largest single gain — would separate the 22.6% of pairs currently scoring above 0.95 | Needs a richer data source |
| Diversity re-ranking (MMR) | Directly addresses coverage by penalising recommendations too similar to ones already chosen | Moderate |
| Sentence embeddings instead of TF-IDF | Captures semantic similarity rather than exact shared tags — "heist thriller" and "crime caper" would relate | Moderate |
| Hybrid with collaborative filtering | Adds popularity and quality signals | Requires user data this dataset lacks |
| Popularity tie-break within clusters | Cheap fix for the arbitrary-ranking problem | Low, but needs a popularity signal |
| Evaluation against human judgement | A labelled relevance set would validate the proxy metrics | High |

---

## Conclusion

This task built a content-based recommendation system using TF-IDF vectorisation and cosine similarity
over five features: genres, director, country, content type and audience category.

**The system works.** Genre overlap reached 0.945 against a random baseline of 0.093, with content type
matched perfectly.

**Three decisions shaped the outcome.** Feature weighting was tested rather than assumed, and the test
changed the final design — the non-monotonic result at 5× genre weight would not have been predicted.
The query title is excluded by index rather than sort position, avoiding a silent failure that the common
tutorial approach introduces when feature soups tie at exactly 1.0. Similarity is computed on demand,
avoiding a 309 MB precomputation for no loss in capability.

**The honest assessment.** Relevance is strong, but coverage sits under 20% and 22.6% of recommendations
score above 0.95 — near-identical feature profiles the model cannot separate. Both trace to the same
root cause: five categorical features are not enough to describe a film. The missing `description` column
is the ceiling on this approach, and no amount of tuning moves it.

A system evaluated only on relevance would have looked perfect. Reporting coverage alongside it is what
makes the result meaningful rather than merely favourable.

---

## Handover

| Task | What this contributes |
|---|---|
| **Task 5** — Classification | The feature soup does **not** transfer. It is built mainly from genres, and Task 5's leakage audit shows Netflix's genre taxonomy encodes content type — reusing it there would leak the label. The TF-IDF technique itself transfers; Task 5 applies it to title text instead, where it was tested and found to add no usable signal |
| **Task 6** — Dashboard | `recommend()` is self-contained and can be wired to a title selector as an interactive demo |

## Files

| File | Description |
|---|---|
| `recommendation_system.ipynb` | Full implementation and evaluation, 52 cells, 5 visualisations |
| `report.md` | This document |

**Input:** `netflix_cleaned.csv` from Task 1.
