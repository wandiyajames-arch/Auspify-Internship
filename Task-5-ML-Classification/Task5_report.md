# Task 5 — Machine Learning Classification Model: Report

**Auspify Technologies — Data Science Internship Program**

| | |
|---|---|
| **Intern** | Wandiya James |
| **Task** | Task 5 (Advanced) — Machine Learning Classification Model |
| **Dataset** | `netflix_cleaned.csv` — 8,786 titles, 28 columns (output of Task 1) |
| **Notebook** | `ml_classification.ipynb` |
| **Target** | Content type — Movie vs TV Show |

---

## Objective

Build and compare machine learning models that classify Netflix content from the available features.

## The target and the number that matters

Content type is the natural target: complete for every row, meaningful, and reasonably balanced.

| Class | Count | Share |
|---|---|---|
| Movie | 6,123 | 69.7% |
| TV Show | 2,663 | 30.3% |

**The majority baseline is 69.7%.** A classifier that ignores its inputs entirely and always answers
"Movie" would be right roughly seven times in ten. Every model has to beat that, and accuracy only
slightly above it is barely learning anything.

Quoting accuracy without quoting the baseline is one of the most common ways classification results get
oversold.

---

## The main work was feature selection, not modelling

This target turned out to be surrounded by traps. Before selecting anything, every candidate feature was
checked against one question: **could this column only be known because we already know the answer?**

### Leakage check 1 — `duration_unit`

| | Season | min |
|---|---|---|
| **Movie** | 0 | 6,123 |
| **TV Show** | 2,663 | 0 |

A perfect separator. Every movie is measured in minutes, every TV show in seasons. This column does not
*predict* the type — it **is** the type, relabelled.

The same applies to `movie_minutes` and `tv_seasons`, created in Task 1 by splitting on this exact
distinction, and to `duration_value`, whose scale differs so sharply between the two (3–312 against 1–17)
that it gives the answer away almost as reliably.

### Leakage check 2 — the genre taxonomy

19 of the 42 genre tags contain the word "TV" (*Crime TV Shows*, *TV Dramas*, *Reality TV*, *Kids' TV*…).

| | No TV genre | Has TV genre |
|---|---|---|
| **Movie** | 6,123 | **0** |
| **TV Show** | 104 | 2,559 |

Not a single movie carries a TV genre tag. Netflix's genre vocabulary is built around content type, so
these tags restate the label.

**The obvious fix does not work.** Dropping the TV-prefixed genres leaves this list: *Horror Movies*,
*Romantic Movies*, *Classic Movies*, *Independent Movies*, *International Movies*. The word was removed
from one side of the taxonomy but the other side still says **Movies** explicitly.

And there is leakage by **absence**:

| Non-TV genres per title | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **Movie** | **0** | 1,476 | 2,240 | 2,407 |
| **TV Show** | **2,043** | 616 | 4 | 0 |

2,043 TV shows have zero non-TV genres, while no movie has zero. A model would simply learn "all genre
flags off means TV show" — still reading the label.

**Conclusion: the genre taxonomy cannot be used for this target at all.** Not the TV tags, not the
remaining tags, not a count of them. This is a real cost — genre is the richest descriptive field in the
dataset, and Task 3 built an entire recommender on it.

### Leakage check 3 — `director_known`

| | Director missing | Director known |
|---|---|---|
| **Movie** | 2.8% | 97.2% |
| **TV Show** | **90.6%** | 9.4% |

A borderline case, and it deserves a different verdict from the first two. This is not a logical identity
like `duration_unit` — it is a **metadata artifact**, reflecting Netflix's practice of not assigning a
single director to a series.

It is genuinely available at prediction time, so it is not leakage in the strict sense. But a model
leaning on it has learned something about Netflix's cataloguing conventions rather than about content.
Rather than decide by assertion, its contribution was **measured** (see the progression table below).

---

## Methodology

**Four feature sets** were built to quantify what each exclusion costs, rather than presenting the final
model alone.

**Stratified 80/20 split** with a fixed seed. With a 70/30 class split, a random partition can easily
produce a test set with a different balance from the training set; `stratify=y` preserves the proportion
in both halves.

**Scaling fitted on the training set only.** Logistic regression is sensitive to feature scale —
`release_year` runs in the thousands while binary flags are 0 or 1. Fitting the scaler on all the data
first would leak test-set statistics into training, a subtler version of the same problem the audit was
about.

**Four algorithms with different inductive biases**, so the comparison is informative rather than four
variations on one idea: Logistic Regression (linear boundary), Decision Tree (threshold splits), Random
Forest (bagged trees), Gradient Boosting (sequential error correction). `class_weight="balanced"` was set
where supported.

---

## Results

### Model comparison — honest feature set

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | vs baseline |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7065 | 0.5124 | 0.6604 | 0.5770 | 0.7639 | +0.010 |
| Decision Tree | 0.6860 | 0.4891 | **0.8011** | 0.6074 | 0.7851 | **−0.011** |
| **Random Forest** | 0.7423 | 0.5560 | 0.7448 | **0.6367** | 0.8101 | +0.046 |
| Gradient Boosting | **0.7622** | **0.7003** | 0.3771 | 0.4902 | **0.8178** | +0.065 |

**No single model won outright, and the disagreement is informative.**

The Decision Tree is **negative against the baseline** — 68.6% accuracy against 69.7%, meaning it is less
accurate than always answering "Movie". Yet its recall is 0.801, the highest of the four. That is
`class_weight="balanced"` working exactly as instructed: it pushes the model to find TV shows, and the
price is more movies wrongly flagged. Judged on accuracy it looks broken; judged on F1 it is the
second-best model here.

Gradient Boosting takes accuracy by playing safe on the majority class — its recall is only 0.377, so it
**misses nearly two thirds of all TV shows**. Random Forest takes F1 with a far more usable balance of
0.556 precision against 0.745 recall.

Which is "best" depends entirely on whether missing a TV show or mislabelling a movie costs more — a
question the data cannot answer.

### Cross-validation — is the result stable?

| Model | CV F1 (mean) | CV F1 (sd) | Test F1 |
|---|---|---|---|
| Logistic Regression | 0.5525 | 0.0146 | 0.5770 |
| Decision Tree | 0.5904 | 0.0040 | 0.6074 |
| Random Forest | 0.6020 | 0.0064 | 0.6367 |
| Gradient Boosting | 0.4959 | **0.0415** | 0.4902 |

Cross-validated F1 sits close to the single-split test F1 for every model, so the test result was not a
lucky partition.

The standard deviations are worth a look too. Three models sit around 0.004–0.015, but Gradient Boosting
is at 0.042 — several times more variable across folds. Combined with its weak recall, that makes its
headline accuracy advantage look less solid than it first appears.

### Confusion matrix — Random Forest (best F1)

| | Predicted Movie | Predicted TV Show |
|---|---|---|
| **Actual Movie** | 908 | 317 |
| **Actual TV Show** | 136 | 397 |

The errors are lopsided. More movies are wrongly flagged as TV shows (317) than TV shows missed (136) —
the balanced class weighting trading precision for recall.

Structurally, the reason is simple: without genre tags or duration, an American adult drama released in
2018 and added in 2019 looks essentially identical whether it is a film or a series. The features that
would separate them are exactly the ones the audit ruled out.

### All four models compared

Test set: 1,225 Movies, 533 TV Shows.

| Model | Movies correct | Movies called TV | TV missed | TV correct | % of TV missed | % of movies false-flagged |
|---|---|---|---|---|---|---|
| Logistic Regression | 890 | 335 | 181 | 352 | 34.0% | 27.3% |
| Decision Tree | 779 | 446 | 106 | 427 | **19.9%** | 36.4% |
| **Random Forest** | 908 | 317 | 136 | 397 | 25.5% | 25.9% |
| Gradient Boosting | 1,139 | 86 | 332 | 201 | **62.3%** | **7.0%** |

**These are not better and worse versions of each other — they are four positions on one trade-off.**

Gradient Boosting flags almost nothing as a TV show: a 7.0% false-alarm rate, by far the cleanest, at the
cost of missing **62.3% of all TV shows**. Its high accuracy comes almost entirely from being right about
movies, which are 70% of the data. For a user asking "is this a series?", missing nearly two in three is
close to useless.

The Decision Tree is the opposite — 19.9% missed, the best recall of the four, but false-flagging over a
third of all movies.

Random Forest is the only model where the two error rates sit close together (25.5% and 25.9%). It is best
at neither individually, and that balance is precisely why it has the highest F1.

**Which model is right depends on which mistake costs more.** A content-tagging system that must not
mislabel films would prefer Gradient Boosting's 7% false-alarm rate; an audit trying to find every series
would prefer the Decision Tree's 20% miss rate. With no stated cost for either error, the balanced option
is the defensible default — which is why Random Forest is the model reported and saved.

---

## The central result: what each exclusion costs

| Feature set | Features | Accuracy | F1 | ROC-AUC | Verdict |
|---|---|---|---|---|---|
| A: Everything | 67 | **1.0000** | 1.0000 | 1.0000 | Meaningless — the label in disguise |
| B: No duration | 65 | 0.9983 | 0.9972 | 1.0000 | Still leaking through genre |
| C: No duration or genres | 22 | 0.9471 | 0.9105 | 0.9667 | Mostly the director artifact |
| **D: Content features only** | 21 | **0.7622** | 0.4902 | 0.8178 | **The real figure** |

**Read that table from the top down and it tells the whole story of this task.**

Set A scores a flawless 1.0000 on every metric. It looks like a triumph and is worth nothing —
`duration_unit` is the label wearing a different name. A model can be perfect and useless at the same
time.

Set B drops duration but keeps genres, and still scores 99.8% because the genre taxonomy restates the
type in both directions.

Set C drops genres too, and the score falls to 94.7% — most of what remains comes from
`director_known`, the metadata artifact.

Set D, the honest set, reaches **76.2%** against a 69.7% baseline, with F1 falling from 1.00 to 0.49.

**That final figure is the real one.** A 6.5-point gain over the baseline is modest — and it is the only
number here describing actual predictive learning rather than a relabelled column.

A submission reporting 99.8% would look far better at a glance and would be describing nothing at all.

---

## A second feature tested and rejected: title text

With genre and duration ruled out, one source of signal remained untried: the **title itself**, vectorised
with TF-IDF — the technique Task 3 used for the recommender.

**The vocabulary was screened for leakage first.** One genuine leak emerged: the token `series` appears in
20 titles, every one a TV show, and was removed. The remaining high-ratio tokens are franchise names
(*transformers*, *beyblade*) appearing in only one class — not leakage, but memorisable identifiers that
invite overfitting.

**Tested across 8 random train/test splits**, because a single split would not settle a difference this
small:

| Model | Metric | Mean change | Improved in |
|---|---|---|---|
| Gradient Boosting | accuracy | **+0.0065** | 7 of 8 seeds |
| Gradient Boosting | F1 | −0.0002 | 2 of 8 seeds |
| Random Forest | accuracy | −0.0058 | 0 of 8 seeds |
| Random Forest | F1 | **−0.0171** | 0 of 8 seeds |

**The accuracy gain is real but hollow.** Gradient Boosting gains 0.7 points consistently — in 7 of 8
seeds — yet its F1 does not move at all. That combination has only one explanation: the extra accuracy
comes from predicting "Movie" more often, not from identifying TV shows any better. On a 70/30 target,
leaning harder on the majority class raises accuracy while achieving nothing.

Random Forest is worse off outright, losing F1 in **8 of 8 seeds**.

This is the same lesson as the model comparison, arriving from a different direction: **accuracy alone is
untrustworthy on an imbalanced target.** Judged on accuracy, title text would have looked like a modest
success worth adopting.

It also makes sense on reflection. A title is a *name*, not a description. Nothing about the words in
*Ozark* indicates whether it is a film or a series — only familiarity with the work does, and that is
knowledge the model does not have.

**Title features were not adopted.**

---

## Overfitting check — the best-accuracy model is the only one overfitting

Cross-validation showed the test scores were stable. It does not show whether a model has *memorised* its
training data — for that, training score has to be compared against test score directly.

| Model | Train F1 | Test F1 | Gap | Train acc | Test acc |
|---|---|---|---|---|---|
| Logistic Regression | 0.5622 | 0.5770 | −0.015 | 0.6880 | 0.7065 |
| Decision Tree | 0.5984 | 0.6074 | −0.009 | 0.6747 | 0.6860 |
| Random Forest | 0.6409 | 0.6367 | +0.004 | 0.7385 | 0.7423 |
| **Gradient Boosting** | 0.5539 | 0.4902 | **+0.064** | 0.7921 | 0.7622 |

Three models have a gap of effectively zero — Logistic Regression and the Decision Tree actually score
slightly *better* on test than on training, which happens with regularised or depth-limited models on
noisy data.

**Gradient Boosting is the exception**, with a training F1 0.064 above its test F1 and training accuracy
nearly 3 points above test. It is the only model fitting noise.

That makes **three independent diagnostics pointing the same way**:

| Diagnostic | Gradient Boosting | Reading |
|---|---|---|
| Recall | 0.377 | Misses nearly two thirds of TV shows |
| CV standard deviation | 0.042 | Several times more variable than the others |
| Train/test F1 gap | +0.064 | The only model overfitting |

Its accuracy of 0.762 is the highest figure in this task, and all three diagnostics say not to trust it.
**Random Forest is the model to report**, despite scoring 2 points lower on the metric most people quote.

---

## Feature importances — and a correction to how they were measured

`feature_importances_` on a tree ensemble is **Gini importance** — how much each feature reduced impurity
across all splits. It is the default, and it has a known bias: features with more distinct values offer
more split points, so they accumulate importance whether or not they genuinely help.

This feature set is exactly where that bias bites — three continuous features (`release_year` with 74
distinct values, `year_added` with 14, `month_added` with 12) against eighteen binary flags.

**Permutation importance** avoids the problem by shuffling one column at a time and measuring how much F1
actually drops, on the test set. Both were computed:

| Feature | Gini | Gini rank | Permutation (F1 drop) | Perm rank | Shift |
|---|---|---|---|---|---|
| `release_year` | 0.244 | 1 | 0.132 | 1 | — |
| `country_Pakistan` | 0.153 | 2 | 0.038 | 3 | −1 |
| `country_India` | 0.127 | 3 | 0.033 | 4 | −1 |
| `year_added` | 0.089 | 4 | 0.051 | **2** | **+2** |
| `month_added` | 0.081 | 5 | 0.010 | **10** | **−5** |
| `country_South Korea` | 0.064 | 6 | 0.014 | 8 | −2 |
| `country_Japan` | 0.051 | 7 | 0.024 | 5 | +2 |
| `audience_Kids` | 0.031 | 8 | 0.020 | 6 | +2 |

**The headline conclusion survives both methods.** `release_year` is first either way, and country
features cluster near the top — with Pakistan and India leading, exactly the contrast Task 2 identified
(83% TV shows against 92% movies). The model rediscovered that structure independently.

**But `month_added` falls from 5th to 10th.** Gini gave it 0.081, which looked meaningful; shuffling it
costs only 0.010 in F1. Its apparent importance was an artifact of having twelve values to split on, not
evidence that the month a title was added predicts anything — which agrees with Task 2's separate finding
that seasonality here is weak and inconsistent.

`year_added` moves the other way, rising from 4th to 2nd: fewer split points, but the model genuinely
relies on it.

**The practical lesson:** the convenient default is slightly misleading. When a feature set mixes
continuous and binary columns — as almost every real one does — permutation importance is the one to
report.

---

## Saving and reusing the model

A trained model that exists only inside a notebook session is not usable. Three things are saved together
as a single pipeline, and omitting any one breaks it silently rather than loudly:

| What | Why |
|---|---|
| The fitted model | The thing that predicts |
| The fitted scaler | New data must be scaled with the **training** statistics, not its own |
| The feature names, in order | A model takes positional input — a reordered column gives a confidently wrong answer with no error |

Saved as `output/netflix_type_classifier.pkl` (7.9 MB — a Random Forest stores 300 fitted trees).

A `predict_type()` helper then takes human-readable arguments (release year, year added, month, country,
audience) and builds the 21-feature vector itself, which is what keeps the column-ordering trap shut.

### The model surfaced something the EDA missed

Six scenarios were tested with known-plausible expectations. Five matched. The sixth did not:

| Scenario | Predicted | P(TV Show) | Prior expectation |
|---|---|---|---|
| Pakistani adult drama, 2020 | TV Show | 0.887 | TV Show ✓ |
| Indian adult film, 2019 | Movie | 0.268 | Movie ✓ |
| US adult title, 2018 | Movie | 0.437 | Movie ✓ |
| South Korean teen title, 2021 | TV Show | 0.813 | TV Show ✓ |
| Old US film, 1995 | Movie | 0.252 | Movie ✓ |
| **US kids title, 2020** | **TV Show** | **0.753** | Movie ✗ |

The expectation was Movie, since the catalogue is 70% movies overall. Checking the data instead of
assuming the model was wrong:

| US audience category | Movie | TV Show |
|---|---|---|
| Adults | 77.6% | 22.4% |
| **Kids** | 56.8% | **43.2%** |
| Older Kids | 76.3% | 23.7% |
| Teens | 70.9% | 29.1% |

Kids is by far the most TV-heavy US category. Narrowed to titles released from 2018 onward — which is what
the scenario described — US kids content is **majority TV Show outright: 61 shows against 50 movies**.

**The model was right and the expectation was wrong.** It found a real pattern that Task 2 never reported:
that analysis examined how the audience mix changed over time, and how the movie/TV balance shifted, but
never crossed those two variables.

A model disagreeing with your expectation is either a bug or a finding, and the only way to tell is to go
back to the data. Here it was a finding — and it is a reminder that EDA is never exhaustive. There are
always crossings you did not try.

---

## Key findings

**1. Three separate leakage sources had to be identified and removed.** `duration_unit` is a perfect
separator; the genre taxonomy restates type in both directions; `director_known` is a metadata artifact
contributing most of the signal in Set C.

**2. Genre leaks by absence as well as presence.** Dropping the TV-prefixed tags is not enough — 2,043
TV shows have zero non-TV genres while no movie does, so the pattern of missing flags carries the label.
This is the subtlest of the three and the easiest to miss.

**3. The honest model gains 6.5 points over baseline.** 76.2% against 69.7%. Modest, and real.

**4. Algorithms converge; features are the limit.** Accuracy spans only about 7 points across four very
different algorithms, and ROC-AUC spans even less (0.76 to 0.82). When models with such different
inductive biases land this close, swapping in something more powerful would not help much.

**5. The wide F1 spread alongside the narrow accuracy spread is the signature of an imbalanced problem.**
The models are not really disagreeing about the data — they are sitting at different points on the same
precision/recall trade-off.

**6. Title text adds no usable signal**, confirmed across 8 seeds.

**7. The best-accuracy model is the only one overfitting.** Gradient Boosting has a train/test F1 gap of
+0.064 while the other three sit at roughly zero. Together with its weak recall and high CV variance, that
is three independent reasons to report Random Forest instead.

**8. The default feature-importance measure overstated one feature.** `month_added` ranks 5th on Gini
importance and 10th on permutation importance — an artifact of its twelve distinct values. The
high-cardinality bias in Gini importance is worth knowing about on any mixed feature set.

**9. The model found a pattern the EDA missed.** US kids content is 43.2% TV shows against 22.4% for
adults, and majority TV Show for 2018+ releases. Task 2 examined audience mix and type balance separately
but never crossed them.

---

## Limitations

**1. The most informative features had to be excluded.** Duration and genre are the two fields that best
describe Netflix content, and both encode the target. Not a modelling failure — a property of how Netflix
structures its metadata — but it caps what any classifier can achieve here.

**2. A 6.5-point gain is modest.** Movie versus TV Show is genuinely hard to predict from release
metadata alone.

**3. Precision and recall trade off sharply.** Gradient Boosting reaches 0.70 precision but only 0.38
recall; Random Forest balances better at 0.56/0.74. No configuration achieved both, reflecting genuinely
overlapping classes rather than poor tuning.

**4. `director_known` is a metadata artifact**, reflecting cataloguing convention rather than content.

**5. Country features are coarse.** Only the top 12 countries are encoded; everything else falls into an
unrepresented remainder.

**6. Title text was tested and rejected** (above) — not an untried gap, but a measured negative result.

**7. Single split, single seed for the main comparison.** Cross-validation checks stability, but a fuller
treatment would repeat the whole pipeline across several seeds, as the title-text experiment did.

---

## Improvement attempts — measured, not estimated

Eight standard next moves were run and compared against the current model, all on identical test rows.
**Not one beat it on F1.**

| # | Approach | F1 | ROC-AUC | F1 change |
|---|---|---|---|---|
| **0** | **Current model** | **0.6367** | 0.8101 | — |
| 5 | + random oversampling | 0.6341 | 0.8093 | −0.003 |
| 3 | + country target encoding | 0.6329 | **0.8212** | −0.004 |
| 4 | + country × audience interactions | 0.6309 | 0.8046 | −0.006 |
| 2 | + country frequency encoding | 0.6285 | 0.8107 | −0.008 |
| 6 | + hyperparameter tuning (12 configs) | 0.6273 | 0.8121 | −0.009 |
| 1 | + threshold tuning (0.52) | 0.6268 | 0.8101 | −0.010 |
| 7 | + soft-voting ensemble | 0.6025 | 0.8122 | −0.034 |

Each failure is informative:

**Threshold tuning** should have been the cheap win on an imbalanced target — but `class_weight="balanced"`
has already shifted the decision boundary, so 0.50 is effectively tuned already. The two techniques do the
same job and you only get paid once.

**Hyperparameter tuning scored below the hand-set configuration.** Its cross-validated F1 (0.605) was lower
than the test F1 it achieved, so the search mildly overfit the folds. When a parameter search cannot beat a
first guess, the parameters are not the constraint.

**The ensemble gained accuracy (0.765, the highest of any model here) and lost 0.034 F1** — it averaged in
Gradient Boosting's bias toward the majority class, the same trap documented throughout this task.

**Interactions hurt**, despite Task 2 finding that country and audience interact. Fifteen sparse binary
crosses over 8,786 rows gave the trees more noise than signal.

**Country target encoding is the one honourable mention.** Its F1 is marginally lower but it lifts ROC-AUC
from 0.810 to 0.821 — and AUC measures ranking quality independently of any threshold. If the goal were
ranking titles by likelihood rather than classifying them, this is the one change worth keeping.

### What remains

These are the changes this dataset cannot support:

| Improvement | Status |
|---|---|
| **Add plot descriptions and cast** | Not possible — the columns do not exist in this version of the dataset. Would be the largest gain by far |
| **Pick a target without taxonomy leakage** (e.g. audience category, which allows genre to be used) | Not attempted. Sidesteps the core constraint, but the new target correlates strongly with genre tags, so it partly trades one problem for another |
| ~~Add title text via TF-IDF~~ | Tested across 8 seeds — no usable gain |

**Nine configurations, four algorithms and two feature-engineering families all land within a few points of
each other.** That convergence is the clearest possible evidence that the constraint is the **features**,
not the modelling. More data would move this; more tuning will not.

---

## On what "good" means here

It is tempting to treat the 99.8% from feature set A as the result and mention the caveat in passing.
That would be the wrong way round.

The 76.2% figure is the finding. Everything above it came from features that restate the label, and a
model trained on those would collapse the moment it met data where duration and genre were not already
type-specific.

The title-text experiment is the same trap in miniature. Judged on accuracy alone it looked like a small
win worth adopting; judged on F1 it turned out to be the model retreating toward the majority class. Both
cases reward looking at the second metric.

---

## Conclusion

This task built and compared four classification models predicting whether a Netflix title is a Movie or
a TV Show.

**The main work was the leakage audit.** Three feature groups encode the target, and removing them took
accuracy from a flawless 1.0000 down to an honest 0.7622 — a 6.5-point gain over the 69.7% majority
baseline.

**Random Forest is the model to report, not the one with the best accuracy.** Gradient Boosting scored
highest on accuracy (0.762) but found only 38% of TV shows, varied most across CV folds, and was the only
model overfitting (+0.064 train/test F1 gap). Three independent diagnostics agreed. Random Forest took F1
(0.637) with a usable 0.56/0.74 precision–recall balance and essentially no overfitting, so it is the one
saved and reused.

**Feature importances validated the exploratory work — once measured properly.** Country and release year
dominate, with Pakistan and India at the top, exactly the contrast Task 2 identified. Switching from Gini
to permutation importance left that conclusion intact but dropped `month_added` from 5th to 10th, since
its apparent weight came from having twelve distinct values rather than from predictive value.

**And the model found something the EDA had missed.** It predicted TV Show for a 2020 US kids title against
an expectation of Movie. Checking the data showed the model right: US kids content is 43.2% TV shows
against 22.4% for adults, and majority TV Show for 2018+ releases. Task 2 examined audience mix and type
balance separately but never crossed them.

**The honest assessment.** A 76.2% classifier is a modest result. It would have been easy to report 99.8%
by leaving duration in the feature set, and that number would have described nothing but a relabelled
column. The value in this task is the audit that produced the lower figure, not the figure itself.

---

## Summary

| | |
|---|---|
| Target | Content type (Movie vs TV Show) |
| Majority baseline | 69.7% |
| Best by accuracy | Gradient Boosting (0.762) — but overfitting, not reported |
| **Model reported** | **Random Forest** (F1 0.637, accuracy 0.742) |
| Gain over baseline | ~6.5 points |
| Key finding | Duration and genre both leak the target and cannot be used |
| Top features | `release_year`, `country_Pakistan`, `country_India` |
| Also tested | Title TF-IDF across 8 seeds — no usable gain |
| Overfitting | Gradient Boosting only (+0.064 train/test F1 gap) |
| Importance method | Permutation importance preferred over Gini |
| Saved artifact | `output/netflix_type_classifier.pkl` for reuse in Task 6 |

## Files

| File | Description |
|---|---|
| `ml_classification.ipynb` | Full implementation, leakage audit, evaluation and improvement experiments, 111 cells, 15 visualisations |
| `output/netflix_type_classifier.pkl` | Saved pipeline — model, scaler and feature order (7.9 MB) |
| `report.md` | This document |

**Input:** `netflix_cleaned.csv` from Task 1.
