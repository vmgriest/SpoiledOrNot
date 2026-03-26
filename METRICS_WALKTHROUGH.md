# Standard classification metrics – what each one means

After you run `python baseline.py` (or `uv run python baseline.py`), the script prints the same four metric blocks **twice** when the dataset includes an official `test/` folder: first for the **validation** holdout (20% of `train`), then for the **test** split. If there is no `test/` folder, you only see validation metrics. Here’s what each part represents and how to read it.

---

## 1. Accuracy

**What it is:** The fraction of all images in the **current split** (validation and/or test — whichever section you’re reading) that the model predicted correctly.

**Formula:**  
`Accuracy = (number of correct predictions) / (total number of samples)`

**How to read it:**  
- **High (e.g. 0.95):** The model is right most of the time overall.  
- **Low (e.g. 0.60):** The model is wrong often; worth checking other metrics and the confusion matrix.

**Limitation:** If one class is much more frequent (e.g. 90% “fresh”), a model that always predicts “fresh” can get 90% accuracy but be useless. So we also use per-class metrics and the confusion matrix.

---

## 2. Precision, recall, F1 (per class)

These are computed **for each class** (e.g. apple, banana, rottenapples, …).

### Precision (for one class)

**What it is:** Among all samples the model predicted as this class, how many were actually this class?

**Formula:**  
`Precision = True Positives / (True Positives + False Positives)`

**Intuition:**  
- **High precision:** When the model says “rotten”, it’s usually right (few false alarms).  
- **Low precision:** The model often calls other things “rotten” (many false positives).

### Recall (for one class)

**What it is:** Among all samples that are truly this class, how many did the model correctly predict as this class?

**Formula:**  
`Recall = True Positives / (True Positives + False Negatives)`

**Intuition:**  
- **High recall:** We catch most of that class (e.g. we don’t miss many rotten fruits).  
- **Low recall:** We miss a lot of that class (e.g. many rotten items predicted as fresh).

### F1 (for one class)

**What it is:** A single number that balances precision and recall (harmonic mean).

**Formula:**  
`F1 = 2 × (Precision × Recall) / (Precision + Recall)`

**Intuition:**  
- Use F1 when you care about both “don’t cry wolf” (precision) and “don’t miss real cases” (recall).  
- **Macro avg:** Average of F1 over all classes (each class weighted equally), so small classes matter as much as large ones.

---

## 3. Confusion matrix

**What it is:** A table of counts: for each pair (true class, predicted class), how many samples fell there.

**Layout:**
- **Rows** = true label (what the image really is).  
- **Columns** = predicted label (what the model said).  
- **Entry [i, j]** = number of samples with true class `i` that were predicted as class `j`.

**How to read it:**
- **Diagonal (e.g. [apple, apple]):** Correct predictions for that class. Big numbers on the diagonal = good.  
- **Off-diagonal (e.g. [apple, rottenapples]):** Confusions.  
  - e.g. Row “orange”, column “rottenoranges” = true oranges predicted as rotten oranges (could be acceptable or a confusion depending on your task).  
  - Row “rottenapples”, column “apple” = dangerous: rotten predicted as fresh.

**Use it to:** See which classes get mixed up (e.g. banana vs rottenbanana) and where to improve data or the model.

---

## 4. ROC–AUC

**What it is:** A measure of how well the model’s **scores** (e.g. softmax probabilities) separate the classes, not just whether the top class is correct.

**For binary (e.g. fresh vs rotten):**
- We take the probability of the “positive” class (e.g. rotten) as a “freshness score”.  
- **AUC** = area under the curve when we plot True Positive Rate vs False Positive Rate at different score thresholds.  
- **AUC = 1.0:** Perfect ranking (all positives have higher score than all negatives).  
- **AUC = 0.5:** Random.  
- **AUC < 0.5:** Worse than random (often means labels or model are flipped).

**For multi-class (your dataset: apple, banana, orange, rottenapples, …):**
- We use “one-vs-rest”: for each class we treat it as positive and the rest as negative, compute AUC, then average (macro).  
- So we get one number that summarizes how well the softmax probabilities separate each class from the others.

**Intuition:**  
- High AUC means the model’s confidence (probability) tends to be higher for the correct class.  
- Useful when you might use a **threshold** later (e.g. “flag if P(rotten) > 0.8”); AUC tells you how good the scores are across all thresholds.

---

## Quick reference

| Metric           | Answers the question |
|------------------|------------------------|
| **Accuracy**     | How many did we get right overall? |
| **Precision**    | When we say class X, how often are we right? |
| **Recall**       | Of all true X, how many did we find? |
| **F1**           | Balance of precision and recall for a class. |
| **Confusion**    | Which classes get confused with which? |
| **ROC–AUC**      | How well do the model’s scores rank/separate classes? |

Running `baseline.py` will print these with short explanations in the terminal; this file is a longer reference for each part.
