> **Note:** This is a study guide covering concept explanations, decision reasoning, step-by-step walkthrough, and key findings for the IDOOU Budget Predictor project.

# Step 2: Investigate an ML Model on the Problematic Dataset

> **What this step does:** Train two models on the biased dataset, evaluate their performance and fairness metrics, and confirm that high accuracy can coexist with severe discrimination.

---

## Context: Why Train on the "Problematic" Dataset?

Step 1 showed that the raw data has a near-perfect association between education and budget, and that the dataset-level fairness metrics (SPD = −0.99, DI = 0.009) are far outside acceptable ranges. This step asks: what happens when we just train a model on this data, as-is?

The answer is the central lesson of the step — a model can achieve 99%+ accuracy while being completely useless for the unprivileged group. This is not a modelling failure. It's what happens when a well-functioning model faithfully learns a biased dataset.

---

## Project Walkthrough

### 1. Train Two Models

Two classifiers are trained on the same data: **Gaussian Naive Bayes (GNB)** and **Logistic Regression (LR)**.

---

#### Concept Note: Gaussian Naive Bayes

GNB is a probabilistic classifier based on Bayes' theorem. It estimates the probability of each class given the input features, then predicts the most likely class. The "Naive" part means it assumes all features are independent of each other — which is almost never true in real data, but works surprisingly well in practice.

It's fast, requires very little data, and tends to be well-calibrated when the independence assumption roughly holds.

---

#### Concept Note: Logistic Regression

Despite the name, Logistic Regression is a classification model. It learns a weighted sum of input features and passes it through a sigmoid function to produce a probability between 0 and 1. Each feature gets a coefficient — positive means it pushes toward the positive class, negative pushes away.

It's one of the most interpretable models available: you can directly read off which features matter and in which direction.

---

**Why two models?** If both models — with different assumptions and mechanisms — produce the same fairness metrics, it strongly suggests the problem is in the data, not in the choice of algorithm. That's exactly what we find here.

---

### 2. Sweep Decision Thresholds

Both models output a probability (e.g., "72% chance this user has budget ≥$300"). The **decision threshold** is the cutoff above which you predict the positive class. The default is 0.5, but that's often not optimal — especially when class distributions are imbalanced.

---

#### Concept Note: Decision Threshold Sweeping

Instead of fixing the threshold at 0.5, the code evaluates the model at 50 different thresholds between 0.01 and 0.5, recording fairness and performance metrics at each point. The threshold that maximises **balanced accuracy** is then selected.

**Why balanced accuracy instead of regular accuracy?**

Regular accuracy can be misleading when one class dominates. If 90% of users are in the <$300 class, a model that predicts <$300 for everyone gets 90% accuracy — while being completely wrong about every ≥$300 user.

Balanced accuracy is the average of the True Positive Rate and True Negative Rate:

$$\text{Balanced Accuracy} = \frac{TPR + TNR}{2}$$

This gives equal weight to both classes regardless of how many examples each has, so it penalises a model that ignores the minority class.

---

**What the threshold sweep found:**

| Model | Best threshold | Balanced accuracy |
|-------|---------------|-------------------|
| Gaussian Naive Bayes | 0.01 | 99.69% |
| Logistic Regression | 0.33 | 99.78% |

GNB's optimal threshold of 0.01 is a red flag — it means GNB is classifying almost everything as the positive class to maximise its balanced accuracy score. That's poor calibration, not good performance. LR's threshold of 0.33 is much more reasonable.

---

### 3. Evaluate Performance

Both models score above 99% accuracy on the test set.

| Model | Accuracy |
|-------|----------|
| Gaussian Naive Bayes | 99.69% |
| Logistic Regression | 99.72% |

These numbers look excellent. This is exactly the trap.

---

#### Concept Note: The Confusion Matrix in a Fairness Context

A confusion matrix breaks predictions into four cells:

|  | Predicted Negative | Predicted Positive |
|--|---|---|
| **Actually Negative** | True Negative (TN) | False Positive (FP) |
| **Actually Positive** | False Negative (FN) | True Positive (TP) |

In this project, the positive class is budget ≥$300. A **False Negative** is a user who actually has a high budget but is predicted as low-budget. In a fairness context, false negatives are particularly harmful for the unprivileged group — they represent real users being denied a positive outcome they deserve.

The stratified breakdown tells the full story:

| Model | Group | TPR (True Positive Rate) | FPR | Accuracy |
|-------|-------|--------------------------|-----|----------|
| GNB | Bachelor's Degree | 1.000 | 0.021 | 100.0% |
| GNB | Master's Degree | 1.000 | 0.070 | 99.9% |
| **GNB** | **High School Grad** | **0.039** | 0.002 | 98.9% |
| LR | Bachelor's Degree | 1.000 | 0.021 | 100.0% |
| LR | Master's Degree | 1.000 | 0.070 | 99.9% |
| **LR** | **High School Grad** | **0.000** | 0.000 | 99.0% |

- Post-HS users: perfectly identified when they have high budgets (TPR = 100%)
- HS Grads (GNB): only 3.9% of high-budget users are correctly identified
- HS Grads (LR): **zero** high-budget HS grad users are correctly predicted — every single one is misclassified as low-budget

The 99%+ overall accuracy is real — but it's built entirely on correctly predicting the dominant classes (young users and post-HS users as low/high budget respectively). The HS grad minority is being silently failed.

---

### 4. Evaluate Fairness Metrics

---

#### Concept Note: Average Odds Difference and Equal Opportunity Difference

Step 1 introduced SPD and Disparate Impact, which measure disparity in the *rate of positive outcomes*. These two metrics go deeper — they measure whether the model makes *errors* at equal rates across groups.

**Equal Opportunity Difference (EOD):**

Compares the True Positive Rate between groups:

$$\text{EOD} = TPR_{\text{unprivileged}} - TPR_{\text{privileged}}$$

- **Fair range:** −0.1 to +0.1
- A value of −1.0 means the unprivileged group's TPR is 1.0 lower than the privileged group's — i.e., every high-budget HS grad is missed by the model.

This is specifically about fairness of *opportunity* — are users who deserve the positive outcome equally likely to receive it?

**Average Odds Difference (AOD):**

Averages the difference in TPR *and* FPR between groups:

$$\text{AOD} = \frac{(FPR_{\text{unpriv}} - FPR_{\text{priv}}) + (TPR_{\text{unpriv}} - TPR_{\text{priv}})}{2}$$

- **Fair range:** −0.1 to +0.1
- Captures error disparity across both types of mistakes, not just false negatives.

---

**Fairness metrics on the test set:**

| Metric | GNB | LR | Fair range |
|--------|-----|----|------------|
| Statistical Parity Difference | −0.989 | −0.993 | −0.1 to +0.1 |
| Disparate Impact | 0.002 | 0.000 | ≥ 0.8 |
| Average Odds Difference | −0.505 | −0.525 | −0.1 to +0.1 |
| Equal Opportunity Difference | −0.961 | **−1.000** | −0.1 to +0.1 |

LR's EOD of −1.000 is as bad as it gets. Every HS grad with a genuine high budget is being classified as low-budget. The model has learned that "HS grad" means "low budget" with such confidence that it overrides any other signal.

---

### 5. Choose a Model

Both models perform identically on fairness metrics. The tiebreaker is calibration and interpretability.

**Why Logistic Regression wins:**
- GNB's optimal threshold of 0.01 — classifying almost everything as positive — is a sign of poor probability calibration, not good performance. A well-calibrated model shouldn't need a threshold that extreme.
- LR's threshold of 0.33 is sensible and its coefficients are directly interpretable, which matters when explaining model behaviour in a fairness-focused analysis.

---

## Key Findings

| Finding | Value |
|---------|-------|
| GNB accuracy | 99.69% |
| LR accuracy | 99.72% |
| LR Equal Opportunity Difference | −1.000 — HS grads with high budgets are **never** correctly predicted |
| LR Average Odds Difference | −0.525 — HS grads are disadvantaged across both error types |
| LR Disparate Impact | 0.000 — unprivileged group receives positive outcome at 0% the rate of privileged group |
| Identical metrics across both models | Confirms the bias is in the data, not the algorithm |

---

## Ethical Flag

Both algorithms, with different mathematical assumptions, arrived at the same discriminatory outcome. This rules out "bad model choice" as the explanation. The problem is upstream — in the data distribution itself. High overall accuracy actively obscures this: without the stratified breakdown, a 99.72% accuracy score would look like a success.

This is why fairness auditing cannot be replaced by accuracy metrics alone. Step 5 applies bias mitigation (Reweighing) to address the root cause in the training data.
