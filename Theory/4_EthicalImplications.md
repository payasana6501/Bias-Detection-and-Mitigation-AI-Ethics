> **Note:** This is a study guide covering concept explanations, decision reasoning, step-by-step walkthrough, and key findings for the IDOOU Budget Predictor project.

# Step 6: Ethical Implications

> **What this step does:** Reflect on what the model does, what it gets wrong, who it harms, and what responsibilities remain — both technical and organisational.

---

## Context: Why Ethical Analysis Comes Last (But Shouldn't Be an Afterthought)

The preceding steps were technical: clean data, train model, measure fairness, mitigate bias. This step steps back and asks a different set of questions — not "does the model work?" but "should it work this way, and what are the consequences if it does?"

Ethical analysis is placed at the end because it requires the full picture: you can't meaningfully discuss harms until you know what the model actually does, who it fails, and by how much. But in a real project, these questions should be raised at the *start* — before data is collected, before features are chosen, before a line of code is written.

---

## 1. The Core Ethical Tension: Demographic Proxies

The model predicts budget using demographic features — age, education level, gender. These are **proxies**: stand-ins for something the model can't directly observe (actual spending behaviour or financial capacity). Proxies are a practical necessity — you often don't have direct access to the thing you want to predict. But they carry a structural risk.

---

#### Concept Note: Statistical Discrimination

Statistical discrimination is when a model (or a person) applies a group-level statistical pattern to an individual, even when individual-level evidence would say otherwise. It's technically accurate at the group level and individually unfair at the same time.

Example from this project: HS grads as a group have lower budgets in this dataset. The model learns this and applies it to *every* HS grad — including the ones who genuinely have high budgets. The individual pays the price for their group's average. The model is statistically grounded and individually wrong.

This is distinct from random error. Random errors affect everyone roughly equally. Statistical discrimination systematically disadvantages one group — and does so with high confidence, because the model is certain it's applying a real pattern.

---

In this project, the model conflates three things that frequently co-occur in the dataset but are not causally linked:
- Being 18–24 years old
- Having a high school education
- Having a budget under $300

These correlate strongly in the training data. But correlation is not causation, and a pattern true for a group is not guaranteed to be true for any individual within it. A 35-year-old HS grad has a different life context than an 18-year-old HS grad — the model doesn't know this, and before mitigation, it didn't try to find out.

---

## 2. Human-in-the-Loop

---

#### Concept Note: Human-in-the-Loop

Human-in-the-loop (HITL) means keeping a human decision-maker in the prediction pipeline — not just at design time, but at inference time. The model makes a prediction; a human (often the user themselves) has the opportunity to confirm, correct, or override it before it affects an outcome.

HITL is especially important when:
- The model makes predictions about individuals using group-level proxies
- The cost of a wrong prediction falls directly on the user
- The user has information the model doesn't (their actual preferences, context, circumstances)

---

**Applied here:** Users should be prompted to confirm or correct the model's budget prediction before activity recommendations are generated. A user who knows their actual budget shouldn't be locked into a demographic estimate. Additionally, users should be able to opt out of providing demographic information entirely — the app should degrade gracefully with partial inputs rather than refusing to function.

This matters beyond fairness: it also affects trust. A user who sees a wrong budget prediction and can't correct it will lose confidence in the app. A user who can correct it gains agency, and the correction improves future model performance if fed back as training signal.

---

## 3. Model Failures and Residual Bias

The honest accounting of what the model still gets wrong after mitigation:

**Before Reweighing:**
- Equal Opportunity Difference = −1.000: every single HS grad with a genuine high budget was misclassified as low budget — a complete failure for this subgroup
- Disparate Impact = 0.009: HS grads received the positive outcome at less than 1% the rate of post-HS users

**After Reweighing:**
- Equal Opportunity Difference = 0.000 ✓ — corrected fully
- Average Odds Difference = −0.003 ✓ — within fair range
- Statistical Parity Difference = −0.9471 — still far outside range
- Disparate Impact = 0.044 — still far outside range

The two metrics that didn't improve (SPD, Disparate Impact) reflect the genuine underlying distribution of the dataset — HS grads simply appear in the ≥$300 class far less often than post-HS users. No mitigation technique can change this without changing the data itself. Deploying this model means accepting that at the population level, HS grads will receive the positive outcome significantly less often — not because the model is wrong about individuals, but because the data reflects real-world socioeconomic inequality.

That is not just a technical limitation.

---

## 4. Potential Harms

The direct harm in this application is **under-personalisation**: HS grad users with genuine high budgets receive activity recommendations calibrated to a lower budget than they actually have. They see cheaper options, miss higher-value activities they could afford, and have a worse app experience than users with identical budgets but different education levels.

The subtler harm is **reinforcement of stereotypes**: every time the model predicts a HS grad as low-budget, it acts as if education level determines financial capacity. At scale, this shapes what content gets surfaced to which groups — and normalises the assumption that it should.

---

## 5. Business Consequences

**Positive impact:** Accurate budget prediction enables personalised recommendations at scale — improving engagement, surfacing relevant activities, and increasing conversion. Users who receive well-matched recommendations are more likely to book, return, and recommend the app.

**Negative impact:** If HS grad users consistently receive under-personalised recommendations, they disengage. At ~17% of the dataset, this is a non-trivial user segment. More seriously: if the bias becomes public — through user complaints, press coverage, or regulatory scrutiny — the reputational damage extends well beyond that segment. A bias-against-education story is straightforward to communicate and hard to recover from. Regulatory exposure (particularly under EU AI Act or similar frameworks treating demographic-based prediction as high-risk) adds legal and compliance risk on top.

The business case for fairness here isn't altruistic — it's that the cost of the bias, if it surfaces publicly, exceeds the cost of fixing it.

---

## 6. Caveats and Recommendations

**What the model is missing:**

- **No behavioural signals.** Budget is predicted entirely from demographics. Past spending history, stated preferences, or in-app behaviour would be far stronger and less discriminatory signals. Demographic proxies are a weak substitute for actual evidence about an individual.
- **Age skew.** 18–24 year olds are 49% of the dataset; 66–92 year olds are ~5.7%. The model may not generalise well to older users, who are severely underrepresented.
- **Education imbalance.** Post-HS users are 3× more represented than HS grads. Even with Reweighing, the model has seen far more examples of one group.

**What further analyses would strengthen this:**

---

#### Concept Note: Counterfactual Fairness

Counterfactual fairness asks: *if this individual had belonged to a different group — everything else being equal — would the prediction have changed?*

In practice: take an HS grad user and change only their education level to Bachelor's Degree, keeping all other features identical. If the model's prediction flips, the model is making decisions based on group membership rather than individual circumstances — and is not counterfactually fair.

This is a stronger fairness criterion than the group-level metrics used in this project, because it operates at the individual level.

---

#### Concept Note: Individual Fairness

Individual fairness requires that *similar individuals receive similar predictions*. Two users who are alike in all relevant ways (spending history, preferences, stated budget) should receive the same recommendation, regardless of their demographic group.

This is harder to measure than group fairness because it requires defining what "similar" means in your domain — a non-trivial problem. But it's closer to the intuitive notion of fairness most users care about: "treat me as an individual, not as a demographic category."

---

Both counterfactual fairness testing and individual fairness analysis go beyond what this project implements, but they represent the next level of rigour for a production deployment of this model.

---

## Summary: What This Project Demonstrates

| Question | Answer |
|----------|--------|
| Does the model work? | Yes — 99%+ accuracy |
| Is it fair out of the box? | No — complete failure for HS grads before mitigation |
| Does mitigation fix it? | Partially — 3 of 5 fairness metrics corrected; 2 reflect structural data imbalance |
| Is the residual bias acceptable? | That's an ethical and business decision, not a technical one |
| What would make it more robust? | Behavioural features, more representative data, counterfactual and individual fairness testing |
| Who bears responsibility? | The organisation deploying the model — technical mitigation reduces but does not eliminate that responsibility |
