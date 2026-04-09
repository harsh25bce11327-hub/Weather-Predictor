# Weather Prediction Using Naive Bayes

**ACADEMIC PROJECT REPORT · AI & MACHINE LEARNING**
*A probability-based approach to rainfall forecasting using Bayesian classification on historical weather data.*

---

## 1. Introduction

Predicting weather is one of the oldest scientific challenges humanity has grappled with. This project builds a probability-based model to predict whether it will rain, drawing on historical weather data and a foundational AI technique: Bayesian classification.

Rather than replicating complex atmospheric simulations, the goal here is conceptual clarity — showing how a simple probabilistic model, when applied thoughtfully, can yield surprisingly useful and interpretable results.

---

## 2. Why This Problem Matters

- **Agriculture** — Farmers depend on accurate rainfall forecasts for crop planning, irrigation decisions, and protecting yields from unexpected weather events.
- **Urban planning** — City governments and emergency services need reliable forecasts to prepare infrastructure for floods, droughts, and seasonal extremes.
- **Daily life** — Individuals rely on weather prediction for everyday decisions: travel, outdoor events, and managing personal comfort and safety.
- **Academic value** — For students of AI and ML, this problem offers a grounded, real-world context to explore probabilistic reasoning and classification.

> *"Simple models can still be effective — the key lies in understanding your assumptions and applying them honestly."*

---

## 3. Approach & Methodology

The classifier is built on the **Naive Bayes algorithm**, a probabilistic method grounded in Bayes' Theorem. The theorem describes how to update the probability of a hypothesis as new evidence is observed:

**Bayes' Theorem:**

```
P(H | E) = [ P(E | H) × P(H) ] / P(E)
```

Where:
- `H` = Hypothesis (Rain / No Rain)
- `E` = Evidence (humidity, temperature, wind speed, pressure)

The word *Naive* in the name refers to the assumption that all features are independent of one another — a simplification that makes computation tractable even when it isn't strictly true.

### How the Model Was Built

| Step | Action |
|------|--------|
| 1. Gather Data | Collected historical weather records containing temperature, humidity, wind speed, and atmospheric pressure readings across many days. |
| 2. Clean Data | Addressed missing or inconsistent values. Converted continuous numerical features into discrete categories (Low, Medium, High) to simplify probability calculations. |
| 3. Select Features | Identified the most informative attributes. Through analysis, humidity and pressure emerged as the strongest predictors of rainfall. |
| 4. Train Model | Applied the Naive Bayes algorithm to learn class-conditional probabilities from the labelled training data. |
| 5. Evaluate | Used accuracy scores and a confusion matrix together to assess model performance across both Rain and No Rain classes. |

---

## 4. Key Decisions

### Why Naive Bayes?
Chosen for its simplicity, speed, and interpretability. A Naive Bayes model makes its reasoning visible — you can trace exactly why it predicts rain on a given day.

### Discretisation
Continuous features were binned into categories (Low / Medium / High) to simplify the conditional probability tables and reduce the risk of overfitting sparse data.

### Evaluation Strategy
Accuracy alone can be misleading with imbalanced classes. Using the confusion matrix alongside accuracy gave a more honest picture of where the model succeeded and failed.

---

## 5. Challenges Faced

**Data quality**
Several records contained missing or inconsistent values, requiring careful preprocessing before any training could begin.

**The independence assumption**
Naive Bayes treats features as independent, but in weather data humidity, temperature, and pressure are naturally correlated. This known limitation affects precision but does not prevent useful predictions.

**Class imbalance**
Non-rainy days significantly outnumbered rainy days in the dataset, which skewed the model's baseline probabilities and required attention during evaluation.

---

## 6. Results

| Metric | Value |
|--------|-------|
| Model Accuracy | 75–80% (varies by dataset) |
| Top Predictor 1 | Humidity — strongest indicator |
| Top Predictor 2 | Pressure — second strongest |

Despite its simplicity, the model produced interpretable and practically useful results. Humidity and atmospheric pressure were identified as the dominant features driving rainfall predictions — findings that align well with domain intuition. The confusion matrix revealed that the model performed better at predicting dry days than rainy ones, a pattern consistent with the class imbalance in the training data.

---

## 7. Learnings

- Probability theory is not just abstract mathematics — it is a practical foundation for real AI and ML systems used in production every day.
- Simpler models can be highly effective when applied with domain knowledge and honest evaluation. Complexity is not always a virtue.
- Understanding a model's assumptions is as important as understanding its mechanics. The independence assumption in Naive Bayes shapes every prediction it makes.

---

## 8. Conclusion

This project demonstrated that probability-based classifiers can be meaningfully applied to real-world weather prediction. While the Naive Bayes model does not approach the sophistication of modern numerical forecasting systems, it serves as an excellent vehicle for learning the fundamentals of machine learning.

The exercise successfully bridged probabilistic theory with hands-on implementation — making abstract concepts like prior probabilities, likelihoods, and posterior estimates concrete and tangible.

Working through real data, cleaning it, making deliberate modelling choices, and evaluating outcomes honestly: these are skills that transfer directly into any future machine learning work.
