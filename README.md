# ML Assignment 2 — Breast Cancer Classification with Streamlit

## a. Problem Statement

The goal of this assignment is to build, evaluate, and deploy multiple
classification models on a single dataset, and then expose the results
through an interactive Streamlit web application. Specifically, the task is
to predict whether a breast tumor is **malignant** or **benign** based on
features computed from a digitized image of a fine needle aspirate (FNA) of
a breast mass. This is a binary classification problem, and five different
classifiers are trained and compared on the same dataset so that their
relative strengths and weaknesses can be studied.

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** UCI Machine Learning Repository (also available as a built-in
  loader in scikit-learn — `sklearn.datasets.load_breast_cancer`)
  https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Instances:** 569 (well above the minimum requirement of 500)
- **Features:** 30 numeric features (well above the minimum requirement of
  12) — these are computed from digitized images of cell nuclei and describe
  characteristics like radius, texture, perimeter, area, smoothness,
  compactness, concavity, symmetry, and fractal dimension (each reported as
  a mean, standard error, and "worst" value).
- **Target variable:** Binary — malignant (0) or benign (1)
- **Class balance:** 212 malignant, 357 benign

The data was split into an 80/20 train-test split (stratified on the
target so both classes are proportionally represented), and all features
were standardized using `StandardScaler` (fit on the training data only)
before being fed into the models. The held-out 20% test split (114 rows) is
saved as `test_data.csv` and is what gets uploaded into the Streamlit app to
demonstrate the models.

## c. GitHub Repository Link

> `<PASTE YOUR GITHUB REPO LINK HERE AFTER YOU PUSH THE CODE>`

## d. Models Used

All five models below were trained on the same 80% training split and
evaluated on the same 20% held-out test split (114 samples) of the Breast
Cancer Wisconsin dataset.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9932 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | This came out as the best performer across almost every metric, which honestly surprised me a bit going in — I expected the ensemble to win. Once the features are standardized, this dataset is close to linearly separable in the transformed space, so a simple linear decision boundary works really well here. Its MCC of 0.96 is the highest of all five models, meaning it's not just accurate but genuinely well-balanced across both classes. |
| Decision Tree | Clearly the weakest model here, both in accuracy and AUC. A single tree tends to overfit the training data and split on noisy feature interactions, so it doesn't generalize as smoothly as the other models on unseen data. Its recall (0.90) is also the lowest, meaning it misses more actual malignant cases than the others — not ideal for a medical screening context. |
| kNN | Performs solidly and ties with Random Forest on accuracy, precision, recall and F1. Since all features were scaled beforehand, distance-based comparisons are meaningful, which is likely why it does well. It doesn't build an explicit decision boundary the way logistic regression does, so it's a bit more sensitive to local noise in the feature space, which is probably why its AUC (0.9788) trails logistic regression slightly. |
| Naive Bayes | Interesting case — its AUC (0.9868) is actually quite high, close to the top models, but its accuracy and F1 are noticeably lower. This tells me the model ranks/scores instances reasonably well but its default probability threshold of 0.5 isn't optimal for this dataset, likely because the conditional independence assumption between the 30 features doesn't really hold (many of the radius/perimeter/area features are highly correlated with each other). |
| Random Forest (Ensemble) | Performs very close to kNN on accuracy/precision/recall/F1, and has the second-highest AUC (0.9932) of all five models — very close to Logistic Regression. As an ensemble of decision trees, it fixes the main weakness of the single Decision Tree by averaging out the overfitting, which is exactly what we see when comparing its numbers to the standalone Decision Tree above. |
| **Overall Winner for your dataset?** | **Logistic Regression.** It has the top score on 5 out of 6 metrics (Accuracy, Precision, Recall, F1, MCC) and is a very close second on AUC. Given that it's also the simplest and most interpretable model of the five, this is a great outcome — sometimes the simpler model really is the better one, especially when the underlying classes are well-separated after scaling. |

## Repository Structure

```
project-folder/
│-- app.py                     # Streamlit application
│-- requirements.txt
│-- README.md
│-- test_data.csv              # held-out test set used by the app
│-- metrics_summary.csv        # comparison table (raw)
│-- model/
│   │-- train_models.py        # trains all 5 models + saves artifacts
│   │-- scaler.pkl
│   │-- feature_names.pkl
│   │-- target_names.json
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest.pkl
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py   # regenerates the model/*.pkl artifacts + test_data.csv
streamlit run app.py
```

## Live Streamlit App Link

> `<PASTE YOUR DEPLOYED STREAMLIT APP LINK HERE>`

## BITS Virtual Lab Screenshot

> `<INSERT SCREENSHOT OF EXECUTION ON BITS VIRTUAL LAB HERE>`
