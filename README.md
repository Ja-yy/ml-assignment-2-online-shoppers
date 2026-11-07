# Online Shoppers Purchase Prediction (ML Assignment 2)

## a. Problem statement

An online store logs every browsing session: how many pages of each type the
visitor opened, how long they spent on them, bounce and exit rates, and when
and how they arrived. Only about one session in six ends in a purchase.

The task is binary classification: from a session's browsing behaviour,
predict whether it ends in a purchase (`Revenue = True`) or not. Five
classifiers are trained on the same data and compared on six metrics. The
trained models are then served in a Streamlit app that scores an uploaded test
CSV without retraining.

## b. Dataset description

**Source:** UCI Machine Learning Repository, *Online Shoppers Purchasing
Intention Dataset* (id 468), Sakar, Polat, Katircioglu & Kastro, 2018.
<https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset>

| Property | Value |
|---|---|
| Instances | 12,330 sessions (12,205 after removing 125 exact duplicate rows) |
| Features | 17 (10 numeric, 7 categorical) |
| Target | `Revenue`, boolean: True = purchase |
| Class balance | 10,297 no purchase (84.4%) / 1,908 purchase (15.6%) |
| Missing values | none |
| Period | one year of sessions; each session belongs to a different user |

**Features**

| Feature | Type | Meaning |
|---|---|---|
| `Administrative`, `Informational`, `ProductRelated` | numeric (count) | pages of each category visited in the session |
| `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration` | numeric (seconds) | time spent on each page category |
| `BounceRates` | numeric | average bounce rate (Google Analytics) of the pages visited |
| `ExitRates` | numeric | average exit rate of the pages visited |
| `PageValues` | numeric | average value of the pages visited before a transaction |
| `SpecialDay` | numeric 0–1 | closeness of the visit to a special day such as Mother's Day or Valentine's Day |
| `Month` | categorical | month of the visit (10 values, no Jan/Apr; June is spelled `June`) |
| `OperatingSystems`, `Browser`, `Region`, `TrafficType` | categorical (integer codes) | client and traffic-source codes |
| `VisitorType` | categorical | Returning_Visitor / New_Visitor / Other |
| `Weekend` | boolean | visit happened on a weekend |

**Preprocessing (inside each saved pipeline)**

- Exact duplicate rows dropped before splitting, so no copy of a test row is
  in the training set.
- Stratified 80/20 split, `random_state=42`: 9,764 training rows, 2,441 test
  rows. The test rows are saved as `test_data.csv`.
- The seven count, duration and `PageValues` columns are very right-skewed,
  so they get `log1p` and then standardisation. The rate columns are
  standardised.
- `Month`, `VisitorType`, `Weekend` and the four integer-coded columns are
  one-hot encoded. `Browser = 13` is a label, not a larger number than
  `Browser = 2`.
- Preprocessing and classifier are saved together as one sklearn `Pipeline`
  per model. The scaler and encoder are fitted only on the training split,
  and the app gets the exact same transformation.

## c. GitHub Repository Link

`<GITHUB REPO LINK>`

```
ml-assignment-2-online-shoppers/
├── app.py                  # Streamlit app: upload CSV -> pick model -> metrics + confusion matrix
├── train_models.py         # trains and tunes all five models, writes model/ and test_data.csv
├── requirements.txt
├── README.md
├── test_data.csv           # held-out 20% test split (17 features + Revenue)
├── data/
│   └── online_shoppers_intention.csv   # original UCI file
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── metrics.json        # test-set metrics + confusion matrices
│   └── metadata.json       # feature list, best hyper-parameters, library versions
└── .streamlit/config.toml  # app theme
```

Run locally:

```bash
pip install -r requirements.txt
python train_models.py      # about 2 minutes; rewrites model/ and test_data.csv
streamlit run app.py
```

## d. Models used

All five models use the same split and the same preprocessing. Each one's
hyper-parameters were picked by 5-fold stratified cross-validation on the
training split only, using F1 as the score. With 84% negatives, a model that
always says "no purchase" already gets 84.4% accuracy, so tuning on accuracy
would favour models that ignore buyers.

| Model | Tuned setting |
|---|---|
| Logistic Regression | C = 1.0 |
| Decision Tree | max_depth = 5, min_samples_leaf = 30 |
| KNN | k = 31, distance-weighted |
| Naive Bayes (Gaussian) | var_smoothing = 0.1 |
| Support Vector Machine (RBF) | C = 2.0 |

**Comparison table** (held-out test set, 2,441 sessions, positive class =
purchase):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9000 | 0.9216 | 0.7184 | 0.5942 | 0.6504 | 0.5963 |
| Decision Tree | 0.9017 | 0.9271 | 0.7113 | 0.6257 | 0.6657 | 0.6101 |
| KNN | 0.8955 | 0.9143 | 0.7153 | 0.5524 | 0.6233 | 0.5702 |
| Naive Bayes | 0.7513 | 0.8754 | 0.3735 | 0.8691 | 0.5224 | 0.4520 |
| Support Vector Machine | 0.8992 | 0.9016 | 0.7048 | 0.6126 | 0.6555 | 0.5988 |

AUC uses `predict_proba` for the first four models and the SVM's
`decision_function` margin. AUC depends only on how the sessions are ranked,
so the SVM needs no probability calibration for it.

Confusion matrices on the test set, as [[TN, FP], [FN, TP]]: LR
[[1970, 89], [155, 227]] · DT [[1962, 97], [143, 239]] · KNN
[[1975, 84], [171, 211]] · NB [[1502, 557], [50, 332]] · SVM
[[1961, 98], [148, 234]].

**Observations**

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong for a linear model: 0.90 accuracy, AUC 0.92, and the highest precision (0.718). After `log1p`, `PageValues` has by far the largest coefficient, so a weighted sum of features is enough to separate most buyers. It misses about 40% of buyers (recall 0.594) at the default 0.5 threshold, because the minority class pulls predicted probabilities down. |
| Decision Tree | Best model on this data: top accuracy (0.902), AUC (0.927), F1 (0.666) and MCC (0.610). A shallow tree (depth 5, at least 30 sessions per leaf) generalises well. Deeper trees overfit in cross-validation. `PageValues` carries 78% of the tree's importance, and the tree turns it into simple yes/no splits ("did the visitor reach a page that leads to checkout?"). Bounce rate and November come next. |
| KNN | Weakest of the four "good" models (F1 0.623, MCC 0.570). It is accurate overall but has the lowest recall of all five models (0.552). Buyers are a small, scattered minority, so even 31 distance-weighted neighbours are mostly non-buyers. Its 17 raw features become about 70 one-hot dimensions, and the many category flags dilute the distance signal from `PageValues`. It is also the largest model file, because it stores the whole training set. |
| Naive Bayes | Very different trade-off: highest recall (0.869, it catches 332 of 382 buyers) but precision only 0.374, so 557 non-buyers are flagged as buyers. Accuracy (0.751) and MCC (0.452) are the lowest. Gaussian NB assumes independent, normally distributed features, but page counts and their durations are strongly correlated, and the one-hot columns are not Gaussian. The double-counted evidence makes its probabilities over-confident. It is still useful if missing a buyer costs much more than a false alarm. |
| Support Vector Machine | Close to logistic regression on the threshold-based metrics (F1 0.656, MCC 0.599), but its AUC is the lowest of the non-NB models (0.902). The RBF kernel draws a good boundary, but its margin ranks sessions less well than LR's probabilities. It keeps 2,576 support vectors, so it is bigger and slower to score than the linear model and gains nothing over it here. |
| **Overall winner** | **Decision Tree.** It leads on Accuracy, AUC, F1 and MCC. MCC is the fairest single number here because it uses all four confusion-matrix cells and is not inflated by the 84% majority class. Purchase intent in this data is mostly a threshold effect on `PageValues` plus a few behavioural rules, which matches what a shallow tree models. Logistic Regression is a close and more stable second. Naive Bayes is the choice only if recall matters most. |

## Streamlit app

- **CSV upload** of test data (upload `test_data.csv` from this repo; the app
  also offers it as a download).
- **Model dropdown** with all five classifiers.
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1, MCC for the
  selected model, plus a table and bar chart comparing all five models on the
  uploaded file.
- **Confusion matrix** and **classification report** for the selected model.
- The app only loads the saved `.joblib` pipelines from `model/` and calls
  predict on them. Nothing is refitted on the uploaded data.
