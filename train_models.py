import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "data" / "online_shoppers_intention.csv"
MODEL_DIR = BASE / "model"
TARGET = "Revenue"

# long right tail, so log them before scaling
SKEWED_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "PageValues",
]
RATE_COLS = ["BounceRates", "ExitRates", "SpecialDay"]
# these are ids, not quantities
CAT_COLS = ["Month", "OperatingSystems", "Browser", "Region",
            "TrafficType", "VisitorType", "Weekend"]
ALL_COLS = SKEWED_COLS + RATE_COLS + CAT_COLS


def read_data():
    df = pd.read_csv(CSV_PATH)
    print("raw shape:", df.shape)
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"removed {n_before - len(df)} duplicate rows, now {df.shape}")
    df[TARGET] = df[TARGET].astype(int)
    df["Weekend"] = df["Weekend"].astype(int)
    print("class counts:", df[TARGET].value_counts().to_dict())
    return df


def build_preprocessor():
    return ColumnTransformer([
        ("skewed", Pipeline([
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]), SKEWED_COLS),
        ("rates", StandardScaler(), RATE_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])


CANDIDATES = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, random_state=SEED),
        {"clf__C": [0.1, 1.0, 10.0]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=SEED),
        {"clf__max_depth": [3, 5, 7, 10, None], "clf__min_samples_leaf": [1, 10, 30]},
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"clf__n_neighbors": [5, 11, 21, 31], "clf__weights": ["uniform", "distance"]},
    ),
    "Naive Bayes": (
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]},
    ),
    "Support Vector Machine": (
        SVC(kernel="rbf", random_state=SEED),
        {"clf__C": [0.5, 1.0, 2.0]},
    ),
}

SAVE_AS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def get_scores(model, X):
    # SVC has no predict_proba here, the margin works fine for AUC
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def get_metrics(model, X, y):
    pred = model.predict(X)
    scores = get_scores(model, X)
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, scores),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def main():
    df = read_data()
    X, y = df[ALL_COLS], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print("train", X_train.shape, "test", X_test.shape)

    test_df = X_test.copy()
    test_df["Weekend"] = test_df["Weekend"].astype(bool)
    test_df[TARGET] = y_test.astype(bool)
    test_df.to_csv(BASE / "test_data.csv", index=False)

    MODEL_DIR.mkdir(exist_ok=True)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_metrics, all_params = {}, {}
    for name, (clf, grid) in CANDIDATES.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        # f1 instead of accuracy because only ~15% of sessions are purchases
        search = GridSearchCV(pipe, grid, scoring="f1", cv=folds, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        all_params[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        all_metrics[name] = get_metrics(best, X_test, y_test)
        joblib.dump(best, MODEL_DIR / SAVE_AS[name], compress=3)
        print(f"{name:<24} {all_params[name]}  "
              f"F1 {all_metrics[name]['F1']:.4f}  AUC {all_metrics[name]['AUC']:.4f}")

    (MODEL_DIR / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    (MODEL_DIR / "metadata.json").write_text(json.dumps({
        "dataset": "UCI Online Shoppers Purchasing Intention (id 468)",
        "target": TARGET,
        "features": ALL_COLS,
        "positive_class": "Revenue = True (purchase)",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": SEED,
        "model_files": SAVE_AS,
        "best_params": all_params,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }, indent=2, default=str))

    table = pd.DataFrame({m: {k: v for k, v in r.items() if k != "confusion_matrix"}
                          for m, r in all_metrics.items()}).T.round(4)
    cols = list(table.columns)
    out = ["| ML Model Name | " + " | ".join(cols) + " |",
           "|---|" + "---|" * len(cols)]
    for name, row in table.iterrows():
        out.append(f"| {name} | " + " | ".join(f"{row[c]:.4f}" for c in cols) + " |")
    print("\n" + "\n".join(out))


if __name__ == "__main__":
    main()
