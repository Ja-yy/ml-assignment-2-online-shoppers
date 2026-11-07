import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
meta = json.loads((MODEL_DIR / "metadata.json").read_text())
FEATURES = meta["features"]
TARGET = meta["target"]
MODEL_FILES = meta["model_files"]
LABELS = ["No purchase", "Purchase"]

st.set_page_config(page_title="Will this shopper buy?", page_icon="🛒", layout="wide")


@st.cache_resource
def load_all_models():
    return {name: joblib.load(MODEL_DIR / f) for name, f in MODEL_FILES.items()}


def to_int_flag(col):
    if col.dtype == bool:
        return col.astype(int)
    lookup = {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
    out = col.astype(str).str.strip().str.lower().map(lookup)
    if out.isna().any():
        bad = list(col[out.isna()].unique()[:5])
        raise ValueError(f"column '{col.name}' should be true/false but has {bad}")
    return out.astype(int)


def split_xy(df):
    absent = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if absent:
        raise ValueError("file is missing columns: " + ", ".join(absent))
    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES].copy()
    X["Weekend"] = to_int_flag(X["Weekend"])
    return X, to_int_flag(df[TARGET])


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def run_model(model, X, y):
    pred = model.predict(X)
    auc = roc_auc_score(y, get_scores(model, X)) if y.nunique() == 2 else float("nan")
    return pred, {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


def draw_confusion(y, pred, title):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,
                xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


# sidebar
st.sidebar.header("1. Test data")
uploaded = st.sidebar.file_uploader(
    "Upload a CSV (17 feature columns + Revenue)", type=["csv"])
sample_file = BASE / "test_data.csv"
if sample_file.exists():
    st.sidebar.download_button(
        "Get test_data.csv", sample_file.read_bytes(),
        file_name="test_data.csv", mime="text/csv")

st.sidebar.header("2. Model")
chosen = st.sidebar.selectbox("Classifier", list(MODEL_FILES))
st.sidebar.caption(
    f"Trained on scikit-learn {meta['sklearn']}, running {sklearn.__version__}")

# main page
st.title("🛒 Will this online shopper buy?")
st.markdown(
    "Every row is one browsing session on a shopping site (UCI *Online Shoppers "
    "Purchasing Intention*). The models guess **Revenue**, i.e. whether the "
    "session ended in a purchase. Only about 1 in 6 do, so look at Recall, F1 "
    "and MCC and not just Accuracy.")

if uploaded is None:
    st.info("Upload a CSV from the sidebar to score it. If you don't have one, "
            "download `test_data.csv` from the sidebar and upload that.")
    st.subheader("Scores on the held-out test split")
    saved = pd.DataFrame(json.loads((MODEL_DIR / "metrics.json").read_text())).T
    st.dataframe(saved.drop(columns="confusion_matrix").astype(float).style.format("{:.4f}"),
                 width="stretch")
    st.stop()

try:
    raw = pd.read_csv(uploaded)
    X, y = split_xy(raw)
except Exception as err:
    st.error(f"Couldn't use this file: {err}")
    st.stop()

models = load_all_models()

st.caption(f"{len(X):,} rows, {int(y.sum()):,} purchases ({y.mean():.1%}). "
           "Models are loaded from `model/`, nothing is retrained.")
with st.expander("First 20 rows"):
    st.dataframe(raw.head(20), width="stretch")

# selected model
pred, metrics = run_model(models[chosen], X, y)
st.subheader(chosen)
for box, (k, v) in zip(st.columns(6), metrics.items()):
    box.metric(k, f"{v:.4f}")

left, right = st.columns([1, 1.3])
with left:
    st.pyplot(draw_confusion(y, pred, f"Confusion matrix: {chosen}"))
with right:
    report = classification_report(y, pred, labels=[0, 1], target_names=LABELS,
                                   output_dict=True, zero_division=0)
    st.markdown("**Classification report**")
    st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"), width="stretch")

# every model side by side
st.subheader("All five models on this file")
table = pd.DataFrame({n: run_model(m, X, y)[1] for n, m in models.items()}).T
table.index.name = "ML Model Name"
top = table["MCC"].idxmax()
st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#c8e6c9"),
             width="stretch")
st.markdown(f"Best MCC on this file: **{top}** ({table.loc[top, 'MCC']:.4f}).")

fig, ax = plt.subplots(figsize=(9, 3.2))
table[["Accuracy", "AUC", "F1", "MCC"]].plot.bar(ax=ax, rot=0, width=0.8, colormap="Set2")
ax.set_ylim(0, 1)
ax.set_ylabel("score")
ax.legend(ncol=4, loc="lower right", fontsize=8)
ax.tick_params(axis="x", labelsize=8)
fig.tight_layout()
st.pyplot(fig)

preds_df = raw.loc[X.index].assign(predicted_Revenue=np.where(pred == 1, True, False))
st.download_button(f"Download {chosen} predictions", preds_df.to_csv(index=False),
                   file_name="predictions.csv", mime="text/csv")
