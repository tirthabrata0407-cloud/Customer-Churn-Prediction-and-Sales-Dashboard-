"""Streamlit dashboard entry point. Run with: streamlit run app.py"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from config import CSV_PATH, TARGET_COL
from data_utils import load_data, get_feature_columns
from models import (
    split_and_scale, build_models, train_all_models,
    evaluate_models, get_confusion_and_report, predict_single
)
from clustering import scale_features, kmeans_elbow, kmeans_fit, pca_2d, dbscan_outliers

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")


@st.cache_resource
def train_cached(X_train_s, y_train):
    models = build_models(y_train)
    return train_all_models(models, X_train_s, y_train)


# ---------------- Load data ----------------
st.sidebar.title("⚙️ Data Source")
uploaded = st.sidebar.file_uploader("Upload a CSV (optional)", type=["csv"])

if uploaded is not None:
    df = load_data(uploaded)
    st.sidebar.success("Using uploaded file.")
else:
    try:
        df = load_data(CSV_PATH)
        st.sidebar.info(f"Using CSV_PATH: {CSV_PATH}")
    except FileNotFoundError:
        st.error(f"Could not find '{CSV_PATH}'. Edit CSV_PATH in config.py or upload a CSV.")
        st.stop()

if TARGET_COL not in df.columns:
    st.error(f"Expected a target column named '{TARGET_COL}' in the dataset.")
    st.stop()

feature_cols = get_feature_columns(df, TARGET_COL)

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "EDA", "Model Comparison", "Customer Segmentation", "Predict a Customer"],
)

st.title("📊 Customer Churn Prediction & Sales Dashboard")

# ---------------- Overview ----------------
if page == "Overview":
    st.subheader("Dataset Snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Churn Rate", f"{df[TARGET_COL].mean()*100:.2f}%")
    st.dataframe(df.head(20), width='stretch')
    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), width='stretch')
    st.info(
        "This dataset has no transaction date/amount fields, so month-over-month "
        "sales-trend analysis isn't available here — this dashboard covers churn "
        "prediction and customer segmentation."
    )

# ---------------- EDA ----------------
elif page == "EDA":
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots(figsize=(4, 3))
    df[TARGET_COL].value_counts().plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_xticklabels(["Retained (0)", "Churned (1)"], rotation=0)
    st.pyplot(fig)

    st.subheader("Feature Distributions by Churn")
    num_cols = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
    chosen = st.multiselect("Pick numeric features", num_cols, default=num_cols[:3])
    for col in chosen:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(data=df, x=col, hue=TARGET_COL, bins=30, kde=True, ax=ax, palette="Set2")
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df[feature_cols + [TARGET_COL]].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# ---------------- Model Comparison ----------------
elif page == "Model Comparison":
    st.subheader("Train & Compare Churn Models")
    st.caption("Class imbalance is handled with class-weighting since churners are a small minority.")

    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    X_train_s, X_test_s, y_train, y_test, scaler = split_and_scale(df, feature_cols, TARGET_COL, test_size)

    with st.spinner("Training models..."):
        trained_models = train_cached(X_train_s, y_train)
        results_df, roc_data = evaluate_models(trained_models, X_test_s, y_test)

    st.dataframe(results_df.style.format({c: "{:.3f}" for c in results_df.columns[1:]}), width='stretch')

    st.subheader("ROC Curves")
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (fpr, tpr) in roc_data.items():
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    st.pyplot(fig)

    st.subheader("Confusion Matrix")
    chosen_model = st.selectbox("Choose a model", list(trained_models.keys()))
    cm, report = get_confusion_and_report(trained_models[chosen_model], X_test_s, y_test)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    st.text(report)

    st.session_state["scaler"] = scaler
    st.session_state["trained_models"] = trained_models
    st.session_state["feature_cols"] = feature_cols

# ---------------- Customer Segmentation ----------------
elif page == "Customer Segmentation":
    st.subheader("Unsupervised Segmentation")
    X_s, scaler = scale_features(df, feature_cols)

    st.markdown("**K-Means — Elbow Method**")
    k_range, inertias = kmeans_elbow(X_s)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(k_range, inertias, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    st.pyplot(fig)

    k_chosen = st.slider("Choose number of clusters (k)", 2, 8, 4)
    km_final = kmeans_fit(X_s, k_chosen)
    df_clusters = df.copy()
    df_clusters["cluster"] = km_final.labels_

    coords = pca_2d(X_s)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(coords[:, 0], coords[:, 1], c=km_final.labels_, cmap="tab10", s=8, alpha=0.6)
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    st.pyplot(fig)

    st.markdown("**Churn rate by cluster**")
    st.dataframe(df_clusters.groupby("cluster")[TARGET_COL].agg(["mean", "count"]))

    st.markdown("---")
    st.markdown("**DBSCAN — anomaly / outlier detection**")
    eps = st.slider("eps", 0.1, 3.0, 0.8, 0.1)
    min_samples = st.slider("min_samples", 2, 20, 5)
    sample, db, n_outliers, n_clusters = dbscan_outliers(df, feature_cols, eps, min_samples)
    st.write(f"Detected **{n_outliers}** outlier points out of {len(sample)} sampled rows "
             f"(across {n_clusters} clusters).")

# ---------------- Predict a Customer ----------------
elif page == "Predict a Customer":
    st.subheader("Score a Single Customer")

    if "trained_models" not in st.session_state:
        st.warning("Go to the 'Model Comparison' page first to train the models.")
        st.stop()

    trained_models = st.session_state["trained_models"]
    scaler = st.session_state["scaler"]
    feature_cols = st.session_state["feature_cols"]

    model_choice = st.selectbox("Model to use", list(trained_models.keys()))

    st.markdown("**Enter customer details:**")
    input_vals = {}
    cols = st.columns(2)
    for i, col in enumerate(feature_cols):
        with cols[i % 2]:
            if df[col].nunique() <= 2 and set(df[col].unique()).issubset({0, 1}):
                input_vals[col] = st.selectbox(col, [0, 1])
            else:
                input_vals[col] = st.number_input(col, value=float(df[col].median()))

    if st.button("Predict Churn"):
        pred, prob = predict_single(trained_models[model_choice], scaler, input_vals, feature_cols)
        if pred == 1:
            st.error(f"⚠️ Likely to churn — probability {prob:.1%}")
        else:
            st.success(f"✅ Likely to stay — churn probability {prob:.1%}")