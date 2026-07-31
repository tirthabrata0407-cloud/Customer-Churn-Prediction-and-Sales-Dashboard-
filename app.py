import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn & Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Customer Churn Prediction & Sales Dashboard")
st.markdown("---")


# -----------------------------------------------------------------------------
# 2. DATA ACQUISITION & PREPROCESSING ENGINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Attempts to load custom files or generates synthetic data matching specifications."""
    try:
        customers = pd.read_csv("cleaned_customer_churn_dataset.csv")
    except Exception:
        # Fallback synthetic dataset matching schema
        np.random.seed(42)
        n = 500
        customer_ids = [f"CUST-{1000 + i}" for i in range(n)]
        tenure = np.random.randint(1, 72, size=n)
        monthly_charges = np.round(np.random.uniform(20.0, 120.0, size=n), 2)
        total_charges = np.round(
            tenure * monthly_charges + np.random.normal(0, 10, size=n), 2
        )
        total_charges = np.maximum(total_charges, monthly_charges)

        contract = np.random.choice(
            ["Month-to-month", "One year", "Two year"],
            size=n,
            p=[0.5, 0.3, 0.2],
        )
        internet_service = np.random.choice(
            ["DSL", "Fiber optic", "No"], size=n, p=[0.4, 0.4, 0.2]
        )
        paperless = np.random.choice(["Yes", "No"], size=n)
        payment_method = np.random.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer",
                "Credit card",
            ],
            size=n,
        )

        churn_prob = (
            0.2
            + 0.3 * (contract == "Month-to-month")
            + 0.2 * (internet_service == "Fiber optic")
            - 0.005 * tenure
        )
        churn_prob = np.clip(churn_prob, 0.05, 0.95)
        churn = np.random.binomial(1, churn_prob)
        churn_labels = np.where(churn == 1, "Yes", "No")

        customers = pd.DataFrame(
            {
                "customerID": customer_ids,
                "tenure": tenure,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "Contract": contract,
                "InternetService": internet_service,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment_method,
                "Churn": churn_labels,
            }
        )

    # Load / Generate Synthetic Transaction Data
    try:
        transactions = pd.read_csv("transactions.csv")
        transactions["date"] = pd.to_datetime(transactions["date"])
    except Exception:
        dates = pd.date_range(start="2025-01-01", end="2026-06-30", freq="D")
        n_tx = 3000
        tx_dates = np.random.choice(dates, size=n_tx)
        tx_cust = np.random.choice(customers["customerID"], size=n_tx)
        tx_amount = np.round(
            np.random.exponential(scale=50, size=n_tx) + 10, 2
        )
        tx_product = np.random.choice(
            [
                "Basic Subscription",
                "Pro Plan",
                "Enterprise Suite",
                "Add-on Module",
            ],
            size=n_tx,
            p=[0.4, 0.3, 0.15, 0.15],
        )

        transactions = pd.DataFrame(
            {
                "transaction_id": [f"TX-{10000+i}" for i in range(n_tx)],
                "customerID": tx_cust,
                "date": pd.to_datetime(tx_dates),
                "amount": tx_amount,
                "product": tx_product,
            }
        )

    return customers, transactions


customers_df, transactions_df = load_data()

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS & NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("🕹️ Navigation")
app_mode = st.sidebar.radio(
    "Select Section",
    [
        "Overview & Key Metrics",
        "Sales Trend Analysis",
        "Churn Prediction & ML",
        "Customer Segmentation (K-Means)",
    ],
)

# -----------------------------------------------------------------------------
# 4. MODULE 1: OVERVIEW & KEY METRICS
# -----------------------------------------------------------------------------
if app_mode == "Overview & Key Metrics":
    st.header("📌 Business Overview & Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    total_customers = len(customers_df)
    churned_customers = len(customers_df[customers_df["Churn"] == "Yes"])
    churn_rate = (churned_customers / total_customers) * 100
    total_revenue = transactions_df["amount"].sum()

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churned_customers:,}")
    col3.metric("Churn Rate", f"{churn_rate:.1f}%")
    col4.metric("Total Revenue", f"${total_revenue:,.2f}")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Customer Churn Distribution")
        fig_churn = px.pie(
            customers_df,
            names="Churn",
            hole=0.4,
            color="Churn",
            color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
        )
        st.plotly_chart(fig_churn, use_container_width=True)

    with c2:
        st.subheader("Contract Type vs Churn")
        fig_contract = px.histogram(
            customers_df,
            x="Contract",
            color="Churn",
            barmode="group",
            color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
        )
        st.plotly_chart(fig_contract, use_container_width=True)

# -----------------------------------------------------------------------------
# 5. MODULE 2: SALES TREND ANALYSIS
# -----------------------------------------------------------------------------
elif app_mode == "Sales Trend Analysis":
    st.header("📈 Sales Trend & Revenue Analysis")

    # Group by Month
    transactions_df["month"] = (
        transactions_df["date"].dt.to_period("M").astype(str)
    )
    monthly_sales = (
        transactions_df.groupby("month")["amount"].sum().reset_index()
    )

    st.subheader("Monthly Revenue Growth")
    fig_line = px.line(
        monthly_sales,
        x="month",
        y="amount",
        markers=True,
        title="Monthly Revenue Trend",
        labels={"month": "Month", "amount": "Sales Revenue ($)"},
    )
    fig_line.update_traces(line_color="#00CC96", line_width=3)
    st.plotly_chart(fig_line, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Revenue by Product Category")
        prod_sales = (
            transactions_df.groupby("product")["amount"]
            .sum()
            .reset_index()
            .sort_values(by="amount", ascending=False)
        )
        fig_bar = px.bar(
            prod_sales, x="product", y="amount", color="product", text_auto=".2s"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.subheader("Sales Heatmap (Year x Month)")
        transactions_df["year"] = transactions_df["date"].dt.year
        transactions_df["month_num"] = transactions_df["date"].dt.month
        pivot_sales = transactions_df.pivot_table(
            index="year",
            columns="month_num",
            values="amount",
            aggfunc="sum",
            fill_value=0,
        )
        fig_heat = px.imshow(
            pivot_sales,
            labels=dict(x="Month", y="Year", color="Sales ($)"),
            aspect="auto",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. MODULE 3: CHURN PREDICTION MODELS
# -----------------------------------------------------------------------------
elif app_mode == "Churn Prediction & ML":
    st.header("🤖 Machine Learning - Churn Risk Engine")

    df_ml = customers_df.copy()
    if "customerID" in df_ml.columns:
        df_ml = df_ml.drop("customerID", axis=1)

    cat_cols = df_ml.select_dtypes(include=["object"]).columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col])
        encoders[col] = le

    X = df_ml.drop("Churn", axis=1)
    y = df_ml["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    st.subheader("Random Forest Classifier Evaluation")
    m1, m2 = st.columns(2)
    m1.metric("Model Accuracy", f"{acc * 100:.2f}%")
    m2.metric("ROC-AUC Score", f"{auc:.3f}")

    st.markdown("---")
    st.subheader("🔮 Predict Individual Customer Churn Risk")

    c1, c2, c3 = st.columns(3)
    with c1:
        tenure_val = st.slider("Tenure (Months)", 1, 72, 12)
        contract_val = st.selectbox(
            "Contract Type", ["Month-to-month", "One year", "Two year"]
        )
    with c2:
        monthly_val = st.number_input("Monthly Charges ($)", 20.0, 150.0, 70.0)
        internet_val = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
    with c3:
        paperless_val = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_val = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer",
                "Credit card",
            ],
        )

    # Input preprocessing
    single_input = pd.DataFrame(
        [
            {
                "tenure": tenure_val,
                "MonthlyCharges": monthly_val,
                "TotalCharges": tenure_val * monthly_val,
                "Contract": encoders["Contract"].transform([contract_val])[0],
                "InternetService": encoders["InternetService"].transform(
                    [internet_val]
                )[0],
                "PaperlessBilling": encoders["PaperlessBilling"].transform(
                    [paperless_val]
                )[0],
                "PaymentMethod": encoders["PaymentMethod"].transform(
                    [payment_val]
                )[0],
            }
        ]
    )

    churn_prob = model.predict_proba(single_input)[0][1]

    st.write("---")
    if churn_prob > 0.5:
        st.error(
            f"⚠️ **High Churn Risk!** Predicted Probability: **{churn_prob*100:.1f}%**"
        )
    else:
        st.success(
            f"✅ **Low Churn Risk!** Predicted Probability: **{churn_prob*100:.1f}%**"
        )

# -----------------------------------------------------------------------------
# 7. MODULE 4: UNSUPERVISED CUSTOMER SEGMENTATION
# -----------------------------------------------------------------------------
elif app_mode == "Customer Segmentation (K-Means)":
    st.header("🎯 Customer Segmentation (K-Means Clustering)")

    features = ["tenure", "MonthlyCharges", "TotalCharges"]
    X_seg = customers_df[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seg)

    k = st.slider("Select Number of Clusters (K)", 2, 6, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    customers_df["Cluster"] = kmeans.fit_predict(X_scaled)
    customers_df["Cluster"] = customers_df["Cluster"].astype(str)

    st.subheader("3D Interactive Segment Visualization")
    fig_3d = px.scatter_3d(
        customers_df,
        x="tenure",
        y="MonthlyCharges",
        z="TotalCharges",
        color="Cluster",
        hover_data=["customerID", "Contract", "Churn"],
        title="Customer Clusters based on Tenure & Charges",
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    st.subheader("Segment Summary Statistics")
    cluster_summary = (
        customers_df.groupby("Cluster")[features].mean().reset_index()
    )
    st.dataframe(cluster_summary.style.highlight_max(axis=0, color="#d1e7dd"))