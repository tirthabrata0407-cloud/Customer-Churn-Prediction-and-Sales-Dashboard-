import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


def split_and_scale(df, feature_cols, target_col, test_size=0.2):
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler


def build_models(y_train):
    """Returns a dict of untrained, class-weighted models (churn is a minority class)."""
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=200, random_state=42
        ),
    }
    if XGB_AVAILABLE:
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=6, scale_pos_weight=pos_weight, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
    if LGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        )
    return models

def train_all_models(models, X_train_s, y_train):
    for model in models.values():
        model.fit(X_train_s, y_train)
    return models


def evaluate_models(trained_models, X_test_s, y_test):
    """Returns (results_df, roc_data_dict)."""
    results = []
    roc_data = {}
    for name, model in trained_models.items():
        preds = model.predict(X_test_s)
        probs = model.predict_proba(X_test_s)[:, 1]
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1": f1_score(y_test, preds, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probs),
        })
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_data[name] = (fpr, tpr)

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    return results_df, roc_data


def get_confusion_and_report(model, X_test_s, y_test):
    preds = model.predict(X_test_s)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)
    return cm, report


def predict_single(model, scaler, input_dict, feature_cols):
    X_input = pd.DataFrame([input_dict])[feature_cols]
    X_input_s = scaler.transform(X_input)
    prob = model.predict_proba(X_input_s)[0, 1]
    pred = model.predict(X_input_s)[0]
    return pred, prob