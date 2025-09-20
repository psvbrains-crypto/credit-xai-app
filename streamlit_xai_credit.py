import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    calibration_curve,
)
from sklearn.calibration import CalibratedClassifierCV
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    equal_opportunity_difference,
    false_positive_rate_difference,
)
import warnings

warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Sophisticated Explainable AI & Fairness Tool for Credit Bias Evaluation")

st.markdown("""
Upload a mortgage or credit scoring dataset to train a robust explainable model. Explore fairness diagnostics, interactive SHAP explanations, calibration, and adversarial robustness all in one app designed to meet the highest academic and societal standards.
""")

# Sidebar for user inputs and configuration
with st.sidebar:
    st.header("Settings")
    test_size_pct = st.slider("Test set size (%)", 10, 50, 30)
    noise_intensity = st.slider("Adversarial noise intensity (%)", 0, 50, 10) / 100

uploaded_file = st.file_uploader("Upload credit data CSV with binary target & sensitive attribute", type=["csv"])

if uploaded_file:
    # Load data, show basic info and detect issues
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    st.subheader("Data Preview")
    st.dataframe(df.head())

    st.subheader("Data Summary")
    st.write(df.describe(include='all'))

    # Select target & sensitive features with validation
    target_col = st.selectbox("Select binary target column", options=df.columns)
    sensitive_col = st.selectbox("Select sensitive attribute column", options=[c for c in df.columns if c != target_col])

    # Safe convert target to numeric ints (0/1)
    y_raw = df[target_col].copy()
    y = pd.to_numeric(y_raw, errors='coerce')
    if y.isnull().any():
        st.warning(f"Warning: Target column '{target_col}' had non-numeric values which were coerced to NaN.")
    y = y.dropna().astype(int)

    # Filter full dataset to only rows where target is valid
    df = df.loc[y.index]
    y = y.loc[y.index]

    if y.nunique() != 2:
        st.error("Target column must contain exactly two unique classes (binary).")
        st.stop()

    # Features preparation
    X = df.drop(columns=[target_col])

    # Encode categorical independent variables
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    if cat_cols:
        X_encoded = pd.get_dummies(X[cat_cols], drop_first=True)
        X_num = X[num_cols].reset_index(drop=True)
        X_prepared = pd.concat([X_num, X_encoded], axis=1)
    else:
        X_prepared = X[num_cols]

    X_prepared.fillna(X_prepared.median(), inplace=True)

    st.subheader("Processed Features Sample")
    st.dataframe(X_prepared.head())

    # Remove rows with any missing features or target values
    valid_idx = X_prepared.dropna().index.intersection(y.index)
    X_prepared = X_prepared.loc[valid_idx]
    y = y.loc[valid_idx]

    if len(y) == 0:
        st.error("No data left after removing missing values. Check dataset.")
        st.stop()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size=test_size_pct / 100, stratify=y, random_state=42)

    # Reproducible seed
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # Calibrated classifier for reliable probabilities
    calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid')
    calibrated_clf.fit(X_train, y_train)

    y_pred = calibrated_clf.predict(X_test)
    y_proba = calibrated_clf.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.subheader("Model Performance Metrics")
    st.write(f"Accuracy: **{acc:.3f}**")
    st.write(f"AUC (ROC): **{auc:.3f}**")
    st.text(classification_report(y_test, y_pred))

    # Confusion matrix visualization
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=calibrated_clf.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # Calibration curve plot
    st.subheader("Calibration Curve")
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    fig = px.line(x=prob_pred, y=prob_true, title='Calibration Curve', labels={'x':'Mean predicted probability', 'y':'Fraction of positives'})
    fig.add_scatter(x=[0,1], y=[0,1], mode='lines', name='Perfectly calibrated', line=dict(dash='dash'))
    st.plotly_chart(fig)

    # Fairness metrics calculate with error-safe wrapper
    sensitive_test = df.loc[X_test.index, sensitive_col]
    sensitive_test = sensitive_test.astype(str)
    y_test_int = y_test.astype(int)
    y_pred_int = y_pred.astype(int)

    def safe_metric(func, y_true, y_pred, sensitive_features):
        try:
            return func(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
        except Exception as e:
            st.warning(f"Fairness metric error: {e}")
            return None

    st.subheader("Fairness and Bias Diagnostics")

    dp_diff = safe_metric(demographic_parity_difference, y_test_int, y_pred_int, sensitive_test)
    eo_diff = safe_metric(equalized_odds_difference, y_test_int, y_pred_int, sensitive_test)
    eoport_diff = safe_metric(equal_opportunity_difference, y_test_int, y_pred_int, sensitive_test)
    fpr_diff = safe_metric(false_positive_rate_difference, y_test_int, y_pred_int, sensitive_test)

    if dp_diff is not None:
        st.write(f"Demographic Parity Difference: {dp_diff:.4f}")
    if eo_diff is not None:
        st.write(f"Equalized Odds Difference: {eo_diff:.4f}")
    if eoport_diff is not None:
        st.write(f"Equal Opportunity Difference: {eoport_diff:.4f}")
    if fpr_diff is not None:
        st.write(f"False Positive Rate Difference: {fpr_diff:.4f}")

    # SHAP explainability
    st.subheader("SHAP Feature Importance & Explanations")
    explainer = shap.TreeExplainer(calibrated_clf)
    shap_vals = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals[1], X_test, show=False)
    st.pyplot(plt.gcf())

    feature_for_plot = st.selectbox("Feature for SHAP dependence plot", X_prepared.columns)
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(feature_for_plot, shap_vals[1], X_test, show=False)
    st.pyplot(plt.gcf())

    instance_idx = st.slider("Sample index for local SHAP explanation", 0, X_test.shape[0]-1, 0)
    force_plot_mat = shap.force_plot(
        explainer.expected_value[1],
        shap_vals[1][instance_idx],
        X_test.iloc[instance_idx],
        matplotlib=True,
        show=False,
    )
    st.pyplot(force_plot_mat)

    # Feature-level bias contributions
    st.subheader("Feature-level Bias Contributions")
    group_vals = sensitive_test.values
    unique_groups = np.unique(group_vals)

    if len(unique_groups) != 2:
        st.warning("Feature-level bias calculation currently requires exactly 2 sensitive groups.")
    else:
        group_A = group_vals == unique_groups[0]
        group_B = group_vals == unique_groups[1]

        shap_contrib_A = np.mean(shap_vals[1][group_A], axis=0)
        shap_contrib_B = np.mean(shap_vals[1][group_B], axis=0)
        bias_contrib = shap_contrib_A - shap_contrib_B

        bias_df = pd.DataFrame({
            'Feature': X_prepared.columns,
            unique_groups[0]: shap_contrib_A,
            unique_groups[1]: shap_contrib_B,
            'Bias Difference': bias_contrib,
        }).sort_values(by='Bias Difference', ascending=False)

        st.dataframe(bias_df)

        fig_bias = px.bar(bias_df, x='Feature', y='Bias Difference', title="SHAP Bias Contribution Difference")
        st.plotly_chart(fig_bias)

    # Adversarial robustness via noise injection
    st.subheader("Adversarial Robustness Test")
    X_noise = X_test.copy()
    X_noise += np.random.normal(0, noise_intensity, X_noise.shape)
    y_pred_noise = calibrated_clf.predict(X_noise)
    acc_noise = accuracy_score(y_test, y_pred_noise)
    st.write(f"Accuracy with noise intensity {noise_intensity*100:.1f}%: {acc_noise:.3f}")

    # Multi-stakeholder feedback section (placeholder)
    st.subheader("Annotate Cases for Human Review")
    feedback_instance = st.slider("Choose test instance index to review", 0, X_test.shape[0]-1, 0)
    feedback_note = st.text_area(f"Feedback for instance {feedback_instance}")
    if st.button("Submit Feedback"):
        st.success("Feedback saved (simulated) — integrate with backend to persist.")

else:
    st.info("Upload a CSV with binary target and at least one sensitive attribute column.")
