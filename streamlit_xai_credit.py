# MIT-Ready Explainable AI Credit Scoring App with compatibility fix
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import warnings

warnings.filterwarnings("ignore")

st.title("Advanced Explainable AI & Fairness App for Credit Bias Evaluation")

st.markdown("""
Upload a mortgage or credit scoring dataset. This app trains a robust model, quantifies and visualizes bias, and gives feature-level explanations. All metrics, SHAP plots, and fairness diagnostics are included.
""")

# Step 1: File upload
uploaded_file = st.file_uploader("Upload credit data CSV", type=["csv"])
if uploaded_file:
    # Step 2: Load & summarize data
    df = pd.read_csv(uploaded_file)
    st.write("Data Sample:", df.head())
    
    # Step 3: Target and sensitive attribute selection
    target_col = st.selectbox("Select target (outcome) column", options=df.columns)
    sensitive_col = st.selectbox("Select sensitive attribute for fairness analysis (e.g. gender/race)", options=[c for c in df.columns if c != target_col])
    
    # Step 4: Feature preparation
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify categorical columns and encode using pandas.get_dummies for compatibility
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    if cat_cols:
        X_encoded = pd.get_dummies(X[cat_cols], drop_first=True)
        X_num = X[num_cols].reset_index(drop=True)
        X_prepared = pd.concat([X_num, X_encoded], axis=1)
    else:
        X_prepared = X[num_cols]
        
    # Fill missing for robustness
    X_prepared.fillna(X_prepared.median(), inplace=True)
    st.write("Processed Feature Sample:", X_prepared.head())
    
    # Binary target check
    if y.nunique() > 2:
        st.error("Currently supports binary target only. Please select a binary outcome.")
        st.stop()
    
    # Split data
    test_size_pct = st.slider("Test set size (%)", 10, 50, 30)
    X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size=test_size_pct/100, random_state=42)
    
    # Step 5: Train classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict and metrics
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    st.markdown(f"**Accuracy:** {acc:.3f}")
    st.markdown(f"**ROC AUC:** {auc:.3f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    # Step 6: Fairness metrics
    sensitive_test = df.loc[X_test.index, sensitive_col].values
    dp_diff = demographic_parity_difference(y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_test)
    eo_diff = equalized_odds_difference(y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_test)
    st.subheader("Fairness Diagnostics:")
    st.write(f"**Demographic Parity Difference:** {dp_diff:.4f} (Values close to 0 suggest fairness)")
    st.write(f"**Equalized Odds Difference:** {eo_diff:.4f} (Values close to 0 suggest fairness)")
    
    # Step 7: SHAP explanations (global and local)
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_test)
    st.subheader("Global SHAP Feature Importance")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals[1], X_test, show=False)
    st.pyplot(plt.gcf())
    st.subheader("Dependence Plot")
    feature_for_plot = st.selectbox("Select feature for dependence plot", X_prepared.columns)
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(feature_for_plot, shap_vals[1], X_test, show=False)
    st.pyplot(plt.gcf())
    st.subheader("Local SHAP Explanations")
    instance_idx = st.slider(f"Test instance to explain (0-{X_test.shape[0]-1})", 0, X_test.shape[0]-1, 0)
    force_plot = shap.force_plot(
        explainer.expected_value[1],
        shap_vals[1][instance_idx],
        X_test.iloc[instance_idx],
        matplotlib=True,
        show=False
    )
    st.pyplot(force_plot)
    
    # Step 8: Feature-level bias decomposition
    st.subheader("Feature-level Bias Contribution")
    group_A = sensitive_test == np.unique(sensitive_test)[0]
    group_B = ~group_A
    shap_contrib_A = np.mean(shap_vals[1][group_A], axis=0)
    shap_contrib_B = np.mean(shap_vals[1][group_B], axis=0)
    bias_contrib = shap_contrib_A - shap_contrib_B
    bias_df = pd.DataFrame({
        'Feature': X_prepared.columns,
        'Group_A_Contribution': shap_contrib_A,
        'Group_B_Contribution': shap_contrib_B,
        'Bias_Difference': bias_contrib
    }).sort_values(by='Bias_Difference', ascending=False)
    st.dataframe(bias_df)
    fig = px.bar(
        bias_df, x='Feature', y='Bias_Difference',
        title="SHAP Feature-Level Bias Difference (Group A vs Group B)"
    )
    st.plotly_chart(fig)
    
    # Step 9: Adversarial robustness prototype (random noise test)
    st.subheader("Adversarial Robustness Test")
    X_noise = X_test.copy()
    noise_intensity = st.slider("Noise intensity (%)", 0, 50, 10)/100
    X_noise += np.random.normal(0, noise_intensity, X_noise.shape)
    y_pred_noise = clf.predict(X_noise)
    acc_noise = accuracy_score(y_test, y_pred_noise)
    st.write(f"Accuracy under adversarial noise: {acc_noise:.3f}")
    
    # Step 10: Multi-stakeholder feedback
    st.subheader("Annotate Cases for Human Review")
    feedback_instance = st.slider("Label test instance for review", 0, X_test.shape[0]-1, 0)
    feedback_note = st.text_input(f"Feedback for instance {feedback_instance}", "")
    if st.button("Save Feedback"):
        st.success("Feedback saved (simulated). For real deployments, connect to a backend or database.")
else:
    st.info("Upload a CSV with binary target and at least one sensitive attribute column.")
