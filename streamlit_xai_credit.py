# streamlit_xai_credit.py

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Title and description
st.title("Explainable AI Credit Scoring Bias Detection Tool")
st.markdown("""
This app lets you upload your credit scoring dataset to train a model and explore explainability analyses
with SHAP values. Discover which features influence credit risk predictions to understand and reduce bias.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your credit data CSV file", type=["csv"])
if uploaded_file:

    # Load data
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Data Sample:")
        st.dataframe(data.head())
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Select target variable
    target_col = st.selectbox("Select the target (outcome) column", options=data.columns)
    if target_col is None:
        st.warning("Please select the target column to proceed.")
        st.stop()

    # Features/target split
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Show type info and missing values
    st.write("Feature Types:")
    st.write(X.dtypes)
    st.write("---")
    st.write("Missing values count in each feature:")
    st.write(X.isna().sum())

    # Fill missing values with median for simplicity
    X = X.fillna(X.median())

    # Some basic data validation
    if y.nunique() > 2:
        st.warning("Target variable has more than two classes; this app currently supports binary classification only.")
        st.stop()

    # Split data for training and testing
    test_size = st.slider("Test set size %", 10, 50, 30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    st.write(f"**Test Set Accuracy:** {acc:.3f}")
    st.write(f"**Test Set ROC AUC:** {auc:.3f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Explainability using SHAP
    st.header("Explainability with SHAP")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)

    st.subheader("Global Feature Importance (SHAP Summary Plot)")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[1], X_test, show=False)
    st.pyplot(plt.gcf())

    st.subheader("SHAP Dependence Plot")
    selected_feature = st.selectbox("Select feature for dependence plot", X.columns)
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(selected_feature, shap_values[1], X_test, show=False)
    st.pyplot(plt.gcf())

    st.subheader("Local Explainability (Force Plot)")
    select_index = st.slider(f"Select sample index to explain (0 to {len(X_test)-1})", 0, len(X_test) - 1, 0)
    selected_sample = X_test.iloc[select_index]

    st.write(f"Features of sample #{select_index}:")
    st.write(selected_sample)

    # Display force plot for the selected instance
    force_plot = shap.force_plot(
        explainer.expected_value[1],
        shap_values[1][select_index, :],
        selected_sample,
        matplotlib=True,
        show=False
    )
    st.pyplot(force_plot)

else:
    st.info("Upload a CSV file to get started.")

