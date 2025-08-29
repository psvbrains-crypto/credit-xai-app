# streamlit_xai_credit.py

import streamlit as st
import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Hide Streamlit warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Explainable AI: Credit Scoring Bias Detection")

# Upload CSV data
uploaded_file = st.file_uploader("Upload credit data CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    # Select target column
    target_col = st.selectbox("Select target variable (credit approval)", data.columns)

    # Split data into features and target
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Train/test split
    test_size = st.slider("Test set size (%)", 10, 50, 30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict & accuracy
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.write(f"Model Accuracy on test set: {acc:.2f}")

    # SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    st.write("### Global Feature Importance (SHAP Summary Plot)")
    shap.summary_plot(shap_values[1], X_test, show=False)
    st.pyplot(bbox_inches='tight')

    # Select one observation to explain
    idx = st.slider(f"Select test sample to explain (0 to {len(X_test)-1})", 0, len(X_test)-1, 0)
    obs = X_test.iloc[[idx]]

    st.write(f"### Local Explanation for Sample #{idx}")
    st.write(obs)

    shap.force_plot(explainer.expected_value[1], shap_values[1][idx,:], obs, matplotlib=True, show=False)
    st.pyplot(bbox_inches='tight')

else:
    st.info("Please upload a CSV file with your credit data to begin.")

