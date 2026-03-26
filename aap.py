import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr
import numpy as np

st.set_page_config(page_title="Statistical Analysis App", layout="wide")

st.title("Correlation & Regression Analysis")
st.markdown("Upload your dataset. Stop guessing and let the math do the work.")

# File Uploader
uploaded_file = st.file_uploader("Upload your data (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Read data
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("Data loaded. Don't break it.")
        st.write(df.head())
    except Exception as e:
        st.error(f"Failed to read file: {e}. Check your formatting.")
        st.stop()

    # Variable Selection
    st.subheader("1. Variable Selection")
    columns = df.columns.tolist()
    
    target_var = st.selectbox("Select Dependent Variable (Y)", options=columns)
    
    # Remove target from independent options
    indep_options = [col for col in columns if col != target_var]
    predictor_vars = st.multiselect("Select Independent Variables (X)", options=indep_options, default=indep_options)

    if not predictor_vars:
        st.warning("Select at least one independent variable.")
        st.stop()

    df_selected = df[[target_var] + predictor_vars].dropna()
    if df_selected.empty:
        st.error("Your selected columns contain too many missing values. Clean your data.")
        st.stop()

    st.markdown("---")
    
    # Correlation Analysis
    st.subheader("2. Correlation Analysis & Heatmap")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Correlation Matrix**")
        corr_matrix = df_selected.corr()
        
        # Plot Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.markdown("**Statistical Significance (p-values)**")
        st.markdown("If p > 0.05, the correlation is likely garbage. Pay attention.")
        
        p_values = pd.DataFrame(index=df_selected.columns, columns=df_selected.columns)
        for r in df_selected.columns:
            for c in df_selected.columns:
                _, p = pearsonr(df_selected[r], df_selected[c])
                p_values.loc[r, c] = p
                
        # Format p-values to scientific notation for readability
        st.dataframe(p_values.astype(float).style.format("{:.4e}"))

    st.markdown("---")
    
    # Regression Analysis
    st.subheader("3. Multiple Linear Regression")
    
    X = df_selected[predictor_vars]
    Y = df_selected[target_var]
    
    # Add constant for intercept
    X_with_const = sm.add_constant(X)
    
    try:
        model = sm.OLS(Y, X_with_const).fit()
        st.text(model.summary())
        
        # Diagnostics
        st.markdown("**Residual Diagnostics**")
        fig_resid, ax_resid = plt.subplots(figsize=(8, 4))
        sns.histplot(model.resid, kde=True, ax=ax_resid)
        ax_resid.set_title("Residual Distribution (Should be normally distributed)")
        st.pyplot(fig_resid)
        
    except Exception as e:
        st.error(f"Regression failed: {e}. Check for multicollinearity or constant variables.")