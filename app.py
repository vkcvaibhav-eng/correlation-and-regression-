import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr
import numpy as np
import io

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
        st.success("Data loaded. Proceed with analysis.")
    except Exception as e:
        st.error(f"Failed to read file: {e}. Check your formatting.")
        st.stop()

    # Variable Selection
    st.subheader("1. Variable Selection")
    columns = df.columns.tolist()
    
    target_var = st.selectbox("Select Dependent Variable (Y)", options=columns)
    
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
    
    # Calculate Correlation and P-values
    corr_matrix = df_selected.corr()
    p_values = pd.DataFrame(index=df_selected.columns, columns=df_selected.columns)
    
    for r in df_selected.columns:
        for c in df_selected.columns:
            if r == c:
                p_values.loc[r, c] = 0.0
            else:
                _, p = pearsonr(df_selected[r], df_selected[c])
                p_values.loc[r, c] = p

    # Create annotation matrix with asterisks
    annot_labels = np.empty_like(corr_matrix, dtype=str)
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            val = corr_matrix.iloc[i, j]
            p_val = p_values.iloc[i, j]
            
            if i == j:
                stars = ""
            elif p_val <= 0.001:
                stars = "***"
            elif p_val <= 0.01:
                stars = "**"
            elif p_val <= 0.05:
                stars = "*"
            else:
                stars = ""
                
            annot_labels[i, j] = f"{val:.2f}{stars}"

    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("**Correlation Matrix (with Significance)**")
        st.markdown("*p≤0.05, **p≤0.01, ***p≤0.001")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=annot_labels, fmt="", cmap="coolwarm", linewidths=0.5, ax=ax, annot_kws={"size": 10})
        st.pyplot(fig)
        
        # Download button for Heatmap
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
        st.download_button(
            label="Download Heatmap (PNG)",
            data=buf.getvalue(),
            file_name="correlation_heatmap.png",
            mime="image/png"
        )
        
    with col2:
        st.markdown("**Exact P-values**")
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
        
        # Construct Regression Equation
        params = model.params
        r_squared = model.rsquared
        
        eq_parts = [f"{params.iloc[0]:.4f}"] # Intercept
        for i, col_name in enumerate(X.columns):
            coef = params.iloc[i+1]
            sign = "+" if coef >= 0 else "-"
            eq_parts.append(f"{sign} {abs(coef):.4f}({col_name})")
            
        equation_str = f"Y = {' '.join(eq_parts)} (R² = {r_squared:.4f})"
        
        st.markdown("**Regression Equation:**")
        st.info(equation_str)
        
        st.text(model.summary())
        
        # Download button for Regression Summary
        summary_download_text = f"Regression Equation:\n{equation_str}\n\n{model.summary().as_text()}"
        st.download_button(
            label="Download Regression Summary (TXT)",
            data=summary_download_text,
            file_name="regression_summary.txt",
            mime="text/plain"
        )
        
        # Diagnostics
        st.markdown("**Residual Diagnostics**")
        fig_resid, ax_resid = plt.subplots(figsize=(8, 4))
        sns.histplot(model.resid, kde=True, ax=ax_resid)
        ax_resid.set_title("Residual Distribution (Should be normally distributed)")
        st.pyplot(fig_resid)
        
    except Exception as e:
        st.error(f"Regression failed: {e}. Check for multicollinearity or constant variables.")
