import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr
import numpy as np
import io # Added for downloading files

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
        st.success("Data loaded successfully.")
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
    
    # Calculate Correlation and P-values
    corr_matrix = df_selected.corr()
    p_values = pd.DataFrame(index=df_selected.columns, columns=df_selected.columns)
    annot_matrix = pd.DataFrame(index=df_selected.columns, columns=df_selected.columns)

    for r in df_selected.columns:
        for c in df_selected.columns:
            corr_val = corr_matrix.loc[r, c]
            if r == c:
                p_values.loc[r, c] = 0.0
                annot_matrix.loc[r, c] = f"{corr_val:.2f}"
            else:
                _, p = pearsonr(df_selected[r], df_selected[c])
                p_values.loc[r, c] = p
                
                # Add significance stars (* p<0.05, ** p<0.01)
                stars = ""
                if p < 0.01:
                    stars = "**"
                elif p < 0.05:
                    stars = "*"
                
                annot_matrix.loc[r, c] = f"{corr_val:.2f}{stars}"
    
    with col1:
        st.markdown("**Correlation Matrix (* p<0.05, ** p<0.01)**")
        
        # Plot Heatmap
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        # Note: fmt="" is required here because we are passing strings (numbers + stars)
        sns.heatmap(corr_matrix, annot=annot_matrix, cmap="coolwarm", fmt="", linewidths=0.5, ax=ax_corr)
        st.pyplot(fig_corr)
        
        # Download Button for Heatmap
        buf = io.BytesIO()
        fig_corr.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        st.download_button(
            label="Download Heatmap (PNG)",
            data=buf.getvalue(),
            file_name="correlation_heatmap.png",
            mime="image/png"
        )
        
    with col2:
        st.markdown("**Statistical Significance (p-values)**")
        st.markdown("Exact p-values for reference.")
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
        
        # Build Regression Equation String
        params = model.params
        r_squared = model.rsquared
        
        eq_parts = []
        if 'const' in params:
            eq_parts.append(f"{params['const']:.4f}")
            
        for name, coef in params.items():
            if name != 'const':
                sign = "+" if coef >= 0 else "-"
                eq_parts.append(f"{sign} {abs(coef):.4f}({name})")
                
        equation_str = f"**Y ({target_var}) = " + " ".join(eq_parts) + f"**  *(R² = {r_squared:.4f})*"
        
        st.markdown("### Regression Equation")
        st.markdown(equation_str)
        
        st.text(model.summary())
        
        # Download Button for Regression Summary
        summary_text = f"Regression Equation:\nY = {' '.join(eq_parts)} (R-squared = {r_squared:.4f})\n\n{model.summary().as_text()}"
        st.download_button(
            label="Download Regression Summary (TXT)",
            data=summary_text,
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
