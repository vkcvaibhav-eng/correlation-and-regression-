import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
import numpy as np
import io

st.set_page_config(page_title="Statistical Analysis App", layout="wide")

st.title("Agricultural Correlation & Regression Analysis")
st.markdown("Upload your dataset to generate publication-ready tables, heatmaps, and regression models.")

# --- Sidebar: Guide & Sample Data ---
with st.sidebar:
    st.header("📋 Data Upload Guide")
    st.markdown("""
    **Format your CSV/Excel file correctly:**
    * **Rows** should represent observations (e.g., Weeks, Months, or specific dates).
    * **Columns** should represent variables (e.g., Population count, Max Temp, Rainfall).
    * Do not include special characters or spaces in column headers.
    """)
    
    # Generate Sample Agri Data (e.g., Pest population vs Weather)
    sample_data = pd.DataFrame({
        'Observation_Week': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Pest_Population': [12, 15, 22, 28, 35, 40, 38, 25, 18, 10],
        'Max_Temperature_C': [30, 32, 35, 36, 38, 37, 34, 31, 29, 28],
        'Min_Temperature_C': [15, 18, 22, 24, 26, 25, 23, 19, 16, 14],
        'Relative_Humidity_pct': [40, 45, 50, 55, 60, 65, 70, 60, 50, 45],
        'Rainfall_mm': [0, 0, 5, 10, 50, 100, 120, 40, 10, 0]
    })
    
    csv_sample = sample_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample CSV",
        data=csv_sample,
        file_name="sample_agri_data.csv",
        mime="text/csv",
        help="Download a sample dataset showing pest population vs weather parameters."
    )

# --- Main App ---
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
        
        with st.expander("Preview Raw Data"):
            st.dataframe(df.head())
            
    except Exception as e:
        st.error(f"Failed to read file: {e}. Check your formatting.")
        st.stop()

    # Variable Selection
    st.subheader("1. Variable Selection")
    
    # Filter only numeric columns for analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    target_var = st.selectbox("Select Dependent Variable (Y) [e.g., Population]", options=numeric_cols)
    
    # Remove target from independent options
    indep_options = [col for col in numeric_cols if col != target_var]
    predictor_vars = st.multiselect("Select Independent Variables (X) [e.g., Weather Parameters]", options=indep_options, default=indep_options)

    if not predictor_vars:
        st.warning("Select at least one independent variable.")
        st.stop()

    df_selected = df[[target_var] + predictor_vars].dropna()
    if df_selected.empty:
        st.error("Your selected columns contain too many missing values. Clean your data.")
        st.stop()

    st.markdown("---")
    
    # Publication-Ready Correlation Table (Target vs Predictors)
    st.subheader("2. Publication-Ready Correlation Table")
    st.markdown("Generates a standard table showing the correlation of weather/abiotic parameters against your dependent variable.")
    
    col_method, col_label = st.columns(2)
    with col_method:
        corr_method = st.selectbox("Select Correlation Method", ["Pearson (Parametric)", "Spearman (Non-Parametric Rank)"])
    with col_label:
        column_label = st.text_input("Data Column Label (e.g., Year, Location, or 'Correlation (r)')", value="2025")

    # Calculate 1-to-many correlation
    table_results = []
    for var in predictor_vars:
        if "Pearson" in corr_method:
            r, p = pearsonr(df_selected[var], df_selected[target_var])
        else:
            r, p = spearmanr(df_selected[var], df_selected[target_var])
            
        stars = ""
        if p < 0.01:
            stars = "**"
        elif p < 0.05:
            stars = "*"
            
        table_results.append({
            "Parameters": var,
            column_label: f"{r:.2f}{stars}"
        })
        
    agri_table_df = pd.DataFrame(table_results)
    
    st.markdown(f"**Population correlation with the weather parameter ({corr_method.split()[0]})**")
    st.table(agri_table_df.set_index("Parameters"))
    st.caption("*Significant at P < 0.05, **Significant at P < 0.01")
    
    # Download Table
    csv_table = agri_table_df.to_csv(index=False)
    st.download_button(
        label="Download Table as CSV",
        data=csv_table,
        file_name="population_weather_correlation.csv",
        mime="text/csv"
    )

    st.markdown("---")
    
    # Heatmap & Cross-Correlation
    st.subheader("3. Cross-Correlation Analysis & Heatmap")
    
    col1, col2 = st.columns([1, 1])
    
    # Calculate Full Correlation Matrix
    if "Pearson" in corr_method:
        corr_matrix = df_selected.corr(method='pearson')
    else:
        corr_matrix = df_selected.corr(method='spearman')
        
    p_values = pd.DataFrame(index=df_selected.columns, columns=df_selected.columns)
    annot_matrix = pd.DataFrame(index=df_selected.columns, columns=df_selected.columns)

    for r in df_selected.columns:
        for c in df_selected.columns:
            corr_val = corr_matrix.loc[r, c]
            if r == c:
                p_values.loc[r, c] = 0.0
                annot_matrix.loc[r, c] = f"{corr_val:.2f}"
            else:
                if "Pearson" in corr_method:
                    _, p = pearsonr(df_selected[r], df_selected[c])
                else:
                    _, p = spearmanr(df_selected[r], df_selected[c])
                p_values.loc[r, c] = p
                
                stars = ""
                if p < 0.01:
                    stars = "**"
                elif p < 0.05:
                    stars = "*"
                
                annot_matrix.loc[r, c] = f"{corr_val:.2f}{stars}"
    
    with col1:
        st.markdown("**Correlation Matrix (* p<0.05, ** p<0.01)**")
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=annot_matrix, cmap="coolwarm", fmt="", linewidths=0.5, ax=ax_corr, vmin=-1, vmax=1)
        st.pyplot(fig_corr)
        
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
        st.dataframe(p_values.astype(float).style.format("{:.4e}"))

    st.markdown("---")
    
    # Regression Analysis
    st.subheader("4. Multiple Linear Regression")
    
    X = df_selected[predictor_vars]
    Y = df_selected[target_var]
    
    X_with_const = sm.add_constant(X)
    
    try:
        model = sm.OLS(Y, X_with_const).fit()
        
        params = model.params
        r_squared = model.rsquared
        
        eq_parts = []
        if 'const' in params:
            eq_parts.append(f"{params['const']:.4f}")
            
        for name, coef in params.items():
            if name != 'const':
                sign = "+" if coef >= 0 else "-"
                eq_parts.append(f"{sign} {abs(coef):.4f}({name})")
                
        equation_str = f"**Y ({target_var}) = " + " ".join(eq_parts) + f"** *(R² = {r_squared:.4f})*"
        
        st.markdown("### Regression Equation")
        st.markdown(equation_str)
        st.text(model.summary())
        
        summary_text = f"Regression Equation:\nY = {' '.join(eq_parts)} (R-squared = {r_squared:.4f})\n\n{model.summary().as_text()}"
        st.download_button(
            label="Download Regression Summary (TXT)",
            data=summary_text,
            file_name="regression_summary.txt",
            mime="text/plain"
        )
        
        st.markdown("**Residual Diagnostics**")
        fig_resid, ax_resid = plt.subplots(figsize=(8, 4))
        sns.histplot(model.resid, kde=True, ax=ax_resid)
        ax_resid.set_title("Residual Distribution")
        st.pyplot(fig_resid)
        
    except Exception as e:
        st.error(f"Regression failed: {e}. Check for multicollinearity or constant variables.")
