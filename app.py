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

    st.markdown("---")
    
    # Publication-Ready Correlation Table (Target vs Predictors)
    st.subheader("2. Publication-Ready Correlation Table")
    st.markdown("Generates a standard table showing the correlation of weather/abiotic parameters against your dependent variable.")
    
    col_method, col_label = st.columns(2)
    with col_method:
        corr_method = st.selectbox("Select Correlation Method", ["Pearson (Parametric)", "Spearman (Non-Parametric Rank)"])
    with col_label:
        column_label = st.text_input("Data Column Label (e.g., Year, Location, or 'Correlation (r)')", value="2025")

    # --- EDUCATIONAL GUIDE EXPANDER ---
    with st.expander("🤔 Which method should I choose? (Pearson vs. Spearman)"):
        st.markdown("""
        ### 1. Pearson Correlation (The "Straight Line" Test)
        Pearson measures how well your data fits a perfectly **straight line** (a linear relationship).
        * **How it works:** It uses the actual raw, numerical values of your data to calculate the relationship.
        * **The Catch:** It assumes your data is normally distributed (parametric) and moves at a constant rate. It is also highly sensitive to outliers—one weird data point can throw off your entire result.
        * **When to use it:** When you are confident your variables increase or decrease together at a steady, constant pace.

        ### 2. Spearman Correlation (The "Directional" Test)
        Spearman measures whether variables move in the same **direction**, regardless of the exact rate (a monotonic relationship).
        * **How it works:** Instead of using the raw numbers, it ranks your data from lowest to highest (1st, 2nd, 3rd...) and calculates the correlation of those ranks.
        * **The Catch:** Because it uses ranks instead of raw values, it is considered non-parametric. It doesn't care if your data is normally distributed, and it easily ignores extreme outliers.
        * **When to use it:** When your data doesn't follow a perfect straight line but still trends together.

        ---
        ### 🐛 A Practical Example
        Imagine you are tracking a pest population against rising daily temperatures:
        * **Scenario A:** For every 1°C increase, the population increases by exactly 50. This is a perfect **linear** relationship. Both Pearson and Spearman will show a high correlation (near 1.0).
        * **Scenario B:** As the temperature rises, the population explodes exponentially. It goes up by 10, then 50, then 500. The relationship is always going *up*, but the rate of growth is accelerating. **Spearman** will still show a near-perfect 1.0 correlation because the ranks are perfectly aligned (higher temp = higher rank in population). **Pearson**, however, will give a much lower score because the data forms a curve, not a straight line.
        """)

    df_selected = df[[target_var] + predictor_vars].dropna()

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
    st.subheader("4. Multiple Linear Regression (OLS)")
    
    X = df_selected[predictor_vars]
    Y = df_selected[target_var]
    
    X_with_const = sm.add_constant(X)
    
    try:
        model = sm.OLS(Y, X_with_const).fit()

        # Check multicollinearity before trusting the OLS predictive equation.
        vif_df = pd.DataFrame()
        high_vif_df = pd.DataFrame()
        has_multicollinearity = False

        if len(predictor_vars) > 1:
            X_vif = X.loc[:, X.nunique(dropna=True) > 1]
            if X_vif.shape[1] > 1:
                vif_values = []

                for variable in X_vif.columns:
                    try:
                        other_variables = [col for col in X_vif.columns if col != variable]
                        vif_model = sm.OLS(X_vif[variable], sm.add_constant(X_vif[other_variables])).fit()
                        r_squared = vif_model.rsquared
                        vif_value = np.inf if r_squared >= 0.999999 else 1 / (1 - r_squared)
                    except Exception:
                        vif_value = np.inf
                    vif_values.append(vif_value)

                vif_df = pd.DataFrame({
                    "Variable": X_vif.columns,
                    "VIF": vif_values
                })
                high_vif_df = vif_df[vif_df["VIF"] >= 5]
                has_multicollinearity = not high_vif_df.empty
        
        # --- 1. Clean Summary Stats ---
        st.markdown("#### Model Summary Statistics")
        col_r, col_ar, col_f, col_p = st.columns(4)
        col_r.metric("R-squared", f"{model.rsquared:.3f}")
        col_ar.metric("Adj. R-squared", f"{model.rsquared_adj:.3f}")
        col_f.metric("F-statistic", f"{model.fvalue:.3f}")
        col_p.metric("Prob (F-stat)", f"{model.f_pvalue:.4e}")
        
        # --- 2. Clean Coefficients Table ---
        st.markdown("#### Coefficients Table")
        
        # Extract data into a clean dataframe
        coef_df = pd.DataFrame({
            "Coefficient": model.params,
            "Std Error": model.bse,
            "t-value": model.tvalues,
            "P > |t|": model.pvalues
        })
        
        # Add Confidence Intervals
        conf_int = model.conf_int()
        coef_df["[0.025"] = conf_int[0]
        coef_df["0.975]"] = conf_int[1]
        
        # Format the dataframe for display
        def highlight_pvals(val):
            color = 'lightgreen' if val < 0.05 else ''
            return f'background-color: {color}'
            
        # FIXED: Changed 'applymap' to 'map' for newer Pandas versions
        formatted_coef_df = coef_df.style.map(highlight_pvals, subset=['P > |t|']).format({
            "Coefficient": "{:.4f}",
            "Std Error": "{:.4f}",
            "t-value": "{:.3f}",
            "P > |t|": "{:.4f}",
            "[0.025": "{:.4f}",
            "0.975]": "{:.4f}"
        })
        
        st.dataframe(formatted_coef_df, use_container_width=True)
        st.caption("Rows highlighted in green indicate statistically significant variables (P < 0.05).")

        # --- 3. Automated Interpretation Guide ---
        st.markdown("### 🧠 Automated Interpretation Guide")
        
        # Overall Model Health
        if model.f_pvalue < 0.05:
            st.success(f"**Overall Model Health:** Good. The Prob (F-statistic) is {model.f_pvalue:.4f}, which is less than 0.05. This means your independent variables, taken together, reliably predict changes in '{target_var}'.")
        else:
            st.error(f"**Overall Model Health:** Poor. The Prob (F-statistic) is {model.f_pvalue:.4f}, which is greater than 0.05. This model is not statistically significant.")

        st.info(f"**Variance Explained:** Your R-squared is {model.rsquared:.3f}. This means {model.rsquared*100:.1f}% of the variance in '{target_var}' is explained by your selected parameters. (Note: Look at Adj. R-squared ({model.rsquared_adj:.3f}) if you have a small dataset with many variables, as R-squared can be artificially inflated).")

        # Significant Variables
        sig_vars = coef_df[coef_df["P > |t|"] < 0.05].index.tolist()
        if "const" in sig_vars:
            sig_vars.remove("const")
            
        st.markdown("**Specific Parameter Impacts:**")
        if len(sig_vars) > 0:
            for var in sig_vars:
                coef_val = coef_df.loc[var, 'Coefficient']
                direction = "increases" if coef_val > 0 else "decreases"
                st.markdown(f"* **{var}:** Significant (P = {coef_df.loc[var, 'P > |t|']:.4f}). Assuming all other parameters are held constant, for every 1 unit increase in {var}, the {target_var} **{direction}** by {abs(coef_val):.4f} units.")
        else:
            st.markdown("* *None of the individual predictor variables are statistically significant (P < 0.05) in this specific combination.*")

        # Predictive Equation
        eq_parts = []
        if 'const' in model.params:
            eq_parts.append(f"{model.params['const']:.4f}")
        for name, coef in model.params.items():
            if name != 'const':
                sign = "+" if coef >= 0 else "-"
                eq_parts.append(f"{sign} {abs(coef):.4f}({name})")
                
        equation_str = f"**{target_var} = " + " ".join(eq_parts) + "**"
        
        st.markdown("---")
        st.markdown("### 🧮 Predictive Equation")
        st.markdown(equation_str)

        if has_multicollinearity:
            high_vif_names = ", ".join(high_vif_df["Variable"].tolist())
            st.warning(
                f"**Multicollinearity Warning:** This Multiple Linear Regression (OLS) equation has a multicollinearity issue. "
                f"The predictor(s) with high VIF are: **{high_vif_names}**. "
                "This means the OLS equation may be unstable and the individual coefficients/P-values may be misleading. "
                "Please use the **5. Stepwise Regression** section below for the refined predictive equation."
            )

            with st.expander("View Multicollinearity Diagnostics (VIF)"):
                formatted_vif_df = vif_df.style.map(
                    lambda val: 'background-color: #ffcccc' if val >= 5 else '',
                    subset=['VIF']
                ).format({"VIF": "{:.3f}"})
                st.dataframe(formatted_vif_df, use_container_width=True)
                st.caption("VIF >= 5 indicates possible multicollinearity; VIF >= 10 indicates severe multicollinearity.")
        elif len(predictor_vars) > 1:
            st.success("No major multicollinearity issue detected in the selected predictors based on VIF < 5.")
        
        # Expander for Raw Output
        with st.expander("View Raw OLS Summary (Classic statsmodels output)"):
            st.text(model.summary())
            
        summary_text = f"Regression Equation:\n{target_var} = {' '.join(eq_parts)} \n\n{model.summary().as_text()}"
        st.download_button(
            label="Download Regression Summary (TXT)",
            data=summary_text,
            file_name="regression_summary.txt",
            mime="text/plain"
        )
        
        st.markdown("**Residual Diagnostics**")
        fig_resid, ax_resid = plt.subplots(figsize=(8, 4))
        sns.histplot(model.resid, kde=True, ax=ax_resid)
        ax_resid.set_title("Residual Distribution (Should ideally look like a bell curve)")
        st.pyplot(fig_resid)

        # ==========================================
        # 5. STEPWISE REGRESSION (BACKWARD ELIMINATION)
        # ==========================================
        st.markdown("---")
        st.subheader("5. Stepwise Regression (Automated Feature Selection)")

        # --- Educational Explanation for the User ---
        st.info("""
        **What is the Multicollinearity Issue?**
        In agricultural data, plant traits often grow proportionally together (e.g., as the panicle gets longer, the flag leaf also gets longer). When you put these highly correlated traits into a standard regression model, the math gets "confused."
        * **The Result:** The overall model will say "Yes, these traits affect the pest population," but the individual P-values will show that *none* of them are significant. The model cannot figure out which specific trait deserves the credit.

        **Why use Stepwise Regression?**
        To fix this overlap, we use **Stepwise Regression (Backward Elimination)**.
        Instead of guessing which variables to keep, this algorithm acts as an automatic filter. It starts with all your variables and removes the least significant one (highest P-value). It recalculates the model and repeats this process until only the strictly significant, independent drivers (P < 0.05) remain.
        """)

        # Backward Elimination Algorithm
        def backward_elimination(data, target, significance_level=0.05):
            features = data.columns.tolist()
            elimination_history = []

            while len(features) > 0:
                features_with_constant = sm.add_constant(data[features])
                temp_model = sm.OLS(target, features_with_constant).fit()
                p_values = temp_model.pvalues[1:]  # Exclude the constant's p-value
                max_p_value = p_values.max()

                if max_p_value >= significance_level:
                    excluded_feature = p_values.idxmax()
                    features.remove(excluded_feature)
                    elimination_history.append((excluded_feature, max_p_value))
                else:
                    break

            return features, elimination_history

        # Run the algorithm
        final_features, history = backward_elimination(X, Y, 0.05)

        # Display what was removed
        if history:
            st.markdown("**Variables Automatically Removed to fix Multicollinearity:**")
            for var, p_val in history:
                st.write(f"- Dropped `{var}` (P-value: {p_val:.4f})")
        else:
            st.success("No variables were dropped. All initial predictors are statistically significant!")

        # Fit the Final Stepwise Model
        if len(final_features) > 0:
            st.markdown(f"#### Final Refined Model for predicting '{target_var}'")

            X_stepwise = X[final_features]
            X_stepwise_const = sm.add_constant(X_stepwise)
            stepwise_model = sm.OLS(Y, X_stepwise_const).fit()

            # Clean Stats for Stepwise Model
            col_rs, col_ars, col_fs, col_ps = st.columns(4)
            col_rs.metric("Final R-squared", f"{stepwise_model.rsquared:.3f}")
            col_ars.metric("Final Adj. R-squared", f"{stepwise_model.rsquared_adj:.3f}")
            col_fs.metric("F-statistic", f"{stepwise_model.fvalue:.3f}")
            col_ps.metric("Prob (F-stat)", f"{stepwise_model.f_pvalue:.4e}")

            # Coefficients Table
            step_coef_df = pd.DataFrame({
                "Coefficient": stepwise_model.params,
                "Std Error": stepwise_model.bse,
                "t-value": stepwise_model.tvalues,
                "P > |t|": stepwise_model.pvalues
            })

            formatted_step_coef_df = step_coef_df.style.map(highlight_pvals, subset=['P > |t|']).format({
                "Coefficient": "{:.4f}",
                "Std Error": "{:.4f}",
                "t-value": "{:.3f}",
                "P > |t|": "{:.4f}"
            })

            st.dataframe(formatted_step_coef_df, use_container_width=True)

            # Generate Final Equation
            step_eq_parts = []
            if 'const' in stepwise_model.params:
                step_eq_parts.append(f"{stepwise_model.params['const']:.4f}")
            for name, coef in stepwise_model.params.items():
                if name != 'const':
                    sign = "+" if coef >= 0 else "-"
                    step_eq_parts.append(f"{sign} {abs(coef):.4f}({name})")

            st.success(f"**Final Predictive Equation (Cleaned of Multicollinearity):**\n\n**{target_var} = " + " ".join(step_eq_parts) + "**")

        else:
            st.warning("After backward elimination, no variables were found to be statistically significant at the P < 0.05 level.")
        
    except Exception as e:
        st.error(f"Regression failed: {e}. Check for multicollinearity or constant variables.")
