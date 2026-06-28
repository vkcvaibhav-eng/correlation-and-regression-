import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
from scipy.stats import pearsonr, shapiro, spearmanr
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson


st.set_page_config(page_title="Statistical Analysis App", layout="wide")


REQUIRED_MULTI_YEAR_COLUMNS = [
    "Year",
    "SMW",
    "Pest_Population",
    "Tmax",
    "Tmin",
    "RH_Max",
    "RH_Min",
    "RH_Avg",
    "Wind_kmph",
    "BSSH_hr",
    "Rainfall_mm",
]

WEATHER_COLUMN_KEYWORDS = [
    "tmax",
    "tmin",
    "temp",
    "temperature",
    "rh",
    "humidity",
    "wind",
    "ws",
    "bssh",
    "sunshine",
    "rain",
    "rainfall",
    "rf",
    "vp",
    "vapour",
    "vapor",
    "evap",
]


def build_single_year_sample_data():
    return pd.DataFrame({
        "Observation_Week": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Pest_Population": [12, 15, 22, 28, 35, 40, 38, 25, 18, 10],
        "Max_Temperature_C": [30, 32, 35, 36, 38, 37, 34, 31, 29, 28],
        "Min_Temperature_C": [15, 18, 22, 24, 26, 25, 23, 19, 16, 14],
        "Relative_Humidity_pct": [40, 45, 50, 55, 60, 65, 70, 60, 50, 45],
        "Rainfall_mm": [0, 0, 5, 10, 50, 100, 120, 40, 10, 0],
    })


def build_multi_year_template_data():
    rows = []
    base_values = {
        2022: [0, 2, 5, 10, 18, 30, 42, 35, 22, 12, 4],
        2023: [1, 3, 8, 16, 28, 44, 39, 27, 15, 7, 2],
        2024: [0, 1, 6, 12, 24, 36, 48, 40, 25, 11, 3],
    }
    smw_values = list(range(30, 41))

    for year, populations in base_values.items():
        year_shift = year - 2022
        for index, smw in enumerate(smw_values):
            rows.append({
                "Year": year,
                "SMW": smw,
                "Pest_Population": populations[index],
                "Tmax": 31.2 + index * 0.45 + year_shift * 0.20,
                "Tmin": 22.0 + index * 0.25 + year_shift * 0.15,
                "RH_Max": 78 - index * 0.80 + year_shift,
                "RH_Min": 48 + index * 0.55 - year_shift * 0.20,
                "RH_Avg": 63 + index * 0.10 + year_shift * 0.20,
                "Wind_kmph": 4.2 + (index % 4) * 0.35 + year_shift * 0.10,
                "BSSH_hr": 6.5 + (index % 5) * 0.25 - year_shift * 0.05,
                "Rainfall_mm": [12, 8, 0, 18, 32, 46, 74, 52, 28, 10, 4][index] + year_shift * 3,
            })

    return pd.DataFrame(rows)


def read_uploaded_dataset(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def coerce_numeric_columns(df, columns):
    df_copy = df.copy()
    for column in columns:
        df_copy[column] = pd.to_numeric(df_copy[column], errors="coerce")
    return df_copy


def significance_stars(p_value):
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def safe_corr(x_values, y_values, method):
    corr_df = pd.DataFrame({"x": x_values, "y": y_values}).dropna()
    if len(corr_df) < 3:
        return np.nan, np.nan
    if corr_df["x"].nunique(dropna=True) < 2 or corr_df["y"].nunique(dropna=True) < 2:
        return np.nan, np.nan

    try:
        if method == "Pearson":
            r_value, p_value = pearsonr(corr_df["x"], corr_df["y"])
        else:
            r_value, p_value = spearmanr(corr_df["x"], corr_df["y"])
        return r_value, p_value
    except Exception:
        return np.nan, np.nan


def format_corr_value(r_value, p_value):
    if pd.isna(r_value):
        return "NA"
    return f"{r_value:.2f}{significance_stars(p_value)}"


def format_number(value, digits=3):
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def style_with_pvalue_highlight(styler, subset):
    def highlight_pvals(val):
        color = "lightgreen" if pd.notna(val) and val < 0.05 else ""
        return f"background-color: {color}"

    if hasattr(styler, "map"):
        return styler.map(highlight_pvals, subset=subset)
    return styler.applymap(highlight_pvals, subset=subset)


def make_readable_model_term(term):
    term_text = str(term)
    if term_text == "Intercept":
        return "const"
    if term_text.startswith("Q(\"") and term_text.endswith("\")"):
        return term_text[3:-2]
    if term_text.startswith("C(Year)[T.") and term_text.endswith("]"):
        year_value = term_text.replace("C(Year)[T.", "").rstrip("]")
        if year_value.endswith(".0"):
            year_value = year_value[:-2]
        return f"Year_{year_value}_adjustment"
    return term_text


def build_regression_equation(model, dependent_var, exclude_terms=None):
    exclude_terms = set(exclude_terms or [])
    eq_parts = []

    for term, coef in model.params.items():
        readable_term = make_readable_model_term(term)
        if term in exclude_terms or readable_term in exclude_terms:
            continue
        if pd.isna(coef) or not np.isfinite(coef):
            continue

        if readable_term in ["const", "Intercept"]:
            eq_parts.insert(0, f"{coef:.4f}")
            continue

        sign = "+" if coef >= 0 else "-"
        eq_parts.append(f"{sign} {abs(coef):.4f}({readable_term})")

    if not eq_parts:
        return f"{dependent_var} = NA"
    return f"{dependent_var} = " + " ".join(eq_parts)


def build_equation_from_model(model, target_var):
    return build_regression_equation(model, target_var)


def get_reference_year_from_model(model, model_df):
    if model is None or model_df is None or "Year" not in model_df.columns:
        return "automatically selected by statsmodels"

    all_years = sorted(pd.to_numeric(model_df["Year"], errors="coerce").dropna().unique())
    adjusted_years = []
    for term in model.params.index:
        term_text = str(term)
        if term_text.startswith("C(Year)[T.") and term_text.endswith("]"):
            year_value = term_text.replace("C(Year)[T.", "").rstrip("]")
            adjusted_year = pd.to_numeric(pd.Series([year_value]), errors="coerce").iloc[0]
            if pd.notna(adjusted_year):
                adjusted_years.append(adjusted_year)

    reference_years = [year for year in all_years if year not in adjusted_years]
    if reference_years:
        reference_year = reference_years[0]
        return int(reference_year) if float(reference_year).is_integer() else reference_year
    return "automatically selected by statsmodels"


def build_model_equation_download_text(model, model_formula, equation_text, interpretation, model_label):
    lines = [
        model_label,
        "",
        "Model formula:",
        model_formula or "NA",
        "",
        "Human-readable equation:",
        equation_text or "NA",
        "",
        "Short interpretation:",
        interpretation or "NA",
    ]

    if model is not None and hasattr(model, "rsquared"):
        lines.extend([
            "",
            f"R-squared: {format_number(model.rsquared, 3)}",
            f"Adjusted R-squared: {format_number(model.rsquared_adj, 3)}",
        ])
    if model is not None and hasattr(model, "aic"):
        lines.append(f"AIC: {format_number(model.aic, 3)}")
    if model is not None and hasattr(model, "bic"):
        lines.append(f"BIC: {format_number(model.bic, 3)}")

    return "\n".join(lines)


def render_model_equation_section(
    model,
    dependent_var,
    model_formula=None,
    model_label="Model Equation",
    interpretation="",
    key_prefix="model_equation",
    model_df=None,
    expander=False,
):
    equation_text = build_regression_equation(model, dependent_var)

    def render_content():
        if model_formula:
            st.markdown("**Model formula:**")
            st.code(model_formula, language="text")
        st.markdown("#### Human-readable Model Equation")
        st.code(equation_text, language="text")
        if model_df is not None and "C(Year)" in (model_formula or ""):
            reference_year = get_reference_year_from_model(model, model_df)
            st.info(
                f"In this model, the base/reference year is {reference_year}. Coefficients of C(Year) represent increase or decrease in pest population compared with the base year after adjusting for weather variables."
            )
        if interpretation:
            st.info(interpretation)
        download_text = build_model_equation_download_text(
            model,
            model_formula,
            equation_text,
            interpretation,
            model_label,
        )
        st.download_button(
            label="Download Model Equation",
            data=text_bytes(download_text),
            file_name=f"{key_prefix}_model_equation.txt",
            mime="text/plain",
            key=f"{key_prefix}_download_model_equation",
        )

    if expander:
        with st.expander("View full model equation"):
            render_content()
    else:
        render_content()

    return equation_text


def render_mixed_model_equation_section(model, dependent_var, model_formula, key_prefix):
    fixed_effect_equation = build_regression_equation(
        model,
        dependent_var,
        exclude_terms=["Group Var"],
    )
    detailed_equation = f"{fixed_effect_equation} + u(Year) + error"
    interpretation = (
        "This equation includes both fixed weather effects and random year effect, which is more suitable for multi-year journal-level inference."
    )

    st.markdown("#### Human-readable Model Equation")
    st.code(f"{dependent_var} = fixed weather effects + random year effect + error", language="text")
    st.code(detailed_equation, language="text")
    st.info(
        "In the mixed model, weather parameters are treated as fixed effects and Year is treated as a random effect. Therefore, the equation includes an additional random year component. u(Year) = random year-to-year variation."
    )

    try:
        random_rows = []
        for year, values in model.random_effects.items():
            if hasattr(values, "iloc"):
                random_value = values.iloc[0]
            elif isinstance(values, (list, tuple, np.ndarray)):
                random_value = values[0]
            else:
                random_value = float(values)
            random_rows.append({
                "Year": year,
                "Random intercept": random_value,
            })
        if random_rows:
            random_effects_df = pd.DataFrame(random_rows)
            st.dataframe(random_effects_df, use_container_width=True, hide_index=True)
    except Exception:
        pass

    download_text = build_model_equation_download_text(
        model,
        model_formula,
        detailed_equation,
        interpretation,
        "Linear Mixed Model Equation",
    )
    st.download_button(
        label="Download Model Equation",
        data=text_bytes(download_text),
        file_name=f"{key_prefix}_model_equation.txt",
        mime="text/plain",
        key=f"{key_prefix}_download_model_equation",
    )

    return detailed_equation


def quote_formula_column(column_name):
    safe_name = str(column_name).replace("\\", "\\\\").replace('"', '\\"')
    return f'Q("{safe_name}")'


def build_formula(dependent_var, weather_vars, include_year_effect=False):
    predictors = [quote_formula_column(var) for var in weather_vars]
    if include_year_effect:
        predictors.append("C(Year)")
    return f"{quote_formula_column(dependent_var)} ~ " + " + ".join(predictors)


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def text_bytes(text):
    return text.encode("utf-8")


def prepare_numeric_analysis_df(df, required_columns=None):
    required_columns = required_columns or []
    numeric_df = df.copy()
    numeric_cols = []

    for column in numeric_df.columns:
        converted = pd.to_numeric(numeric_df[column], errors="coerce")
        if column in required_columns or converted.notna().sum() > 0:
            numeric_df[column] = converted
            numeric_cols.append(column)

    return numeric_df, numeric_cols


def get_default_multi_year_variables(df, key_prefix):
    numeric_df, numeric_cols = prepare_numeric_analysis_df(df, required_columns=["Year", "SMW"])
    dependent_options = [col for col in numeric_cols if col not in ["Year", "SMW"] and numeric_df[col].notna().sum() > 0]

    if not dependent_options:
        return numeric_df, None, []

    default_dep_index = dependent_options.index("Pest_Population") if "Pest_Population" in dependent_options else 0
    dependent_var = st.selectbox(
        "Select dependent variable (pest population)",
        options=dependent_options,
        index=default_dep_index,
        key=f"{key_prefix}_dependent_var",
    )

    weather_options = [
        col for col in dependent_options
        if col != dependent_var and numeric_df[col].nunique(dropna=True) > 1
    ]

    default_weather = [
        col for col in weather_options
        if any(keyword in str(col).strip().lower() for keyword in WEATHER_COLUMN_KEYWORDS)
    ]

    weather_vars = st.multiselect(
        "Select independent weather variables",
        options=weather_options,
        default=default_weather,
        key=f"{key_prefix}_weather_vars",
    )

    return numeric_df, dependent_var, weather_vars


def remove_constant_predictors(reg_df, predictors):
    active_predictors = [
        predictor for predictor in predictors
        if predictor in reg_df.columns and reg_df[predictor].nunique(dropna=True) > 1
    ]
    removed_predictors = [predictor for predictor in predictors if predictor not in active_predictors]
    return active_predictors, removed_predictors


def get_significant_predictors(pvalues, alpha=0.05):
    significant = []
    for name, p_value in pvalues.items():
        name_text = str(name)
        if name_text == "const" or name_text.startswith("C(Year)"):
            continue
        if pd.notna(p_value) and p_value < alpha:
            significant.append(name_text)
    return significant


def build_coefficient_table(model):
    return pd.DataFrame({
        "Term": model.params.index,
        "Coefficient": model.params.values,
        "Std Error": model.bse.values,
        "t-value": model.tvalues.values,
        "P > |t|": model.pvalues.values,
    })


def format_regression_summary_df(summary_df):
    formatted_df = summary_df.copy()
    for column in ["R-squared", "Adjusted R-squared", "F-statistic"]:
        if column in formatted_df.columns:
            formatted_df[column] = formatted_df[column].apply(
                lambda value: format_number(value, 3) if pd.to_numeric(pd.Series([value]), errors="coerce").notna().iloc[0] else value
            )
    if "Prob(F)" in formatted_df.columns:
        formatted_df["Prob(F)"] = formatted_df["Prob(F)"].apply(
            lambda value: f"{float(value):.4e}" if pd.to_numeric(pd.Series([value]), errors="coerce").notna().iloc[0] else value
        )
    return formatted_df


def fit_pooled_year_ols(df, dependent_var, weather_vars):
    pooled_df = coerce_numeric_columns(df, ["Year", dependent_var] + weather_vars)
    pooled_df = pooled_df[["Year", dependent_var] + weather_vars].dropna()
    active_vars, removed_vars = remove_constant_predictors(pooled_df, weather_vars)

    if not active_vars:
        return None, None, pooled_df, None, "No non-constant weather predictors were available for the pooled model.", removed_vars

    if len(pooled_df) < len(active_vars) + pooled_df["Year"].nunique() + 2:
        warning = "Pooled regression may be unstable because complete observations are limited for the selected predictors and year effects."
    else:
        warning = ""

    try:
        formula = build_formula(dependent_var, active_vars, include_year_effect=True)
        pooled_model = smf.ols(formula=formula, data=pooled_df).fit()
        display_formula = f"{dependent_var} ~ " + " + ".join(active_vars) + " + C(Year)"
        return pooled_model, display_formula, pooled_df, build_coefficient_table(pooled_model), warning, removed_vars
    except Exception as exc:
        return None, None, pooled_df, None, f"Pooled regression failed: {exc}", removed_vars


def create_lagged_weather_df(df, dependent_var, weather_vars):
    lagged_df = coerce_numeric_columns(df, ["Year", "SMW", dependent_var] + weather_vars)
    lagged_df = lagged_df.sort_values(["Year", "SMW"]).copy()
    lagged_columns = {"Same-week variables only": [], "Lag 1 variables": [], "Lag 2 variables": []}

    for weather_var in weather_vars:
        same_col = f"{weather_var}_same"
        lag1_col = f"{weather_var}_lag1"
        lag2_col = f"{weather_var}_lag2"
        lagged_df[same_col] = lagged_df[weather_var]
        lagged_df[lag1_col] = lagged_df.groupby("Year", group_keys=False)[weather_var].shift(1)
        lagged_df[lag2_col] = lagged_df.groupby("Year", group_keys=False)[weather_var].shift(2)
        lagged_columns["Same-week variables only"].append(same_col)
        lagged_columns["Lag 1 variables"].append(lag1_col)
        lagged_columns["Lag 2 variables"].append(lag2_col)

    return lagged_df, lagged_columns


def get_best_lag_predictors(lag_numeric_df):
    best_predictors = []
    label_to_suffix = {
        "Same week": "_same",
        "1-week lag": "_lag1",
        "2-week lag": "_lag2",
    }

    if lag_numeric_df.empty:
        return best_predictors

    for _, row in lag_numeric_df.iterrows():
        best_lag = row.get("Best lag", "NA")
        suffix = label_to_suffix.get(best_lag)
        if suffix:
            best_predictors.append(f"{row['Parameter']}{suffix}")
    return best_predictors


def fit_lag_based_ols(df, dependent_var, weather_vars, lag_mode, lag_numeric_df=None):
    lagged_df, lagged_columns = create_lagged_weather_df(df, dependent_var, weather_vars)
    if lag_mode == "Best lag variables from lag correlation table":
        selected_predictors = get_best_lag_predictors(lag_numeric_df if lag_numeric_df is not None else pd.DataFrame())
    else:
        selected_predictors = lagged_columns.get(lag_mode, [])

    selected_predictors = [predictor for predictor in selected_predictors if predictor in lagged_df.columns]
    model_df = lagged_df[["Year", dependent_var] + selected_predictors].dropna()
    active_predictors, removed_predictors = remove_constant_predictors(model_df, selected_predictors)

    if not active_predictors:
        return None, None, model_df, None, "No non-constant lagged predictors were available for the lag-based model.", removed_predictors

    if len(model_df) < len(active_predictors) + model_df["Year"].nunique() + 2:
        return None, None, model_df, None, "Lag-based model could not be fitted because complete observations were fewer than predictors + year effects + 2.", removed_predictors

    try:
        formula = build_formula(dependent_var, active_predictors, include_year_effect=True)
        lag_model = smf.ols(formula=formula, data=model_df).fit()
        display_formula = f"{dependent_var} ~ " + " + ".join(active_predictors) + " + C(Year)"
        return lag_model, display_formula, model_df, build_coefficient_table(lag_model), "", removed_predictors
    except Exception as exc:
        return None, None, model_df, None, f"Lag-based model failed: {exc}", removed_predictors


def render_sidebar():
    with st.sidebar:
        st.header("📋 Data Upload Guide")
        st.markdown("""
        **Format your CSV/Excel file correctly:**
        * **Rows** should represent observations (e.g., Weeks, Months, SMW, Months, or specific dates).
        * **Columns** should represent variables (e.g., Population count, Max Temp, Rainfall).
        * For multi-year analysis, use long-format data with one row per Year-SMW observation.
        * Do not include special characters or spaces in column headers.
        """)

        sample_data = build_single_year_sample_data()
        st.download_button(
            label="📥 Download Sample CSV",
            data=sample_data.to_csv(index=False),
            file_name="sample_agri_data.csv",
            mime="text/csv",
            help="Download a sample single-year dataset showing pest population vs weather parameters.",
        )

        multi_year_template = build_multi_year_template_data()
        st.download_button(
            label="📥 Download Multi-Year Population Dynamics Template",
            data=multi_year_template.to_csv(index=False),
            file_name="multi_year_population_dynamics_template.csv",
            mime="text/csv",
            help="Download a long-format multi-year template with Year, SMW, pest population, and weather parameters.",
        )


def render_single_year_analysis(df):
    st.subheader("1. Variable Selection")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Single-year analysis requires at least two numeric columns.")
        return

    target_var = st.selectbox(
        "Select Dependent Variable (Y) [e.g., Population]",
        options=numeric_cols,
        key="single_target_var",
    )

    indep_options = [col for col in numeric_cols if col != target_var]
    predictor_vars = st.multiselect(
        "Select Independent Variables (X) [e.g., Weather Parameters]",
        options=indep_options,
        default=indep_options,
        key="single_predictor_vars",
    )

    if not predictor_vars:
        st.warning("Select at least one independent variable.")
        return

    st.markdown("---")

    st.subheader("2. Publication-Ready Correlation Table")
    st.markdown("Generates a standard table showing the correlation of weather/abiotic parameters against your dependent variable.")
    st.warning("Correlation indicates association only and should not be interpreted as direct causation.")

    col_method, col_label = st.columns(2)
    with col_method:
        corr_method = st.selectbox(
            "Select Correlation Method",
            ["Pearson (Parametric)", "Spearman (Non-Parametric Rank)"],
            key="single_corr_method",
        )
    with col_label:
        column_label = st.text_input(
            "Data Column Label (e.g., Year, Location, or 'Correlation (r)')",
            value="2025",
            key="single_column_label",
        )

    with st.expander("🤔 Which method should I choose? (Pearson vs. Spearman)"):
        st.markdown("""
        ### 1. Pearson Correlation (The "Straight Line" Test)
        Pearson measures how well your data fits a perfectly **straight line** (a linear relationship).
        * **How it works:** It uses the actual raw, numerical values of your data to calculate the relationship.
        * **The Catch:** It assumes your data is normally distributed (parametric) and moves at a constant rate. It is also highly sensitive to outliers-one unusual data point can throw off your entire result.
        * **When to use it:** When you are confident your variables increase or decrease together at a steady, constant pace.

        ### 2. Spearman Correlation (The "Directional" Test)
        Spearman measures whether variables move in the same **direction**, regardless of the exact rate (a monotonic relationship).
        * **How it works:** Instead of using the raw numbers, it ranks your data from lowest to highest (1st, 2nd, 3rd...) and calculates the correlation of those ranks.
        * **The Catch:** Because it uses ranks instead of raw values, it is considered non-parametric. It does not require normally distributed data and is less affected by extreme outliers.
        * **When to use it:** When your data does not follow a perfect straight line but still trends together.

        ---
        ### 🐛 A Practical Example
        Imagine you are tracking a pest population against rising daily temperatures:
        * **Scenario A:** For every 1°C increase, the population increases by exactly 50. This is a perfect **linear** relationship. Both Pearson and Spearman will show a high correlation (near 1.0).
        * **Scenario B:** As the temperature rises, the population increases slowly at first and then rapidly. The relationship is always going up, but the rate of growth is accelerating. **Spearman** will still show a high correlation because the ranks are aligned, while **Pearson** may be lower because the data forms a curve.
        """)

    df_selected = df[[target_var] + predictor_vars].dropna()
    if len(df_selected) < 3:
        st.warning("Correlation and regression require at least three complete observations.")
        return

    table_results = []
    corr_method_short = "Pearson" if "Pearson" in corr_method else "Spearman"
    for var in predictor_vars:
        r_value, p_value = safe_corr(df_selected[var], df_selected[target_var], corr_method_short)
        table_results.append({
            "Parameters": var,
            column_label: format_corr_value(r_value, p_value),
        })

    agri_table_df = pd.DataFrame(table_results)

    st.markdown(f"**Population correlation with the weather parameter ({corr_method.split()[0]})**")
    st.table(agri_table_df.set_index("Parameters"))
    st.caption("*Significant at P < 0.05, **Significant at P < 0.01")

    st.download_button(
        label="Download Table as CSV",
        data=agri_table_df.to_csv(index=False),
        file_name="population_weather_correlation.csv",
        mime="text/csv",
        key="single_corr_download",
    )

    st.markdown("---")

    st.subheader("3. Cross-Correlation Analysis & Heatmap")

    st.markdown("**Heatmap Editing Options**")
    cmap_options = {
        "Coolwarm (Default)": "coolwarm",
        "Viridis": "viridis",
        "Magma": "magma",
        "YlGnBu": "YlGnBu",
        "RdYlGn": "RdYlGn",
        "Spectral": "Spectral",
        "Blues": "Blues",
        "Greens": "Greens",
    }
    heatmap_style_col, heatmap_font_col = st.columns(2)
    with heatmap_style_col:
        selected_cmap_label = st.selectbox(
            "Select Heatmap Colour Palette",
            options=list(cmap_options.keys()),
            key="single_heatmap_cmap",
        )
    with heatmap_font_col:
        heatmap_font_size = st.slider(
            "Select Heatmap Font Size",
            min_value=8,
            max_value=20,
            value=10,
            step=1,
            key="single_heatmap_font_size",
        )

    selected_cmap = cmap_options[selected_cmap_label]
    col1, col2 = st.columns([1, 1])

    if corr_method_short == "Pearson":
        corr_matrix = df_selected.corr(method="pearson")
    else:
        corr_matrix = df_selected.corr(method="spearman")

    p_values = pd.DataFrame(index=df_selected.columns, columns=df_selected.columns)
    annot_matrix = pd.DataFrame(index=df_selected.columns, columns=df_selected.columns)

    for r_name in df_selected.columns:
        for c_name in df_selected.columns:
            corr_val = corr_matrix.loc[r_name, c_name]
            if r_name == c_name:
                p_values.loc[r_name, c_name] = 0.0
                annot_matrix.loc[r_name, c_name] = f"{corr_val:.2f}"
            else:
                _, p_value = safe_corr(df_selected[r_name], df_selected[c_name], corr_method_short)
                p_values.loc[r_name, c_name] = p_value
                annot_matrix.loc[r_name, c_name] = format_corr_value(corr_val, p_value)

    with col1:
        st.markdown("**Correlation Matrix (* p<0.05, ** p<0.01)**")
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=annot_matrix,
            cmap=selected_cmap,
            fmt="",
            linewidths=0.5,
            ax=ax_corr,
            vmin=-1,
            vmax=1,
            annot_kws={"size": heatmap_font_size, "fontfamily": "Times New Roman"},
        )
        ax_corr.tick_params(axis="x", labelsize=heatmap_font_size)
        ax_corr.tick_params(axis="y", labelsize=heatmap_font_size)
        plt.setp(ax_corr.get_xticklabels(), rotation=45, ha="right", fontfamily="Times New Roman")
        plt.setp(ax_corr.get_yticklabels(), rotation=0, fontfamily="Times New Roman")
        if ax_corr.collections and ax_corr.collections[0].colorbar:
            ax_corr.collections[0].colorbar.ax.tick_params(labelsize=heatmap_font_size)
            plt.setp(ax_corr.collections[0].colorbar.ax.get_yticklabels(), fontfamily="Times New Roman")
        st.pyplot(fig_corr)

        buf = io.BytesIO()
        fig_corr.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        st.download_button(
            label="Download Heatmap (PNG)",
            data=buf.getvalue(),
            file_name="correlation_heatmap.png",
            mime="image/png",
            key="single_heatmap_download",
        )

    with col2:
        st.markdown("**Statistical Significance (p-values)**")
        st.dataframe(p_values.astype(float).style.format("{:.4e}"))

    st.markdown("---")

    st.subheader("4. Multiple Linear Regression (OLS)")

    X = df_selected[predictor_vars]
    Y = df_selected[target_var]

    X_with_const = sm.add_constant(X, has_constant="add")

    try:
        model = sm.OLS(Y, X_with_const).fit()

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
                        vif_model = sm.OLS(X_vif[variable], sm.add_constant(X_vif[other_variables], has_constant="add")).fit()
                        r_squared = vif_model.rsquared
                        vif_value = np.inf if r_squared >= 0.999999 else 1 / (1 - r_squared)
                    except Exception:
                        vif_value = np.inf
                    vif_values.append(vif_value)

                vif_df = pd.DataFrame({
                    "Variable": X_vif.columns,
                    "VIF": vif_values,
                })
                high_vif_df = vif_df[vif_df["VIF"] >= 5]
                has_multicollinearity = not high_vif_df.empty

        st.markdown("#### Model Summary Statistics")
        col_r, col_ar, col_f, col_p = st.columns(4)
        col_r.metric("R-squared", f"{model.rsquared:.3f}")
        col_ar.metric("Adj. R-squared", f"{model.rsquared_adj:.3f}")
        col_f.metric("F-statistic", f"{model.fvalue:.3f}")
        col_p.metric("Prob (F-stat)", f"{model.f_pvalue:.4e}")

        st.markdown("#### Coefficients Table")

        coef_df = pd.DataFrame({
            "Coefficient": model.params,
            "Std Error": model.bse,
            "t-value": model.tvalues,
            "P > |t|": model.pvalues,
        })

        conf_int = model.conf_int()
        coef_df["[0.025"] = conf_int[0]
        coef_df["0.975]"] = conf_int[1]

        formatted_coef_df = style_with_pvalue_highlight(
            coef_df.style,
            subset=["P > |t|"],
        ).format({
            "Coefficient": "{:.4f}",
            "Std Error": "{:.4f}",
            "t-value": "{:.3f}",
            "P > |t|": "{:.4f}",
            "[0.025": "{:.4f}",
            "0.975]": "{:.4f}",
        })

        st.dataframe(formatted_coef_df, use_container_width=True)
        st.caption("Rows highlighted in green indicate statistically significant variables (P < 0.05).")

        st.markdown("### 🧠 Automated Interpretation Guide")

        if model.f_pvalue < 0.05:
            st.success(f"**Overall Model Health:** Good. The Prob (F-statistic) is {model.f_pvalue:.4f}, which is less than 0.05. This means your independent variables, taken together, reliably predict changes in '{target_var}'.")
        else:
            st.error(f"**Overall Model Health:** Poor. The Prob (F-statistic) is {model.f_pvalue:.4f}, which is greater than 0.05. This model is not statistically significant.")

        st.info(f"**Variance Explained:** Your R-squared is {model.rsquared:.3f}. This means {model.rsquared * 100:.1f}% of the variance in '{target_var}' is explained by your selected parameters. Look at Adj. R-squared ({model.rsquared_adj:.3f}) when the dataset is small or many predictors are used.")

        sig_vars = coef_df[coef_df["P > |t|"] < 0.05].index.tolist()
        if "const" in sig_vars:
            sig_vars.remove("const")

        st.markdown("**Specific Parameter Impacts:**")
        if len(sig_vars) > 0:
            for var in sig_vars:
                coef_val = coef_df.loc[var, "Coefficient"]
                direction = "increases" if coef_val > 0 else "decreases"
                st.markdown(f"* **{var}:** Significant (P = {coef_df.loc[var, 'P > |t|']:.4f}). Assuming all other parameters are held constant, for every 1 unit increase in {var}, the {target_var} **{direction}** by {abs(coef_val):.4f} units.")
        else:
            st.markdown("* *None of the individual predictor variables are statistically significant (P < 0.05) in this specific combination.*")

        equation_str = build_equation_from_model(model, target_var)

        st.markdown("---")
        st.markdown("### 🧮 Predictive Equation")
        st.markdown(f"**{equation_str}**")

        if has_multicollinearity:
            high_vif_names = ", ".join(high_vif_df["Variable"].tolist())
            st.warning(
                f"**Multicollinearity Warning:** This Multiple Linear Regression (OLS) equation has a multicollinearity issue. "
                f"The predictor(s) with high VIF are: **{high_vif_names}**. "
                "This means the OLS equation may be unstable and the individual coefficients/P-values may be misleading. "
                "Please use the **5. Stepwise Regression** section below for the refined predictive equation."
            )

            with st.expander("View Multicollinearity Diagnostics (VIF)"):
                formatted_vif_df = style_with_pvalue_highlight(
                    vif_df.style,
                    subset=["VIF"],
                ).format({"VIF": "{:.3f}"})
                st.dataframe(formatted_vif_df, use_container_width=True)
                st.caption("VIF >= 5 indicates possible multicollinearity; VIF >= 10 indicates severe multicollinearity.")
        elif len(predictor_vars) > 1:
            st.success("No major multicollinearity issue detected in the selected predictors based on VIF < 5.")

        with st.expander("View Raw OLS Summary (Classic statsmodels output)"):
            st.text(model.summary())

        summary_text = f"Regression Equation:\n{equation_str}\n\n{model.summary().as_text()}"
        st.download_button(
            label="Download Regression Summary (TXT)",
            data=summary_text,
            file_name="regression_summary.txt",
            mime="text/plain",
            key="single_regression_download",
        )

        st.markdown("**Residual Diagnostics**")
        fig_resid, ax_resid = plt.subplots(figsize=(8, 4))
        sns.histplot(model.resid, kde=True, ax=ax_resid)
        ax_resid.set_title("Residual Distribution (Should ideally look like a bell curve)")
        st.pyplot(fig_resid)

        st.markdown("---")
        st.subheader("5. Stepwise Regression (Automated Feature Selection)")
        st.warning("Stepwise regression is useful for variable selection but may be unstable in small datasets. Biological relevance should also be considered.")

        st.info("""
        **What is the Multicollinearity Issue?**
        In agricultural data, plant traits often grow proportionally together (e.g., as the panicle gets longer, the flag leaf also gets longer). When you put these highly correlated traits into a standard regression model, the math gets "confused."
        * **The Result:** The overall model will say "Yes, these traits affect the pest population," but the individual P-values may show that none of them are significant. The model cannot figure out which specific trait deserves the credit.

        **Why use Stepwise Regression?**
        To fix this overlap, we use **Stepwise Regression (Backward Elimination)**.
        It starts with all selected variables and removes the least significant one (highest P-value). It recalculates the model and repeats this process until only the statistically significant independent drivers (P < 0.05) remain.
        """)

        def backward_elimination(data, target, significance_level=0.05):
            features = data.columns.tolist()
            elimination_history = []

            while len(features) > 0:
                features_with_constant = sm.add_constant(data[features], has_constant="add")
                temp_model = sm.OLS(target, features_with_constant).fit()
                p_values = temp_model.pvalues.drop(labels=["const"], errors="ignore")
                if p_values.empty:
                    break
                max_p_value = p_values.max()

                if max_p_value >= significance_level:
                    excluded_feature = p_values.idxmax()
                    features.remove(excluded_feature)
                    elimination_history.append((excluded_feature, max_p_value))
                else:
                    break

            return features, elimination_history

        final_features, history = backward_elimination(X, Y, 0.05)

        if history:
            st.markdown("**Variables Automatically Removed to fix Multicollinearity:**")
            for var, p_val in history:
                st.write(f"- Dropped `{var}` (P-value: {p_val:.4f})")
        else:
            st.success("No variables were dropped. All initial predictors are statistically significant!")

        if len(final_features) > 0:
            st.markdown(f"#### Final Refined Model for predicting '{target_var}'")

            X_stepwise = X[final_features]
            X_stepwise_const = sm.add_constant(X_stepwise, has_constant="add")
            stepwise_model = sm.OLS(Y, X_stepwise_const).fit()

            col_rs, col_ars, col_fs, col_ps = st.columns(4)
            col_rs.metric("Final R-squared", f"{stepwise_model.rsquared:.3f}")
            col_ars.metric("Final Adj. R-squared", f"{stepwise_model.rsquared_adj:.3f}")
            col_fs.metric("F-statistic", f"{stepwise_model.fvalue:.3f}")
            col_ps.metric("Prob (F-stat)", f"{stepwise_model.f_pvalue:.4e}")

            step_coef_df = pd.DataFrame({
                "Coefficient": stepwise_model.params,
                "Std Error": stepwise_model.bse,
                "t-value": stepwise_model.tvalues,
                "P > |t|": stepwise_model.pvalues,
            })

            formatted_step_coef_df = style_with_pvalue_highlight(
                step_coef_df.style,
                subset=["P > |t|"],
            ).format({
                "Coefficient": "{:.4f}",
                "Std Error": "{:.4f}",
                "t-value": "{:.3f}",
                "P > |t|": "{:.4f}",
            })

            st.dataframe(formatted_step_coef_df, use_container_width=True)

            final_equation_str = build_equation_from_model(stepwise_model, target_var)
            st.success(f"**Final Predictive Equation (Cleaned of Multicollinearity):**\n\n**{final_equation_str}**")

            st.markdown("---")
            st.info("### How to Interpret and Report These Results")

            st.markdown(f"""
            #### 1. Understanding the Equation
            The equation above tells the biological story of how the selected predictors impact the **{target_var}**.
            * **The Constant:** This is the baseline starting point (the mathematical anchor).
            * **The Signs (+ or -):** A positive **(+)** sign means the predictor increases the **{target_var}**. A negative **(-)** sign means the predictor decreases the **{target_var}**.
            * **The Numbers (Coefficients):** This represents the exact change in **{target_var}** for every 1-unit change in that specific predictor, while the other predictors in the model are held constant.

            #### 2. Understanding the R-squared ({stepwise_model.rsquared:.2f})
            An R-squared value of **{stepwise_model.rsquared:.2f}** means that **{stepwise_model.rsquared * 100:.1f}%** of the variation in **{target_var}** is explained by the specific predictors left in this refined model. The remaining **{100 - stepwise_model.rsquared * 100:.1f}%** represents unexplained variation, such as natural environmental noise, unmeasured factors, or biological variability.
            """)

            st.markdown("**Reference: Acceptable R-squared Ranges in Agricultural Entomology**")
            r2_guide_data = {
                "R-squared Range": ["> 0.70", "0.50 - 0.70", "0.30 - 0.50", "< 0.30"],
                "Interpretation for Biological Data": [
                    "Excellent: Very strong predictive power. Usually seen in highly controlled lab or greenhouse settings.",
                    "Good: Strong biological relationship. Excellent outcome for field-level screening trials.",
                    "Moderate/Acceptable: Very common for field ecological data due to high environmental noise such as weather, predators, and microclimate variation.",
                    "Weak: A trend exists if P < 0.05, but the pest population is mostly driven by unmeasured factors.",
                ],
            }
            r2_df = pd.DataFrame(r2_guide_data)
            st.table(r2_df.set_index("R-squared Range"))

            st.markdown("""
            #### 3. How to report this in your findings
            You can copy and paste the following template directly into your thesis, project report, or publication draft:
            """)

            report_template = f"The stepwise regression model yielded a refined predictive equation (R-squared = {stepwise_model.rsquared:.2f}), indicating that {stepwise_model.rsquared * 100:.1f}% of the variation in {target_var} is explained by this specific combination of selected predictors. The final predictive equation was formulated as: {final_equation_str}."

            st.code(report_template, language="text")

        else:
            st.warning("After backward elimination, no variables were found to be statistically significant at the P < 0.05 level.")

    except Exception as e:
        st.error(f"Regression failed: {e}. Check for multicollinearity, constant variables, or insufficient complete observations.")


def validate_multi_year_dataset(df):
    missing = [column for column in ["Year", "SMW"] if column not in df.columns]
    if missing:
        return False, "Multi-year analysis requires 'Year', 'SMW', and data from more than one year."

    year_series = pd.to_numeric(df["Year"], errors="coerce").dropna()
    if year_series.nunique() <= 1:
        return False, "Multi-year analysis requires 'Year', 'SMW', and data from more than one year."

    _, numeric_cols = prepare_numeric_analysis_df(df, required_columns=["Year", "SMW"])
    analysis_numeric_cols = [col for col in numeric_cols if col not in ["Year", "SMW"]]
    if len(analysis_numeric_cols) < 2:
        return False, "Multi-year analysis requires 'Year', 'SMW', and data from more than one year."

    return True, "Dataset passed multi-year validation."


def is_multi_year_long_format_dataset(df):
    if df is None or "Year" not in df.columns or "SMW" not in df.columns:
        return False

    year_series = pd.to_numeric(df["Year"], errors="coerce").dropna()
    return year_series.nunique() > 1


def build_year_wise_population_summary(df, dependent_var):
    summary_rows = []
    working_df = coerce_numeric_columns(df, ["Year", "SMW", dependent_var]).dropna(subset=["Year", "SMW"])

    for year, year_df in working_df.groupby("Year", sort=True):
        year_df = year_df.sort_values("SMW")
        dep_series = pd.to_numeric(year_df[dependent_var], errors="coerce")
        valid_dep_df = year_df.assign(_dependent=dep_series).dropna(subset=["_dependent"])

        if valid_dep_df.empty:
            summary_rows.append({
                "Year": int(year) if float(year).is_integer() else year,
                "First appearance SMW": "NA",
                "Peak SMW": "NA",
                "Peak population": "NA",
                "Mean population": "NA",
                "Minimum population": "NA",
                "Number of observations": 0,
            })
            continue

        appearance_df = valid_dep_df[valid_dep_df["_dependent"] > 0]
        first_appearance = appearance_df["SMW"].iloc[0] if not appearance_df.empty else "Not observed"
        peak_idx = valid_dep_df["_dependent"].idxmax()
        peak_row = valid_dep_df.loc[peak_idx]

        summary_rows.append({
            "Year": int(year) if float(year).is_integer() else year,
            "First appearance SMW": first_appearance,
            "Peak SMW": peak_row["SMW"],
            "Peak population": peak_row["_dependent"],
            "Mean population": valid_dep_df["_dependent"].mean(),
            "Minimum population": valid_dep_df["_dependent"].min(),
            "Number of observations": int(valid_dep_df["_dependent"].count()),
        })

    summary_df = pd.DataFrame(summary_rows)
    numeric_display_cols = ["Peak population", "Mean population", "Minimum population"]
    for column in numeric_display_cols:
        if column in summary_df.columns:
            converted_column = pd.to_numeric(summary_df[column], errors="coerce")
            summary_df[column] = converted_column.where(converted_column.notna(), summary_df[column])
    return summary_df


def build_year_wise_correlation_table(df, dependent_var, weather_vars, corr_method):
    years = sorted(pd.to_numeric(df["Year"], errors="coerce").dropna().unique())
    rows = []
    numeric_records = []

    for weather_var in weather_vars:
        row = {"Parameter": weather_var}
        numeric_row = {"Parameter": weather_var}

        for year in years:
            year_df = df[pd.to_numeric(df["Year"], errors="coerce") == year]
            r_value, p_value = safe_corr(year_df[weather_var], year_df[dependent_var], corr_method)
            year_label = str(int(year)) if float(year).is_integer() else str(year)
            row[year_label] = format_corr_value(r_value, p_value)
            numeric_row[year_label] = r_value

        pooled_r, pooled_p = safe_corr(df[weather_var], df[dependent_var], corr_method)
        row["Pooled"] = format_corr_value(pooled_r, pooled_p)
        numeric_row["Pooled"] = pooled_r

        rows.append(row)
        numeric_records.append(numeric_row)

    return pd.DataFrame(rows), pd.DataFrame(numeric_records)


def build_direction_consistency_table(correlation_numeric_df):
    rows = []
    for _, corr_row in correlation_numeric_df.iterrows():
        parameter = corr_row["Parameter"]
        year_values = corr_row.drop(labels=["Parameter", "Pooled"], errors="ignore")
        valid_values = pd.to_numeric(year_values, errors="coerce").dropna()

        if valid_values.empty:
            direction = "Insufficient data"
            interpretation = "Interpretation not reliable"
        elif (valid_values > 0).all():
            direction = "Positive in all years"
            interpretation = "Consistently favourable association"
        elif (valid_values < 0).all():
            direction = "Negative in all years"
            interpretation = "Consistently unfavourable association"
        else:
            direction = "Mixed direction across years"
            interpretation = "Year-dependent or unstable association"

        rows.append({
            "Parameter": parameter,
            "Correlation direction": direction,
            "Biological interpretation": interpretation,
        })

    return pd.DataFrame(rows)


def build_year_wise_regression_summary(df, dependent_var, weather_vars):
    rows = []
    warnings = []
    equation_details = []
    working_df = coerce_numeric_columns(df, ["Year", dependent_var] + weather_vars)

    for year, year_df in working_df.groupby("Year", sort=True):
        year_label = int(year) if not pd.isna(year) and float(year).is_integer() else year
        reg_df = year_df[[dependent_var] + weather_vars].dropna()
        active_vars, removed_vars = remove_constant_predictors(reg_df, weather_vars)

        if removed_vars:
            warnings.append(f"{year_label}: constant predictor(s) removed before regression: {', '.join(removed_vars)}.")

        if not active_vars:
            warnings.append(f"Skipped {year_label}: no non-constant predictors were available after missing-value removal.")
            rows.append({
                "Year": year_label,
                "Regression equation": "Skipped: no non-constant predictors",
                "R-squared": "NA",
                "Adjusted R-squared": "NA",
                "F-statistic": "NA",
                "Prob(F)": "NA",
                "Significant variables at p < 0.05": "NA",
                "Number of observations": int(len(reg_df)),
            })
            continue

        if len(reg_df) < len(active_vars) + 2:
            message = f"Skipped {year_label}: observations ({len(reg_df)}) were less than predictors + 2 ({len(active_vars) + 2})."
            warnings.append(message)
            rows.append({
                "Year": year_label,
                "Regression equation": "Skipped: insufficient observations",
                "R-squared": "NA",
                "Adjusted R-squared": "NA",
                "F-statistic": "NA",
                "Prob(F)": "NA",
                "Significant variables at p < 0.05": "NA",
                "Number of observations": int(len(reg_df)),
            })
            continue

        try:
            X = sm.add_constant(reg_df[active_vars], has_constant="add")
            y = reg_df[dependent_var]
            model = sm.OLS(y, X).fit()

            significant_vars = get_significant_predictors(model.pvalues)
            equation_text = build_regression_equation(model, dependent_var)
            model_formula = f"{dependent_var} ~ " + " + ".join(active_vars)
            interpretation = (
                "This equation predicts pest population from selected weather parameters. Positive coefficients indicate favourable association, while negative coefficients indicate unfavourable association when other variables are held constant."
            )
            equation_details.append({
                "Year": year_label,
                "Model formula": model_formula,
                "Human-readable equation": equation_text,
                "Interpretation": interpretation,
                "Download text": build_model_equation_download_text(
                    model,
                    model_formula,
                    equation_text,
                    interpretation,
                    f"Year-wise OLS Model Equation - {year_label}",
                ),
            })

            rows.append({
                "Year": year_label,
                "Regression equation": equation_text,
                "R-squared": model.rsquared,
                "Adjusted R-squared": model.rsquared_adj,
                "F-statistic": model.fvalue,
                "Prob(F)": model.f_pvalue,
                "Significant variables at p < 0.05": ", ".join(significant_vars) if significant_vars else "None",
                "Number of observations": int(len(reg_df)),
            })
        except Exception as exc:
            message = f"Skipped {year_label}: regression failed ({exc})."
            warnings.append(message)
            rows.append({
                "Year": year_label,
                "Regression equation": "Skipped: regression failed",
                "R-squared": "NA",
                "Adjusted R-squared": "NA",
                "F-statistic": "NA",
                "Prob(F)": "NA",
                "Significant variables at p < 0.05": "NA",
                "Number of observations": int(len(reg_df)),
            })

    return format_regression_summary_df(pd.DataFrame(rows)), warnings, equation_details


def render_pooled_regression(df, dependent_var, weather_vars):
    pooled_model, display_formula, pooled_df, coefficient_df, warning, removed_vars = fit_pooled_year_ols(
        df,
        dependent_var,
        weather_vars,
    )

    if removed_vars:
        st.warning(f"Constant predictor(s) removed from pooled regression: {', '.join(removed_vars)}.")
    if warning:
        st.warning(warning)
    if pooled_model is not None and coefficient_df is not None:
        col_r, col_ar, col_f, col_p = st.columns(4)
        col_r.metric("R-squared", f"{pooled_model.rsquared:.3f}")
        col_ar.metric("Adjusted R-squared", f"{pooled_model.rsquared_adj:.3f}")
        col_f.metric("F-statistic", f"{pooled_model.fvalue:.3f}")
        col_p.metric("Prob(F)", f"{pooled_model.f_pvalue:.4e}")

        formatted_coef_df = style_with_pvalue_highlight(
            coefficient_df.set_index("Term").style,
            subset=["P > |t|"],
        ).format({
            "Coefficient": "{:.4f}",
            "Std Error": "{:.4f}",
            "t-value": "{:.3f}",
            "P > |t|": "{:.4f}",
        })
        st.dataframe(formatted_coef_df, use_container_width=True)

        significant_vars = get_significant_predictors(pooled_model.pvalues)
        if significant_vars:
            st.success(f"Significant variables at p < 0.05: {', '.join(significant_vars)}")
        else:
            st.info("No weather variables were significant at p < 0.05 in the pooled year-adjusted model.")

        st.markdown("**Final model formula:**")
        st.code(display_formula, language="text")
        st.info("This pooled model adjusts for year-to-year variation by treating Year as a categorical factor using C(Year).")
        render_model_equation_section(
            pooled_model,
            dependent_var,
            model_formula=display_formula,
            model_label="Pooled OLS with Year Effect Equation",
            interpretation="This equation adjusts the prediction for year-to-year variation using C(Year).",
            key_prefix="multi_pooled_year_effect",
            model_df=pooled_df,
            expander=True,
        )

        return pooled_model, display_formula

    return None, None


def build_lag_correlation_table(df, dependent_var, weather_vars, corr_method, selected_lags):
    working_df = coerce_numeric_columns(df, ["Year", "SMW", dependent_var] + weather_vars)
    working_df = working_df.sort_values(["Year", "SMW"])
    rows = []
    numeric_rows = []
    lag_labels = {
        0: "Same week",
        1: "1-week lag",
        2: "2-week lag",
    }

    for weather_var in weather_vars:
        row = {"Parameter": weather_var}
        numeric_row = {"Parameter": weather_var}
        lag_values = {}

        for lag in [0, 1, 2]:
            label = lag_labels[lag]
            if lag not in selected_lags:
                row[label] = "Not selected"
                numeric_row[label] = np.nan
                continue

            if lag == 0:
                lagged_values = working_df[weather_var]
            else:
                lagged_values = working_df.groupby("Year", group_keys=False)[weather_var].shift(lag)

            r_value, p_value = safe_corr(lagged_values, working_df[dependent_var], corr_method)
            row[label] = format_corr_value(r_value, p_value)
            numeric_row[label] = r_value
            lag_values[label] = r_value

        valid_lag_values = {
            label: value
            for label, value in lag_values.items()
            if pd.notna(value) and np.isfinite(value)
        }
        if valid_lag_values:
            best_lag = max(valid_lag_values, key=lambda label: abs(valid_lag_values[label]))
        else:
            best_lag = "NA"
        row["Best lag"] = best_lag
        numeric_row["Best lag"] = best_lag
        rows.append(row)
        numeric_rows.append(numeric_row)

    return pd.DataFrame(rows), pd.DataFrame(numeric_rows)


def build_scientific_interpretation(population_summary_df, corr_numeric_df, direction_df, pooled_model, dependent_var, lag_numeric_df=None):
    interpretation_parts = []

    if not population_summary_df.empty and "Peak population" in population_summary_df.columns:
        peak_df = population_summary_df.copy()
        peak_df["_peak_population_numeric"] = pd.to_numeric(peak_df["Peak population"], errors="coerce")
        peak_df = peak_df.dropna(subset=["_peak_population_numeric"])
        if not peak_df.empty:
            peak_row = peak_df.loc[peak_df["_peak_population_numeric"].idxmax()]
            interpretation_parts.append(
                f"The peak activity of {dependent_var} was observed during SMW {peak_row['Peak SMW']} in {peak_row['Year']}, with a peak population of {format_number(peak_row['_peak_population_numeric'], 2)}."
            )

    pooled_corr = corr_numeric_df[["Parameter", "Pooled"]].copy() if "Pooled" in corr_numeric_df.columns else pd.DataFrame()
    if not pooled_corr.empty:
        pooled_corr["Pooled"] = pd.to_numeric(pooled_corr["Pooled"], errors="coerce")
        positive_df = pooled_corr[pooled_corr["Pooled"] > 0].dropna()
        negative_df = pooled_corr[pooled_corr["Pooled"] < 0].dropna()

        if not positive_df.empty:
            pos_row = positive_df.loc[positive_df["Pooled"].idxmax()]
            interpretation_parts.append(
                f"The strongest positive pooled association was recorded with {pos_row['Parameter']} (r = {pos_row['Pooled']:.2f}), indicating a favourable association under the pooled condition."
            )

        if not negative_df.empty:
            neg_row = negative_df.loc[negative_df["Pooled"].idxmin()]
            interpretation_parts.append(
                f"The strongest negative pooled association was recorded with {neg_row['Parameter']} (r = {neg_row['Pooled']:.2f}), indicating an unfavourable association under the pooled condition."
            )

    if not direction_df.empty:
        consistent_df = direction_df[
            direction_df["Correlation direction"].isin(["Positive in all years", "Negative in all years"])
        ]
        if not consistent_df.empty:
            consistent_factors = ", ".join(consistent_df["Parameter"].tolist())
            interpretation_parts.append(
                f"The direction consistency analysis showed that {consistent_factors} had stable year-wise association with the pest population."
            )

    if pooled_model is not None:
        interpretation_parts.append(
            f"The pooled regression model explained {pooled_model.rsquared * 100:.1f}% variation in {dependent_var} after adjusting for year-to-year variation."
        )

    if lag_numeric_df is not None and not lag_numeric_df.empty:
        lag_long = lag_numeric_df.melt(
            id_vars=["Parameter", "Best lag"],
            value_vars=[col for col in ["Same week", "1-week lag", "2-week lag"] if col in lag_numeric_df.columns],
            var_name="Lag period",
            value_name="Correlation",
        ).dropna(subset=["Correlation"])
        if not lag_long.empty:
            lag_long["Absolute correlation"] = lag_long["Correlation"].abs()
            strongest_lag = lag_long.loc[lag_long["Absolute correlation"].idxmax()]
            interpretation_parts.append(
                f"The lag analysis indicated that {strongest_lag['Parameter']} during {strongest_lag['Lag period']} had the strongest delayed association with {dependent_var} (r = {strongest_lag['Correlation']:.2f})."
            )

    if not interpretation_parts:
        return "A scientific interpretation could not be generated because the selected dataset did not produce sufficient valid analytical outputs."

    return " ".join(interpretation_parts)


def render_model_metrics(model, include_aic_bic=False):
    metric_count = 6 if include_aic_bic else 4
    columns = st.columns(metric_count)
    columns[0].metric("R-squared", f"{model.rsquared:.3f}")
    columns[1].metric("Adjusted R-squared", f"{model.rsquared_adj:.3f}")
    columns[2].metric("F-statistic", f"{model.fvalue:.3f}")
    columns[3].metric("Prob(F)", f"{model.f_pvalue:.4e}")
    if include_aic_bic:
        columns[4].metric("AIC", f"{model.aic:.3f}")
        columns[5].metric("BIC", f"{model.bic:.3f}")


def render_coefficient_download(coefficient_df, key_prefix, file_name):
    if coefficient_df is None or coefficient_df.empty:
        return
    st.download_button(
        label=f"Download {file_name.replace('_', ' ').replace('.csv', '')}",
        data=csv_bytes(coefficient_df),
        file_name=file_name,
        mime="text/csv",
        key=f"{key_prefix}_coef_download",
    )


def fit_mixedlm_model(df, dependent_var, weather_vars):
    model_df = coerce_numeric_columns(df, ["Year", dependent_var] + weather_vars)
    model_df = model_df[["Year", dependent_var] + weather_vars].dropna()
    active_vars, removed_vars = remove_constant_predictors(model_df, weather_vars)

    if not active_vars:
        return None, None, model_df, None, "No non-constant weather predictors were available for the mixed model.", removed_vars

    if model_df["Year"].nunique() < 2 or len(model_df) < len(active_vars) + 3:
        return None, None, model_df, None, "Mixed model could not be fitted for this dataset. This may occur due to small sample size, too many predictors, constant variables or singular matrix. Use pooled OLS with Year effect as an alternative.", removed_vars

    try:
        formula = build_formula(dependent_var, active_vars, include_year_effect=False)
        mixed_model = smf.mixedlm(formula=formula, data=model_df, groups=model_df["Year"]).fit(
            reml=False,
            method="lbfgs",
            maxiter=300,
            disp=False,
        )
        display_formula = f"{dependent_var} ~ " + " + ".join(active_vars) + " + (1 | Year)"
        return mixed_model, display_formula, model_df, build_coefficient_table(mixed_model), "", removed_vars
    except Exception:
        return None, None, model_df, None, "Mixed model could not be fitted for this dataset. This may occur due to small sample size, too many predictors, constant variables or singular matrix. Use pooled OLS with Year effect as an alternative.", removed_vars


def build_diagnostic_summary(model):
    rows = []
    residuals = np.asarray(model.resid, dtype=float)
    finite_mask = np.isfinite(residuals)
    residuals = residuals[finite_mask]

    if len(residuals) < 3:
        return pd.DataFrame([{
            "Diagnostic": "Residual diagnostics",
            "Statistic": "NA",
            "P-value": "NA",
            "Interpretation": "Diagnostics require at least three valid residuals.",
        }])

    try:
        dw_value = durbin_watson(residuals)
        if 1.5 <= dw_value <= 2.5:
            dw_interpretation = "Residual autocorrelation was not strongly indicated."
        elif dw_value < 1.5:
            dw_interpretation = "Positive residual autocorrelation may be present."
        else:
            dw_interpretation = "Negative residual autocorrelation may be present."
        rows.append({
            "Diagnostic": "Durbin-Watson autocorrelation",
            "Statistic": f"{dw_value:.3f}",
            "P-value": "NA",
            "Interpretation": dw_interpretation,
        })
    except Exception as exc:
        rows.append({
            "Diagnostic": "Durbin-Watson autocorrelation",
            "Statistic": "NA",
            "P-value": "NA",
            "Interpretation": f"Could not calculate Durbin-Watson statistic ({exc}).",
        })

    try:
        exog = np.asarray(model.model.exog)
        exog = exog[finite_mask, :]
        if exog.shape[1] >= 2:
            bp_lm, bp_lm_pvalue, bp_f, bp_f_pvalue = het_breuschpagan(residuals, exog)
            bp_interpretation = "Heteroscedasticity was indicated." if bp_f_pvalue < 0.05 else "Heteroscedasticity was not strongly indicated."
            rows.append({
                "Diagnostic": "Breusch-Pagan heteroscedasticity",
                "Statistic": f"{bp_f:.3f}",
                "P-value": f"{bp_f_pvalue:.4e}",
                "Interpretation": bp_interpretation,
            })
        else:
            rows.append({
                "Diagnostic": "Breusch-Pagan heteroscedasticity",
                "Statistic": "NA",
                "P-value": "NA",
                "Interpretation": "Breusch-Pagan test requires at least one predictor plus intercept.",
            })
    except Exception as exc:
        rows.append({
            "Diagnostic": "Breusch-Pagan heteroscedasticity",
            "Statistic": "NA",
            "P-value": "NA",
            "Interpretation": f"Could not calculate Breusch-Pagan test ({exc}).",
        })

    try:
        if 3 <= len(residuals) <= 5000:
            shapiro_stat, shapiro_pvalue = shapiro(residuals)
            normality_interpretation = "Residual normality was not strongly rejected." if shapiro_pvalue >= 0.05 else "Residuals deviated significantly from normality."
            rows.append({
                "Diagnostic": "Shapiro-Wilk residual normality",
                "Statistic": f"{shapiro_stat:.3f}",
                "P-value": f"{shapiro_pvalue:.4e}",
                "Interpretation": normality_interpretation,
            })
        else:
            rows.append({
                "Diagnostic": "Shapiro-Wilk residual normality",
                "Statistic": "NA",
                "P-value": "NA",
                "Interpretation": "Shapiro-Wilk test was not run because sample size was outside the suitable range.",
            })
    except Exception as exc:
        rows.append({
            "Diagnostic": "Shapiro-Wilk residual normality",
            "Statistic": "NA",
            "P-value": "NA",
            "Interpretation": f"Could not calculate Shapiro-Wilk test ({exc}).",
        })

    return pd.DataFrame(rows)


def render_residual_diagnostic_plots(model):
    residuals = np.asarray(model.resid, dtype=float)
    fitted = np.asarray(model.fittedvalues, dtype=float)
    finite_mask = np.isfinite(residuals) & np.isfinite(fitted)
    residuals = residuals[finite_mask]
    fitted = fitted[finite_mask]

    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
        sns.histplot(residuals, kde=True, ax=ax_hist)
        ax_hist.set_title("Residual Histogram")
        ax_hist.set_xlabel("Residual")
        st.pyplot(fig_hist)

    with plot_col2:
        fig_scatter, ax_scatter = plt.subplots(figsize=(7, 4))
        ax_scatter.scatter(fitted, residuals)
        ax_scatter.axhline(0, color="red", linestyle="--", linewidth=1)
        ax_scatter.set_title("Residuals vs Fitted Values")
        ax_scatter.set_xlabel("Fitted values")
        ax_scatter.set_ylabel("Residuals")
        st.pyplot(fig_scatter)

    fig_qq, ax_qq = plt.subplots(figsize=(7, 4))
    sm.qqplot(residuals, line="45", ax=ax_qq)
    ax_qq.set_title("Q-Q Plot of Residuals")
    st.pyplot(fig_qq)


def build_model_comparison_table(model_records):
    rows = []
    row_counts = []

    for record in model_records:
        model = record.get("model")
        coefficient_df = record.get("coefficients")
        model_df = record.get("data")
        if model is None:
            continue

        if hasattr(model, "rsquared"):
            r_squared_note = f"{model.rsquared:.3f}"
        else:
            r_squared_note = "Not directly available"

        if coefficient_df is not None and not coefficient_df.empty:
            significant_terms = coefficient_df[
                (coefficient_df["P > |t|"] < 0.05)
                & (coefficient_df["Term"] != "const")
                & (~coefficient_df["Term"].astype(str).str.startswith("C(Year)"))
            ]["Term"].astype(str).tolist()
        else:
            significant_terms = []

        nobs = int(len(model_df)) if model_df is not None else int(getattr(model, "nobs", 0))
        row_counts.append(nobs)

        rows.append({
            "Model": record["name"],
            "R-squared or marginal note": r_squared_note,
            "AIC": format_number(getattr(model, "aic", np.nan), 3),
            "BIC": format_number(getattr(model, "bic", np.nan), 3),
            "Significant predictors": ", ".join(significant_terms) if significant_terms else "None",
            "Remarks": record.get("remarks", ""),
            "Number of observations": nobs,
        })

    comparison_df = pd.DataFrame(rows)
    row_warning = ""
    if len(set(row_counts)) > 1:
        row_warning = "Models were fitted on different numbers of rows because lag creation and missing-value removal changed the available observations."
    return comparison_df, row_warning


def build_journal_conclusion(dependent_var, pooled_model=None, lag_model=None, mixed_model=None, lag_mode=None):
    parts = [
        "Correlation results were treated as preliminary because correlation indicated association only and did not establish direct causation."
    ]

    if pooled_model is not None:
        parts.append(
            f"The year-adjusted pooled model explained {pooled_model.rsquared * 100:.1f} per cent variation in {dependent_var}, indicating that weather parameters and year-to-year seasonal variation contributed to population fluctuation."
        )

    if mixed_model is not None:
        parts.append(
            "The mixed model with Year as a random effect supported interpretation after accounting for year-wise baseline differences."
        )

    if lag_model is not None:
        parts.append(
            f"The lag-based model using {lag_mode.lower()} explained {lag_model.rsquared * 100:.1f} per cent variation, suggesting that delayed weather influence should be considered while interpreting pest build-up."
        )

    parts.append(
        "Therefore, journal-level inference should be based on year effect, lag effect, residual diagnostics and biological plausibility, and the terms associated with, influenced, explained variation and contributed to population fluctuation should be preferred over causal wording."
    )
    return " ".join(parts)


def render_journal_level_model(df):
    st.subheader("SECTION 1: Journal-level explanation")
    st.info(
        "Correlation and ordinary regression are useful for preliminary population dynamics, but multi-year weekly pest data may have year-to-year variation, lag effects, multicollinearity and autocorrelation. Therefore, journal-level interpretation should be supported by diagnostics, year-adjusted models and, where possible, mixed models."
    )
    st.warning("Journal-level interpretation should be based on diagnostics, year effect, lag effect and biological plausibility, not only statistical significance.")

    is_valid, validation_message = validate_multi_year_dataset(df)
    if not is_valid:
        st.warning(validation_message)
        return

    numeric_df, dependent_var, weather_vars = get_default_multi_year_variables(df, "journal")
    if dependent_var is None:
        st.warning("Select a dataset with a numeric dependent pest population column.")
        return
    if not weather_vars:
        st.warning("Select at least one weather variable before running journal-level models.")
        return

    analysis_df = coerce_numeric_columns(numeric_df, ["Year", "SMW", dependent_var] + weather_vars)

    st.subheader("SECTION 2: Model options")
    selected_options = st.multiselect(
        "Select journal-level analyses to run",
        [
            "Pooled OLS with Year effect",
            "Linear Mixed Model with Year as random effect",
            "Lag-based pooled model",
            "Model comparison and diagnostics",
        ],
        default=[
            "Pooled OLS with Year effect",
            "Linear Mixed Model with Year as random effect",
            "Lag-based pooled model",
            "Model comparison and diagnostics",
        ],
        key="journal_model_options",
    )

    model_records = []
    pooled_model = None
    pooled_coef_df = None
    pooled_model_df = None
    mixed_model = None
    mixed_coef_df = None
    mixed_model_df = None
    lag_model = None
    lag_coef_df = None
    lag_model_df = None
    lag_mode = None
    diagnostic_summary_df = pd.DataFrame()

    if "Pooled OLS with Year effect" in selected_options:
        st.markdown("#### Pooled OLS with Year effect")
        pooled_model, pooled_formula, pooled_model_df, pooled_coef_df, pooled_warning, pooled_removed = fit_pooled_year_ols(
            analysis_df,
            dependent_var,
            weather_vars,
        )
        if pooled_removed:
            st.warning(f"Constant predictor(s) removed from pooled model: {', '.join(pooled_removed)}.")
        if pooled_warning:
            st.warning(pooled_warning)
        if pooled_model is not None and pooled_coef_df is not None:
            st.code(pooled_formula, language="text")
            render_model_metrics(pooled_model, include_aic_bic=True)
            st.dataframe(
                style_with_pvalue_highlight(pooled_coef_df.set_index("Term").style, subset=["P > |t|"]).format({
                    "Coefficient": "{:.4f}",
                    "Std Error": "{:.4f}",
                    "t-value": "{:.3f}",
                    "P > |t|": "{:.4f}",
                }),
                use_container_width=True,
            )
            render_model_equation_section(
                pooled_model,
                dependent_var,
                model_formula=pooled_formula,
                model_label="Journal-Level Pooled OLS with Year Effect Equation",
                interpretation="This equation adjusts the prediction for year-to-year variation using C(Year).",
                key_prefix="journal_pooled_year_effect",
                model_df=pooled_model_df,
                expander=True,
            )
            model_records.append({
                "name": "Pooled OLS with Year effect",
                "model": pooled_model,
                "coefficients": pooled_coef_df,
                "data": pooled_model_df,
                "remarks": "Year-adjusted fixed-effect model.",
            })

    if "Linear Mixed Model with Year as random effect" in selected_options:
        st.subheader("SECTION 3: Linear Mixed Model")
        mixed_model, mixed_formula, mixed_model_df, mixed_coef_df, mixed_warning, mixed_removed = fit_mixedlm_model(
            analysis_df,
            dependent_var,
            weather_vars,
        )
        if mixed_removed:
            st.warning(f"Constant predictor(s) removed from mixed model: {', '.join(mixed_removed)}.")
        if mixed_warning:
            st.warning(mixed_warning)
        if mixed_model is not None and mixed_coef_df is not None:
            st.code(mixed_formula, language="text")
            mixed_cols = st.columns(3)
            mixed_cols[0].metric("AIC", f"{mixed_model.aic:.3f}")
            mixed_cols[1].metric("BIC", f"{mixed_model.bic:.3f}")
            mixed_cols[2].metric("Log-likelihood", f"{mixed_model.llf:.3f}")
            st.dataframe(
                style_with_pvalue_highlight(mixed_coef_df.set_index("Term").style, subset=["P > |t|"]).format({
                    "Coefficient": "{:.4f}",
                    "Std Error": "{:.4f}",
                    "t-value": "{:.3f}",
                    "P > |t|": "{:.4f}",
                }),
                use_container_width=True,
            )
            st.info("The mixed model estimated weather effects while allowing baseline pest population to vary by year.")
            render_mixed_model_equation_section(
                mixed_model,
                dependent_var,
                mixed_formula,
                "journal_mixed_model",
            )
            render_coefficient_download(mixed_coef_df, "journal_mixed", "mixed_model_coefficient_table.csv")
            model_records.append({
                "name": "Linear Mixed Model with Year random effect",
                "model": mixed_model,
                "coefficients": mixed_coef_df,
                "data": mixed_model_df,
                "remarks": "Random intercept for Year; R-squared not directly available.",
            })

    if "Lag-based pooled model" in selected_options:
        st.subheader("SECTION 4: Lag-based model")
        lag_corr_method = st.radio(
            "Correlation method for best-lag selection",
            ["Pearson", "Spearman"],
            horizontal=True,
            key="journal_lag_corr_method",
        )
        _, journal_lag_numeric_df = build_lag_correlation_table(
            analysis_df,
            dependent_var,
            weather_vars,
            lag_corr_method,
            [0, 1, 2],
        )
        lag_mode = st.radio(
            "Select lagged variable set",
            [
                "Same-week variables only",
                "Lag 1 variables",
                "Lag 2 variables",
                "Best lag variables from lag correlation table",
            ],
            key="journal_lag_mode",
        )
        lag_model, lag_formula, lag_model_df, lag_coef_df, lag_warning, lag_removed = fit_lag_based_ols(
            analysis_df,
            dependent_var,
            weather_vars,
            lag_mode,
            journal_lag_numeric_df,
        )
        if lag_removed:
            st.warning(f"Constant lagged predictor(s) removed from lag model: {', '.join(lag_removed)}.")
        if lag_warning:
            st.warning(lag_warning)
        if lag_model is not None and lag_coef_df is not None:
            st.code(lag_formula, language="text")
            render_model_metrics(lag_model, include_aic_bic=True)
            st.dataframe(
                style_with_pvalue_highlight(lag_coef_df.set_index("Term").style, subset=["P > |t|"]).format({
                    "Coefficient": "{:.4f}",
                    "Std Error": "{:.4f}",
                    "t-value": "{:.3f}",
                    "P > |t|": "{:.4f}",
                }),
                use_container_width=True,
            )
            st.info("Lagged variables were created within each year only, so values did not cross from one year into another.")
            render_model_equation_section(
                lag_model,
                dependent_var,
                model_formula=lag_formula,
                model_label="Lag-Based OLS with Year Effect Equation",
                interpretation="Lagged variables indicate the effect of previous week or previous two weeks weather on current pest population.",
                key_prefix="journal_lag_based",
                model_df=lag_model_df,
                expander=True,
            )
            render_coefficient_download(lag_coef_df, "journal_lag", "lag_model_coefficient_table.csv")
            model_records.append({
                "name": "Lag-based OLS with Year effect",
                "model": lag_model,
                "coefficients": lag_coef_df,
                "data": lag_model_df,
                "remarks": f"Lag mode: {lag_mode}.",
            })

    if "Model comparison and diagnostics" in selected_options:
        st.subheader("SECTION 5: Residual diagnostics")
        diagnostic_models = {}
        if pooled_model is not None:
            diagnostic_models["Pooled OLS with Year effect"] = pooled_model
        if lag_model is not None:
            diagnostic_models["Lag-based OLS with Year effect"] = lag_model

        if diagnostic_models:
            selected_diagnostic_model = st.selectbox(
                "Select model for residual diagnostics",
                options=list(diagnostic_models.keys()),
                key="journal_diagnostic_model",
            )
            diagnostic_model = diagnostic_models[selected_diagnostic_model]
            render_residual_diagnostic_plots(diagnostic_model)
            diagnostic_summary_df = build_diagnostic_summary(diagnostic_model)
            st.dataframe(diagnostic_summary_df, use_container_width=True, hide_index=True)
            st.download_button(
                label="Download diagnostic summary",
                data=csv_bytes(diagnostic_summary_df),
                file_name="diagnostic_summary.csv",
                mime="text/csv",
                key="journal_diagnostic_download",
            )
        else:
            st.warning("No OLS model was available for residual diagnostics.")

        st.subheader("SECTION 6: Model comparison")
        comparison_df, comparison_warning = build_model_comparison_table(model_records)
        if comparison_warning:
            st.warning(comparison_warning)
        if not comparison_df.empty:
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            st.info("Lower AIC/BIC indicates better relative model fit. Models fitted on very different datasets should be compared cautiously.")
            st.download_button(
                label="Download model comparison table",
                data=csv_bytes(comparison_df),
                file_name="model_comparison_table.csv",
                mime="text/csv",
                key="journal_model_comparison_download",
            )
        else:
            st.warning("No fitted model was available for comparison.")

    st.subheader("SECTION 7: Journal-level conclusion")
    journal_conclusion = build_journal_conclusion(
        dependent_var,
        pooled_model=pooled_model,
        lag_model=lag_model,
        mixed_model=mixed_model,
        lag_mode=lag_mode,
    )
    st.write(journal_conclusion)
    st.code(journal_conclusion, language="text")
    st.download_button(
        label="Download journal-level interpretation text",
        data=text_bytes(journal_conclusion),
        file_name="journal_level_interpretation.txt",
        mime="text/plain",
        key="journal_interpretation_download",
    )


def render_multi_year_analysis(df):
    st.subheader("SECTION 1: Data validation")
    is_valid, validation_message = validate_multi_year_dataset(df)

    if not is_valid:
        st.warning(validation_message)
        return

    st.success(validation_message)

    st.subheader("SECTION 2: Variable selection")

    numeric_df, dependent_var, weather_vars = get_default_multi_year_variables(df, "multi")

    if dependent_var is None:
        st.warning("Select a dataset with a numeric dependent pest population column.")
        return
    if not weather_vars:
        st.warning("Select at least one weather variable for multi-year analysis.")
        return

    corr_method = st.radio(
        "Select correlation method",
        ["Pearson", "Spearman"],
        horizontal=True,
        key="multi_corr_method",
    )
    st.warning("Correlation indicates association only and should not be interpreted as direct causation.")

    analysis_df = coerce_numeric_columns(numeric_df, ["Year", "SMW", dependent_var] + weather_vars)

    st.subheader("SECTION 3: Year-wise population summary")
    population_summary_df = build_year_wise_population_summary(analysis_df, dependent_var)
    st.dataframe(population_summary_df, use_container_width=True, hide_index=True)
    st.download_button(
        label="Download year-wise population summary",
        data=csv_bytes(population_summary_df),
        file_name="year_wise_population_summary.csv",
        mime="text/csv",
        key="download_population_summary",
    )

    st.subheader("SECTION 4: Year-wise correlation table")
    correlation_table_df, correlation_numeric_df = build_year_wise_correlation_table(
        analysis_df,
        dependent_var,
        weather_vars,
        corr_method,
    )
    st.dataframe(correlation_table_df, use_container_width=True, hide_index=True)
    st.caption("*Significant at P < 0.05, **Significant at P < 0.01")
    st.download_button(
        label="Download year-wise correlation table",
        data=csv_bytes(correlation_table_df),
        file_name="year_wise_correlation_table.csv",
        mime="text/csv",
        key="download_year_wise_correlation",
    )

    st.subheader("SECTION 5: Direction consistency table")
    direction_df = build_direction_consistency_table(correlation_numeric_df)
    st.dataframe(direction_df, use_container_width=True, hide_index=True)
    st.download_button(
        label="Download direction consistency table",
        data=csv_bytes(direction_df),
        file_name="direction_consistency_table.csv",
        mime="text/csv",
        key="download_direction_consistency",
    )

    st.subheader("SECTION 6: Year-wise regression equations")
    year_regression_df, regression_warnings, year_equation_details = build_year_wise_regression_summary(
        analysis_df,
        dependent_var,
        weather_vars,
    )
    for warning in regression_warnings:
        st.warning(warning)
    st.dataframe(year_regression_df, use_container_width=True, hide_index=True)
    st.download_button(
        label="Download year-wise regression summary",
        data=csv_bytes(year_regression_df),
        file_name="year_wise_regression_summary.csv",
        mime="text/csv",
        key="download_year_wise_regression",
    )
    if year_equation_details:
        st.markdown("#### Human-readable Model Equation")
        with st.expander("View full model equation"):
            for detail in year_equation_details:
                st.markdown(f"**Year {detail['Year']}**")
                st.code(detail["Human-readable equation"], language="text")
        st.info(
            "This equation predicts pest population from selected weather parameters. Positive coefficients indicate favourable association, while negative coefficients indicate unfavourable association when other variables are held constant."
        )
        st.download_button(
            label="Download Model Equation",
            data=text_bytes("\n\n".join(detail["Download text"] for detail in year_equation_details)),
            file_name="year_wise_ols_model_equations.txt",
            mime="text/plain",
            key="download_year_wise_model_equations",
        )

    st.subheader("SECTION 7: Pooled regression with Year effect")
    st.warning("Simple pooled analysis may hide year-to-year variation. Year-adjusted or mixed models are preferred for multi-year inference.")
    pooled_model, pooled_formula = render_pooled_regression(analysis_df, dependent_var, weather_vars)

    st.subheader("SECTION 8: Lag correlation analysis")
    run_lag_analysis = st.checkbox(
        "Run optional lag correlation analysis",
        value=True,
        key="run_lag_analysis",
    )
    lag_table_df = pd.DataFrame()
    lag_numeric_df = pd.DataFrame()
    if run_lag_analysis:
        selected_lags = st.multiselect(
            "Select lag values",
            options=[0, 1, 2],
            default=[0, 1, 2],
            format_func=lambda lag: "0 week" if lag == 0 else f"{lag} week",
            key="selected_lags",
        )
        if selected_lags:
            lag_table_df, lag_numeric_df = build_lag_correlation_table(
                analysis_df,
                dependent_var,
                weather_vars,
                corr_method,
                selected_lags,
            )
            st.dataframe(lag_table_df, use_container_width=True, hide_index=True)
            st.download_button(
                label="Download lag correlation table",
                data=csv_bytes(lag_table_df),
                file_name="lag_correlation_table.csv",
                mime="text/csv",
                key="download_lag_correlation",
            )
        else:
            st.warning("Select at least one lag value to run lag correlation analysis.")

    st.subheader("SECTION 9: Ready-to-use scientific interpretation")
    interpretation_text = build_scientific_interpretation(
        population_summary_df,
        correlation_numeric_df,
        direction_df,
        pooled_model,
        dependent_var,
        lag_numeric_df,
    )
    st.write(interpretation_text)
    st.code(interpretation_text, language="text")

    st.subheader("SECTION 10: Download buttons")
    st.markdown("Use the download buttons shown below each table above to export the generated CSV outputs.")


def render_student_requirement_guide():
    st.subheader("Student Requirement Guide")

    st.markdown("### Analysis Requirement Table")
    requirement_df = pd.DataFrame({
        "Section": [
            "Year-wise peak incidence",
            "Year-wise correlation",
            "Pooled correlation",
            "OLS regression",
            "Stepwise regression",
            "VIF/multicollinearity",
            "Lag correlation",
            "Year effect model",
            "Mixed model",
            "Residual diagnostics",
            "Model comparison",
        ],
        "Ph.D. thesis": [
            "Required",
            "Required",
            "Required",
            "Required",
            "Useful",
            "Required",
            "Recommended",
            "Recommended",
            "Optional",
            "Recommended",
            "Optional",
        ],
        "High-value journal": [
            "Required",
            "Required",
            "Supporting only",
            "Supporting only",
            "Use carefully",
            "Required",
            "Required",
            "Required",
            "Strongly recommended",
            "Required",
            "Required",
        ],
    })
    st.dataframe(requirement_df, use_container_width=True, hide_index=True)

    st.markdown("### Meaning Table")
    meaning_df = pd.DataFrame({
        "Term": [
            "Required",
            "Recommended",
            "Useful",
            "Optional",
            "Supporting only",
            "Use carefully",
            "Strongly recommended",
        ],
        "Meaning": [
            "Must include",
            "Strongly advisable",
            "Good to include",
            "Include only if data are suitable",
            "Do not make final conclusion only from this analysis",
            "Result may be unstable; biological meaning must also be considered",
            "Very important for journal-level statistical strength",
        ],
    })
    st.dataframe(meaning_df, use_container_width=True, hide_index=True)

    st.info(
        "Correlation and OLS regression are acceptable for thesis-level explanation, but high-value journals usually require year effect, lag analysis, diagnostics and mixed/model comparison support."
    )


render_sidebar()

st.title("Agricultural Correlation & Regression Analysis")
st.markdown("Upload your dataset to generate publication-ready tables, heatmaps, and regression models.")

uploaded_file = st.file_uploader("Upload your data (CSV or Excel)", type=["csv", "xlsx"])

df = None
uploaded_is_multi_year = False
if uploaded_file is not None:
    try:
        df = read_uploaded_dataset(uploaded_file)
        uploaded_is_multi_year = is_multi_year_long_format_dataset(df)
        st.success("Data loaded successfully.")
        if uploaded_is_multi_year:
            st.info(
                "Multi-year long-format dataset detected. Single-Year analysis is disabled for this upload to avoid misleading results; use the Multi-Year Population Dynamics or Journal-Level Model tabs."
            )
        with st.expander("Preview Raw Data"):
            st.dataframe(df.head())
    except Exception as exc:
        st.error(f"Failed to read file: {exc}. Check your formatting.")

single_year_tab, multi_year_tab, journal_tab, student_guide_tab = st.tabs([
    "Single-Year Population Dynamics",
    "Multi-Year Population Dynamics",
    "Journal-Level Model",
    "Student Requirement Guide",
])

with single_year_tab:
    if df is None:
        st.info("Upload a CSV or Excel file to begin single-year population dynamics analysis.")
    elif uploaded_is_multi_year:
        st.warning(
            "This uploaded file contains 'Year' and 'SMW' with data from more than one year, so it is being treated as a multi-year population dynamics dataset."
        )
        st.info(
            "Single-Year Population Dynamics is not run for multi-year files because it can incorrectly select 'Year' as a variable and produce misleading tables. Please use the Multi-Year Population Dynamics tab, the Journal-Level Model tab, or upload/filter data for one year only."
        )
    else:
        render_single_year_analysis(df)

with multi_year_tab:
    if df is None:
        st.info("Upload a long-format CSV or Excel file containing Year, SMW, pest population, and weather variables.")
    else:
        render_multi_year_analysis(df)

with journal_tab:
    if df is None:
        st.info("Upload a long-format multi-year dataset to run journal-level models.")
    else:
        render_journal_level_model(df)

with student_guide_tab:
    render_student_requirement_guide()
