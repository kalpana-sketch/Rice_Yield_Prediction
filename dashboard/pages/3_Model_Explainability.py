
# ============================================================
# PHASE 5.8 — MODEL EXPLAINABILITY DASHBOARD
# SHAP + LIME
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Model Explainability | Odisha Rice",
    page_icon="🔍",
    layout="wide"
)

# Project paths
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
LIME_DIR = ARTIFACTS / "lime"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(path):

    if path.exists():
        return pd.read_csv(path)

    return None


def show_figure(filename, caption=None, width="stretch"):

    path = FIGURES / filename

    if path.exists():
        st.image(str(path), caption=caption, width=width)
    else:
        st.warning(f"Figure not found: {filename}")


def show_lime_figure(pattern, caption):

    matches = list(FIGURES.glob(pattern))

    if matches:
        st.image(str(matches[0]), caption=caption, width="stretch")
    else:
        st.warning(f"LIME figure not found: {pattern}")


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

grouped_train = load_csv(
    ARTIFACTS / "shap_importance_grouped_train.csv"
)

grouped_test = load_csv(
    ARTIFACTS / "shap_importance_grouped_test.csv"
)

season_train = load_csv(
    ARTIFACTS / "shap_by_season_train.csv"
)

season_test = load_csv(
    ARTIFACTS / "shap_by_season_test.csv"
)

district_train = load_csv(
    ARTIFACTS / "shap_by_district_train.csv"
)

waterfall_cases = load_csv(
    ARTIFACTS / "waterfall_cases.csv"
)

lime_summary = load_csv(
    LIME_DIR / "lime_cases_summary.csv"
)


# ============================================================
# HEADER
# ============================================================

st.title("🔍 Model Explainability")

st.markdown(
    """
    Understand how the trained Random Forest model generates
    rice yield predictions using SHAP and LIME.

    This page presents global feature importance, feature
    contributions, and explanations of individual predictions.
    """
)

st.info(
    "All explanations are based on the previously verified "
    "model and saved Phase 2–4 artifacts. No model retraining "
    "or new SHAP/LIME calculations are performed here."
)


# ============================================================
# SECTION 1 — GLOBAL SHAP FEATURE IMPORTANCE
# ============================================================

st.header("1. Global Feature Importance")

st.markdown(
    """
    SHAP measures how features contribute to the predictions
    made by the trained model.

    Method B is used as the primary grouped importance measure:
    it sums signed encoded-feature contributions within each
    original variable before calculating their average absolute value.
    """
)

if grouped_train is not None and grouped_test is not None:

    train_plot = grouped_train[
        ["variable", "B_mean_abs_of_group_sum"]
    ].copy()

    test_plot = grouped_test[
        ["variable", "B_mean_abs_of_group_sum"]
    ].copy()

    train_plot["Dataset"] = "Training"
    test_plot["Dataset"] = "Test"

    combined = pd.concat(
        [train_plot, test_plot],
        ignore_index=True
    )

    combined = combined.rename(
        columns={
            "B_mean_abs_of_group_sum": "Mean Absolute SHAP (kg/ha)"
        }
    )

    fig = px.bar(
        combined,
        x="Mean Absolute SHAP (kg/ha)",
        y="variable",
        color="Dataset",
        barmode="group",
        orientation="h",
        title="Grouped SHAP Importance: Training vs Test",
        labels={"variable": "Original Feature"}
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View grouped SHAP importance tables"):

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Training")
            st.dataframe(
                grouped_train,
                use_container_width=True,
                hide_index=True
            )

        with col2:
            st.subheader("Test")
            st.dataframe(
                grouped_test,
                use_container_width=True,
                hide_index=True
            )

else:
    st.warning("Grouped SHAP CSV files are missing.")


# ============================================================
# SECTION 2 — ENCODED FEATURE IMPORTANCE
# ============================================================

st.header("2. Encoded Feature Importance")

st.markdown(
    """
    The original 12 variables are transformed into 43 encoded
    features by the preprocessing pipeline.

    This view shows the importance of individual encoded columns.
    """
)

show_figure(
    "shap_bar_encoded_test.png",
    "Top encoded SHAP features — test observations"
)

show_figure(
    "shap_bar_encoded_train.png",
    "Top encoded SHAP features — training observations"
)


# ============================================================
# SECTION 3 — SHAP BEESWARM
# ============================================================

st.header("3. SHAP Summary — Beeswarm Plots")

st.markdown(
    """
    Beeswarm plots show the distribution of SHAP contributions
    across observations.

    - Positive SHAP values push predictions upward.
    - Negative SHAP values push predictions downward.
    - Each point represents an observation.
    """
)

tab1, tab2 = st.tabs(["Test Data", "Training Data"])

with tab1:

    show_figure(
        "shap_beeswarm_test.png",
        "SHAP beeswarm — test data"
    )

    show_figure(
        "shap_beeswarm_numeric_test.png",
        "SHAP beeswarm — numerical features, test data"
    )

with tab2:

    show_figure(
        "shap_beeswarm_train.png",
        "SHAP beeswarm — training data"
    )

    show_figure(
        "shap_beeswarm_numeric_train.png",
        "SHAP beeswarm — numerical features, training data"
    )


# ============================================================
# SECTION 4 — FEATURE DEPENDENCE
# ============================================================

st.header("4. SHAP Feature Dependence")

st.markdown(
    """
    This plot explores how numerical feature values relate to
    their SHAP contributions in the model.
    """
)

show_figure(
    "shap_dependence_top_numeric.png",
    "SHAP dependence plots for selected numerical features"
)

st.caption(
    "These relationships describe model behaviour and do not "
    "establish causal effects."
)


# ============================================================
# SECTION 5 — SEASON-WISE SHAP ANALYSIS
# ============================================================

st.header("5. Season-wise SHAP Analysis")

if season_train is not None and season_test is not None:

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Training seasons")

        show_figure(
            "shap_season_heatmap_train.png",
            "Season-wise mean grouped SHAP contributions — training"
        )

    with col2:

        st.subheader("Test seasons")

        show_figure(
            "shap_season_heatmap_test.png",
            "Season-wise mean grouped SHAP contributions — test"
        )

    with st.expander("View season-wise SHAP values"):

        st.subheader("Training")
        st.dataframe(
            season_train,
            use_container_width=True
        )

        st.subheader("Test")
        st.dataframe(
            season_test,
            use_container_width=True
        )

else:
    st.warning("Season-wise SHAP files are missing.")


# ============================================================
# SECTION 6 — DISTRICT-WISE SHAP ANALYSIS
# ============================================================

st.header("6. District-wise SHAP Analysis")

st.markdown(
    """
    This section displays average grouped SHAP contributions
    across districts in the training observations.
    """
)

if district_train is not None:

    show_figure(
        "shap_district_heatmap_train.png",
        "District-wise mean grouped SHAP contributions — training"
    )

    with st.expander("View district-wise SHAP values"):

        st.dataframe(
            district_train,
            use_container_width=True
        )

else:
    st.warning("District-wise SHAP data is missing.")


# ============================================================
# SECTION 7 — INDIVIDUAL SHAP EXPLANATIONS
# ============================================================

st.header("7. Individual Prediction Explanations")

st.markdown(
    """
    Waterfall plots explain how individual feature contributions
    move a prediction away from the SHAP base value.
    """
)

if waterfall_cases is not None:

    case_names = waterfall_cases["case"].tolist()

    selected_case = st.selectbox(
        "Select an observation",
        case_names
    )

    selected_row = waterfall_cases[
        waterfall_cases["case"] == selected_case
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Actual Yield",
        f"{selected_row['actual']:,.2f} kg/ha"
    )

    col2.metric(
        "Predicted Yield",
        f"{selected_row['predicted']:,.2f} kg/ha"
    )

    col3.metric(
        "Prediction Error",
        f"{selected_row['error']:+,.2f} kg/ha"
    )

    row_number = int(selected_row["row"])

    if row_number == 92:
        figure_name = "waterfall_grouped_largest_underprediction.png"

    elif row_number == 59:
        figure_name = "waterfall_grouped_largest_overprediction.png"

    elif row_number == 50:
        figure_name = "waterfall_grouped_typical_row.png"

    else:
        figure_name = "waterfall_grouped_Kharif_row.png"

    show_figure(
        figure_name,
        f"Grouped SHAP waterfall — {selected_case}"
    )

    st.caption(
        "Positive contributions increase the prediction relative "
        "to the base value; negative contributions decrease it."
    )

else:
    st.warning("Waterfall case summary is missing.")


# ============================================================
# SECTION 8 — LIME LOCAL EXPLANATIONS
# ============================================================

st.header("8. LIME Local Explanations")

st.markdown(
    """
    LIME explains an individual prediction by fitting a local
    interpretable model around the selected observation.

    The following examples are the four observations analysed
    during Phase 3.
    """
)

if lime_summary is not None:

    lime_cases = lime_summary["case"].tolist()

    selected_lime_case = st.selectbox(
        "Select a LIME observation",
        lime_cases,
        key="lime_case_selector"
    )

    lime_row = lime_summary[
        lime_summary["case"] == selected_lime_case
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Model Prediction",
        f"{lime_row['model_prediction']:,.2f} kg/ha"
    )

    col2.metric(
        "LIME Local R²",
        f"{lime_row['local_r2']:.3f}"
    )

    col3.metric(
        "District",
        str(lime_row["district"])
    )

    row_number = int(lime_row["row"])

    if row_number == 92:
        pattern = "lime_bar_largest_underprediction_row92.png"

    elif row_number == 59:
        pattern = "lime_bar_largest_overprediction_row59.png"

    elif row_number == 50:
        pattern = "lime_bar_typical_row_row50.png"

    else:
        pattern = "lime_bar_Kharif_row_row64.png"

    show_lime_figure(
        pattern,
        f"LIME explanation — {selected_lime_case}"
    )

    if row_number in [50, 64]:

        st.warning(
            "Kharif was not present in the training data. "
            "The season was held fixed during LIME sampling. "
            "Its missing LIME weight must not be interpreted "
            "as absence of model influence."
        )

    if lime_row["local_r2"] < 0.20:

        st.warning(
            "This LIME explanation has a weak local fit. "
            "Interpret its feature weights with caution."
        )

    elif lime_row["local_r2"] < 0.70:

        st.info(
            "This LIME explanation has a moderate local fit. "
            "It is an approximation of the original model."
        )

else:
    st.warning("LIME case summary is missing.")


# ============================================================
# SECTION 9 — SHAP AND LIME LIMITATIONS
# ============================================================

st.header("9. Interpretation and Limitations")

with st.expander("Important research limitations", expanded=True):

    st.markdown(
        """
        **SHAP limitations**

        1. SHAP explains model behaviour, not causal relationships.
        2. Correlated variables may share explanatory contributions.
        3. Training explanations are in-sample.
        4. Test explanations are based on 174 observations.
        5. Kharif was absent from the training seasons. Its encoded
           season contribution is not a learned Kharif effect.
        6. The primary SHAP method is Tree-Path-Dependent.
           The interventional SHAP cross-check did not pass
           the additivity verification and is not used here.

        **LIME limitations**

        1. LIME is a local surrogate approximation.
        2. Explanations depend on sampling and kernel settings.
        3. Independent feature sampling may create unrealistic
           combinations of agricultural conditions.
        4. Some selected observations have weak local R².
        5. Kharif was held fixed during LIME sampling.

        **Research interpretation**

        These results describe associations learned by the
        Random Forest model. They should not be interpreted as
        causal effects or agricultural recommendations.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Odisha Rice Yield Analytics | SHAP and LIME Explainability"
)

st.caption(
    "Explanations describe the trained model's behaviour, "
    "not causal relationships."
)
