
# ============================================================
# PHASE 5.7 — RICE YIELD PREDICTION
# ============================================================

import streamlit as st
import pandas as pd

from utils import (
    load_dataset,
    load_model,
    load_metadata,
    predict_yield,
    is_unseen_season,
    format_yield,
    NUMERICAL_FEATURES,
    TRAINING_SEASONS
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Yield Prediction | Odisha Rice",
    page_icon="🌾",
    layout="wide"
)


# ============================================================
# LOAD DATA AND METADATA
# ============================================================

try:
    df = load_dataset()
    metadata = load_metadata()
    model = load_model()

except Exception as e:
    st.error(f"Unable to load prediction resources: {e}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🌾 Rice Yield Prediction")

st.markdown(
    """
    Estimate rice yield using the trained Random Forest regression
    model. Enter the district, season and agricultural conditions
    to generate a prediction.
    """
)

st.info(
    "This prediction uses the original trained model and its "
    "12 input features. The model is not retrained on this page."
)


# ============================================================
# SIDEBAR — INPUT INFORMATION
# ============================================================

st.sidebar.header("Prediction Information")

st.sidebar.caption(
    "Enter values that represent the agricultural conditions "
    "for the observation you want to estimate."
)


# ============================================================
# FEATURE RANGES
# ============================================================

district_options = metadata["training_districts"]

season_options = (
    TRAINING_SEASONS + ["Kharif"]
)

def feature_range(feature):

    minimum = float(df[feature].min())
    maximum = float(df[feature].max())
    median = float(df[feature].median())

    return minimum, maximum, median


# ============================================================
# INPUT FORM
# ============================================================

st.subheader("Enter Agricultural Conditions")

with st.form("yield_prediction_form"):

    st.markdown("### 1. Location and Season")

    col1, col2 = st.columns(2)

    with col1:

        district = st.selectbox(
            "District",
            options=district_options
        )

    with col2:

        season = st.selectbox(
            "Season",
            options=season_options
        )

    st.markdown("### 2. Agricultural and Environmental Features")

    st.caption(
        "Values are initially populated using the dataset median. "
        "Adjust them to match your intended scenario."
    )

    input_values = {}

    feature_labels = {
        "area": ("Cultivated Area (hectares)", 2),
        "year": ("Year", 0),
        "avg_temperature_c": ("Average Temperature (°C)", 2),
        "rainfall_mm": ("Rainfall (mm)", 2),
        "humidity_pct": ("Humidity (%)", 2),
        "soil_ph": ("Soil pH", 2),
        "nitrogen_kg_ha": ("Nitrogen (kg/ha)", 2),
        "phosphorus_kg_ha": ("Phosphorus (kg/ha)", 2),
        "potassium_kg_ha": ("Potassium (kg/ha)", 2),
        "irrigation_pct": ("Irrigation (%)", 2)
    }

    left, right = st.columns(2)

    for index, feature in enumerate(NUMERICAL_FEATURES):

        minimum, maximum, median = feature_range(feature)

        label, decimals = feature_labels[feature]

        column = left if index % 2 == 0 else right

        with column:

            if feature == "year":

                value = st.number_input(
                    label,
                    min_value=int(minimum),
                    max_value=int(maximum),
                    value=int(median),
                    step=1
                )

            else:

                value = st.number_input(
                    label,
                    min_value=minimum,
                    max_value=maximum,
                    value=median,
                    step=0.1,
                    format=f"%.{decimals}f"
                )

            input_values[feature] = value

    st.write("")

    submitted = st.form_submit_button(
        "🌾 Predict Rice Yield",
        use_container_width=True,
        type="primary"
    )


# ============================================================
# GENERATE PREDICTION
# ============================================================

if submitted:

    prediction_input = {
        "area": input_values["area"],
        "year": input_values["year"],
        "avg_temperature_c": input_values["avg_temperature_c"],
        "rainfall_mm": input_values["rainfall_mm"],
        "humidity_pct": input_values["humidity_pct"],
        "soil_ph": input_values["soil_ph"],
        "nitrogen_kg_ha": input_values["nitrogen_kg_ha"],
        "phosphorus_kg_ha": input_values["phosphorus_kg_ha"],
        "potassium_kg_ha": input_values["potassium_kg_ha"],
        "irrigation_pct": input_values["irrigation_pct"],
        "district": district,
        "season": season
    }

    try:

        with st.spinner("Generating yield prediction..."):

            predicted_yield = float(
                predict_yield(prediction_input)[0]
            )

        st.divider()

        st.subheader("Prediction Result")

        st.metric(
            "Estimated Rice Yield",
            format_yield(predicted_yield)
        )

        st.success(
            f"Estimated yield for {district}, {season}: "
            f"{predicted_yield:,.2f} kg/ha"
        )

        if is_unseen_season(season):

            st.warning(
                "Kharif was not present in the model's training "
                "data. Its season category is therefore unseen by "
                "the trained encoder. The prediction is exploratory "
                "and should be interpreted with additional caution."
            )

        else:

            st.info(
                f"The selected season ({season}) was represented "
                "in the model's training data."
            )

        st.markdown("### Input Summary")

        summary = pd.DataFrame(
            [prediction_input]
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "The predicted value is a statistical model estimate. "
            "It is not a guarantee of actual agricultural yield."
        )

    except Exception as e:

        st.error(
            f"Prediction could not be generated: {e}"
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("About the prediction model"):

    st.write(
        "Model: Random Forest Regressor"
    )

    st.write(
        f"Number of trees: "
        f"{metadata['rf_params']['n_estimators']}"
    )

    st.write(
        f"Maximum tree depth: "
        f"{metadata['rf_params']['max_depth']}"
    )

    st.write(
        "Original training period: 2014–2021"
    )

    st.write(
        "Evaluation period: 2022–2024"
    )

    st.write(
        "Test RMSE: 541.77 kg/ha"
    )

    st.write(
        "Test MAE: 435.91 kg/ha"
    )

    st.write(
        "Test R²: 0.4165"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Odisha Rice Yield Analytics | Random Forest Prediction"
)

st.caption(
    "Predictions are estimates based on historical training data "
    "and should not be interpreted as causal or guaranteed outcomes."
)
