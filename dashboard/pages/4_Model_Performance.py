
# ============================================================
# ODISHA RICE YIELD ANALYTICS
# PAGE 4 — MODEL PERFORMANCE
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from utils import (
    load_dataset,
    load_model,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    BASELINE_METRICS
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Model Performance | Odisha Rice",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Performance Evaluation")

st.caption(
    "Evaluation of the trained Random Forest regression model "
    "using a time-aware training and testing split."
)

st.info(
    "This page evaluates the existing trained model. "
    "No model retraining or modification is performed."
)


# ============================================================
# 2. LOAD DATA AND MODEL
# ============================================================

try:
    data = load_dataset()
    model = load_model()

except Exception as e:
    st.error(f"Unable to load project resources: {e}")
    st.stop()


# ============================================================
# 3. RECREATE ORIGINAL TIME-AWARE SPLIT
# ============================================================

train_data = data[data["year"] <= 2021].copy()

test_data = data[data["year"] >= 2022].copy()

X_train = train_data[FEATURE_COLUMNS]
y_train = train_data[TARGET_COLUMN]

X_test = test_data[FEATURE_COLUMNS]
y_test = test_data[TARGET_COLUMN]


# ============================================================
# 4. GENERATE PREDICTIONS
# ============================================================

try:
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

except Exception as e:
    st.error(f"Prediction generation failed: {e}")
    st.stop()


# ============================================================
# 5. CALCULATE PERFORMANCE METRICS
# ============================================================

def calculate_metrics(actual, predicted):

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mae = mean_absolute_error(
        actual, predicted
    )

    r2 = r2_score(
        actual, predicted
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


train_metrics = calculate_metrics(
    y_train, train_predictions
)

test_metrics = calculate_metrics(
    y_test, test_predictions
)


# ============================================================
# 6. PREPARE EVALUATION DATAFRAME
# ============================================================

train_results = pd.DataFrame({
    "Year": train_data["year"].values,
    "Actual": y_train.values,
    "Predicted": train_predictions,
    "Residual": (
        y_train.values - train_predictions
    ),
    "Dataset": "Training"
})

test_results = pd.DataFrame({
    "Year": test_data["year"].values,
    "Actual": y_test.values,
    "Predicted": test_predictions,
    "Residual": (
        y_test.values - test_predictions
    ),
    "Dataset": "Testing"
})

all_results = pd.concat(
    [train_results, test_results],
    ignore_index=True
)


# ============================================================
# 7. DATASET OVERVIEW
# ============================================================

st.divider()

st.subheader("1. Evaluation Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Observations",
    f"{len(data):,}"
)

c2.metric(
    "Training Observations",
    f"{len(train_data):,}"
)

c3.metric(
    "Testing Observations",
    f"{len(test_data):,}"
)

c4.metric(
    "Input Features",
    len(FEATURE_COLUMNS)
)

st.caption(
    "Training period: 2014–2021 | "
    "Testing period: 2022–2024"
)


# ============================================================
# 8. MODEL PERFORMANCE METRICS
# ============================================================

st.divider()

st.subheader("2. Model Performance Metrics")

st.markdown(
    "The following metrics measure the prediction accuracy "
    "of the trained Random Forest model."
)

metric_table = pd.DataFrame({
    "Metric": ["RMSE (kg/ha)", "MAE (kg/ha)", "R² Score"],
    "Training": [
        train_metrics["RMSE"],
        train_metrics["MAE"],
        train_metrics["R2"]
    ],
    "Testing": [
        test_metrics["RMSE"],
        test_metrics["MAE"],
        test_metrics["R2"]
    ]
})

st.dataframe(
    metric_table.round(4),
    use_container_width=True,
    hide_index=True
)

m1, m2, m3 = st.columns(3)

m1.metric(
    "Test RMSE",
    f'{test_metrics["RMSE"]:,.2f} kg/ha'
)

m2.metric(
    "Test MAE",
    f'{test_metrics["MAE"]:,.2f} kg/ha'
)

m3.metric(
    "Test R²",
    f'{test_metrics["R2"]:.4f}'
)


# Verify test metrics against the original baseline

st.subheader("Baseline Verification")

rmse_match = np.isclose(
    test_metrics["RMSE"],
    BASELINE_METRICS["rmse"],
    atol=0.01
)

mae_match = np.isclose(
    test_metrics["MAE"],
    BASELINE_METRICS["mae"],
    atol=0.01
)

r2_match = np.isclose(
    test_metrics["R2"],
    BASELINE_METRICS["r2"],
    atol=0.0001
)

if rmse_match and mae_match and r2_match:

    st.success(
        "All three test metrics match the original "
        "verified baseline."
    )

else:

    st.warning(
        "The recalculated metrics do not fully match "
        "the original baseline. Check the dataset, "
        "saved model and evaluation split."
    )

    st.write({
        "RMSE matches": rmse_match,
        "MAE matches": mae_match,
        "R² matches": r2_match
    })


# ============================================================
# 9. ACTUAL VS PREDICTED
# ============================================================

st.divider()

st.subheader("3. Actual vs. Predicted Rice Yield")

st.markdown(
    "Points closer to the diagonal reference line represent "
    "more accurate predictions."
)

fig_actual = go.Figure()

fig_actual.add_trace(
    go.Scatter(
        x=y_test,
        y=test_predictions,
        mode="markers",
        name="Test observations",
        marker=dict(
            color="#22A06B",
            size=9,
            opacity=0.7
        )
    )
)

min_value = min(
    float(y_test.min()),
    float(test_predictions.min())
)

max_value = max(
    float(y_test.max()),
    float(test_predictions.max())
)

fig_actual.add_trace(
    go.Scatter(
        x=[min_value, max_value],
        y=[min_value, max_value],
        mode="lines",
        name="Perfect prediction",
        line=dict(
            color="#FF6B6B",
            dash="dash"
        )
    )
)

fig_actual.update_layout(
    xaxis_title="Actual Rice Yield (kg/ha)",
    yaxis_title="Predicted Rice Yield (kg/ha)",
    template="plotly_dark",
    height=500,
    legend_title="Legend"
)

st.plotly_chart(
    fig_actual,
    use_container_width=True
)


# ============================================================
# 10. RESIDUAL ANALYSIS
# ============================================================

st.divider()

st.subheader("4. Residual Analysis")

st.markdown(
    "Residual = Actual Yield − Predicted Yield. "
    "Positive residuals indicate underprediction, "
    "while negative residuals indicate overprediction."
)

fig_residual = go.Figure()

fig_residual.add_trace(
    go.Scatter(
        x=test_predictions,
        y=test_results["Residual"],
        mode="markers",
        marker=dict(
            color="#4C9BE8",
            size=9,
            opacity=0.7
        ),
        name="Test residuals"
    )
)

fig_residual.add_hline(
    y=0,
    line_dash="dash",
    line_color="#FF6B6B"
)

fig_residual.update_layout(
    xaxis_title="Predicted Rice Yield (kg/ha)",
    yaxis_title="Residual (kg/ha)",
    template="plotly_dark",
    height=450
)

st.plotly_chart(
    fig_residual,
    use_container_width=True
)


# ============================================================
# 11. RESIDUAL DISTRIBUTION
# ============================================================

st.subheader("5. Distribution of Prediction Errors")

fig_hist = px.histogram(
    test_results,
    x="Residual",
    nbins=25,
    title="Distribution of Test Residuals",
    labels={
        "Residual": "Residual (kg/ha)"
    },
    template="plotly_dark"
)

fig_hist.add_vline(
    x=0,
    line_dash="dash",
    line_color="#FF6B6B"
)

fig_hist.update_layout(
    height=400
)

st.plotly_chart(
    fig_hist,
    use_container_width=True
)


# ============================================================
# 12. YEAR-WISE TEST PERFORMANCE
# ============================================================

st.divider()

st.subheader("6. Year-wise Test Performance")

year_rows = []

for year in sorted(test_results["Year"].unique()):

    year_data = test_results[
        test_results["Year"] == year
    ]

    year_rows.append({
        "Year": int(year),
        "Observations": len(year_data),
        "RMSE": np.sqrt(
            mean_squared_error(
                year_data["Actual"],
                year_data["Predicted"]
            )
        ),
        "MAE": mean_absolute_error(
            year_data["Actual"],
            year_data["Predicted"]
        ),
        "R²": (
            r2_score(
                year_data["Actual"],
                year_data["Predicted"]
            )
            if len(year_data) >= 2
            else np.nan
        )
    })

year_metrics = pd.DataFrame(year_rows)

st.dataframe(
    year_metrics.round(3),
    use_container_width=True,
    hide_index=True
)

fig_year = px.bar(
    year_metrics,
    x="Year",
    y="RMSE",
    text_auto=".1f",
    title="Test RMSE by Year",
    labels={
        "RMSE": "RMSE (kg/ha)"
    },
    template="plotly_dark"
)

fig_year.update_layout(
    height=400
)

st.plotly_chart(
    fig_year,
    use_container_width=True
)


# ============================================================
# 13. LARGEST PREDICTION ERRORS
# ============================================================

st.divider()

st.subheader("7. Observations with the Largest Prediction Errors")

error_table = test_results.copy()

error_table["Absolute Error"] = (
    error_table["Residual"].abs()
)

error_table = error_table.sort_values(
    "Absolute Error",
    ascending=False
)

st.dataframe(
    error_table[
        [
            "Year",
            "Actual",
            "Predicted",
            "Residual",
            "Absolute Error"
        ]
    ].head(10).round(2),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 14. RESEARCH INTERPRETATION
# ============================================================

st.divider()

st.subheader("8. Research Interpretation")

st.markdown(
    f"""
    - The model was trained using **{len(train_data)} observations**
      from 2014–2021.
    - Evaluation was performed on **{len(test_data)} observations**
      from 2022–2024.
    - The test RMSE is **{test_metrics['RMSE']:,.2f} kg/ha**.
    - The test MAE is **{test_metrics['MAE']:,.2f} kg/ha**.
    - The test R² score is **{test_metrics['R2']:.4f}**.

    The test results represent the model's performance on later
    years that were excluded from training.
    """
)

st.warning(
    "The year-wise metrics are descriptive. Individual years "
    "contain relatively few observations, so their R² values "
    "may be unstable or undefined."
)

with st.expander("Important Evaluation Limitations"):

    st.markdown(
        """
        1. This is a time-aware evaluation, not random splitting.
        2. The test period contains later years than the training period.
        3. Evaluation metrics describe predictive performance, not causality.
        4. A positive R² does not guarantee accurate predictions
           for every district or season.
        5. Kharif was not represented in the original training seasons.
           Predictions involving Kharif require additional caution.
        6. These results should be interpreted as an academic
           research evaluation, not guaranteed agricultural outcomes.
        """
    )


# ============================================================
# 15. DOWNLOAD TEST PREDICTIONS
# ============================================================

st.divider()

st.subheader("9. Download Evaluation Results")

download_data = test_results.copy()

csv_data = download_data.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Test Predictions CSV",
    data=csv_data,
    file_name="rice_yield_test_predictions.csv",
    mime="text/csv"
)


# ============================================================
# 16. FOOTER
# ============================================================

st.divider()

st.caption(
    "Odisha Rice Yield Analytics | Model Performance Evaluation"
)

st.caption(
    "Predictions are estimates based on historical data "
    "and should not be interpreted as guaranteed outcomes."
)
