# Odisha Rice Yield Prediction

A machine-learning research project for analyzing and predicting
**district-level rice yield in Odisha** using historical agricultural,
climate, soil, nutrient, irrigation, and seasonal information.

The project includes data preparation and exploratory analysis,
leakage-aware machine-learning workflows, time-aware model evaluation,
explainable AI using SHAP and LIME, robustness checks, and an
interactive Streamlit dashboard.

## 🚀 Live Dashboard

**[Open Odisha Rice Yield
Analytics](https://riceyieldprediction-wgowxbc5trijckm4fai95q.streamlit.app/)**

The dashboard provides: - Historical rice-yield analysis across Odisha
districts and seasons. - District-level filters and agricultural trend
visualizations. - Yield prediction using the saved Random Forest
pipeline. - SHAP- and LIME-based model explanations. - Model performance
and evaluation results.

## 📌 Project Objective

The objective is to predict **rice yield in kg/hectare (`yield_kg_ha`)**
for Odisha districts and seasons using historical agricultural and
environmental information.

The project is framed as a **future-year prediction problem**.
Therefore, the evaluation methodology avoids random train/test splitting
and uses chronological validation to better represent prediction on
future years.

## 📂 Repository Structure

``` text
Rice_Yield_Prediction/
│
├── artifacts/
│   ├── final_model_rf.joblib
│   ├── model metadata and evaluation outputs
│   ├── SHAP analysis outputs
│   ├── LIME explanation outputs
│   └── robustness analysis outputs
│
├── dashboard/
│   ├── app.py
│   ├── utils.py
│   └── pages/
│       ├── 1_District_Analysis.py
│       ├── 2_Yield_Prediction.py
│       ├── 3_Model_Explainability.py
│       └── 4_Model_Performance.py
│
├── Odisha_Rice_ML_Audited_Corrected.ipynb
├── clean_rice.csv
├── metadata.csv
├── odisha_rice_integrated_eda_clean.csv
├── Integrated_Odisha_Rice_Yield_Dataset_2014_2024.csv
├── district-season-and-crop-wise-area-production-and-yield-statistics-for-odisha.csv
├── requirements.txt
├── .gitignore
└── README.md
```

The `artifacts/` directory contains the saved model and outputs required
by the dashboard. The dashboard loads these existing artifacts; it does
not retrain the model when a user opens the app.

## 📊 Data Preparation & Integration

Two datasets are integrated:

### Dataset 1

District-, season-, and crop-wise agricultural statistics for Odisha,
including area, production, and yield.

### Dataset 2

An integrated Odisha rice dataset containing environmental and
agricultural variables such as rainfall, temperature, humidity, soil pH,
nutrients, and irrigation.

The datasets are connected using: - `fiscal_year` -
`district_lgd_code` - `crop` - `season`

Only **Rice** observations are retained. `State Total` records are
removed before merging because the project focuses on district-level
observations.

The final inner merge produces **869 matched district-level
observations**. After removing observations with missing target values,
**839 observations** remain for supervised machine learning.

## 🎯 Target Variable

The modeling target is:

``` text
yield_kg_ha
```

It is calculated consistently as:

``` text
Yield (kg/ha) = Production (tonnes) × 1000 / Area (ha)
```

This was used instead of the raw `reported_crop_yield` field because the
source data contains inconsistent unit representation.

## 🔎 Exploratory Data Analysis

The data preparation and EDA work covers: - Dataset structure and data
types - Missing-value analysis - Duplicate checks - District-name
validation - Merge validation - Target distribution - Outlier analysis
using the IQR rule - Year-wise observations - Yield trends over time -
Seasonal yield analysis - District-level performance - Climate and soil
feature distributions - Feature-vs-yield relationships - Area
vs. yield - Rainfall and temperature trends - Soil pH and N/P/K
analysis - Irrigation analysis - Correlation analysis - Area
vs. production validation

### Important EDA decisions

Outliers are not automatically removed because unusual agricultural
yields may represent genuine observations.

`production` is retained for data validation and EDA but is excluded
from the ML feature set because it is mathematically related to the
target.

## 🧠 Machine Learning

This is a **regression problem** because the target is a continuous
numerical value.

### Features

The ML pipeline uses 12 raw predictors.

#### Numerical features

-   `area`
-   `year`
-   `avg_temperature_c`
-   `rainfall_mm`
-   `humidity_pct`
-   `soil_ph`
-   `nitrogen_kg_ha`
-   `phosphorus_kg_ha`
-   `potassium_kg_ha`
-   `irrigation_pct`

#### Categorical features

-   `district`
-   `season`

### Excluded variables

-   `yield_kg_ha` --- target
-   `production` --- target leakage
-   `reported_crop_yield` --- redundant/target-derived yield field
-   `district_lgd_code` --- identifier
-   `district_source` --- redundant district-name field
-   `fiscal_year` --- redundant with `year`

## ⚙️ Preprocessing

The ML workflow uses a `ColumnTransformer` and `Pipeline`.

### Numerical data

`StandardScaler`

### Categorical data

`OneHotEncoder(handle_unknown="ignore")`

The complete preprocessing and model are kept inside a pipeline to avoid
data leakage during cross-validation.

The 12 raw predictors become **43 features after preprocessing/one-hot
encoding**.

## 🕒 Chronological Train/Test Split

A random train/test split is intentionally avoided.

  Dataset             Years   Observations
  ------------ ------------ --------------
  Training       2014--2021            665
  Final test     2022--2024            174
  Total          2014--2024            839

This better represents the intended use case:

**Train on historical years → predict future years.**

## 🔁 Time-Aware Cross-Validation

Model selection uses **year-aware forward validation**.

The validation folds keep complete years together, ensuring that a
validation year occurs after its corresponding training years. Five
time-aware folds are used.

Hyperparameter tuning is performed using `RandomizedSearchCV` on the
training period only. The final 2022--2024 test set is kept separate
from model selection.

## 🤖 Models Compared

The project compares: - Linear Regression - Decision Tree Regressor -
Random Forest Regressor - Gradient Boosting Regressor

Random Forest and Gradient Boosting are additionally tuned using
time-aware randomized search.

Model selection is based on the **lowest mean validation RMSE**, with
validation MAE used as a secondary criterion.

## 🏆 Final Model & Performance

The final selected model is a **Tuned Random Forest Regressor**.

  Metric                      Result
  ------------------- --------------
  Selection CV RMSE     639.40 kg/ha
  Test RMSE             541.77 kg/ha
  Test MAE              435.91 kg/ha
  Test R²                     0.4165

The final model is selected using training-period cross-validation
rather than choosing the model based on final test performance.

## 🔬 Explainable AI & Robustness Analysis

The project extends beyond predictive performance by examining how the
fitted model produces its predictions.

### Phase 2 --- SHAP Analysis

SHAP (SHapley Additive exPlanations) is used to examine global and local
feature contributions.

The analysis includes: - Global feature importance. - Train and test
feature-ranking comparison. - Grouped and encoded feature-level
comparisons. - Additivity checks. - Comparison of Tree-Path-Dependent
and Interventional explainers.

**Tree-Path-Dependent SHAP** is used as the primary explanation method.
The Interventional approach was not selected as primary because its
additivity checks showed larger discrepancies for this fitted pipeline.

The leading global features included season, year, district, area, and
humidity. Feature contributions describe model behaviour; they do not
establish causal relationships.

### Phase 3 --- LIME Analysis

LIME (Local Interpretable Model-agnostic Explanations) is used to
explain individual predictions.

The workflow: - Checks that the LIME prediction wrapper matches the
saved model pipeline. - Generates local explanations for selected
examples. - Examines both overpredictions and underpredictions. -
Includes examples from different districts and seasons. - Records local
fidelity scores and the limitations of each explanation.

Some selected examples have low local R². These explanations should
therefore be interpreted cautiously rather than treated as highly
faithful summaries of the model.

### Phase 4 --- Robustness Analysis

The robustness analysis examines: - SHAP explanation consistency across
training and test data. - Agreement between grouped and encoded SHAP
feature rankings. - LIME sensitivity to random seeds, sample counts, and
kernel widths. - Directional agreement between eligible SHAP and LIME
local explanations.

The analysis found strong agreement in global SHAP feature rankings
across the evaluated comparisons. Local explanations were less
consistent, reinforcing the need to interpret individual explanations
with care.

Detailed outputs and verification summaries are saved in `artifacts/`.

## ⚠️ Important Data-Quality Finding

The training data contains these season labels:

``` text
Autumn
Summer
Winter
```

The test period also contains:

``` text
Kharif
```

There are **60 test observations** with the previously unseen `Kharif`
label.

Because the categorical encoder uses:

``` python
OneHotEncoder(handle_unknown="ignore")
```

the model continues to run, but those observations receive no learned
season-specific signal. The dashboard also communicates this limitation
when relevant.

This is an important limitation and a potential area for future feature
engineering.

## 📊 Interactive Streamlit Dashboard

The project includes a multi-page dashboard built with Streamlit.

### Dashboard pages

1.  **Overview** --- project summary, dataset statistics, and
    selected-data overview.
2.  **District Analysis** --- historical yield patterns and
    district-level comparisons.
3.  **Yield Prediction** --- interactive prediction using the saved
    Random Forest pipeline.
4.  **Model Explainability** --- SHAP and LIME explanation outputs.
5.  **Model Performance** --- model evaluation metrics and performance
    visualizations.

### Run the dashboard locally

Clone the repository:

``` bash
git clone https://github.com/kalpana-sketch/Rice_Yield_Prediction.git
cd Rice_Yield_Prediction
```

Create and activate a virtual environment:

``` bash
python -m venv .venv
```

On Windows PowerShell:

``` powershell
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

``` bash
pip install -r requirements.txt
```

Start Streamlit from the repository root:

``` bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser, usually at
`http://localhost:8501`.

### Live deployment

The dashboard is deployed using Streamlit Community Cloud:

**<https://riceyieldprediction-wgowxbc5trijckm4fai95q.streamlit.app/>**

## 🚧 Limitations

### 1. Limited dataset size

The modeling dataset contains 839 observations, which is relatively
small for learning detailed district- and season-specific patterns.

### 2. Limited temporal history

Only 2014--2024 is available, with 2022--2024 used as the final holdout
period.

### 3. Season-label drift

`Kharif` appears in the test period but not in the training period.

### 4. District-level aggregation

Environmental and soil variables are available at district level rather
than finer spatial levels such as blocks or farms.

### 5. Missing explanatory variables

The dataset does not include potentially important variables such as: -
Crop variety - Pest/disease incidence - Exact sowing/harvest dates -
Extreme weather events - More detailed soil characteristics

### 6. Future-year uncertainty

The test set represents only three future years relative to the training
period. Performance on years beyond 2024 is not established.

### 7. Moderate explanatory power

The final Test R² of **0.4165** means that a substantial portion of the
variation in the holdout data remains unexplained.

Therefore, this model should be treated as a **research/academic
prediction model**, not as a production agricultural decision system.

### 8. Explainability limitations

SHAP and LIME explain model behaviour, not causal relationships. Local
LIME fidelity varies by observation, and explanations should be
interpreted with the reported robustness checks and limitations in mind.

## 🛠️ Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   SHAP
-   LIME
-   Joblib
-   Streamlit
-   Plotly
-   Jupyter Notebook

## 📦 Dependencies

The pinned Python dependencies used by the dashboard are listed in
[`requirements.txt`](requirements.txt).

Install them with:

``` bash
pip install -r requirements.txt
```

## ✅ Methodology Verification

The ML workflow includes automated checks for: - Target leakage -
Feature consistency - Chronological train/test separation -
Preprocessing pipeline structure - Year-aware cross-validation -
Hyperparameter tuning - Final model selection - Metric calculations -
Feature-importance dimensions - Residual calculations

The final verification reported:

``` text
All audited checks passed.
```

## 📌 Key Takeaway

This project demonstrates a complete machine-learning workflow for
Odisha rice-yield prediction:

``` text
Raw Data
   ↓
Data Cleaning
   ↓
Dataset Integration
   ↓
Exploratory Data Analysis
   ↓
Target Engineering
   ↓
Leakage Prevention
   ↓
Feature Preprocessing
   ↓
Chronological Train/Test Split
   ↓
Time-Aware Cross-Validation
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Final Model Selection
   ↓
Test Evaluation
   ↓
SHAP & LIME Explainability
   ↓
Robustness Analysis
   ↓
Interactive Streamlit Dashboard
```

The final tuned Random Forest achieves a **Test RMSE of 541.77 kg/ha**,
**Test MAE of 435.91 kg/ha**, and **Test R² of 0.4165** on the
2022--2024 holdout period.

The project emphasizes leakage-aware evaluation, transparent reporting
of limitations, explainability, and responsible interpretation of model
predictions.

## 👩‍💻 Authors

-   **Kalpana Pal (23052398)**
-   **Ridhima (23052415)**
-   **Mayank (23052401)**
-   **Naman**
-   **Mihir**
-   **Shauryan**

B.Tech --- Computer Science & Engineering

This project was developed as a machine-learning/data-science project
focused on agricultural yield prediction and responsible model
evaluation.

## 🔗 Project Links

-   **GitHub Repository:** [Rice Yield
    Prediction](https://github.com/kalpana-sketch/Rice_Yield_Prediction)
-   **Live Dashboard:** [Odisha Rice Yield
    Analytics](https://riceyieldprediction-wgowxbc5trijckm4fai95q.streamlit.app/)
