# Odisha Rice Yield Prediction

A machine learning project for analyzing and predicting **rice yield in
Odisha** using district-level agricultural, climate, soil, nutrient,
irrigation, and seasonal information.

The project is organized into two Jupyter notebooks:

1.  **EDA & Data Preparation** --- integrates the source datasets,
    validates the data, performs exploratory data analysis, and creates
    the ML-ready dataset.
2.  **Machine Learning** --- builds leakage-safe preprocessing
    pipelines, compares regression models, performs time-aware
    validation and hyperparameter tuning, evaluates the final model, and
    performs residual/error analysis.

------------------------------------------------------------------------

## 📌 Project Objective

The objective is to predict **rice yield in kg/hectare (`yield_kg_ha`)**
for Odisha districts and seasons using historical agricultural and
environmental information.

The project is designed as a **future-year prediction problem**, so the
evaluation methodology avoids random train/test splitting and instead
uses chronological validation.

------------------------------------------------------------------------

## 📂 Project Structure

``` text
Odisha-Rice-Yield/
│
├── Odisha_Rice_Complete_Combined_EDA_Final.ipynb
├── Odisha_Rice_ML_Audited_Corrected.ipynb
│
├── district-season-and-crop-wise-area-production-and-yield-statistics-for-odisha.csv
├── Integrated_Odisha_Rice_Yield_Dataset_2014_2024.csv
├── odisha_rice_integrated_eda_clean.csv
│
└── README.md
```

> The cleaned CSV is generated at the end of the EDA notebook and is
> used as the input to the ML notebook.

------------------------------------------------------------------------

## 📊 Data Preparation & Integration

Two datasets are integrated:

### Dataset 1

District/season/crop-wise agricultural statistics for Odisha.

### Dataset 2

An integrated Odisha Rice dataset containing environmental and
agricultural variables.

The datasets are connected using:

-   `fiscal_year`
-   `district_lgd_code`
-   `crop`
-   `season`

Only **Rice** observations are retained.

`State Total` records are removed before merging because the project
focuses on district-level observations.

The final inner merge produces **869 matched district-level
observations**. After removing observations with missing target values,
**839 observations** remain for supervised machine learning.

------------------------------------------------------------------------

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

------------------------------------------------------------------------

## 🔎 Exploratory Data Analysis

The EDA notebook covers:

-   Dataset structure and data types
-   Missing-value analysis
-   Duplicate checks
-   District-name validation
-   Merge validation
-   Target distribution
-   Outlier analysis using the IQR rule
-   Year-wise observations
-   Yield trends over time
-   Seasonal yield analysis
-   District-level performance
-   Climate and soil feature distributions
-   Feature-vs-yield relationships
-   Area vs. yield
-   Rainfall and temperature trends
-   Soil pH and N/P/K analysis
-   Irrigation analysis
-   Correlation analysis
-   Area vs. production validation

### Important EDA decisions

Outliers are not automatically removed because unusual agricultural
yields may represent genuine observations.

`production` is retained for data validation and EDA but is excluded
from the ML feature set because it is mathematically related to the
target.

------------------------------------------------------------------------

## 🧠 Machine Learning

This is a **regression problem** because the target is a continuous
numerical value.

### Features

The ML pipeline uses:

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

The following are excluded:

-   `yield_kg_ha` --- target
-   `production` --- target leakage
-   `reported_crop_yield` --- redundant/target-derived yield field
-   `district_lgd_code` --- identifier
-   `district_source` --- redundant district-name field
-   `fiscal_year` --- redundant with `year`

------------------------------------------------------------------------

## ⚙️ Preprocessing

The ML notebook uses a `ColumnTransformer` and `Pipeline`.

### Numerical data

``` text
StandardScaler
```

### Categorical data

``` text
OneHotEncoder(handle_unknown="ignore")
```

The complete preprocessing and model are kept inside a pipeline to avoid
data leakage during cross-validation.

The 12 raw predictors become **43 features after preprocessing/one-hot
encoding**.

------------------------------------------------------------------------

## 🕒 Chronological Train/Test Split

A random train/test split is intentionally avoided.

The data is split chronologically:

  Dataset             Years   Observations
  ------------ ------------ --------------
  Training       2014--2021            665
  Final Test     2022--2024            174
  Total          2014--2024            839

This better represents the intended use case:

> Train on historical years → predict future years.

------------------------------------------------------------------------

## 🔁 Time-Aware Cross-Validation

Model selection uses **year-aware forward validation**.

The validation folds keep complete years together, ensuring that a
validation year occurs after its corresponding training years.

Five time-aware folds are used.

Hyperparameter tuning is performed using `RandomizedSearchCV` on the
training period only.

The final 2022--2024 test set is kept separate from model selection.

------------------------------------------------------------------------

## 🤖 Models

The project compares:

-   Linear Regression
-   Decision Tree Regressor
-   Random Forest Regressor
-   Gradient Boosting Regressor

Random Forest and Gradient Boosting are additionally tuned using
time-aware randomized search.

Model selection is based on the **lowest mean validation RMSE**, with
validation MAE used as a secondary criterion.

------------------------------------------------------------------------

## 🏆 Final Model

The final selected model is:

**Tuned Random Forest Regressor**

### Final performance

  Metric                          Result
  ------------------- ------------------
  Selection CV RMSE     **639.40 kg/ha**
  Test RMSE             **541.77 kg/ha**
  Test MAE              **435.91 kg/ha**
  Test R²                     **0.4165**

The final model is selected using training-period cross-validation
rather than choosing the model based on final test performance.

------------------------------------------------------------------------

## 📈 Model Interpretation

The project also includes:

-   Feature importance analysis
-   Prediction vs. actual analysis
-   Residual analysis
-   Absolute error analysis
-   Error analysis by year
-   Error analysis by season
-   Error analysis by district
-   Train/test performance comparison

Feature importance is interpreted as **model reliance**, not as proof of
causal relationships.

------------------------------------------------------------------------

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
season-specific signal.

This is an important limitation and a potential area for future feature
engineering.

------------------------------------------------------------------------

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

The dataset does not include potentially important variables such as:

-   Crop variety
-   Pest/disease incidence
-   Exact sowing/harvest dates
-   Extreme weather events
-   More detailed soil characteristics

### 6. Future-year uncertainty

The test set represents only three future years relative to the training
period. Performance on years beyond 2024 is not established.

### 7. Moderate explanatory power

The final Test R² of **0.4165** means that a substantial portion of the
variation in the holdout data remains unexplained.

Therefore, this model should be treated as a **research/academic
prediction model**, not as a production agricultural decision system.

------------------------------------------------------------------------

## 🛠️ Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Jupyter Notebook

------------------------------------------------------------------------

## ▶️ How to Run

### 1. Clone/download the project

Place all notebooks and input CSV files in the same project directory.

### 2. Create a virtual environment

``` bash
python -m venv .venv
```

Activate it on Windows:

``` bash
.venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Run the EDA notebook

Open:

``` text
Odisha_Rice_Complete_Combined_EDA_Final.ipynb
```

Run it from top to bottom.

This creates:

``` text
odisha_rice_integrated_eda_clean.csv
```

### 5. Run the ML notebook

Open:

``` text
Odisha_Rice_ML_Audited_Corrected.ipynb
```

Run it from top to bottom.

The ML notebook uses the cleaned CSV generated by the EDA notebook.

------------------------------------------------------------------------

## ✅ Methodology Verification

The ML notebook contains automated checks for:

-   Target leakage
-   Feature consistency
-   Chronological train/test separation
-   Preprocessing pipeline structure
-   Year-aware cross-validation
-   Hyperparameter tuning
-   Final model selection
-   Metric calculations
-   Feature-importance dimensions
-   Residual calculations

The final verification reports:

``` text
All audited checks passed.
```

------------------------------------------------------------------------

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
Feature Importance & Error Analysis
```

The final tuned Random Forest achieves a **Test RMSE of 541.77 kg/ha**,
**Test MAE of 435.91 kg/ha**, and **Test R² of 0.4165** on the
2022--2024 holdout period.

------------------------------------------------------------------------

## 👩‍💻 Author

**Kalpana Pal (23052398)**
**Ridhima (23052415)**
**Mayank (23052401)**
**Naman**
**Mihir**
**Shauryan**


B.Tech --- Computer Science & Engineering

This project was developed as a machine-learning/data-science project
focused on agricultural yield prediction and responsible model
evaluation.
