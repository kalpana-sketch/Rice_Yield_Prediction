
# ============================================================
# ODISHA RICE YIELD ANALYTICS
# MAIN DASHBOARD HOMEPAGE
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px

from utils import (
    load_dataset,
    DASHBOARD_TITLE,
    DASHBOARD_SUBTITLE,
    format_yield
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Odisha Rice Yield Analytics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7faf7;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.dashboard-title {
    font-size: 2.4rem;
    font-weight: 750;
    color: #164b35;
    margin-bottom: 0.2rem;
}

.dashboard-subtitle {
    font-size: 1rem;
    color: #64756b;
    margin-bottom: 1.5rem;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e1ebe3;
    box-shadow: 0 2px 8px rgba(0,0,0,0.035);
}

.metric-label {
    font-size: 0.9rem;
    color: #65756a;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #164b35;
}

.metric-note {
    font-size: 0.8rem;
    color: #849188;
}

.section-heading {
    color: #164b35;
    font-weight: 700;
    font-size: 1.25rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD DATA
# ============================================================

try:
    df = load_dataset()

except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()


# ============================================================
# 4. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("# 🌾 Odisha Rice")
    st.caption("Yield Analytics Platform")

    st.divider()

    st.markdown("### Dashboard Filters")

    years = sorted(
        df["year"].dropna().unique().tolist()
    )

    districts = sorted(
        df["district"].dropna().unique().tolist()
    )

    seasons = sorted(
        df["season"].dropna().unique().tolist()
    )

    selected_years = st.multiselect(
        "Select year",
        options=years,
        default=years
    )

    selected_districts = st.multiselect(
        "Select districts",
        options=districts,
        default=districts
    )

    selected_seasons = st.multiselect(
        "Select seasons",
        options=seasons,
        default=seasons
    )

    if st.button("Reset filters", use_container_width=True):
        st.rerun()

    st.divider()

    st.markdown("### About the Project")

    st.caption(
        "District-Level Rice Yield Prediction in Odisha "
        "Using Time-Aware Machine Learning Regression."
    )

    st.caption("Study period: 2014–2024")


# ============================================================
# 5. APPLY FILTERS
# ============================================================

filtered_df = df[
    df["year"].isin(selected_years)
    & df["district"].isin(selected_districts)
    & df["season"].isin(selected_seasons)
].copy()


# ============================================================
# 6. PAGE HEADER
# ============================================================

st.markdown(
    f'<div class="dashboard-title">🌾 {DASHBOARD_TITLE}</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="dashboard-subtitle">{DASHBOARD_SUBTITLE}</div>',
    unsafe_allow_html=True
)

st.info(
    "Explore historical rice yield patterns across Odisha's "
    "districts and seasons using the project's agricultural dataset."
)


# ============================================================
# 7. KPI CARDS
# ============================================================

total_observations = len(df)

total_districts = df["district"].nunique()

study_period = (
    f"{int(df['year'].min())}–{int(df['year'].max())}"
)

total_features = 12

average_yield = df["yield_kg_ha"].mean()

k1, k2, k3, k4 = st.columns(4)

metrics = [
    (k1, "Total Observations", f"{total_observations:,}", "Dataset records"),
    (k2, "Districts", str(total_districts), "Across Odisha"),
    (k3, "Study Period", study_period, "Historical coverage"),
    (k4, "Input Features", str(total_features), "10 numerical + 2 categorical")
]

for column, label, value, note in metrics:

    with column:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


st.write("")


# ============================================================
# 8. FILTER SUMMARY
# ============================================================

st.markdown(
    '<div class="section-heading">Selected Data Overview</div>',
    unsafe_allow_html=True
)

if filtered_df.empty:

    st.warning(
        "No observations match the selected filters. "
        "Please change your filter selections."
    )

    st.stop()

c1, c2, c3 = st.columns(3)

c1.metric(
    "Filtered Observations",
    f"{len(filtered_df):,}"
)

c2.metric(
    "Average Rice Yield",
    format_yield(filtered_df["yield_kg_ha"].mean())
)

c3.metric(
    "Average Production",
    f"{filtered_df['production'].mean():,.2f} tonnes"
)


st.divider()


# ============================================================
# 9. YEARLY RICE YIELD TREND
# ============================================================

st.markdown(
    '<div class="section-heading">Rice Yield Trend Across Years</div>',
    unsafe_allow_html=True
)

yearly_yield = (
    filtered_df
    .groupby("year", as_index=False)
    .agg(
        average_yield=("yield_kg_ha", "mean"),
        observations=("yield_kg_ha", "count")
    )
)

fig_trend = px.line(
    yearly_yield,
    x="year",
    y="average_yield",
    markers=True,
    hover_data=["observations"],
    labels={
        "year": "Year",
        "average_yield": "Average Yield (kg/ha)",
        "observations": "Observations"
    }
)

fig_trend.update_traces(
    line=dict(color="#23865b", width=3),
    marker=dict(size=9)
)

fig_trend.update_layout(
    template="plotly_white",
    height=400,
    margin=dict(l=20, r=20, t=20, b=20),
    hovermode="x unified"
)

st.plotly_chart(
    fig_trend,
    use_container_width=True
)


# ============================================================
# 10. DISTRICT-WISE YIELD
# ============================================================

st.markdown(
    '<div class="section-heading">District-wise Average Rice Yield</div>',
    unsafe_allow_html=True
)

district_yield = (
    filtered_df
    .groupby("district", as_index=False)
    .agg(
        average_yield=("yield_kg_ha", "mean"),
        observations=("yield_kg_ha", "count")
    )
    .sort_values("average_yield", ascending=True)
)

fig_district = px.bar(
    district_yield,
    x="average_yield",
    y="district",
    orientation="h",
    color="average_yield",
    color_continuous_scale="Greens",
    hover_data=["observations"],
    labels={
        "average_yield": "Average Yield (kg/ha)",
        "district": "District",
        "observations": "Observations"
    }
)

fig_district.update_layout(
    template="plotly_white",
    height=650,
    margin=dict(l=20, r=20, t=20, b=20),
    coloraxis_showscale=False
)

st.plotly_chart(
    fig_district,
    use_container_width=True
)


# ============================================================
# 11. SEASON-WISE ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-heading">Season-wise Yield Distribution</div>',
    unsafe_allow_html=True
)

left, right = st.columns(2)

with left:

    fig_season = px.box(
        filtered_df,
        x="season",
        y="yield_kg_ha",
        color="season",
        color_discrete_sequence=[
            "#23865b",
            "#75ad85",
            "#b3cfae",
            "#e3b85c"
        ],
        labels={
            "season": "Season",
            "yield_kg_ha": "Rice Yield (kg/ha)"
        }
    )

    fig_season.update_layout(
        template="plotly_white",
        showlegend=False,
        height=400
    )

    st.plotly_chart(
        fig_season,
        use_container_width=True
    )


# ============================================================
# 12. YIELD DISTRIBUTION
# ============================================================

with right:

    fig_distribution = px.histogram(
        filtered_df,
        x="yield_kg_ha",
        nbins=30,
        color_discrete_sequence=["#23865b"],
        labels={
            "yield_kg_ha": "Rice Yield (kg/ha)"
        }
    )

    fig_distribution.update_layout(
        template="plotly_white",
        height=400,
        showlegend=False
    )

    st.plotly_chart(
        fig_distribution,
        use_container_width=True
    )


# ============================================================
# 13. DATASET PREVIEW
# ============================================================

with st.expander("View filtered dataset"):

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.caption(
    "Odisha Rice Yield Analytics | Academic Research Dashboard | "
    "Historical data: 2014–2024"
)

st.caption(
    "Note: Descriptive trends represent observed data and do not "
    "establish causal relationships."
)
