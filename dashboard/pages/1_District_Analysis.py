
# ============================================================
# PHASE 5.6 — DISTRICT ANALYSIS
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px

from utils import load_dataset, format_yield


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="District Analysis | Odisha Rice",
    page_icon="📍",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

try:
    df = load_dataset()
except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("📍 District-wise Rice Yield Analysis")

st.markdown(
    """
    Explore historical rice yield performance across Odisha's
    30 districts. Compare district rankings, seasonal patterns
    and changes over time.
    """
)

st.divider()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("District Analysis Filters")

all_districts = sorted(df["district"].unique())

selected_districts = st.sidebar.multiselect(
    "Select districts",
    all_districts,
    default=all_districts
)

all_years = sorted(df["year"].unique())

selected_years = st.sidebar.multiselect(
    "Select years",
    all_years,
    default=all_years
)

all_seasons = sorted(df["season"].unique())

selected_seasons = st.sidebar.multiselect(
    "Select seasons",
    all_seasons,
    default=all_seasons
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered = df[
    df["district"].isin(selected_districts)
    & df["year"].isin(selected_years)
    & df["season"].isin(selected_seasons)
].copy()

if filtered.empty:
    st.warning("No data available for the selected filters.")
    st.stop()


# ============================================================
# DISTRICT SUMMARY
# ============================================================

district_summary = (
    filtered
    .groupby("district", as_index=False)
    .agg(
        average_yield=("yield_kg_ha", "mean"),
        median_yield=("yield_kg_ha", "median"),
        minimum_yield=("yield_kg_ha", "min"),
        maximum_yield=("yield_kg_ha", "max"),
        observations=("yield_kg_ha", "count"),
        average_area=("area", "mean"),
        average_production=("production", "mean")
    )
    .sort_values("average_yield", ascending=False)
    .reset_index(drop=True)
)

district_summary.insert(
    0,
    "rank",
    range(1, len(district_summary) + 1)
)


# ============================================================
# KPI CARDS
# ============================================================

st.subheader("District Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Districts Selected",
    district_summary["district"].nunique()
)

c2.metric(
    "Average Yield",
    format_yield(filtered["yield_kg_ha"].mean())
)

c3.metric(
    "Highest Yield District",
    district_summary.iloc[0]["district"]
)

c4.metric(
    "Observations",
    f"{len(filtered):,}"
)

st.divider()


# ============================================================
# DISTRICT RANKING
# ============================================================

st.subheader("1. District-wise Average Yield Ranking")

ranking = district_summary.sort_values(
    "average_yield",
    ascending=True
)

fig_ranking = px.bar(
    ranking,
    x="average_yield",
    y="district",
    orientation="h",
    color="average_yield",
    color_continuous_scale="Greens",
    hover_data=[
        "observations",
        "median_yield",
        "average_production"
    ],
    labels={
        "average_yield": "Average Yield (kg/ha)",
        "district": "District",
        "observations": "Observations"
    }
)

fig_ranking.update_layout(
    template="plotly_white",
    height=max(500, len(ranking) * 24),
    coloraxis_showscale=False,
    margin=dict(l=20, r=20, t=20, b=20)
)

st.plotly_chart(
    fig_ranking,
    use_container_width=True
)


# ============================================================
# TOP AND BOTTOM DISTRICTS
# ============================================================

st.subheader("2. Highest and Lowest Yield Districts")

top_col, bottom_col = st.columns(2)

with top_col:

    st.markdown("#### Top 5 Districts")

    st.dataframe(
        district_summary.head(5)[
            ["rank", "district", "average_yield", "observations"]
        ],
        use_container_width=True,
        hide_index=True
    )

with bottom_col:

    st.markdown("#### Bottom 5 Districts")

    st.dataframe(
        district_summary.tail(5).sort_values("average_yield")[
            ["rank", "district", "average_yield", "observations"]
        ],
        use_container_width=True,
        hide_index=True
    )


st.divider()


# ============================================================
# DISTRICT YIELD DISTRIBUTION
# ============================================================

st.subheader("3. Yield Distribution Across Districts")

fig_box = px.box(
    filtered,
    x="district",
    y="yield_kg_ha",
    color="district",
    points="outliers",
    labels={
        "district": "District",
        "yield_kg_ha": "Yield (kg/ha)"
    }
)

fig_box.update_layout(
    template="plotly_white",
    height=550,
    showlegend=False,
    xaxis_tickangle=-55
)

st.plotly_chart(
    fig_box,
    use_container_width=True
)


st.divider()


# ============================================================
# YEARLY TRENDS
# ============================================================

st.subheader("4. Yearly Yield Trends by District")

trend_data = (
    filtered
    .groupby(
        ["year", "district"],
        as_index=False
    )
    .agg(
        average_yield=("yield_kg_ha", "mean")
    )
)

fig_trend = px.line(
    trend_data,
    x="year",
    y="average_yield",
    color="district",
    markers=True,
    labels={
        "year": "Year",
        "average_yield": "Average Yield (kg/ha)",
        "district": "District"
    }
)

fig_trend.update_layout(
    template="plotly_white",
    height=550,
    hovermode="x unified"
)

st.plotly_chart(
    fig_trend,
    use_container_width=True
)


st.divider()


# ============================================================
# SEASON-WISE COMPARISON
# ============================================================

st.subheader("5. Seasonal Yield Comparison")

season_summary = (
    filtered
    .groupby(
        ["district", "season"],
        as_index=False
    )
    .agg(
        average_yield=("yield_kg_ha", "mean"),
        observations=("yield_kg_ha", "count")
    )
)

fig_season = px.bar(
    season_summary,
    x="district",
    y="average_yield",
    color="season",
    barmode="group",
    hover_data=["observations"],
    labels={
        "district": "District",
        "average_yield": "Average Yield (kg/ha)",
        "season": "Season"
    }
)

fig_season.update_layout(
    template="plotly_white",
    height=550,
    xaxis_tickangle=-45
)

st.plotly_chart(
    fig_season,
    use_container_width=True
)

st.caption(
    "Kharif is present in the 2023–2024 test period but was "
    "not represented in model training. These charts describe "
    "observed dataset values, not a learned Kharif model effect."
)


st.divider()


# ============================================================
# COMPLETE DISTRICT SUMMARY
# ============================================================

st.subheader("6. District Summary Table")

st.dataframe(
    district_summary.round(2),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DOWNLOAD
# ============================================================

csv_data = district_summary.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download District Summary CSV",
    data=csv_data,
    file_name="odisha_district_yield_summary.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "District analysis | Odisha Rice Yield Research Project"
)

st.caption(
    "District averages are descriptive statistics and should "
    "not be interpreted as causal effects."
)
