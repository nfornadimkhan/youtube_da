import streamlit as st
import pandas as pd
import numpy as np
import nltk
import base64
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# Uncomment if you haven't downloaded NLTK tokenizers before
# nltk.download('punkt')

# -----------------------------------------------------------------
# 1. STREAMLIT PAGE SETUP & CUSTOM CSS
# -----------------------------------------------------------------
st.set_page_config(
    page_title="YouTube Analytics Dashboard with PCoA",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main { padding: 0rem 1rem; }
.stButton>button { width: 100%; }
.reportview-container { background: #f0f2f6 }
.sidebar .sidebar-content { background: #f0f2f6 }
h1, h2, h3 { color: #1f77b4; }
.stAlert { background-color: rgba(255, 255, 255, 0.8); }
/* Make dataframes a bit bigger by default */
[data-testid="stDataFrameContainer"] {
    max-height: 600px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------
# 2. LOAD YOUR LOCAL YOUTUBE CSV
#    Adjust 'CSV_PATH' if needed.
# -----------------------------------------------------------------
CSV_PATH = "youtube_data/cleaned_videos_data.csv"  # <-- change to your actual CSV filename/path

st.title("📊 Integrated YouTube Analytics Dashboard")
st.write("""
**This app first shows the PCoA analysis from your 'My Analysis with YT' code snippet.**  
Then it provides advanced analytics (demand, supply, trend, etc.) below.
""")

try:
    df = pd.read_csv(CSV_PATH)
    st.success(f"Data loaded successfully from '{CSV_PATH}'!")
except FileNotFoundError:
    st.error(f"Could not find '{CSV_PATH}'. Please place your CSV in the same folder or update the path.")
    st.stop()

# -----------------------------------------------------------------
# 3. DATA PREPROCESSING (COMMON FOR BOTH PCoA & ADVANCED ANALYTICS)
# -----------------------------------------------------------------
if "published_date" not in df.columns:
    st.error("CSV must contain a 'published_date' column.")
    st.stop()

df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
df["days_since_published"] = (datetime.now() - df["published_date"]).dt.days
df["recency_weight"] = 1 / (1 + df["days_since_published"]/365)

required_cols = ["category", "keyword", "view_count", "engagement_rate"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

# -----------------------------------------------------------------
# 4. FIRST SECTION - PCoA ("MY ANALYSIS WITH YT")
# -----------------------------------------------------------------

st.header("1. Principal Coordinate Analysis (PCoA) with Opportunity Zone")

# A) BUILD A SIMPLE FEATURE MATRIX FOR PCoA
#    We'll pick some numeric features we want to reduce
#    (Here we demonstrate using 'view_count' & 'engagement_rate' 
#     but you can add more as desired, e.g., recency_weight.)
numeric_features = ["view_count", "engagement_rate", "days_since_published"]
df_pcoa = df.dropna(subset=numeric_features).copy()

# Group by (category, keyword) to get aggregated features
agg_df = df_pcoa.groupby(["category", "keyword"], as_index=False).agg({
    "view_count": "mean",
    "engagement_rate": "mean",
    "days_since_published": "mean"
})

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(agg_df[numeric_features])

# Create a distance matrix (using Euclidean for demonstration)
dist_matrix = pairwise_distances(X_scaled, metric="euclidean")

# PCoA can be approximated with PCA on the distance matrix
# We'll do classical MDS style approach: 
#    1) Convert distances to a kernel matrix
#    2) Do PCA on that. 
# For simplicity, let's do a direct PCA on X_scaled or a simpler approach.
# We'll do a direct PCA on X_scaled as a demonstration (like a "PCoA-lite").
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(X_scaled)
pc1 = pca_coords[:, 0]
pc2 = pca_coords[:, 1]
explained_var_ratio = pca.explained_variance_ratio_

# B) BUILD THE PLOTLY FIGURE (from your snippet)
# We'll define some color mapping for categories
unique_cats = agg_df["category"].unique().tolist()
color_map = px.colors.qualitative.Dark24  # up to 24 unique colors
colors = dict(zip(unique_cats, color_map*10))  # repeat if more categories

fig_pcoa = go.Figure()

for category in unique_cats:
    cat_mask = (agg_df["category"] == category)
    
    fig_pcoa.add_trace(go.Scatter(
        x=pc1[cat_mask],
        y=pc2[cat_mask],
        mode="markers",
        name=category,
        marker=dict(
            color=colors[category],
            size=10,
            opacity=0.7
        ),
        hovertemplate=(
            "Keyword: %{customdata[2]}<br>" +
            "Mean View Count: %{customdata[0]:,.1f}<br>" +
            "Mean Engagement: %{customdata[1]:.2f}"
        ),
        customdata=np.column_stack([
            agg_df.loc[cat_mask, "view_count"],
            agg_df.loc[cat_mask, "engagement_rate"],
            agg_df.loc[cat_mask, "keyword"]
        ])
    ))

# Add shape to highlight an "opportunity zone" in top-right quadrant
xmax, ymax = pc1.max(), pc2.max()

fig_pcoa.add_shape(
    type="rect",
    x0=0,
    y0=0,
    x1=xmax,
    y1=ymax,
    line=dict(
        color="rgba(0,100,0,0.5)",
        width=2,
        dash="dash",
    ),
    fillcolor="rgba(0,100,0,0.1)"
)

# Add annotation for the zone
fig_pcoa.add_annotation(
    x=(xmax * 0.6),
    y=(ymax * 0.8),
    text="Premium Opportunities<br>High Demand, Low Competition",
    showarrow=True,
    arrowhead=1,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="rgba(0,100,0,0.5)",
    bgcolor="white",
    bordercolor="rgba(0,100,0,0.5)",
    borderwidth=2,
    borderpad=4,
    font=dict(size=12)
)

fig_pcoa.update_layout(
    title="PCoA Analysis with Premium Opportunity Zone",
    xaxis_title=f"PC1 ({explained_var_ratio[0]*100:.1f}% var explained)",
    yaxis_title=f"PC2 ({explained_var_ratio[1]*100:.1f}% var explained)",
    height=700,
    width=900,
    showlegend=True,
    legend=dict(
        title="Categories",
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    ),
    plot_bgcolor="white"
)
fig_pcoa.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=True)
fig_pcoa.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=True)

st.plotly_chart(fig_pcoa, use_container_width=True)

# C) PRINT TEXT ANALYSIS OF THE "PREMIUM ZONE"
st.subheader("Premium Opportunity Zone Analysis")
pc1_series = pd.Series(pc1, index=agg_df.index)
pc2_series = pd.Series(pc2, index=agg_df.index)
high_opportunity_mask = (pc1_series > 0) & (pc2_series > 0)

for category in unique_cats:
    cat_mask = (agg_df["category"] == category)
    zone_mask = cat_mask & high_opportunity_mask
    count_premium = zone_mask.sum()
    
    st.write(f"**{category} Category**")
    st.write(f"- Keywords in premium zone: `{count_premium}`")
    
    if count_premium > 0:
        # Show top 3 by mean view count or any other metric
        cat_df = agg_df.loc[zone_mask].copy()
        cat_df["pc1"] = pc1_series[zone_mask]
        cat_df["pc2"] = pc2_series[zone_mask]
        # Let's rank them by 'view_count' descending
        cat_df = cat_df.sort_values("view_count", ascending=False)
        st.write("Top 3 premium opportunities (by mean view count):")
        for _, row in cat_df.head(3).iterrows():
            st.write(f"- **{row['keyword']}**: " 
                     f"mean_view={row['view_count']:.1f}, "
                     f"mean_engagement={row['engagement_rate']:.2f}, "
                     f"PC1={row['pc1']:.2f}, PC2={row['pc2']:.2f}")

st.markdown("---")

# -----------------------------------------------------------------
# 5. SECOND SECTION - ADVANCED ANALYTICS
# -----------------------------------------------------------------
st.header("2. Advanced Analytics: Demand, Supply, Trend, Opportunity")

# Re-group by category & keyword for advanced metrics
def calculate_demand_metrics(keyword_group):
    view_velocity = keyword_group["view_count"] / (keyword_group["days_since_published"] + 1)
    weighted_engagement = (keyword_group["engagement_rate"] * keyword_group["recency_weight"]).mean()
    demand_consistency = 1 / (1 + view_velocity.std())
    return pd.Series({
        "avg_daily_views": view_velocity.mean(),
        "weighted_engagement": weighted_engagement,
        "demand_consistency": demand_consistency
    })

keyword_demand = df.groupby(["category", "keyword"]).apply(calculate_demand_metrics)

def normalize_score(series):
    # Protect against division by zero
    if series.min() == series.max():
        return pd.Series([50]*len(series), index=series.index)
    return (series - series.min()) / (series.max() - series.min()) * 100

keyword_demand["demand_score"] = normalize_score(keyword_demand["avg_daily_views"])
video_counts = df.groupby(["category", "keyword"]).size()
keyword_demand["video_count"] = video_counts
keyword_demand["supply_saturation"] = normalize_score(keyword_demand["video_count"])
keyword_demand["opportunity_score"] = (
    keyword_demand["demand_score"] * (100 - keyword_demand["supply_saturation"]) / 100
)

# Trend calculation
def calculate_trend(group):
    recent_mask = group["days_since_published"] <= 90
    if recent_mask.any():
        recent_views = group[recent_mask]["view_count"].mean()
        older_views = group[~recent_mask]["view_count"].mean() if (~recent_mask).any() else recent_views
        trend = ((recent_views / older_views) - 1) * 100
    else:
        trend = 0
    return trend

keyword_demand["trend"] = df.groupby(["category", "keyword"]).apply(calculate_trend)

# ---------------------------
# SIDEBAR FILTERS
# ---------------------------
st.sidebar.title("Dashboard Controls (Advanced)")
all_categories = sorted(set(keyword_demand.index.get_level_values("category")))
selected_categories = st.sidebar.multiselect(
    "Select Categories",
    options=all_categories,
    default=all_categories
)

min_demand = st.sidebar.slider("Minimum Demand Score", 0, 100, 0, step=5)
min_opportunity = st.sidebar.slider("Minimum Opportunity Score", 0, 100, 0, step=5)
min_trend = st.sidebar.slider("Minimum Trend (%)", -100, 300, -100, step=10)

mask = (keyword_demand["demand_score"] >= min_demand) & \
       (keyword_demand["opportunity_score"] >= min_opportunity) & \
       (keyword_demand["trend"] >= min_trend) & \
       (keyword_demand.index.get_level_values("category").isin(selected_categories))

filtered_data = keyword_demand[mask].copy()
# Ensure marker sizes are non-negative
filtered_data["trend_size"] = filtered_data["trend"].apply(lambda x: max(0, x))

# ---------------------------
# SHOW METRICS
# ---------------------------
st.subheader("Overall Channel Metrics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Total Videos",
        f"{len(df):,}",
        f"{(df['days_since_published']<=30).sum():,} new in last 30 days"
    )

with col2:
    total_views = df["view_count"].sum()
    recent_views = df.loc[df["days_since_published"] <= 30, "view_count"].sum()
    st.metric(
        "Total Views",
        f"{total_views:,.0f}",
        f"{recent_views:+,.0f} last 30 days"
    )

with col3:
    avg_eng = df["engagement_rate"].mean()
    recent_eng = df.loc[df["days_since_published"] <= 30, "engagement_rate"].mean()
    diff_eng = (recent_eng - avg_eng) if not np.isnan(recent_eng) else 0
    st.metric(
        "Avg Engagement Rate",
        f"{avg_eng:.2f}%",
        f"{diff_eng:+.2f}% last 30 days"
    )

with col4:
    # Compute overall trend using the same function but on entire df
    def overall_trend_calc(df_all):
        recent_mask = df_all["days_since_published"] <= 90
        if recent_mask.any():
            rec = df_all.loc[recent_mask, "view_count"].mean()
            old = df_all.loc[~recent_mask, "view_count"].mean() if (~recent_mask).any() else rec
            return ((rec/old) - 1)*100
        return 0
    overall_trend = overall_trend_calc(df)
    st.metric(
        "Overall Trend",
        f"{overall_trend:.1f}%",
        "vs previous period"
    )

# ---------------------------
# TABS FOR ANALYSES
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Trend Analysis",
    "Opportunity Matrix",
    "Detailed Metrics",
    "Keyword Analysis"
])

with tab1:
    st.subheader("Top Trending Keywords")
    top_trending = filtered_data.nlargest(10, "trend").reset_index()
    fig_trend = px.bar(
        top_trending,
        x="keyword",
        y="trend",
        color="category",
        title="Top 10 Trending Keywords",
        text=top_trending["trend"].apply(lambda x: f"{x:.1f}%")
    )
    fig_trend.update_layout(
        xaxis_title="Keyword", yaxis_title="Trend (%)",
        xaxis_tickangle=-45
    )
    fig_trend.update_traces(textposition="outside")
    st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    st.subheader("Content Opportunity Matrix")
    fig_matrix = px.scatter(
        filtered_data.reset_index(),
        x="supply_saturation",
        y="demand_score",
        color="category",
        size="trend_size",
        hover_data=["keyword", "opportunity_score", "trend"],
        title="Content Opportunity Matrix"
    )
    # Example shape to highlight an area in the scatter:
    fig_matrix.add_shape(
        type="rect",
        x0=0, y0=50,
        x1=50, y1=100,
        line=dict(color="green", width=2, dash="dash"),
        fillcolor="rgba(0,255,0,0.1)"
    )
    st.plotly_chart(fig_matrix, use_container_width=True)

with tab3:
    st.subheader("Detailed Metrics")
    st.write("Use the scroll to see more rows/columns or apply filters in the sidebar.")
    st.dataframe(
        filtered_data.style.background_gradient(
            subset=["demand_score", "opportunity_score", "trend"],
            cmap="RdYlGn"
        ),
        use_container_width=True
    )

with tab4:
    st.subheader("Keyword Performance Analysis")
    colA, colB = st.columns(2)

    with colA:
        for cat in selected_categories:
            cat_data = filtered_data.loc[cat]
            if cat_data.empty:
                continue
            st.write(f"### **{cat}** Category - Top 5 by Opportunity Score")
            st.write(
                cat_data.nlargest(5, "opportunity_score")[
                    ["demand_score", "opportunity_score", "trend", "video_count"]
                ]
            )

    with colB:
        st.write("### Keyword Trend Over Time")
        all_keywords = filtered_data.index.get_level_values("keyword").unique()
        if len(all_keywords) == 0:
            st.info("No keywords to display. Adjust filters in the sidebar.")
        else:
            selected_keyword = st.selectbox("Select Keyword", options=all_keywords)
            # Filter original df
            kw_data = df[df["keyword"] == selected_keyword].copy()
            kw_data.sort_values("published_date", inplace=True)
            fig_kw_trend = px.line(
                kw_data,
                x="published_date",
                y="view_count",
                title=f"View Count Trend for '{selected_keyword}'"
            )
            st.plotly_chart(fig_kw_trend, use_container_width=True)

# ---------------------------
# DOWNLOAD SECTION
# ---------------------------
st.markdown("---")
st.subheader("Export Filtered Data")
csv_data = filtered_data.to_csv()
b64 = base64.b64encode(csv_data.encode()).decode()
download_link = f'<a href="data:file/csv;base64,{b64}" download="youtube_filtered_analysis.csv">Download (CSV)</a>'
st.markdown(download_link, unsafe_allow_html=True)