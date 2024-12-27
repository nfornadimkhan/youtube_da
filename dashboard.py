import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Plant Breeding YouTube Analysis",
    page_icon="🌱",
    layout="wide"
)

# Custom color settings
category_colors = {'Old': '#FF6B6B', 'Current': '#4ECDC4', 'Modern': '#45B7D1'}
gradient_colors = ['#FF9999', '#FF7979', '#4ECDC4', '#45B7D1', '#2E86AB', '#07004D']

@st.cache_data
def load_data():
    df = pd.read_csv('youtube_data/all_videos_data.csv')
    df['published_date'] = pd.to_datetime(df['published_date'])
    df['year'] = df['published_date'].dt.year
    current_time = pd.Timestamp.now(tz='UTC')
    df['engagement_rate'] = (df['like_count'] + df['comment_count']) / df['view_count']
    df['views_per_day'] = df['view_count'] / ((current_time - df['published_date']).dt.total_seconds() / (24 * 60 * 60))
    return df

# Load data
df = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Overview", "Content Evolution", "Engagement Analysis", "Keyword Analysis"]
)

# Overview Page
if page == "Overview":
    st.title("🌱 Plant Breeding YouTube Content Analysis")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Videos", len(df))
    with col2:
        st.metric("Total Views", f"{df['view_count'].sum():,}")
    with col3:
        st.metric("Average Engagement Rate", f"{df['engagement_rate'].mean():.2%}")
    with col4:
        st.metric("Date Range", f"{df['published_date'].min().year} - {df['published_date'].max().year}")
    
    # Category Distribution
    st.subheader("Content Distribution by Category")
    fig = px.pie(df, names='category', values='view_count', 
                 color='category', color_discrete_map=category_colors)
    st.plotly_chart(fig, use_container_width=True)
    
    # Basic Statistics
    st.subheader("Category Statistics")
    stats = df.groupby('category').agg({
        'view_count': ['count', 'mean', 'sum'],
        'engagement_rate': 'mean'
    }).round(2)
    st.dataframe(stats)

# Content Evolution Page
elif page == "Content Evolution":
    st.title("📈 Content Evolution Over Time")
    
    # Yearly trend
    yearly_data = df.groupby(['year', 'category']).size().unstack(fill_value=0)
    fig = px.area(yearly_data, 
                  color_discrete_map=category_colors,
                  title="Video Publication Trends")
    st.plotly_chart(fig, use_container_width=True)
    
    # Views Evolution
    st.subheader("Cumulative Views by Category")
    fig = go.Figure()
    for category in df['category'].unique():
        category_data = df[df['category'] == category].sort_values('published_date')
        fig.add_trace(go.Scatter(
            x=category_data['published_date'],
            y=category_data['view_count'].cumsum(),
            name=category,
            fill='tonexty'
        ))
    st.plotly_chart(fig, use_container_width=True)

# Engagement Analysis Page
elif page == "Engagement Analysis":
    st.title("👥 Engagement Analysis")
    
    # Engagement over time
    st.subheader("Engagement Rate Evolution")
    engagement_data = df.groupby(['year', 'category'])['engagement_rate'].mean().unstack()
    fig = px.line(engagement_data, markers=True,
                  color_discrete_map=category_colors)
    st.plotly_chart(fig, use_container_width=True)
    
    # Engagement distribution
    st.subheader("Engagement Rate Distribution")
    fig = px.box(df, x='category', y='engagement_rate', 
                 color='category', color_discrete_map=category_colors)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top engaging videos
    st.subheader("Top 10 Most Engaging Videos")
    top_engaging = df.nlargest(10, 'engagement_rate')[
        ['title', 'category', 'view_count', 'engagement_rate']]
    st.dataframe(top_engaging)

# Keyword Analysis Page
elif page == "Keyword Analysis":
    st.title("🔍 Keyword Analysis")
    
    # Keyword frequency by category
    st.subheader("Top Keywords by Category")
    selected_category = st.selectbox("Select Category", df['category'].unique())
    
    top_keywords = df[df['category'] == selected_category].groupby('keyword').agg({
        'view_count': 'sum',
        'video_id': 'count'
    }).sort_values('view_count', ascending=False).head(10)
    
    fig = px.bar(top_keywords, x=top_keywords.index, y='view_count',
                 title=f"Top Keywords in {selected_category} Category")
    st.plotly_chart(fig, use_container_width=True)
    
    # Keyword trends over time
    st.subheader("Keyword Trends Over Time")
    selected_keywords = st.multiselect(
        "Select Keywords to Compare",
        df['keyword'].unique(),
        default=df['keyword'].value_counts().head().index.tolist()
    )
    
    if selected_keywords:
        keyword_trend = df[df['keyword'].isin(selected_keywords)].groupby(
            ['year', 'keyword'])['view_count'].sum().unstack()
        fig = px.line(keyword_trend, markers=True)
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created with ❤️ using Streamlit") 