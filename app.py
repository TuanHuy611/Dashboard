# -*- coding: utf-8 -*-
"""
ĐỒ ÁN TỔNG HỢP - HKTDL - HK251
DỰ ĐOÁN DOANH THU THEO MÙA, VÙNG VÀ DANH MỤC SẢN PHẨM
SỬ DỤNG RANDOM FOREST VÀ XGBOOST
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ==================== CÀI ĐẶT TRANG ====================
st.set_page_config(
    page_title="Dự đoán Doanh thu - Superstore Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

# Tiêu đề giống bìa báo cáo
st.image("01_logobachkhoa.png", width=150)  # Logo BK (link public)
st.title("ĐỒ ÁN TỔNG HỢP - HKTDL - HK251")
st.markdown("### **Dự đoán doanh thu theo mùa, vùng và danh mục sản phẩm**")
st.markdown("##### Sử dụng Random Forest và XGBoost")
st.markdown("**GVHD:** Vũ Ngọc Tú  |  **Năm:** 2025")
st.divider()

# ==================== LOAD DATA ====================
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y", dayfirst=True, errors='coerce')
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y", dayfirst=True, errors='coerce')
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter
    df["Season"] = df["Month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall"
    })
    return df

df = load_data()

# ==================== SIDEBAR FILTERS ====================
st.sidebar.header("Bộ lọc dữ liệu")
regions = st.sidebar.multiselect("Vùng (Region)", options=df["Region"].unique(), default=df["Region"].unique())
categories = st.sidebar.multiselect("Danh mục (Category)", options=df["Category"].unique(), default=df["Category"].unique())
years = st.sidebar.multiselect("Năm", options=sorted(df["Year"].unique()), default=sorted(df["Year"].unique()))

# Lọc dữ liệu
filtered = df[
    (df["Region"].isin(regions)) &
    (df["Category"].isin(categories)) &
    (df["Year"].isin(years))
]

# ==================== KPIs ====================
col1, col2, col3, col4 = st.columns(4)
total_sales = filtered["Sales"].sum()
total_orders = filtered["Order ID"].nunique()
avg_order_value = total_sales / total_orders if total_orders > 0 else 0


with col1:
    st.metric("Tổng Doanh Thu", f"${total_sales:,.0f}")
with col2:
    st.metric("Số Đơn Hàng", f"{total_orders:,}")
with col3:
    st.metric("Giá Trị ĐH Trung Bình", f"${avg_order_value:,.0f}")

st.divider()

# ==================== TABS ====================
tab1, tab2, tab3, tab4 = st.tabs(["Tổng quan", "Xu hướng theo thời gian", "Theo vùng & mùa", "Theo danh mục sản phẩm"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fig_region = px.bar(filtered.groupby("Region")["Sales"].sum().reset_index(),
                            x="Region", y="Sales", title="Doanh thu theo Vùng")
        st.plotly_chart(fig_region, use_container_width=True)
    with col2:
        fig_cat = px.pie(filtered.groupby("Category")["Sales"].sum().reset_index(),
                         values="Sales", names="Category", title="Tỷ trọng theo Danh mục")
        st.plotly_chart(fig_cat, use_container_width=True)

with tab2:
    trend = filtered.groupby(["Year", "Month"])["Sales"].sum().reset_index()
    trend["Date"] = pd.to_datetime(trend[["Year", "Month"]].assign(day=1))
    fig_trend = px.line(trend, x="Date", y="Sales", title="Xu hướng doanh thu theo tháng")
    st.plotly_chart(fig_trend, use_container_width=True)

    quarterly = filtered.groupby(["Year", "Quarter"])["Sales"].sum().reset_index()
    fig_q = px.bar(quarterly, x="Quarter", y="Sales", color="Year", barmode="group",
                   title="Doanh thu theo Quý")
    st.plotly_chart(fig_q, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        fig_season = px.bar(filtered.groupby("Season")["Sales"].sum().reset_index(),
                            x="Season", y="Sales", title="Doanh thu theo Mùa")
        st.plotly_chart(fig_season, use_container_width=True)
    with col2:
        heatmap = filtered.pivot_table(values="Sales", index="Region", columns="Season", aggfunc="sum", fill_value=0)
        fig_heat = px.imshow(heatmap, text_auto=True, aspect="auto",
                             title="Heatmap: Doanh thu theo Vùng x Mùa")
        st.plotly_chart(fig_heat, use_container_width=True)

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        top_sub = filtered.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False).head(10)
        fig_top_sub = px.bar(x=top_sub.index, y=top_sub.values, title="Top 10 Sub-Category")
        fig_top_sub.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_top_sub, use_container_width=True)
    with col2:
        fig_segment = px.pie(filtered.groupby("Segment")["Sales"].sum().reset_index(),
                             values="Sales", names="Segment", title="Doanh thu theo Phân khúc KH")
        st.plotly_chart(fig_segment, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Dashboard được xây dựng bằng Streamlit + Plotly | Dữ liệu: Superstore Sales (train.csv)")