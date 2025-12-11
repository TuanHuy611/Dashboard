# -*- coding: utf-8 -*-
"""
ĐỒ ÁN TỔNG HỢP - HKTDL - HK251
DỰ ĐOÁN DOANH THU THEO MÙA, VÙNG VÀ DANH MỤC SẢN PHẨM
SỬ DỤNG RANDOM FOREST VÀ XGBOOST + STREAMLIT DASHBOARD
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder

# ==================== CÀI ĐẶT TRANG ====================
st.set_page_config(
    page_title="Dự đoán Doanh thu Superstore",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

st.image("01_logobachkhoa.png", width=150)
st.title("ĐỒ ÁN TỔNG HỢP - KTDL - HK251")
st.markdown("### **Dự đoán doanh thu theo mùa, vùng và danh mục sản phẩm**")
st.markdown("##### Sử dụng Random Forest và XGBoost")
st.markdown("**GVHD:** Vũ Ngọc Tú  |  **Năm:** 2025")
st.divider()

# ==================== LOAD DATA ====================
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")  # File gốc để lấy unique values và freq
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

# Feature names từ file processed (hardcode từ phân tích)
feature_names = [
    'Order_Sales_Sum', 'customer_avg_sales',
    'Ship Mode_First Class', 'Ship Mode_Same Day', 'Ship Mode_Second Class', 'Ship Mode_Standard Class',
    'Category_Furniture', 'Category_Office Supplies', 'Category_Technology',
    'Segment_Consumer', 'Segment_Corporate', 'Segment_Home Office',
    'Region_Central', 'Region_East', 'Region_South', 'Region_West',
    'Season_Fall', 'Season_Spring', 'Season_Summer', 'Season_Winter',
    'City_freq', 'State_freq', 'Sub-Category_freq', 'Product Name_freq'
]

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_models():
    if not os.path.exists("rf_final_best_narrow.joblib") or not os.path.exists("xgb_final_best_narrow.joblib"):
        st.error("Không tìm thấy file model! Đảm bảo rf_final_best_narrow.joblib và xgb_final_best_narrow.joblib ở cùng thư mục.")
        return None, None
    rf_model = joblib.load("rf_final_best_narrow.joblib")
    xgb_model = joblib.load("xgb_final_best_narrow.joblib")
    return rf_model, xgb_model

rf_model, xgb_model = load_models()

# ==================== SIDEBAR FILTERS ====================
st.sidebar.header("Bộ lọc dữ liệu")
regions = st.sidebar.multiselect("Vùng (Region)", options=df["Region"].unique(), default=df["Region"].unique())
categories = st.sidebar.multiselect("Danh mục (Category)", options=df["Category"].unique(), default=df["Category"].unique())
years = st.sidebar.multiselect("Năm", options=sorted(df["Year"].unique()), default=sorted(df["Year"].unique()))

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
with col4:
    st.metric("Số mẫu dữ liệu", f"{len(filtered):,}")

st.divider()

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Tổng quan", "Xu hướng thời gian", "Theo vùng & mùa", "Theo danh mục", "DỰ ĐOÁN DOANH THU", "MODEL EVALUATION"])

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


# ==================== TAB DỰ ĐOÁN ====================
with tab5:
    st.header("DỰ ĐOÁN DOANH THU TƯƠNG LAI")
    st.markdown("#### Chọn điều kiện để dự đoán (dựa trên feature đã train)")

    if rf_model is None or xgb_model is None:
        st.error("Model chưa tải. Kiểm tra file model.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            pred_region = st.selectbox("Vùng (Region)", options=sorted(df["Region"].unique()))
            pred_category = st.selectbox("Danh mục (Category)", options=sorted(df["Category"].unique()))
            pred_subcat = st.selectbox("Sub-Category", options=sorted(df[df["Category"] == pred_category]["Sub-Category"].unique()))
            pred_product = st.selectbox("Product Name", options=sorted(df[df["Sub-Category"] == pred_subcat]["Product Name"].unique()))
        with col2:
            pred_ship_mode = st.selectbox("Ship Mode", options=sorted(df["Ship Mode"].unique()))
            pred_segment = st.selectbox("Segment", options=sorted(df["Segment"].unique()))
            pred_city = st.selectbox("City", options=sorted(df["City"].unique()))
            pred_state = st.selectbox("State", options=sorted(df[df["City"] == pred_city]["State"].unique()))
            pred_month = st.selectbox("Tháng", options=list(range(1,13)), format_func=lambda x: datetime.strptime(str(x), "%m").strftime("%B"))
            pred_season = ["Winter", "Spring", "Summer", "Fall"][(pred_month-1)//3]

        # Numerical inputs (dùng mean từ train_data làm default)
        pred_order_sales_sum = st.number_input("Order Sales Sum (số món hàng trong đơn)", value=1.5069) 
        pred_customer_avg_sales = st.number_input("Customer Avg Sales (Số lần khách hàng đã mua trước đó)", value=1.5104)

        if st.button("DỰ ĐOÁN DOANH THU", type="primary", use_container_width=True):
            with st.spinner("Đang dự đoán..."):
                # Tính freq từ df gốc
                total_rows = len(df)
                city_freq = len(df[df["City"] == pred_city]) / total_rows
                state_freq = len(df[df["State"] == pred_state]) / total_rows
                subcat_freq = len(df[df["Sub-Category"] == pred_subcat]) / total_rows
                product_freq = len(df[df["Product Name"] == pred_product]) / total_rows

                # Tạo dict input
                input_dict = {
                    'Order_Sales_Sum': pred_order_sales_sum,
                    'customer_avg_sales': pred_customer_avg_sales,
                    'Ship Mode': pred_ship_mode,
                    'Category': pred_category,
                    'Segment': pred_segment,
                    'Region': pred_region,
                    'Season': pred_season,
                    'City_freq': city_freq,
                    'State_freq': state_freq,
                    'Sub-Category_freq': subcat_freq,
                    'Product Name_freq': product_freq
                }

                # Tạo DataFrame
                input_df = pd.DataFrame([input_dict])

                # One-hot encode các categorical
                categorical_cols = ['Ship Mode', 'Category', 'Segment', 'Region', 'Season']
                encoder = OneHotEncoder(categories=[
                    sorted(df['Ship Mode'].unique()), 
                    sorted(df['Category'].unique()), 
                    sorted(df['Segment'].unique()), 
                    sorted(df['Region'].unique()), 
                    ['Fall', 'Spring', 'Summer', 'Winter']
                ], drop=None, sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(df[categorical_cols])  # Fit trên df gốc để khớp tên cột
                encoded_cols = encoder.get_feature_names_out(categorical_cols)
                
                input_encoded = pd.DataFrame(encoder.transform(input_df[categorical_cols]), columns=encoded_cols)
                
                # Kết hợp numerical + encoded
                numerical_cols = ['Order_Sales_Sum', 'customer_avg_sales', 'City_freq', 'State_freq', 'Sub-Category_freq', 'Product Name_freq']
                input_final = pd.concat([input_df[numerical_cols], input_encoded], axis=1)
                
                # Align với feature_names (thêm cột thiếu = 0)
                for col in feature_names:
                    if col not in input_final.columns:
                        input_final[col] = 0
                input_final = input_final[feature_names]

                X_input = input_final.values

                # Dự đoán
                rf_pred = rf_model.predict(X_input)[0]
                xgb_pred = xgb_model.predict(X_input)[0]
                ensemble_pred = (rf_pred + xgb_pred) / 2

                st.success("DỰ ĐOÁN THÀNH CÔNG!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Random Forest", f"${rf_pred:,.0f}")
                with col2:
                    st.metric("XGBoost", f"${xgb_pred:,.0f}")
                with col3:
                    st.metric("Ensemble (TB)", f"${ensemble_pred:,.0f}")
                with col4:
                    st.metric("Độ lệch", f"{abs(rf_pred - xgb_pred):,.0f}")

                # Biểu đồ
                fig = go.Figure()
                fig.add_trace(go.Bar(x=["RF", "XGB", "Ensemble"], y=[rf_pred, xgb_pred, ensemble_pred],
                                     text=[f"${v:,.0f}" for v in [rf_pred, xgb_pred, ensemble_pred]], textposition="outside"))
                fig.update_layout(title=f"Dự đoán tháng {pred_month} tại {pred_region} - {pred_category} ({pred_season})")
                st.plotly_chart(fig, use_container_width=True)

# ==================== TAB MODEL EVALUATION ====================
with tab6:
    st.header("ĐÁNH GIÁ MODEL TRÊN TEST DATA")
    st.markdown("Dựa trên metrics từ test_data (3).csv – MAE, RMSE, R2 từ notebook.")
    # Hardcode metrics từ notebook 
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Random Forest")
        st.metric("MAE", "177.752718")  
        st.metric("RMSE", "512.183603")
        st.metric("R2", "0.239567")
    with col2:
        st.subheader("XGBoost")
        st.metric("MAE", "172.237491")
        st.metric("RMSE", "457.964709")
        st.metric("R2", "0.392042")

# Footer
st.markdown("---")
st.caption("Dashboard by Streamlit + Plotly | Model: RF & XGBoost trên dữ liệu processed | Dữ liệu: Superstore Sales")
