# -*- coding: utf-8 -*-
"""
ĐỒ ÁN TỔNG HỢP - HKTDL - HK251
DỰ ĐOÁN DOANH THU THEO MÙA, VÙNG VÀ DANH MỤC SẢN PHẨM
SỬ DỤNG RANDOM FOREST VÀ XGBOOST + STREAMLIT DASHBOARD
"""
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ML & utils
import joblib
from sklearn.preprocessing import OneHotEncoder

# Interactivity & explainability
from streamlit_plotly_events import plotly_events
import shap

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
# -------------------- TAB DỰ ĐOÁN (Persistent points + click -> show SHAP) --------------------
# -------------------- TAB 5: DỰ ĐOÁN (RF + XGB points, click -> model-specific SHAP) --------------------
# -------------------- TAB 5: DỰ ĐOÁN (RF+XGB, prettier colors, robust click) --------------------
# -------------------- TAB 5: DỰ ĐOÁN (Hoàn chỉnh, hover top-features + click -> detail) --------------------
with tab5:
    st.header("DỰ ĐOÁN DOANH THU — RF & XGB (Interactive, persistent points)")

    # helper: month -> season
    def month_to_season(m):
        if m in (12,1,2): return "Winter"
        if m in (3,4,5): return "Spring"
        if m in (6,7,8): return "Summer"
        return "Fall"

    # Locate files
    raw_path = "train.csv"
    proc_train = None
    proc_valid = None
    for p in os.listdir("."):
        ln = p.lower()
        if "train_data" in ln and ln.endswith(".csv"):
            proc_train = p
        if "valid_data" in ln and ln.endswith(".csv"):
            proc_valid = p

    if not os.path.exists(raw_path):
        st.error("Không tìm thấy train.csv (raw). Cần file này để populate UI.")
    elif proc_train is None or proc_valid is None:
        st.error("Không tìm thấy processed train_data*.csv hoặc valid_data*.csv.")
    else:
        # Load raw (for UI + freq)
        train_raw = pd.read_csv(raw_path, low_memory=False)
        if "Order Date" in train_raw.columns:
            train_raw["Order Date"] = pd.to_datetime(train_raw["Order Date"], format="%d/%m/%Y", dayfirst=True, errors='coerce')
        total_train_rows = len(train_raw) if len(train_raw) > 0 else 1

        # Load processed (for training & infer columns)
        df_train_proc = pd.read_csv(proc_train, low_memory=False)
        df_valid_proc = pd.read_csv(proc_valid, low_memory=False)
        st.info(f"Using processed: {proc_train} ({len(df_train_proc):,}) + {proc_valid} ({len(df_valid_proc):,})")

        # Infer X columns
        proc_cols = list(df_train_proc.columns)
        X_cat_cols = [c for c in proc_cols if any(c.startswith(p) for p in ["Region_","Category_","Segment_","Ship Mode_","Ship_Mode_","Season_"])]
        numeric_candidates = ["Order_Sales_Sum","customer_avg_sales","City_freq","State_freq","Sub-Category_freq","Product Name_freq"]
        numeric_present = [c for c in numeric_candidates if c in df_train_proc.columns]

        # UI inputs (from raw)
        st.markdown("### Chọn điều kiện dự đoán")
        c1, c2 = st.columns(2)
        with c1:
            pred_region = st.selectbox("Region", options=sorted(train_raw["Region"].dropna().unique()))
            pred_category = st.selectbox("Category", options=sorted(train_raw["Category"].dropna().unique()))
            pred_subcat = st.selectbox("Sub-Category", options=sorted(train_raw[train_raw["Category"]==pred_category]["Sub-Category"].dropna().unique()))
            pred_product = st.selectbox("Product Name", options=sorted(train_raw[train_raw["Sub-Category"]==pred_subcat]["Product Name"].dropna().unique()))
            pred_ship_mode = st.selectbox("Ship Mode", options=sorted(train_raw["Ship Mode"].dropna().unique()))
        with c2:
            pred_segment = st.selectbox("Segment", options=sorted(train_raw["Segment"].dropna().unique()))
            pred_city = st.selectbox("City", options=sorted(train_raw["City"].dropna().unique()))
            pred_state = st.selectbox("State", options=sorted(train_raw[train_raw["City"]==pred_city]["State"].dropna().unique()))
            pred_month = st.selectbox("Tháng", options=list(range(1,13)), format_func=lambda x: datetime.strptime(str(x), "%m").strftime("%B"))
            pred_season = month_to_season(pred_month)

        # discrete numeric levels
        st.markdown("### Mức rời rạc (0..3)")
        pred_order_sales_sum = st.selectbox("Order_Sales_Sum (0..3)", [0,1,2,3], index=1)
        pred_customer_avg_sales = st.selectbox("customer_avg_sales (0..3)", [0,1,2,3], index=1)

        # compute freqs from raw train.csv
        def compute_freq(col, val):
            if col not in train_raw.columns:
                return 0.0
            return len(train_raw[train_raw[col] == val]) / total_train_rows
        city_freq = compute_freq("City", pred_city)
        state_freq = compute_freq("State", pred_state)
        subcat_freq = compute_freq("Sub-Category", pred_subcat)
        product_freq = compute_freq("Product Name", pred_product)

        st.write(f"City_freq={city_freq:.6f} | State_freq={state_freq:.6f} | Subcat_freq={subcat_freq:.6f} | Product_freq={product_freq:.6f}")

        # session_state init
        ss = st.session_state
        if "models_trained" not in ss: ss["models_trained"] = False
        if "rf_model" not in ss: ss["rf_model"] = None
        if "xgb_model" not in ss: ss["xgb_model"] = None
        if "X_cols" not in ss: ss["X_cols"] = None
        if "pred_points" not in ss: ss["pred_points"] = []  # list of dicts {month, rf_y, xgb_y, rf_payload, xgb_payload}

        # Predict button (train once lazily)
        if st.button("DỰ ĐOÁN & THÊM ĐIỂM (RF + XGB)"):
            if not ss["models_trained"]:
                st.info("Training models on processed train+valid (1 run)...")
                df_model = pd.concat([df_train_proc, df_valid_proc], ignore_index=True)
                if "Sales" not in df_model.columns:
                    st.error("Processed train/valid files must contain 'Sales'.")
                else:
                    X_cols = numeric_present + X_cat_cols
                    ss["X_cols"] = X_cols
                    X_train_full = df_model[X_cols].fillna(0)
                    y_train_full = df_model["Sales"].values

                    from sklearn.ensemble import RandomForestRegressor
                    from xgboost import XGBRegressor

                    rf_params = {'bootstrap': False, 'max_depth': 7, 'max_features': 0.8,
                                 'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 550,
                                 'random_state': 42, 'n_jobs': -1}
                    xgb_params = {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 6,
                                  'min_child_weight': 4, 'n_estimators': 700, 'reg_alpha': 0,
                                  'reg_lambda': 5, 'subsample': 0.9, 'random_state': 42,
                                  'verbosity':0, 'n_jobs':-1, 'objective':'reg:squarederror'}

                    with st.spinner("Training RandomForest..."):
                        rf = RandomForestRegressor(**rf_params)
                        rf.fit(X_train_full, y_train_full)
                        ss["rf_model"] = rf
                    with st.spinner("Training XGBoost..."):
                        xgb = XGBRegressor(**xgb_params)
                        xgb.fit(X_train_full, y_train_full, verbose=False)
                        ss["xgb_model"] = xgb

                    ss["models_trained"] = True
                    st.success("Training completed (models stored in session).")
            else:
                st.info("Using trained models from session_state (no retrain).")

            # Build input row according to X_cols
            if ss["X_cols"] is None:
                st.error("X_cols missing; cannot create input.")
            else:
                X_cols = ss["X_cols"]
                # init zeros
                input_dict = {c: 0 for c in X_cols}
                if "Order_Sales_Sum" in X_cols: input_dict["Order_Sales_Sum"] = pred_order_sales_sum
                if "customer_avg_sales" in X_cols: input_dict["customer_avg_sales"] = pred_customer_avg_sales
                if "City_freq" in X_cols: input_dict["City_freq"] = city_freq
                if "State_freq" in X_cols: input_dict["State_freq"] = state_freq
                if "Sub-Category_freq" in X_cols: input_dict["Sub-Category_freq"] = subcat_freq
                if "Product Name_freq" in X_cols: input_dict["Product Name_freq"] = product_freq

                # helper set one-hot
                def set_oh(pref, val):
                    t = f"{pref}_{val}"
                    if t in input_dict:
                        input_dict[t] = 1
                        return True
                    for col in X_cols:
                        if col.lower().startswith(pref.lower()+"_") and col.lower().endswith(str(val).lower().replace(" ", "_")):
                            input_dict[col] = 1
                            return True
                    return False

                set_oh("Region", pred_region)
                set_oh("Category", pred_category)
                set_oh("Segment", pred_segment)
                set_oh("Ship Mode", pred_ship_mode) or set_oh("Ship_Mode", pred_ship_mode)
                set_oh("Season", pred_season)

                X_input_df = pd.DataFrame([[input_dict.get(c,0) for c in X_cols]], columns=X_cols)

                # Predict
                try:
                    rf_pred = float(ss["rf_model"].predict(X_input_df.values)[0])
                except Exception as e:
                    st.error(f"RF predict error: {e}")
                    rf_pred = 0.0
                try:
                    xgb_pred = float(ss["xgb_model"].predict(X_input_df.values)[0])
                except Exception as e:
                    st.error(f"XGB predict error: {e}")
                    xgb_pred = 0.0

                # SHAP (best-effort)
                try:
                    expl_rf = shap.TreeExplainer(ss["rf_model"])
                    shap_rf_vals = np.array(expl_rf.shap_values(X_input_df.values)).flatten()
                except Exception:
                    shap_rf_vals = np.zeros(len(X_cols))
                try:
                    expl_xgb = shap.TreeExplainer(ss["xgb_model"])
                    shap_xgb_vals = np.array(expl_xgb.shap_values(X_input_df.values)).flatten()
                except Exception:
                    shap_xgb_vals = np.zeros(len(X_cols))

                df_rf_top = pd.DataFrame({"feature": X_cols, "shap": shap_rf_vals, "abs": np.abs(shap_rf_vals)}).sort_values("abs", ascending=False).head(8)
                df_xgb_top = pd.DataFrame({"feature": X_cols, "shap": shap_xgb_vals, "abs": np.abs(shap_xgb_vals)}).sort_values("abs", ascending=False).head(8)

                # create payloads for each model
                rf_payload = {"model":"RF", "pred": rf_pred, "top": df_rf_top.to_dict(orient="records")}
                xgb_payload = {"model":"XGB", "pred": xgb_pred, "top": df_xgb_top.to_dict(orient="records")}

                ss["pred_points"].append({
                    "month": int(pred_month),
                    "rf_y": rf_pred,
                    "xgb_y": xgb_pred,
                    "rf_payload": rf_payload,
                    "xgb_payload": xgb_payload
                })

                st.success("RF & XGB predictions appended to chart.")

        # --- Render chart (two traces) with prettier colors and hover showing top-3 features ---
        pts = ss["pred_points"]
        fig = go.Figure()
        if len(pts) == 0:
            fig.update_layout(template="plotly_white",
                              title="Prediction points (chưa có điểm)",
                              xaxis_title="Month (1..12)", yaxis_title="Predicted Sales")
            plotly_events(fig, click_event=True, key="empty_chart_tab5")
        else:
            months = [p["month"] for p in pts]
            rf_xs = [m - 0.06 for m in months]
            xgb_xs = [m + 0.06 for m in months]
            rf_ys = [p["rf_y"] for p in pts]
            xgb_ys = [p["xgb_y"] for p in pts]

            # build hover strings and customdata
            rf_custom = []
            xgb_custom = []
            rf_hovertext = []
            xgb_hovertext = []

            def format_top_lines(top_list, n=3):
                lines = []
                for i, item in enumerate(top_list[:n], 1):
                    feat = item.get("feature", str(item))
                    shapv = item.get("shap", None)
                    if shapv is not None:
                        lines.append(f"{i}) {feat}: {shapv:+.2f}")
                    else:
                        lines.append(f"{i}) {feat}")
                return "<br>".join(lines) if lines else "No SHAP"

            for p in pts:
                rf_top = p["rf_payload"]["top"] if isinstance(p.get("rf_payload"), dict) else p.get("rf_payload", {}).get("top", [])
                xgb_top = p["xgb_payload"]["top"] if isinstance(p.get("xgb_payload"), dict) else p.get("xgb_payload", {}).get("top", [])

                rf_custom.append(json.dumps(p["rf_payload"]))
                xgb_custom.append(json.dumps(p["xgb_payload"]))

                rf_hovertext.append(f"Model: RF<br>Month: {p['month']}<br>Pred: ${p['rf_y']:,.0f}<br><b>Top features (RF):</b><br>{format_top_lines(rf_top,3)}")
                xgb_hovertext.append(f"Model: XGB<br>Month: {p['month']}<br>Pred: ${p['xgb_y']:,.0f}<br><b>Top features (XGB):</b><br>{format_top_lines(xgb_top,3)}")

            # prettier colors
            color_rf = "#1f77b4"
            color_xgb = "#ff7f0e"

            fig.add_trace(go.Scatter(
                x=rf_xs, y=rf_ys, mode="markers+text", name="RF",
                marker=dict(color=color_rf, symbol="circle", size=10, line=dict(width=1, color="black")),
                text=[f"${v:,.0f}" for v in rf_ys], textposition="top center",
                customdata=rf_custom, hovertext=rf_hovertext, hoverinfo="text",
                hovertemplate="%{hovertext}<extra></extra>"
            ))

            fig.add_trace(go.Scatter(
                x=xgb_xs, y=xgb_ys, mode="markers+text", name="XGB",
                marker=dict(color=color_xgb, symbol="diamond", size=11, line=dict(width=1, color="black")),
                text=[f"${v:,.0f}" for v in xgb_ys], textposition="bottom center",
                customdata=xgb_custom, hovertext=xgb_hovertext, hoverinfo="text",
                hovertemplate="%{hovertext}<extra></extra>"
            ))

            fig.update_layout(template="plotly_white",
                              title="Predictions: RF (blue) vs XGB (orange) — click a point to see full top features",
                              xaxis=dict(title="Month (1..12)", dtick=1, range=[0.5, 12.5]),
                              yaxis=dict(title="Predicted Sales"),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

            clicked = plotly_events(fig, click_event=True, key="pred_chart_tab5")

            # Robust clicked parsing and show detailed table
            if clicked:
                ev = clicked[0]
                payload_obj = None

                # 1) top-level customdata
                if isinstance(ev, dict) and ev.get("customdata") is not None:
                    cd = ev.get("customdata")
                    if isinstance(cd, str):
                        try:
                            payload_obj = json.loads(cd)
                        except:
                            payload_obj = None
                    elif isinstance(cd, dict):
                        payload_obj = cd

                # 2) top-level pointIndex / curveNumber (your env)
                if payload_obj is None and isinstance(ev, dict) and (ev.get("pointIndex") is not None or ev.get("pointNumber") is not None):
                    try:
                        idx = int(ev.get("pointIndex") if ev.get("pointIndex") is not None else ev.get("pointNumber"))
                    except:
                        idx = None
                    curve = ev.get("curveNumber")
                    if idx is not None and "pred_points" in ss and 0 <= idx < len(ss["pred_points"]):
                        rec = ss["pred_points"][idx]
                        if curve is None:
                            payload_obj = rec.get("rf_payload") or rec.get("xgb_payload")
                        else:
                            payload_obj = rec["rf_payload"] if int(curve) == 0 else rec["xgb_payload"]

                # 3) points list structure
                if payload_obj is None and isinstance(ev, dict) and ev.get("points"):
                    first = ev["points"][0]
                    cd = first.get("customdata") or first.get("customData")
                    if cd:
                        if isinstance(cd, str):
                            try:
                                payload_obj = json.loads(cd)
                            except:
                                payload_obj = None
                        elif isinstance(cd, dict):
                            payload_obj = cd
                    if payload_obj is None:
                        # fallback to pointNumber/pointIndex
                        idx = None
                        if first.get("pointIndex") is not None:
                            try: idx = int(first.get("pointIndex"))
                            except: idx = None
                        elif first.get("pointNumber") is not None:
                            try: idx = int(first.get("pointNumber"))
                            except: idx = None
                        curve = first.get("curveNumber")
                        if idx is not None and "pred_points" in ss and 0 <= idx < len(ss["pred_points"]):
                            rec = ss["pred_points"][idx]
                            payload_obj = rec["rf_payload"] if (curve == 0) else rec["xgb_payload"]

                # 4) fallback match by x,y closeness
                if payload_obj is None and isinstance(ev, dict):
                    xval = ev.get("x")
                    yval = ev.get("y")
                    if xval is not None and yval is not None and "pred_points" in ss:
                        found = None
                        for rec in ss["pred_points"]:
                            if rec.get("month") == int(round(xval)):
                                if abs(rec.get("rf_y",0) - float(yval)) < 1e-4:
                                    found = rec.get("rf_payload"); break
                                if abs(rec.get("xgb_y",0)-float(yval)) < 1e-4:
                                    found = rec.get("xgb_payload"); break
                                # pick closest if none exact
                                if abs(rec.get("rf_y",0)-float(yval)) <= abs(rec.get("xgb_y",0)-float(yval)):
                                    found = rec.get("rf_payload")
                                else:
                                    found = rec.get("xgb_payload")
                                break
                        if found is not None:
                            payload_obj = found

                if payload_obj is None:
                    st.warning("Không tìm thấy payload từ event click. In raw event để debug:")
                    st.write(ev)
                else:
                    # parse string if necessary
                    if isinstance(payload_obj, str):
                        try:
                            payload_obj = json.loads(payload_obj)
                        except Exception as e:
                            st.error(f"Không parse được payload JSON: {e}")
                            st.write("Raw payload:", payload_obj)
                            payload_obj = None

                    if payload_obj is not None:
                        st.subheader(f"Chi tiết: model {payload_obj.get('model','')}")
                        st.write(f"Prediction: ${payload_obj.get('pred',0):,.0f}")
                        st.markdown("**Top features (by |SHAP|)**")
                        st.table(pd.DataFrame(payload_obj.get("top", [])))

        # Option to clear all points
        if st.button("XÓA TẤT CẢ ĐIỂM"):
            ss["pred_points"] = []
            st.experimental_rerun()




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
