"""
UAC Predictive Forecasting Dashboard
HHS Unaccompanied Alien Children Program
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UAC Care Load Forecasting | HHS",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700; color: #003366;
        border-bottom: 3px solid #003366; padding-bottom: 0.5rem; margin-bottom: 1rem;
    }
    .metric-card {
        background: #f0f4f8; border-radius: 10px; padding: 1rem;
        border-left: 4px solid #003366;
    }
    .warning-card {
        background: #fff3cd; border-radius: 10px; padding: 1rem;
        border-left: 4px solid #e8a200;
    }
    .alert-card {
        background: #f8d7da; border-radius: 10px; padding: 1rem;
        border-left: 4px solid #c00000;
    }
    .success-card {
        background: #d4edda; border-radius: 10px; padding: 1rem;
        border-left: 4px solid #155724;
    }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading & Preprocessing ─────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program.csv").dropna(how="all")
    df.columns = ["Date", "CBP_Apprehended", "CBP_Custody",
                  "HHS_Transfers", "HHS_Care", "HHS_Discharged"]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["HHS_Care"] = df["HHS_Care"].astype(str).str.replace(",", "").astype(float)

    # Reindex to full daily range & interpolate
    df = df.set_index("Date")
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_idx).interpolate(method="time")
    df.index.name = "Date"
    return df


@st.cache_data
def engineer_features(df, target="HHS_Care"):
    for lag in [1, 7, 14]:
        df[f"lag_{lag}"] = df[target].shift(lag)
    df["roll7_mean"]  = df[target].shift(1).rolling(7).mean()
    df["roll14_mean"] = df[target].shift(1).rolling(14).mean()
    df["roll7_std"]   = df[target].shift(1).rolling(7).std()
    df["net_pressure"]      = df["HHS_Transfers"] - df["HHS_Discharged"]
    df["net_pressure_lag1"] = df["net_pressure"].shift(1)
    df["dow"]   = df.index.dayofweek
    df["month"] = df.index.month
    return df.dropna()


FEATURES = ["lag_1", "lag_7", "lag_14", "roll7_mean", "roll14_mean",
            "roll7_std", "net_pressure", "net_pressure_lag1",
            "dow", "month", "CBP_Apprehended", "HHS_Transfers"]


@st.cache_resource
def train_models(df):
    X = df[FEATURES]
    y = df["HHS_Care"]
    split = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    gb = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                   learning_rate=0.05, random_state=42)
    gb.fit(X_train, y_train)

    models = {"Random Forest": rf, "Gradient Boosting": gb}
    preds  = {name: m.predict(X_test) for name, m in models.items()}

    # Naive & MA baselines
    naive_pred = y_test.shift(1).dropna()
    y_test_n   = y_test.loc[naive_pred.index]
    ma_pred    = y_test.rolling(7).mean().dropna()
    y_test_ma  = y_test.loc[ma_pred.index]

    metrics = {}
    for name, p in [("Naive Persistence", (naive_pred, y_test_n)),
                    ("Moving Average (7d)", (ma_pred, y_test_ma))]:
        yh, ya = p
        metrics[name] = {
            "MAE":  mean_absolute_error(ya, yh),
            "RMSE": np.sqrt(mean_squared_error(ya, yh)),
            "MAPE": np.mean(np.abs((ya - yh) / ya)) * 100,
        }
    for name, p in preds.items():
        metrics[name] = {
            "MAE":  mean_absolute_error(y_test, p),
            "RMSE": np.sqrt(mean_squared_error(y_test, p)),
            "MAPE": np.mean(np.abs((y_test - p) / y_test)) * 100,
        }

    return models, preds, y_test, X_test, metrics


@st.cache_data
def generate_forecast(df, _model, horizon, model_name):
    """Walk-forward multi-step forecast with 90% confidence bands (±1.645*std)."""
    history  = df["HHS_Care"].copy()
    net_hist = df["net_pressure"].copy()
    cbp_mean = df["CBP_Apprehended"].iloc[-30:].mean()
    tr_mean  = df["HHS_Transfers"].iloc[-30:].mean()

    forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1),
                                   periods=horizon, freq="D")
    vals, residuals = [], []

    # Collect residuals from recent test window for CI estimation
    FEATURES_ = FEATURES
    X_recent = df[FEATURES_].iloc[-60:]
    y_recent = df["HHS_Care"].iloc[-60:]
    res = y_recent.values - _model.predict(X_recent)
    res_std = res.std()

    for fdate in forecast_dates:
        row = {
            "lag_1":  history.iloc[-1],
            "lag_7":  history.iloc[-7],
            "lag_14": history.iloc[-14],
            "roll7_mean":  history.iloc[-7:].mean(),
            "roll14_mean": history.iloc[-14:].mean(),
            "roll7_std":   history.iloc[-7:].std(),
            "net_pressure":      net_hist.iloc[-1],
            "net_pressure_lag1": net_hist.iloc[-2],
            "dow":   fdate.dayofweek,
            "month": fdate.month,
            "CBP_Apprehended": cbp_mean,
            "HHS_Transfers":   tr_mean,
        }
        pred = _model.predict(pd.DataFrame([row]))[0]
        vals.append(pred)
        history[fdate]  = pred
        net_hist[fdate] = net_hist.iloc[-1]

    uncertainty_growth = np.array([res_std * np.sqrt(i + 1) for i in range(horizon)])
    z = 1.645  # 90% CI
    return pd.DataFrame({
        "Date":    forecast_dates,
        "Forecast": vals,
        "CI_Upper": np.array(vals) + z * uncertainty_growth,
        "CI_Lower": np.maximum(np.array(vals) - z * uncertainty_growth, 0),
    })


@st.cache_data
def train_discharge_model(df):
    df2 = df.copy()
    df2["lag_1d"]   = df2["HHS_Discharged"].shift(1)
    df2["lag_7d"]   = df2["HHS_Discharged"].shift(7)
    df2["roll7d"]   = df2["HHS_Discharged"].shift(1).rolling(7).mean()
    df2 = df2.dropna()
    Xd = df2[["lag_1d", "lag_7d", "roll7d", "dow", "month", "HHS_Care"]]
    yd = df2["HHS_Discharged"]
    split = int(len(df2) * 0.80)
    m = GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                  learning_rate=0.05, random_state=42)
    m.fit(Xd.iloc[:split], yd.iloc[:split])
    return m, df2


@st.cache_data
def forecast_discharge(df, _model, horizon):
    df2 = df.copy()
    df2["lag_1d"]  = df2["HHS_Discharged"].shift(1)
    df2["lag_7d"]  = df2["HHS_Discharged"].shift(7)
    df2["roll7d"]  = df2["HHS_Discharged"].shift(1).rolling(7).mean()
    df2 = df2.dropna()

    history_d = df["HHS_Discharged"].copy()
    care_last  = df["HHS_Care"].iloc[-1]
    dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1),
                          periods=horizon, freq="D")
    vals = []
    for fdate in dates:
        row = {
            "lag_1d":  history_d.iloc[-1],
            "lag_7d":  history_d.iloc[-7],
            "roll7d":  history_d.iloc[-7:].mean(),
            "dow":     fdate.dayofweek,
            "month":   fdate.month,
            "HHS_Care": care_last,
        }
        pred = _model.predict(pd.DataFrame([row]))[0]
        vals.append(max(pred, 0))
        history_d[fdate] = pred

    res_std = df["HHS_Discharged"].iloc[-60:].std() * 0.3
    unc = np.array([res_std * np.sqrt(i + 1) for i in range(horizon)])
    z = 1.645
    return pd.DataFrame({
        "Date":    dates,
        "Forecast": vals,
        "CI_Upper": np.array(vals) + z * unc,
        "CI_Lower": np.maximum(np.array(vals) - z * unc, 0),
    })


# ─── LOAD DATA ────────────────────────────────────────────────────────────────
df_raw  = load_and_prepare_data()
df_feat = engineer_features(df_raw.copy())
models, preds, y_test, X_test, metrics = train_models(df_feat)
dis_model, _ = train_discharge_model(df_feat)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Seal_of_the_United_States_Department_of_Health_and_Human_Services.svg/200px-Seal_of_the_United_States_Department_of_Health_and_Human_Services.svg.png", width=80)
    st.markdown("## 🏛️ HHS UAC Program")
    st.markdown("**Predictive Forecasting Dashboard**")
    st.divider()

    st.markdown("### ⚙️ Forecast Settings")
    horizon = st.slider("Forecast Horizon (days)", 7, 60, 30, step=7)
    selected_model = st.selectbox("Primary Forecast Model",
                                  ["Random Forest", "Gradient Boosting"])
    show_ci = st.checkbox("Show Confidence Intervals", value=True)

    st.divider()
    st.markdown("### 📅 Historical Data Range")
    st.info(f"**{df_raw.index.min().strftime('%b %d, %Y')}**\nto\n**{df_raw.index.max().strftime('%b %d, %Y')}**")
    st.metric("Total Observations", f"{len(df_feat):,}")

    st.divider()
    st.markdown("### ⚠️ Surge Alert Threshold")
    surge_thresh = st.number_input("Alert if care load exceeds:", 
                                   value=int(df_raw["HHS_Care"].max() * 0.85),
                                   step=100)

# ─── GENERATE FORECASTS ───────────────────────────────────────────────────────
fc_care = generate_forecast(df_feat, models[selected_model], horizon, selected_model)
fc_dis  = forecast_discharge(df_feat, dis_model, horizon)
surge_days = (fc_care["Forecast"] > surge_thresh).sum()

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏛️ UAC Program — Predictive Care Load & Placement Demand Forecasting</div>', 
            unsafe_allow_html=True)
st.markdown("*U.S. Department of Health and Human Services | Office of Refugee Resettlement*")

# ─── KPI ROW ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
current_care   = df_raw["HHS_Care"].iloc[-1]
forecast_peak  = fc_care["Forecast"].max()
avg_discharge  = df_raw["HHS_Discharged"].iloc[-30:].mean()
net_pres_now   = df_raw["HHS_Transfers"].iloc[-1] - df_raw["HHS_Discharged"].iloc[-1]
best_mae       = min(m["MAE"] for m in metrics.values())

k1.metric("Current Children in HHS Care", f"{current_care:,.0f}", 
          f"{current_care - df_raw['HHS_Care'].iloc[-7]:+.0f} (7d)")
k2.metric(f"Peak Forecast ({horizon}d)", f"{forecast_peak:,.0f}",
          f"{forecast_peak - current_care:+.0f} projected")
k3.metric("Avg Daily Discharges (30d)", f"{avg_discharge:.0f}",
          help="Sponsor placements per day")
k4.metric("Current Net Pressure", f"{net_pres_now:+.0f}",
          help="Transfers in minus Discharges out")
k5.metric("Best Model MAE", f"{best_mae:.1f} children",
          help="Mean Absolute Error on test set")

st.divider()

# ─── ALERT BANNER ─────────────────────────────────────────────────────────────
if surge_days > 0:
    st.markdown(f"""
    <div class="warning-card">
    ⚠️ <strong>Surge Warning:</strong> Forecast exceeds the alert threshold of 
    <strong>{surge_thresh:,}</strong> children on <strong>{surge_days}</strong> of the next {horizon} days. 
    Early resource scaling is recommended.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="success-card">
    ✅ <strong>Capacity Status:</strong> No surge projected in the {horizon}-day forecast window. 
    Current care load trend appears stable.
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# ─── TAB LAYOUT ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Care Load Forecast",
    "📤 Discharge Demand",
    "🔁 Model Comparison",
    "🔍 Historical EDA",
    "📋 Data & Export"
])

# ══ TAB 1: Care Load Forecast ════════════════════════════════════════════════
with tab1:
    st.markdown(f"### Future HHS Care Load — {horizon}-Day Forecast ({selected_model})")
    
    # Recent history + forecast
    hist_window = df_raw["HHS_Care"].iloc[-90:]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_window.index, y=hist_window.values,
        name="Historical Care Load", line=dict(color="#003366", width=2),
        hovertemplate="%{x|%b %d, %Y}<br>Children: %{y:,.0f}<extra></extra>"
    ))
    
    if show_ci:
        fig.add_trace(go.Scatter(
            x=pd.concat([fc_care["Date"], fc_care["Date"].iloc[::-1]]),
            y=pd.concat([fc_care["CI_Upper"], fc_care["CI_Lower"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(255,165,0,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="90% Confidence Band", hoverinfo="skip"
        ))
    
    fig.add_trace(go.Scatter(
        x=fc_care["Date"], y=fc_care["Forecast"],
        name=f"Forecast ({selected_model})",
        line=dict(color="#e87722", width=2.5, dash="dash"),
        hovertemplate="%{x|%b %d, %Y}<br>Forecast: %{y:,.0f}<extra></extra>"
    ))
    
    fig.add_hline(y=surge_thresh, line_dash="dot", line_color="red",
                  annotation_text=f"Surge Threshold: {surge_thresh:,}", 
                  annotation_position="bottom right")
    
    fig.add_vline(x=df_raw.index[-1], line_dash="dash", line_color="gray",
                  annotation_text="Today", annotation_position="top right")
    
    fig.update_layout(
        height=420, template="plotly_white",
        xaxis_title="Date", yaxis_title="Children in HHS Care",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=30, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📊 Forecast Summary")
        summary = fc_care.copy()
        summary["Date"] = summary["Date"].dt.strftime("%b %d, %Y")
        summary.columns = ["Date", "Forecast", "Upper (90%)", "Lower (90%)"]
        for col in ["Forecast", "Upper (90%)", "Lower (90%)"]:
            summary[col] = summary[col].apply(lambda x: f"{x:,.0f}")
        st.dataframe(summary, use_container_width=True, height=300)
    with c2:
        st.markdown("#### 🗓️ Week-by-Week Outlook")
        weeks = []
        for i in range(0, min(horizon, 28), 7):
            chunk = fc_care.iloc[i:i+7]
            if len(chunk):
                weeks.append({
                    "Week": f"Week {i//7+1} ({chunk['Date'].iloc[0].strftime('%b %d')}–{chunk['Date'].iloc[-1].strftime('%b %d')})",
                    "Avg Forecast": f"{chunk['Forecast'].mean():,.0f}",
                    "Peak": f"{chunk['Forecast'].max():,.0f}",
                    "Trend": "📈" if chunk['Forecast'].iloc[-1] > chunk['Forecast'].iloc[0] else "📉"
                })
        st.dataframe(pd.DataFrame(weeks), use_container_width=True, hide_index=True)

# ══ TAB 2: Discharge Demand ═══════════════════════════════════════════════════
with tab2:
    st.markdown(f"### Predicted Discharge (Sponsor Placement) Demand — {horizon} Days")
    
    hist_dis = df_raw["HHS_Discharged"].iloc[-90:]
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=hist_dis.index, y=hist_dis.values,
        name="Historical Discharges", marker_color="#003366",
        opacity=0.6,
        hovertemplate="%{x|%b %d}<br>Discharged: %{y:.0f}<extra></extra>"
    ))
    if show_ci:
        fig2.add_trace(go.Scatter(
            x=pd.concat([fc_dis["Date"], fc_dis["Date"].iloc[::-1]]),
            y=pd.concat([fc_dis["CI_Upper"], fc_dis["CI_Lower"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(0,180,100,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="90% CI", hoverinfo="skip"
        ))
    fig2.add_trace(go.Scatter(
        x=fc_dis["Date"], y=fc_dis["Forecast"],
        name="Discharge Forecast",
        line=dict(color="#1a7a4a", width=2.5, dash="dash"),
        hovertemplate="%{x|%b %d}<br>Forecast: %{y:.0f}<extra></extra>"
    ))
    fig2.add_vline(x=df_raw.index[-1], line_dash="dash", line_color="gray",
                   annotation_text="Today", annotation_position="top right")
    fig2.update_layout(
        height=380, template="plotly_white",
        xaxis_title="Date", yaxis_title="Daily Discharges",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=30, b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Forecasted Daily Discharges", f"{fc_dis['Forecast'].mean():.0f}")
    col2.metric("Total Projected Placements", f"{fc_dis['Forecast'].sum():,.0f}")
    col3.metric(f"Net Intake Pressure ({horizon}d)", 
                f"{(df_raw['HHS_Transfers'].iloc[-horizon:].sum() - fc_dis['Forecast'].sum()):+,.0f}",
                help="Positive = Intake exceeds projected discharges (capacity pressure)")

    st.markdown("#### Intake vs. Discharge Balance")
    fig_bal = go.Figure()
    recent_tr  = df_raw["HHS_Transfers"].iloc[-30:]
    recent_dis = df_raw["HHS_Discharged"].iloc[-30:]
    fig_bal.add_trace(go.Scatter(x=recent_tr.index, y=recent_tr.values,
                                  name="Transfers In (Historical)", 
                                  line=dict(color="#cc0000", width=2)))
    fig_bal.add_trace(go.Scatter(x=recent_dis.index, y=recent_dis.values,
                                  name="Discharges Out (Historical)", 
                                  line=dict(color="#003366", width=2)))
    fig_bal.add_trace(go.Scatter(x=fc_dis["Date"], y=fc_dis["Forecast"],
                                  name="Projected Discharges",
                                  line=dict(color="#1a7a4a", width=2, dash="dash")))
    fig_bal.update_layout(height=300, template="plotly_white",
                           legend=dict(orientation="h", y=-0.2),
                           margin=dict(t=30, b=10))
    st.plotly_chart(fig_bal, use_container_width=True)

# ══ TAB 3: Model Comparison ═══════════════════════════════════════════════════
with tab3:
    st.markdown("### Model Performance Comparison on Test Set")
    
    metrics_df = pd.DataFrame(metrics).T.reset_index()
    metrics_df.columns = ["Model", "MAE", "RMSE", "MAPE (%)"]
    metrics_df = metrics_df.sort_values("MAE")
    
    fig_m = make_subplots(rows=1, cols=3,
                          subplot_titles=("MAE (lower=better)", 
                                          "RMSE (lower=better)", 
                                          "MAPE % (lower=better)"))
    colors = ["#003366", "#e87722", "#1a7a4a", "#cc0000"]
    for i, metric_name in enumerate(["MAE", "RMSE", "MAPE (%)"]):
        fig_m.add_trace(
            go.Bar(x=metrics_df["Model"], y=metrics_df[metric_name],
                   marker_color=colors, showlegend=False,
                   text=metrics_df[metric_name].round(2),
                   textposition="outside"),
            row=1, col=i+1
        )
    fig_m.update_layout(height=380, template="plotly_white", margin=dict(t=60))
    st.plotly_chart(fig_m, use_container_width=True)
    
    st.dataframe(metrics_df.style.highlight_min(subset=["MAE","RMSE","MAPE (%)"],
                                                  color="#d4edda"), 
                 use_container_width=True, hide_index=True)
    
    st.markdown("### Actual vs. Predicted — Test Period")
    for m_name, m_preds in preds.items():
        if m_name == selected_model:
            fig_avp = go.Figure()
            fig_avp.add_trace(go.Scatter(
                x=y_test.index, y=y_test.values,
                name="Actual", line=dict(color="#003366", width=2)))
            fig_avp.add_trace(go.Scatter(
                x=y_test.index, y=m_preds,
                name=f"Predicted ({m_name})", 
                line=dict(color="#e87722", width=2, dash="dash")))
            fig_avp.update_layout(height=340, template="plotly_white",
                                   xaxis_title="Date", yaxis_title="Children in HHS Care",
                                   legend=dict(orientation="h", y=-0.2),
                                   margin=dict(t=10, b=10))
            st.plotly_chart(fig_avp, use_container_width=True)
    
    st.markdown("### Feature Importance (Gradient Boosting)")
    fi = pd.Series(models["Gradient Boosting"].feature_importances_, 
                   index=FEATURES).sort_values(ascending=True)
    fig_fi = go.Figure(go.Bar(
        x=fi.values, y=fi.index, orientation="h",
        marker_color="#003366"
    ))
    fig_fi.update_layout(height=380, template="plotly_white",
                          xaxis_title="Importance Score",
                          margin=dict(t=10))
    st.plotly_chart(fig_fi, use_container_width=True)

# ══ TAB 4: Historical EDA ════════════════════════════════════════════════════
with tab4:
    st.markdown("### Historical Trends & Exploratory Analysis")
    
    # Full care load trend
    monthly = df_raw["HHS_Care"].resample("ME").mean()
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df_raw.index, y=df_raw["HHS_Care"],
        name="Daily HHS Care Load", line=dict(color="#b0b8c8", width=1), opacity=0.5))
    fig_trend.add_trace(go.Scatter(
        x=df_raw["HHS_Care"].rolling(30).mean().index,
        y=df_raw["HHS_Care"].rolling(30).mean().values,
        name="30-Day Moving Average", line=dict(color="#003366", width=2.5)))
    fig_trend.update_layout(
        height=360, template="plotly_white",
        title="HHS Children in Care — Full Historical Trend (Jan 2023 – Dec 2025)",
        xaxis_title="Date", yaxis_title="Children",
        legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=10)
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        # Monthly avg
        monthly_avg = df_raw["HHS_Care"].resample("ME").mean().reset_index()
        fig_ma = px.bar(monthly_avg, x="Date", y="HHS_Care",
                        title="Monthly Average Care Load",
                        color="HHS_Care", color_continuous_scale="Blues",
                        labels={"HHS_Care": "Avg Children"})
        fig_ma.update_layout(height=320, template="plotly_white",
                              margin=dict(t=50, b=10), showlegend=False)
        st.plotly_chart(fig_ma, use_container_width=True)
    with c2:
        # Day of week pattern
        dow_avg = df_raw.groupby(df_raw.index.dayofweek)["HHS_Discharged"].mean()
        dow_avg.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        fig_dow = px.bar(x=dow_avg.index, y=dow_avg.values,
                         title="Avg Discharges by Day of Week",
                         labels={"x": "Day", "y": "Avg Discharges"},
                         color=dow_avg.values, color_continuous_scale="Greens")
        fig_dow.update_layout(height=320, template="plotly_white",
                               margin=dict(t=50, b=10), showlegend=False)
        st.plotly_chart(fig_dow, use_container_width=True)
    
    # Flow analysis
    fig_flow = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=("CBP Transfers into HHS",
                                             "HHS Daily Discharges"),
                             vertical_spacing=0.12)
    fig_flow.add_trace(go.Scatter(
        x=df_raw.index, y=df_raw["HHS_Transfers"],
        line=dict(color="#cc0000", width=1.2), name="HHS Transfers"), row=1, col=1)
    fig_flow.add_trace(go.Scatter(
        x=df_raw["HHS_Transfers"].rolling(14).mean().index,
        y=df_raw["HHS_Transfers"].rolling(14).mean().values,
        line=dict(color="#800000", width=2), name="14d MA Transfers"), row=1, col=1)
    fig_flow.add_trace(go.Scatter(
        x=df_raw.index, y=df_raw["HHS_Discharged"],
        line=dict(color="#1a7a4a", width=1.2), name="Discharges"), row=2, col=1)
    fig_flow.add_trace(go.Scatter(
        x=df_raw["HHS_Discharged"].rolling(14).mean().index,
        y=df_raw["HHS_Discharged"].rolling(14).mean().values,
        line=dict(color="#0d4025", width=2), name="14d MA Discharges"), row=2, col=1)
    fig_flow.update_layout(height=420, template="plotly_white",
                            legend=dict(orientation="h", y=-0.12),
                            margin=dict(t=50, b=10))
    st.plotly_chart(fig_flow, use_container_width=True)
    
    # Correlation heatmap
    corr = df_raw[["CBP_Apprehended","CBP_Custody","HHS_Transfers",
                    "HHS_Care","HHS_Discharged"]].corr()
    fig_cor = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                        title="Variable Correlation Matrix",
                        labels=dict(color="Correlation"))
    fig_cor.update_layout(height=380, margin=dict(t=50))
    st.plotly_chart(fig_cor, use_container_width=True)

# ══ TAB 5: Data & Export ══════════════════════════════════════════════════════
with tab5:
    st.markdown("### Raw Dataset Preview")
    display_df = df_raw.reset_index().tail(60)
    display_df.columns = ["Date","CBP Apprehended","CBP Custody",
                           "HHS Transfers","HHS Care","HHS Discharged"]
    display_df["Date"] = display_df["Date"].dt.strftime("%b %d, %Y")
    st.dataframe(display_df, use_container_width=True, height=350)
    
    st.markdown("### Export Forecast Data")
    export_df = fc_care.copy()
    export_df.columns = ["Date","HHS_Care_Forecast","CI_Upper_90","CI_Lower_90"]
    export_df["HHS_Discharged_Forecast"] = fc_dis["Forecast"].values
    csv_out = export_df.to_csv(index=False)
    st.download_button("⬇️ Download Forecast as CSV", csv_out,
                        file_name="UAC_Forecast.csv", mime="text/csv")
    
    st.markdown("### Dataset Summary Statistics")
    st.dataframe(df_raw.describe().round(1), use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.8rem;'>"
    "U.S. Department of Health and Human Services | Office of Refugee Resettlement | "
    "UAC Predictive Forecasting System | For Internal Planning Use Only"
    "</div>",
    unsafe_allow_html=True
)
