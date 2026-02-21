import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import os
from datetime import timedelta

# --- CONFIGURATION ---
st.set_page_config(page_title="LKR AI Forecaster", layout="wide", page_icon="📈")
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000/predict")

@st.cache_data
def load_data():
    """
    Loads processed Central Bank data with smart path detection.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'processed', 'LKR_Forex_Processed.csv')
    
    if not os.path.exists(data_path):
        st.error(f"❌ Data file not found! Looked at: {data_path}")
        st.stop()
        
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# --- UI HEADER ---
st.title("🌍 LKR Exchange Rate AI Forecaster")
st.markdown("### Stochastic Gradient Boosting (LightGBM) Recursive Analysis")
st.divider()

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("Forecast Settings")
available_currencies = sorted(df['Currency'].unique())
selected_curr = st.sidebar.selectbox("Select Currency", available_currencies)

# --- NEW: CURRENCY RESET LOGIC ---
# If the user switches currency, we clear the session state to wipe old graphs
if "last_curr" not in st.session_state:
    st.session_state.last_curr = selected_curr

if st.session_state.last_curr != selected_curr:
    st.session_state.last_curr = selected_curr
    # This forces Streamlit to rerun and clear the previous "Generate Forecast" output
    st.rerun()

forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 1, 30, 14)

# Display Precision Toggle
precision_mode = st.sidebar.radio("Display Precision", ["Standard (2 Dec)", "Exact (4 Dec)"], index=0)
prec = ".4f" if precision_mode == "Exact (4 Dec)" else ".2f"

st.sidebar.markdown("---")

# Macro Stress Testing (The Shock Factor)
st.sidebar.subheader("☢️ Macro Stress Testing")
st.sidebar.info("Simulate a sudden jump/drop in global USD strength (DXY Index).")
usd_shock_pct = st.sidebar.slider("Global USD Shock (%)", -10.0, 10.0, 0.0, step=0.5)
usd_shock_decimal = usd_shock_pct / 100.0

if usd_shock_pct > 0:
    st.sidebar.warning(f"Simulating a {usd_shock_pct}% USD Rally (Bearish for LKR)")
elif usd_shock_pct < 0:
    st.sidebar.success(f"Simulating a {abs(usd_shock_pct)}% USD Drop (Bullish for LKR)")

# --- DATA PREPARATION ---
curr_df = df[df['Currency'] == selected_curr].sort_values('Date')
latest_row = curr_df.iloc[-1].to_dict()
latest_date = curr_df['Date'].iloc[-1]
curr_price = latest_row['LKR_Rate']

latest_row_json = latest_row.copy()
latest_row_json['Date'] = str(latest_date)

# --- MAIN INTERFACE: PREDICTION ---
# Button logic keeps the UI stable and gives the user control
if st.button("Generate Forecast", type="primary"):
    with st.spinner(f"Simulating {forecast_days}-day stochastic trend..."):
        try:
            # API CALL WITH SHOCK FACTOR
            response = requests.post(BACKEND_URL, json={
                "currency": selected_curr,
                "latest_data": latest_row_json,
                "days": forecast_days,
                "usd_shock": usd_shock_decimal
            })
            
            if response.status_code == 200:
                result = response.json()
                forecast_prices = result['forecast_trend']
                upper_b = result['upper_bound']
                lower_b = result['lower_bound']
                final_target_price = forecast_prices[-1]
                
                # --- 1. FORECAST INSIGHTS CARDS ---
                st.subheader("📊 Forecast Insights")
                if usd_shock_pct != 0:
                    st.caption(f"⚠️ Note: These values include a {usd_shock_pct}% artificial USD macro shock.")
                
                c1, c2, c3, c4 = st.columns(4)
                
                high_p = max(forecast_prices)
                low_p = min(forecast_prices)
                vol = (high_p - low_p) / final_target_price * 100

                # Card 1: Today
                c1.metric("Current Rate (Today)", f"Rs. {curr_price:{prec}}")
                
                # Card 2: Final Target (Day N)
                c2.metric(f"Predicted Day {forecast_days}", f"Rs. {final_target_price:{prec}}", f"{final_target_price-curr_price:+.4f}")
                
                # Card 3: Simulation Range
                with c3:
                    st.markdown("<p style='margin-bottom: -1px; font-size: 14px; color: #6e7075;'>Simulation Range</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 19px; font-weight: 500;'>Rs. {low_p:{prec}} | {high_p:{prec}}</p>", unsafe_allow_html=True)
                
                # Card 4: Volatility
                c4.metric("Market Volatility", f"{vol:.2f}%", "High" if vol > 1.2 else "Stable")

                # --- 2. THE CHART ---
                future_dates = [latest_date + timedelta(days=i+1) for i in range(forecast_days)]
                plot_dates = [latest_date] + future_dates
                
                fig = go.Figure()

                # HIGHLIGHTED FORECAST ZONE
                fig.add_vrect(
                    x0=latest_date, x1=future_dates[-1],
                    fillcolor="rgba(173, 216, 230, 0.2)", 
                    layer="below", line_width=0,
                    annotation_text="FORECAST PERIOD", annotation_position="top left"
                )

                # Confidence Interval (95%)
                fig.add_trace(go.Scatter(
                    x=future_dates + future_dates[::-1],
                    y=upper_b + lower_b[::-1],
                    fill='toself',
                    fillcolor='rgba(214, 39, 40, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name="95% Confidence Interval",
                    hoverinfo="skip"
                ))

                # Historical Data
                fig.add_trace(go.Scatter(
                    x=curr_df['Date'], y=curr_df['LKR_Rate'],
                    name="Historical Market Data", 
                    line=dict(color="#1f77b4", width=2)
                ))

                # AI Forecast Trend
                fig.add_trace(go.Scatter(
                    x=plot_dates, y=[curr_price] + forecast_prices,
                    name="AI Forecasted Trend",
                    line=dict(color="#d62728", width=4, dash='dash')
                ))

                # Final Point Marker
                fig.add_trace(go.Scatter(
                    x=[future_dates[-1]], y=[final_target_price],
                    mode='markers+text',
                    marker=dict(color='red', size=12, symbol='star'),
                    text=[f"Rs. {final_target_price:{prec}}"],
                    textposition="top center",
                    name="Target Prediction",
                    showlegend=False
                ))

                fig.update_layout(
                    title=f"{selected_curr}/LKR Forecast Analysis ({forecast_days}-Day)",
                    xaxis_title="Timeline",
                    yaxis_title="Exchange Rate (LKR)",
                    hovermode="x unified",
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(
                        rangeslider=dict(visible=True), 
                        type="date",
                        range=[latest_date - timedelta(days=30), future_dates[-1] + timedelta(days=2)]
                    ),
                    yaxis=dict(fixedrange=False, autorange=True),
                    margin=dict(l=0, r=0, t=80, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.success(f"Generated {forecast_days}-day forecast.")

            else:
                st.error(f"Backend API Error ({response.status_code}): {response.text}")
        
        except Exception as e:
            st.error(f"Connection Failed: {e}")

# SIDEBAR FOOTER
st.sidebar.markdown("---")
st.sidebar.caption(
    "**Methodology:** Stochastic Recursive Multi-step forecasting using LightGBM. "
    "Dynamic feature updates include LKR lags and moving averages."
)