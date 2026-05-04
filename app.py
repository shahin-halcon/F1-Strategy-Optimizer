import streamlit as st
import pandas as pd
import xgboost as xgb
import plotly.graph_objects as go

# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(page_title="F1 Strategy Optimizer", page_icon="🏎️", layout="wide", initial_sidebar_state="expanded")

# Inject Custom CSS for that sleek, dark F1 aesthetic
st.markdown("""
    <style>
    .stApp {
        background-color: #111111;
        color: #FFFFFF;
    }
    /* Style the metric cards */
    div[data-testid="metric-container"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    h1, h2, h3 {
        color: #E10600; /* Official F1 Red */
        font-family: 'Arial', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. LOAD THE AI MODEL ---
@st.cache_resource # This prevents the model from reloading every time you move a slider
def load_model():
    model = xgb.XGBRegressor()
    try:
        model.load_model('f1_tire_model.json')
        return model
    except:
        return None

ai_model = load_model()

# --- 3. THE SIMULATION ENGINE ---
def simulate_stint(start_lap, end_lap, compound, model):
    stint_data = []
    tyre_life = 1
    for lap in range(start_lap, end_lap + 1):
        stint_data.append({
            'LapNumber': lap,
            'TyreLife': tyre_life,
            'Compound_SOFT': 1 if compound == 'SOFT' else 0,
            'Compound_MEDIUM': 1 if compound == 'MEDIUM' else 0,
            'Compound_HARD': 1 if compound == 'HARD' else 0
        })
        tyre_life += 1
    
    df = pd.DataFrame(stint_data)[['LapNumber', 'TyreLife', 'Compound_SOFT', 'Compound_MEDIUM', 'Compound_HARD']]
    predictions = model.predict(df)
    return list(range(start_lap, end_lap + 1)), predictions.tolist()

# --- 4. THE UI LAYOUT ---

# Sidebar Controls
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=100) # F1 Logo
    st.title("Race Parameters")
    pit_loss = st.slider("Pit Stop Time Loss (sec)", 15.0, 30.0, 20.0, step=0.5)
    
    st.markdown("---")
    st.header("Strategy A (1-Stop)")
    strat_a_start = st.selectbox("Start Tire", ["MEDIUM", "SOFT", "HARD"], index=0, key="a_start")
    strat_a_pit = st.slider("Pit Lap", 10, 40, 25, key="a_pit")
    strat_a_end = st.selectbox("End Tire", ["HARD", "MEDIUM", "SOFT"], index=0, key="a_end")

    st.markdown("---")
    st.header("Strategy B (2-Stop)")
    strat_b_start = st.selectbox("Start Tire", ["SOFT", "MEDIUM", "HARD"], index=0, key="b_start")
    strat_b_pit1 = st.slider("First Pit Lap", 5, 25, 15, key="b_pit1")
    strat_b_mid = st.selectbox("Middle Tire", ["MEDIUM", "SOFT", "HARD"], index=0, key="b_mid")
    strat_b_pit2 = st.slider("Second Pit Lap", 26, 45, 35, key="b_pit2")
    strat_b_end = st.selectbox("Final Tire", ["SOFT", "MEDIUM", "HARD"], index=0, key="b_end")

# Main Content Area
st.title("Tire Degradation & Pit-Stop Optimizer")
st.write("Powered by XGBoost Machine Learning | Bahrain Grand Prix Telemetry")

if ai_model is None:
    st.error("⚠️ Could not find 'f1_tire_model.json'. Make sure you ran the model training script first!")
else:
    # --- RUN SIMULATIONS ---
    # Strategy A
    laps_a1, times_a1 = simulate_stint(1, strat_a_pit, strat_a_start, ai_model)
    laps_a2, times_a2 = simulate_stint(strat_a_pit + 1, 50, strat_a_end, ai_model)
    
    # Add pit stop penalty
    times_a1[-1] += pit_loss
    
    total_time_a = sum(times_a1) + sum(times_a2)
    laps_A = laps_a1 + laps_a2
    times_A = times_a1 + times_a2

    # Strategy B
    laps_b1, times_b1 = simulate_stint(1, strat_b_pit1, strat_b_start, ai_model)
    laps_b2, times_b2 = simulate_stint(strat_b_pit1 + 1, strat_b_pit2, strat_b_mid, ai_model)
    laps_b3, times_b3 = simulate_stint(strat_b_pit2 + 1, 50, strat_b_end, ai_model)
    
    # Add pit stop penalties
    times_b1[-1] += pit_loss
    times_b2[-1] += pit_loss
    
    total_time_b = sum(times_b1) + sum(times_b2) + sum(times_b3)
    laps_B = laps_b1 + laps_b2 + laps_b3
    times_B = times_b1 + times_b2 + times_b3

    # --- TOP METRICS ---
    col1, col2, col3 = st.columns(3)
    
    winner = "Strategy A" if total_time_a < total_time_b else "Strategy B"
    delta = abs(total_time_a - total_time_b)

    with col1:
        st.metric(label="Strategy A Total Time", value=f"{total_time_a:.1f} s")
    with col2:
        st.metric(label="Strategy B Total Time", value=f"{total_time_b:.1f} s")
    with col3:
        st.metric(label="🏆 Optimal Strategy", value=winner, delta=f"-{delta:.2f} s faster")

    # --- INTERACTIVE PLOTLY CHART ---
    fig = go.Figure()

    # Add Strategy A line
    fig.add_trace(go.Scatter(x=laps_A, y=times_A, mode='lines', name='Strategy A (1-Stop)', 
                             line=dict(color='#00D2BE', width=3))) # Mercedes-ish Teal
    
    # Add Strategy B line
    fig.add_trace(go.Scatter(x=laps_B, y=times_B, mode='lines', name='Strategy B (2-Stop)', 
                             line=dict(color='#FF8700', width=3))) # McLaren-ish Orange

    fig.update_layout(
        title="Lap Time Evolution & Pit Stop Degradation",
        xaxis_title="Race Lap Number",
        yaxis_title="Predicted Lap Time (Seconds)",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- RAW DATA EXPANDER ---
    with st.expander("View Raw AI Predictions"):
        df_results = pd.DataFrame({
            "Lap": laps_A,
            "Strat A Time": [round(t, 3) for t in times_A],
            "Strat B Time": [round(t, 3) for t in times_B]
        })
        st.dataframe(df_results, use_container_width=True)