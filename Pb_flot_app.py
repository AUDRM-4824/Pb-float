import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import random
from datetime import datetime, timedelta

# Lead flotation lookup tables
COLLECTOR_LOOKUP = {
    0: {"recovery": 10.0, "grade": 45.0, "zn_activation": 0.1, "fe_activation": 0.08},
    25: {"recovery": 30.0, "grade": 48.0, "zn_activation": 0.3, "fe_activation": 0.25},
    50: {"recovery": 46.0, "grade": 55.0, "zn_activation": 0.6, "fe_activation": 0.5},
    75: {"recovery": 58.0, "grade": 45.0, "zn_activation": 1.2, "fe_activation": 1.0},
    100: {"recovery": 70.0, "grade": 40.0, "zn_activation": 2.0, "fe_activation": 1.6},
    150: {"recovery": 75.0, "grade": 30.0, "zn_activation": 3.5, "fe_activation": 2.8}
}

AIR_RATE_LOOKUP = {
    0: {"recovery": 10.0, "grade": 35.0, "zn_flotation": 0.05, "fe_flotation": 0.03},
    150: {"recovery": 35.0, "grade": 50.0, "zn_flotation": 0.15, "fe_flotation": 0.10},
    300: {"recovery": 45.0, "grade": 52.0, "zn_flotation": 0.25, "fe_flotation": 0.18},
    450: {"recovery": 65.0, "grade": 48.0, "zn_flotation": 0.4, "fe_flotation": 0.30},
    600: {"recovery": 75.0, "grade": 30.0, "zn_flotation": 0.6, "fe_flotation": 0.45}
}

SMBS_LOOKUP = {
    0: {"recovery": 100.0, "grade": 45.0, "iron_rejection": 0.0, "zn_depression": 0.0},
    50: {"recovery": 99.0, "grade": 52.0, "iron_rejection": 15.0, "zn_depression": 10.0},
    100: {"recovery": 98.0, "grade": 58.0, "iron_rejection": 35.0, "zn_depression": 25.0},
    150: {"recovery": 92.0, "grade": 62.0, "iron_rejection": 50.0, "zn_depression": 40.0},
    200: {"recovery": 88.0, "grade": 65.0, "iron_rejection": 65.0, "zn_depression": 55.0},
    300: {"recovery": 85.0, "grade": 68.0, "iron_rejection": 80.0, "zn_depression": 70.0}
}

PH_LOOKUP = {
    7.0: {"recovery_multiplier": 1.0, "grade_bonus": -2.0, "zn_selectivity": 1.2},
    7.5: {"recovery_multiplier": 1.0, "grade_bonus": 0.0, "zn_selectivity": 1.0},
    8.0: {"recovery_multiplier": 1.0, "grade_bonus": 2.0, "zn_selectivity": 0.9},
    8.5: {"recovery_multiplier": 1.0, "grade_bonus": 3.5, "zn_selectivity": 0.8},
    9.0: {"recovery_multiplier": 1.0, "grade_bonus": 4.0, "zn_selectivity": 0.7},
    9.5: {"recovery_multiplier": 0.85, "grade_bonus": 3.0, "zn_selectivity": 0.6},
    10.0: {"recovery_multiplier": 0.70, "grade_bonus": 1.0, "zn_selectivity": 0.5}
}

LUPROSET_LOOKUP = {
    0: {"carbon_rejection": 0.0, "recovery_effect": 0.0},
    50: {"carbon_rejection": 20.0, "recovery_effect": -0.5},
    100: {"carbon_rejection": 35.0, "recovery_effect": -1.0},
    200: {"carbon_rejection": 55.0, "recovery_effect": -2.0},
    300: {"carbon_rejection": 70.0, "recovery_effect": -3.5},
    500: {"carbon_rejection": 85.0, "recovery_effect": -5.0}
}

def interpolate_lookup(value, lookup_table):
    """Interpolate between lookup table values"""
    keys = sorted(lookup_table.keys())
    
    if value <= keys[0]:
        return lookup_table[keys[0]]
    if value >= keys[-1]:
        return lookup_table[keys[-1]]
    
    # Find surrounding keys
    for i in range(len(keys) - 1):
        if keys[i] <= value <= keys[i + 1]:
            lower_key, upper_key = keys[i], keys[i + 1]
            break
    
    # Linear interpolation
    weight = (value - lower_key) / (upper_key - lower_key)
    result = {}
    
    for param in lookup_table[lower_key]:
        lower_val = lookup_table[lower_key][param]
        upper_val = lookup_table[upper_key][param]
        result[param] = lower_val + weight * (upper_val - lower_val)
    
    return result

def calculate_performance(collector, air_rate, smbs, ph, luproset, fe_feed_grade, carbon_feed_grade, zn_feed_grade):
    """Calculate lead flotation performance from parameters"""
    
    # Get individual effects
    collector_metrics = interpolate_lookup(collector, COLLECTOR_LOOKUP)
    air_metrics = interpolate_lookup(air_rate, AIR_RATE_LOOKUP)
    smbs_metrics = interpolate_lookup(smbs, SMBS_LOOKUP)
    ph_metrics = interpolate_lookup(ph, PH_LOOKUP)
    luproset_metrics = interpolate_lookup(luproset, LUPROSET_LOOKUP)
    
    # Weighted combination for recovery
    base_recovery = (collector_metrics["recovery"] * 0.55 + 
                    air_metrics["recovery"] * 0.25 + 
                    smbs_metrics["recovery"] * 0.20 +
                    65.0 * 0.10)
    
    # Apply pH multiplier
    recovery = base_recovery * ph_metrics["recovery_multiplier"]
    
    # Luproset slightly reduces recovery due to non-selective effects
    recovery += luproset_metrics["recovery_effect"]
    
    # Lead grade calculation
    base_grade = (collector_metrics["grade"] * 0.50 + 
                 air_metrics["grade"] * 0.25 + 
                 smbs_metrics["grade"] * 0.25 +
                 52.0 * 0.10)
    
    grade = base_grade + ph_metrics["grade_bonus"]
    
    # Iron grade in concentrate - starts from feed grade, affected by collector activation, air rate, and reduced by SMBS
    iron_rejection_factor = smbs_metrics["iron_rejection"] / 100.0
    
    # Collector activation effect on iron (similar to zinc but slightly less aggressive)
    collector_iron_activation = collector_metrics["fe_activation"] * fe_feed_grade * 0.15
    
    # Air rate effect on iron flotation (similar to zinc but slightly less responsive)
    air_iron_flotation = air_metrics["fe_flotation"] * fe_feed_grade
    
    # Base iron from feed, plus collector activation, plus air flotation, minus SMBS depression
    base_iron = fe_feed_grade * (1.0 - iron_rejection_factor)
    iron_grade = base_iron + collector_iron_activation + air_iron_flotation
    
    # Carbon grade in concentrate - starts from feed grade, reduced by Luproset
    carbon_rejection_factor = luproset_metrics["carbon_rejection"] / 100.0
    carbon_grade = carbon_feed_grade * 4 * (1.0 - carbon_rejection_factor)
    
    # Zinc grade calculation
    base_zn_flotation = (collector_metrics["zn_activation"] + 
                        air_metrics["zn_flotation"]) * zn_feed_grade
    
    # pH effect on zinc selectivity (higher pH reduces zinc flotation)
    zn_with_ph = base_zn_flotation * ph_metrics["zn_selectivity"]
    
    # SMBS depression effect on zinc
    zn_depression_factor = smbs_metrics["zn_depression"] / 100.0
    zinc_grade = zn_with_ph * (1.0 - zn_depression_factor)
    
    # Constraints
    recovery = max(0, min(100, recovery))
    grade = max(35, min(75, grade))
    iron_grade = max(0.1, min(fe_feed_grade * 2.5, iron_grade))
    carbon_grade = max(0.1, min((carbon_feed_grade*4), carbon_grade))
    zinc_grade = max(0.01, min(zn_feed_grade*2, zinc_grade))
    
    return recovery, grade, iron_grade, carbon_grade, zinc_grade

def update_feed_grades():
    """Update feed grades with random variation within ±15%"""
    if 'current_fe_grade' not in st.session_state:
        st.session_state.current_fe_grade = 11.0
        st.session_state.current_carbon_grade = 4.5
        st.session_state.current_zn_grade = 10.5
    
    # Random variation within ±15% (scaled up by factor of 15)
    fe_variation = random.uniform(-0.15, 0.15) * st.session_state.current_fe_grade
    carbon_variation = random.uniform(-0.15, 0.15) * st.session_state.current_carbon_grade
    zn_variation = random.uniform(-0.15, 0.15) * st.session_state.current_zn_grade
    
    # Apply variations with bounds checking
    st.session_state.current_fe_grade = max(8.0, min(13.0, 
        st.session_state.current_fe_grade + fe_variation))
    st.session_state.current_carbon_grade = max(3.0, min(6.0, 
        st.session_state.current_carbon_grade + carbon_variation))
    st.session_state.current_zn_grade = max(8.0, min(13.0, 
        st.session_state.current_zn_grade + zn_variation))

def add_to_history(timestamp, recovery, grade, iron_grade, carbon_grade, zinc_grade, 
                   collector, air_rate, smbs, ph, luproset):
    """Add current values to trending history"""
    if 'trend_history' not in st.session_state:
        st.session_state.trend_history = []
    
    pb_zn_ratio = grade / zinc_grade if zinc_grade > 0 else 0
    
    new_point = {
        'timestamp': timestamp,
        'recovery': recovery,
        'grade': grade,
        'iron_grade': iron_grade,
        'carbon_grade': carbon_grade,
        'zinc_grade': zinc_grade,
        'pb_zn_ratio': pb_zn_ratio,
        'collector': collector,
        'air_rate': air_rate,
        'smbs': smbs,
        'ph': ph,
        'luproset': luproset
    }
    
    st.session_state.trend_history.append(new_point)
    
    # Keep only last 50 points for performance
    if len(st.session_state.trend_history) > 50:
        st.session_state.trend_history = st.session_state.trend_history[-50:]

def create_trending_plots():
    """Create real-time trending plots"""
    if 'trend_history' not in st.session_state or len(st.session_state.trend_history) < 2:
        st.info("📊 Adjust the parameters to start tracking changes over time...")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.trend_history)
    
    # Create relative timestamps (seconds from start)
    start_time = df['timestamp'].iloc[0]
    df['time_elapsed'] = (df['timestamp'] - start_time).dt.total_seconds()
    
    # Create two main plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Performance Metrics Trending")
        
        # Performance metrics plot
        fig1 = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Lead Recovery (%)', 'Lead Grade (%)', 
                           'Pb/Zn Selectivity Ratio', 'Zinc in Concentrate (%)'),
            vertical_spacing=0.3,
            horizontal_spacing=0.15
        )
        
        # Lead Recovery
        fig1.add_trace(
            go.Scatter(x=df['time_elapsed'], y=df['recovery'], 
                      name='Recovery', line=dict(color='blue', width=2),
                      mode='lines+markers', marker=dict(size=4)),
            row=1, col=1
        )
        
        # Lead Grade
        fig1.add_trace(
            go.Scatter(x=df['time_elapsed'], y=df['grade'], 
                      name='Grade', line=dict(color='green', width=2),
                      mode='lines+markers', marker=dict(size=4)),
            row=1, col=2
        )
                        
        # Zinc Grade
        fig1.add_trace(
            go.Scatter(x=df['time_elapsed'], y=df['zinc_grade'], 
                      name='Zinc', line=dict(color='purple', width=2),
                      mode='lines+markers', marker=dict(size=4)),
            row=2, col=2
        )
        
        # Update layout
        fig1.update_layout(height=500, showlegend=False, title_text="Performance Trends")
        fig1.update_xaxes(title_text="Time (seconds)")
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("📉 Impurity Tracking")
        
        # Impurities and input parameters
        fig2 = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Iron in Concentrate (%)', 'Carbon in Concentrate (%)',
                           'pH Changes', 'Air Rate (m³/hr)'),
            vertical_spacing=0.3,
            horizontal_spacing=0.15
        )
        
        # Iron Grade
        fig2.add_trace(
            go.Scatter(x=df['time_elapsed'], y=df['iron_grade'], 
                      name='Iron', line=dict(color='orange', width=2),
                      mode='lines+markers', marker=dict(size=4)),
            row=1, col=1
        )
        
        # Carbon Grade
        fig2.add_trace(
            go.Scatter(x=df['time_elapsed'], y=df['carbon_grade'], 
                      name='Carbon', line=dict(color='brown', width=2),
                      mode='lines+markers', marker=dict(size=4)),
            row=1, col=2
        )
        
        # pH
        fig2.add_trace(
            go.Scatter(x=df['time_elapsed'], y=df['ph'], 
                      name='pH', line=dict(color='cyan', width=2),
                      mode='lines+markers', marker=dict(size=4)),
            row=2, col=1
        )
        
        # Air Rate
        fig2.add_trace(
            go.Scatter(x=df['time_elapsed'], y=df['air_rate'], 
                      name='Air Rate', line=dict(color='red', width=2),
                      mode='lines+markers', marker=dict(size=4)),
            row=2, col=2
        )
        
        # Update layout
        fig2.update_layout(height=500, showlegend=False, title_text="Impurities & Key Parameters")
        fig2.update_xaxes(title_text="Time (seconds)")
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Summary statistics
    st.subheader("📊 Session Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Avg Recovery", 
            f"{df['recovery'].mean():.1f}%",
            f"±{df['recovery'].std():.1f}%"
        )
    
    with col2:
        st.metric(
            "Avg Grade", 
            f"{df['grade'].mean():.1f}%",
            f"±{df['grade'].std():.1f}%"
        )
    
       
    with col4:
        st.metric(
            "Avg Iron", 
            f"{df['iron_grade'].mean():.2f}%",
            f"±{df['iron_grade'].std():.2f}%"
        )
    
    with col5:
        st.metric(
            "Data Points", 
            f"{len(df)}",
            f"Last: {df['time_elapsed'].iloc[-1]:.0f}s"
        )

# Initialize session state
if 'dynamic_mode' not in st.session_state:
    st.session_state.dynamic_mode = False
if 'feed_history' not in st.session_state:
    st.session_state.feed_history = []
if 'trend_history' not in st.session_state:
    st.session_state.trend_history = []

# Streamlit App
st.set_page_config(
    page_title="Lead Flotation Simulator",
    page_icon="⛏️",
    layout="wide"
)

st.title("⛏️ Lead Cleaner Flotation Simulator")
st.caption("Real-time Parameter Tracking & Performance Trending")

# Dynamic mode controls
col_mode1, col_mode2, col_mode3 = st.columns([2, 2, 3])

with col_mode1:
    if st.button("🚀 Start Dynamic Mode" if not st.session_state.dynamic_mode else "⏸️ Stop Dynamic Mode"):
        st.session_state.dynamic_mode = not st.session_state.dynamic_mode
        if st.session_state.dynamic_mode:
            if 'current_fe_grade' not in st.session_state:
                st.session_state.current_fe_grade = 11.0
                st.session_state.current_carbon_grade = 4.5
                st.session_state.current_zn_grade = 10.5

with col_mode2:
    if st.session_state.dynamic_mode:
        if st.button("🔄 Update Feed Conditions"):
            update_feed_grades()
            st.session_state.feed_history.append({
                'time': time.time(),
                'fe_grade': st.session_state.current_fe_grade,
                'carbon_grade': st.session_state.current_carbon_grade,
                'zn_grade': st.session_state.current_zn_grade
            })
            if len(st.session_state.feed_history) > 20:
                st.session_state.feed_history = st.session_state.feed_history[-20:]

with col_mode3:
    if st.button("🗑️ Clear Trending Data"):
        st.session_state.trend_history = []
        st.rerun()

# Sidebar controls
st.sidebar.header("Flotation Parameters")

# Convert all sliders to number inputs
collector = st.sidebar.number_input(
    "Collector Dosage (g/t)",
    min_value=0, max_value=150, value=50, step=5,
    help="Xanthate collector for lead mineral hydrophobicity - also activates zinc and iron minerals"
)

air_rate = st.sidebar.number_input(
    "Air Rate (m³/hr)",
    min_value=0, max_value=600, value=300, step=25,
    help="Bubble generation rate - increases flotation of all sulfides including zinc and iron"
)

smbs = st.sidebar.number_input(
    "SMBS Dosage (g/t)",
    min_value=0, max_value=300, value=100, step=10,
    help="Sodium Metabisulfite - depresses iron sulfides and zinc minerals"
)

ph = st.sidebar.number_input(
    "pH",
    min_value=7.0, max_value=10.0, value=8.5, step=0.1, format="%.1f",
    help="Pulp pH - higher pH improves Pb/Zn selectivity"
)

luproset = st.sidebar.number_input(
    "Luproset Dosage (g/t)",
    min_value=0, max_value=500, value=100, step=25,
    help="Luproset depressant - reduces carbon content in lead concentrate"
)

st.sidebar.header("Feed Composition")

# Use dynamic grades if in dynamic mode, otherwise use number inputs
if st.session_state.dynamic_mode and 'current_fe_grade' in st.session_state:
    fe_feed_grade = st.session_state.current_fe_grade
    carbon_feed_grade = st.session_state.current_carbon_grade
    zn_feed_grade = st.session_state.current_zn_grade
    
    st.sidebar.metric(
        "Feed Iron Grade (%)",
        f"{fe_feed_grade:.2f}",
        delta=f"{fe_feed_grade - 11.0:.2f}" if abs(fe_feed_grade - 11.0) > 0.01 else None
    )
    
    st.sidebar.metric(
        "Feed Carbon Grade (%)",
        f"{carbon_feed_grade:.2f}",
        delta=f"{carbon_feed_grade - 4.5:.2f}" if abs(carbon_feed_grade - 4.5) > 0.01 else None
    )
    
    st.sidebar.metric(
        "Feed Zinc Grade (%)",
        f"{zn_feed_grade:.2f}",
        delta=f"{zn_feed_grade - 10.5:.2f}" if abs(zn_feed_grade - 10.5) > 0.01 else None
    )
    
else:
    fe_feed_grade = st.sidebar.number_input(
        "Feed Iron Grade (%)",
        min_value=8.0, max_value=13.0, value=11.0, step=0.1, format="%.1f",
        help="Iron content in feed ore (mainly as pyrite)"
    )

    carbon_feed_grade = st.sidebar.number_input(
        "Feed Carbon Grade (%)",
        min_value=3.0, max_value=6.0, value=4.5, step=0.1, format="%.1f",
        help="Carbon content in feed ore (graphite, organic carbon)"
    )

    zn_feed_grade = st.sidebar.number_input(
        "Feed Zinc Grade (%)",
        min_value=8.0, max_value=13.0, value=10.5, step=0.1, format="%.1f",
        help="Zinc content in feed ore (sphalerite, zinc-bearing minerals)"
    )

# Calculate current performance
recovery, grade, iron_grade, carbon_grade, zinc_grade = calculate_performance(
    collector, air_rate, smbs, ph, luproset, fe_feed_grade, carbon_feed_grade, zn_feed_grade
)

# Add to trending history (only when parameters change significantly)
current_timestamp = datetime.now()
should_add_point = True

if st.session_state.trend_history:
    last_point = st.session_state.trend_history[-1]
    # Check if any parameter changed significantly
    param_changes = [
        abs(collector - last_point['collector']) > 4,
        abs(air_rate - last_point['air_rate']) > 24,
        abs(smbs - last_point['smbs']) > 9,
        abs(ph - last_point['ph']) > 0.05,
        abs(luproset - last_point['luproset']) > 20
    ]
    should_add_point = any(param_changes)

if should_add_point:
    add_to_history(current_timestamp, recovery, grade, iron_grade, carbon_grade, zinc_grade,
                   collector, air_rate, smbs, ph, luproset)

# Main dashboard
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric(
        "Lead Recovery", 
        f"{recovery:.1f}%",
        delta=f"{recovery - 65:.1f}%" if abs(recovery - 65) > 0.1 else None
    )

with col2:
    st.metric(
        "Lead Grade", 
        f"{grade:.1f}%",
        delta=f"{grade - 60:.1f}%" if abs(grade - 60) > 0.1 else None
    )

with col3:
    st.metric(
        "Iron in Conc.", 
        f"{iron_grade:.2f}%",
        delta=f"{iron_grade - 5.0:.2f}%" if abs(iron_grade - 5.0) > 0.01 else None,
        delta_color="inverse"
    )

with col4:
    st.metric(
        "Carbon in Conc.", 
        f"{carbon_grade:.2f}%",
        delta=f"{carbon_grade - 7.0:.2f}%" if abs(carbon_grade - 7.0) > 0.01 else None,
        delta_color="inverse"
    )

with col5:
    st.metric(
        "Zinc in Conc.", 
        f"{zinc_grade:.2f}%",
        delta=f"{zinc_grade - 8.0:.2f}%" if abs(zinc_grade - 8.0) > 0.01 else None,
        delta_color="inverse"
    )

# Add separator
st.markdown("---")

# Real-time trending plots
st.header("📈 Real-Time Parameter Trending")
st.caption("Adjust parameters to see how changes affect performance over time")

create_trending_plots()

# Educational information
with st.expander("📚 How to Use Real-Time Trending"):
    st.write("""
    **Getting Started:**
    1. **Adjust any parameter** in the sidebar to start collecting data points
    2. **Watch the trends** develop as you make changes to parameters
    3. **Experiment** with different combinations to see their effects
    4. **Use "Clear Trending Data"** to reset and start fresh
    
    **What to Look For:**
    - **Recovery vs Grade Trade-offs**: Higher collector often increases recovery but may reduce grade
    - **Selectivity Patterns**: How pH and SMBS changes affect the Pb/Zn ratio
    - **Impurity Control**: How Luproset affects carbon content over time
    - **Parameter Interactions**: How multiple changes combine to affect performance
    
    **Optimization Tips:**
    - Target Pb/Zn ratio > 15 for good selectivity
    - Monitor iron and carbon levels - keep them low
    - pH 8.5-9.0 typically gives best selectivity
    - SMBS is crucial for zinc depression
    """)

# Reset button
if st.button("Reset All Data"):
    st.session_state.feed_history = []
    st.session_state.trend_history = []
    st.session_state.dynamic_mode = False
    st.rerun()