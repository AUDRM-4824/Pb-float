import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import random

# Lead flotation lookup tables
COLLECTOR_LOOKUP = {
    0: {"recovery": 10.0, "grade": 45.0, "zn_activation": 0.1},
    25: {"recovery": 30.0, "grade": 48.0, "zn_activation": 0.3},
    50: {"recovery": 46.0, "grade": 55.0, "zn_activation": 0.6},
    75: {"recovery": 58.0, "grade": 45.0, "zn_activation": 1.2},
    100: {"recovery": 70.0, "grade": 40.0, "zn_activation": 2.0},
    150: {"recovery": 75.0, "grade": 30.0, "zn_activation": 3.5}
}

AIR_RATE_LOOKUP = {
    0: {"recovery": 10.0, "grade": 35.0, "zn_flotation": 0.05},
    25: {"recovery": 35.0, "grade": 50.0, "zn_flotation": 0.15},
    50: {"recovery": 45.0, "grade": 52.0, "zn_flotation": 0.25},
    75: {"recovery": 65.0, "grade": 48.0, "zn_flotation": 0.4},
    100: {"recovery": 75.0, "grade": 30.0, "zn_flotation": 0.6}
}

# SMBS lookup - note the inverse relationship for recovery vs grade
# SMBS also depresses zinc minerals, reducing zinc contamination
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

# Luproset lookup - reduces carbon content in concentrate
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
    
    # Iron grade in concentrate - starts from feed grade, reduced by SMBS
    iron_rejection_factor = smbs_metrics["iron_rejection"] / 100.0
    iron_grade = fe_feed_grade * (1.0 - iron_rejection_factor)
    
    # Carbon grade in concentrate - starts from feed grade, reduced by Luproset
    carbon_rejection_factor = luproset_metrics["carbon_rejection"] / 100.0
    carbon_grade = carbon_feed_grade * 4 * (1.0 - carbon_rejection_factor)
    
    # Zinc grade calculation - NEW
    # Base zinc flotation from collector activation and air rate
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
    iron_grade = max(0.1, min(fe_feed_grade, iron_grade))
    carbon_grade = max(0.1, min((carbon_feed_grade*4), carbon_grade))
    zinc_grade = max(0.01, min(zn_feed_grade*2, zinc_grade))  # Zinc can concentrate slightly
    
    return recovery, grade, iron_grade, carbon_grade, zinc_grade

def update_feed_grades():
    """Update feed grades with random variation within ±1%"""
    if 'current_fe_grade' not in st.session_state:
        st.session_state.current_fe_grade = 11.0
        st.session_state.current_carbon_grade = 4.5
        st.session_state.current_zn_grade = 10.5
    
    # Random variation within ±1%
    fe_variation = random.uniform(-1, 1) * st.session_state.current_fe_grade
    carbon_variation = random.uniform(-1, 1) * st.session_state.current_carbon_grade
    zn_variation = random.uniform(-1, 1) * st.session_state.current_zn_grade
    
    # Apply variations with bounds checking
    st.session_state.current_fe_grade = max(8.0, min(13.0, 
        st.session_state.current_fe_grade + fe_variation))
    st.session_state.current_carbon_grade = max(3.0, min(6.0, 
        st.session_state.current_carbon_grade + carbon_variation))
    st.session_state.current_zn_grade = max(8.0, min(13.0, 
        st.session_state.current_zn_grade + zn_variation))

# Initialize session state for dynamic mode
if 'dynamic_mode' not in st.session_state:
    st.session_state.dynamic_mode = False
if 'feed_history' not in st.session_state:
    st.session_state.feed_history = []

# Streamlit App
st.set_page_config(
    page_title="Lead Flotation Simulator",
    page_icon="⛏️",
    layout="wide"
)

st.title("⛏️ Lead Cleaner Flotation Simulator")
st.caption("Pb cleaners - Manual Training Mode")

# Dynamic mode controls - simplified
col_mode1, col_mode2, col_mode3 = st.columns([2, 2, 3])

with col_mode1:
    if st.button("🚀 Start Dynamic Mode" if not st.session_state.dynamic_mode else "⏸️ Stop Dynamic Mode"):
        st.session_state.dynamic_mode = not st.session_state.dynamic_mode
        if st.session_state.dynamic_mode:
            # Initialize current grades if starting
            if 'current_fe_grade' not in st.session_state:
                st.session_state.current_fe_grade = 11.0
                st.session_state.current_carbon_grade = 4.5
                st.session_state.current_zn_grade = 10.5

with col_mode2:
    if st.session_state.dynamic_mode:
        if st.button("🔄 Update Feed Conditions"):
            update_feed_grades()
            
            # Store history for trending
            st.session_state.feed_history.append({
                'time': time.time(),
                'fe_grade': st.session_state.current_fe_grade,
                'carbon_grade': st.session_state.current_carbon_grade,
                'zn_grade': st.session_state.current_zn_grade
            })
            
            # Keep only last 20 points
            if len(st.session_state.feed_history) > 20:
                st.session_state.feed_history = st.session_state.feed_history[-20:]

with col_mode3:
    if st.session_state.dynamic_mode:
        st.info("🔄 Dynamic mode active - Click 'Update Feed Conditions' to change feed grades")
    else:
        st.info("⚙️ Manual mode - Use sliders to set feed grades")

# Sidebar controls
st.sidebar.header("Flotation Parameters")

collector = st.sidebar.slider(
    "Collector Dosage (g/t)",
    min_value=0, max_value=150, value=50, step=5,
    help="Xanthate collector for lead mineral hydrophobicity - also activates zinc minerals"
)

air_rate = st.sidebar.slider(
    "Air Rate (L/min)",
    min_value=0, max_value=100, value=50, step=5,
    help="Bubble generation rate - increases flotation of all sulfides including zinc"
)

smbs = st.sidebar.slider(
    "SMBS Dosage (g/t)",
    min_value=0, max_value=300, value=100, step=10,
    help="Sodium Metabisulfite - depresses iron sulfides and zinc minerals"
)

ph = st.sidebar.slider(
    "pH",
    min_value=7.0, max_value=10.0, value=8.5, step=0.1,
    help="Pulp pH - higher pH improves Pb/Zn selectivity"
)

luproset = st.sidebar.slider(
    "Luproset Dosage (g/t)",
    min_value=0, max_value=500, value=100, step=25,
    help="Luproset depressant - reduces carbon content in lead concentrate"
)

st.sidebar.header("Feed Composition")

# Use dynamic grades if in dynamic mode, otherwise use sliders
if st.session_state.dynamic_mode and 'current_fe_grade' in st.session_state:
    fe_feed_grade = st.session_state.current_fe_grade
    carbon_feed_grade = st.session_state.current_carbon_grade
    zn_feed_grade = st.session_state.current_zn_grade
    
    # Display current values with visual indicators
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
    fe_feed_grade = st.sidebar.slider(
        "Feed Iron Grade (%)",
        min_value=8.0, max_value=13.0, value=11.0, step=0.1,
        help="Iron content in feed ore (mainly as pyrite)"
    )

    carbon_feed_grade = st.sidebar.slider(
        "Feed Carbon Grade (%)",
        min_value=3.0, max_value=6.0, value=4.5, step=0.1,
        help="Carbon content in feed ore (graphite, organic carbon)"
    )

    zn_feed_grade = st.sidebar.slider(
        "Feed Zinc Grade (%)",
        min_value=8.0, max_value=13.0, value=10.5, step=0.1,
        help="Zinc content in feed ore (sphalerite, zinc-bearing minerals)"
    )

# Calculate current performance
recovery, grade, iron_grade, carbon_grade, zinc_grade = calculate_performance(
    collector, air_rate, smbs, ph, luproset, fe_feed_grade, carbon_feed_grade, zn_feed_grade
)

# Main dashboard - Updated with 6 columns for zinc
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

# Educational information - Updated with zinc information
with st.expander("📚 Understanding Depressants and Selectivity in Lead Flotation"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("""
        **Sodium Metabisulfite (SMBS):**
        
        - **Primary Function**: Selective depressant for iron sulfides and zinc minerals
        - **Chemistry**: Releases SO₂ which adsorbs on mineral surfaces
        - **Grade Effect**: Positive - rejects Fe and Zn gangue minerals
        - **Recovery Effect**: Negative - some lead minerals may also be depressed
        - **Zinc Control**: Effective at reducing zinc contamination
        """)
    
    with col2:
        st.write("""
        **Luproset:**
        
        - **Primary Function**: Selective depressant for carbonaceous materials
        - **Target Minerals**: Graphite, organic carbon, coal particles
        - **Metallurgical Benefit**: Reduces carbon for cleaner smelting
        - **Recovery Effect**: Slight negative impact
        - **Zinc Effect**: Minimal direct effect on zinc minerals
        """)
    
    with col3:
        st.write("""
        **Dynamic Training Mode:**
        
        - **Simulates**: Real ore variability encountered in operations
        - **Feed Changes**: ±1% variation with each button click
        - **Operator Challenge**: Maintain performance despite feed changes
        - **Key Skills**: Quick response to grade variations
        - **Target**: Pb/Zn ratio >15 for good selectivity
        """)

    st.write("""
    **Optimization Strategy for Pb/Zn Selectivity:**
    1. **pH Optimization**: Operate at pH 8.5-9.5 for better Pb/Zn selectivity
    2. **SMBS Control**: Use sufficient SMBS to depress both iron and zinc
    3. **Collector Management**: Optimize collector to minimize zinc activation while maintaining lead recovery
    4. **Air Rate Balance**: Control air rate to avoid excessive zinc flotation
    5. **Monitor Selectivity Index**: Target Pb/Zn ratio > 15 for good selectivity
    6. **Dynamic Response**: Adjust reagents quickly when feed composition changes
    """)


# Reset button
if st.button("Reset All Trends"):
    st.session_state.feed_history = []
    st.session_state.dynamic_mode = False
    st.rerun()