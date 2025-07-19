import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

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

# Streamlit App
st.set_page_config(
    page_title="Lead Flotation Simulator",
    page_icon="⛏️",
    layout="wide"
)

st.title("⛏️ Lead Cleaner Flotation Simulator")
st.caption("Pb cleaners")

# Sidebar controls
st.sidebar.header("Flotation Parameters")

collector = st.sidebar.slider(
    "Collector Dosage (g/t)",
    min_value=0, max_value=150, value=0, step=5,
    help="Xanthate collector for lead mineral hydrophobicity - also activates zinc minerals"
)

air_rate = st.sidebar.slider(
    "Air Rate (L/min)",
    min_value=0, max_value=100, value=0, step=5,
    help="Bubble generation rate - increases flotation of all sulfides including zinc"
)

smbs = st.sidebar.slider(
    "SMBS Dosage (g/t)",
    min_value=0, max_value=300, value=0, step=10,
    help="Sodium Metabisulfite - depresses iron sulfides and zinc minerals"
)

ph = st.sidebar.slider(
    "pH",
    min_value=7.0, max_value=10.0, value=7.5, step=0.1,
    help="Pulp pH - higher pH improves Pb/Zn selectivity"
)

luproset = st.sidebar.slider(
    "Luproset Dosage (g/t)",
    min_value=0, max_value=500, value=0, step=25,
    help="Luproset depressant - reduces carbon content in lead concentrate"
)

st.sidebar.header("Feed Composition")

fe_feed_grade = st.sidebar.slider(
    "Feed Iron Grade (%)",
    min_value=8.0, max_value=13.0, value=11.0, step=0.5,
    help="Iron content in feed ore (mainly as pyrite)"
)

carbon_feed_grade = st.sidebar.slider(
    "Feed Carbon Grade (%)",
    min_value=3.0, max_value=6.0, value=4.5, step=0.1,
    help="Carbon content in feed ore (graphite, organic carbon)"
)

zn_feed_grade = st.sidebar.slider(
    "Feed Zinc Grade (%)",
    min_value=8.0, max_value=13.0, value=10.5, step=0.5,
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
        delta=f"{recovery - 65:.1f}%" if recovery != 65 else None
    )

with col2:
    st.metric(
        "Lead Grade", 
        f"{grade:.1f}%",
        delta=f"{grade - 60:.1f}%" if grade != 60 else None
    )

with col3:
    st.metric(
        "Iron in Conc.", 
        f"{iron_grade:.2f}%",
        delta=f"{iron_grade - fe_feed_grade:.2f}%" if iron_grade != fe_feed_grade else None,
        delta_color="inverse"
    )

with col4:
    st.metric(
        "Carbon in Conc.", 
        f"{carbon_grade:.2f}%",
        delta=f"{carbon_grade - carbon_feed_grade:.2f}%" if carbon_grade != carbon_feed_grade else None,
        delta_color="inverse"
    )

with col5:
    st.metric(
        "Zinc in Conc.", 
        f"{zinc_grade:.2f}%",
        delta=f"{zinc_grade - zn_feed_grade:.2f}%" if zinc_grade != zn_feed_grade else None,
        delta_color="inverse"
    )


# Performance visualization
col1, col2 = st.columns(2)

with col1:
    # Grade-Recovery plot with SMBS effect
    fig1 = go.Figure()
    
    # Generate SMBS curve
    smbs_range = np.linspace(0, 300, 20)
    recoveries = []
    grades = []
    zinc_grades = []
    
    for smbs_val in smbs_range:
        rec, gr, _, _, zn_gr = calculate_performance(collector, air_rate, smbs_val, ph, luproset, 
                                                   fe_feed_grade, carbon_feed_grade, zn_feed_grade)
        recoveries.append(rec)
        grades.append(gr)
        zinc_grades.append(zn_gr)
    
    # Add SMBS curve
    fig1.add_scatter(
        x=recoveries, y=grades,
        mode='lines+markers',
        name='SMBS Effect Curve',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    )
    
    # Add current operating point
    fig1.add_scatter(
        x=[recovery], y=[grade],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name='Current Operation'
    )
    
    # Add target zone
    fig1.add_shape(
        type="rect",
        x0=80, y0=55, x1=92, y1=65,
        fillcolor="lightgreen", opacity=0.3,
        line=dict(color="green", width=2)
    )
    
    fig1.update_layout(
        title="Grade-Recovery Performance<br><sub>Green zone shows typical targets</sub>",
        xaxis_title="Lead Recovery (%)",
        yaxis_title="Lead Grade (%)",
        showlegend=True
    )
    
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Zinc contamination vs collector/air rate
    fig2 = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Collector Effect on Zinc Contamination', 'Air Rate Effect on Zinc Contamination'),
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
    )
    
    # Collector range effect
    collector_range = np.linspace(0, 150, 30)
    collector_recoveries = []
    collector_zinc_grades = []
    
    for coll_val in collector_range:
        rec, _, _, _, zn_gr = calculate_performance(coll_val, air_rate, smbs, ph, luproset, 
                                                  fe_feed_grade, carbon_feed_grade, zn_feed_grade)
        collector_recoveries.append(rec)
        collector_zinc_grades.append(zn_gr)
    
    # Recovery vs Collector
    fig2.add_trace(
        go.Scatter(x=collector_range, y=collector_recoveries, name='Recovery', 
                  line=dict(color='blue')),
        row=1, col=1, secondary_y=False
    )
    
    # Zinc grade vs Collector
    fig2.add_trace(
        go.Scatter(x=collector_range, y=collector_zinc_grades, name='Zinc Grade', 
                  line=dict(color='red', width=3)),
        row=1, col=1, secondary_y=True
    )
    
    # Air rate range effect
    air_range = np.linspace(0, 100, 30)
    air_recoveries = []
    air_zinc_grades = []
    
    for air_val in air_range:
        rec, _, _, _, zn_gr = calculate_performance(collector, air_val, smbs, ph, luproset, 
                                                  fe_feed_grade, carbon_feed_grade, zn_feed_grade)
        air_recoveries.append(rec)
        air_zinc_grades.append(zn_gr)
    
    # Recovery vs Air rate
    fig2.add_trace(
        go.Scatter(x=air_range, y=air_recoveries, name='Recovery', 
                  line=dict(color='blue', dash='dash'), showlegend=False),
        row=2, col=1, secondary_y=False
    )
    
    # Zinc grade vs Air rate
    fig2.add_trace(
        go.Scatter(x=air_range, y=air_zinc_grades, name='Zinc Grade', 
                  line=dict(color='red', dash='dash'), showlegend=False),
        row=2, col=1, secondary_y=True
    )
    
    # Add current points
    fig2.add_vline(x=collector, line_dash="dot", line_color="green", row=1, col=1,
                   annotation_text=f"Current: {collector} g/t")
    fig2.add_vline(x=air_rate, line_dash="dot", line_color="green", row=2, col=1,
                   annotation_text=f"Current: {air_rate} L/min")
    
    fig2.update_layout(height=500, title_text="Zinc Contamination Effects")
    fig2.update_yaxes(title_text="Recovery (%)", secondary_y=False, row=1, col=1)
    fig2.update_yaxes(title_text="Zn Grade (%)", secondary_y=True, row=1, col=1)
    fig2.update_yaxes(title_text="Recovery (%)", secondary_y=False, row=2, col=1)
    fig2.update_yaxes(title_text="Zn Grade (%)", secondary_y=True, row=2, col=1)
    fig2.update_xaxes(title_text="Collector Dosage (g/t)", row=1, col=1)
    fig2.update_xaxes(title_text="Air Rate (L/min)", row=2, col=1)
    
    st.plotly_chart(fig2, use_container_width=True)

# Selectivity analysis
fig3 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('pH Effect on Pb/Zn Selectivity', 'SMBS Effect on Zn Depression'),
    specs=[[{"secondary_y": True}, {"secondary_y": True}]]
)

# pH effects on selectivity
ph_range = np.linspace(7.0, 10.0, 30)
pb_grades = []
zn_grades_ph = []
selectivities = []

for ph_val in ph_range:
    _, pb_gr, _, _, zn_gr = calculate_performance(collector, air_rate, smbs, ph_val, luproset, 
                                                fe_feed_grade, carbon_feed_grade, zn_feed_grade)
    pb_grades.append(pb_gr)
    zn_grades_ph.append(zn_gr)
    selectivities.append(pb_gr / zn_gr if zn_gr > 0 else 100)

fig3.add_trace(
    go.Scatter(x=ph_range, y=pb_grades, name='Pb Grade', line=dict(color='blue')),
    row=1, col=1, secondary_y=False
)
fig3.add_trace(
    go.Scatter(x=ph_range, y=zn_grades_ph, name='Zn Grade', line=dict(color='red')),
    row=1, col=1, secondary_y=True
)

# SMBS effects on zinc
smbs_range = np.linspace(0, 300, 30)
smbs_recoveries = []
smbs_zn_grades = []

for smbs_val in smbs_range:
    rec, _, _, _, zn_gr = calculate_performance(collector, air_rate, smbs_val, ph, luproset, 
                                              fe_feed_grade, carbon_feed_grade, zn_feed_grade)
    smbs_recoveries.append(rec)
    smbs_zn_grades.append(zn_gr)

fig3.add_trace(
    go.Scatter(x=smbs_range, y=smbs_recoveries, name='Recovery', line=dict(color='blue', dash='dash')),
    row=1, col=2, secondary_y=False
)
fig3.add_trace(
    go.Scatter(x=smbs_range, y=smbs_zn_grades, name='Zn Grade', line=dict(color='red', dash='dash')),
    row=1, col=2, secondary_y=True
)

fig3.update_layout(height=400, title_text="Selectivity Control")
fig3.update_yaxes(title_text="Pb Grade (%)", secondary_y=False, row=1, col=1)
fig3.update_yaxes(title_text="Zn Grade (%)", secondary_y=True, row=1, col=1)
fig3.update_yaxes(title_text="Recovery (%)", secondary_y=False, row=1, col=2)
fig3.update_yaxes(title_text="Zn Grade (%)", secondary_y=True, row=1, col=2)
fig3.update_xaxes(title_text="pH", row=1, col=1)
fig3.update_xaxes(title_text="SMBS Dosage (g/t)", row=1, col=2)

st.plotly_chart(fig3, use_container_width=True)

# Real-time trends - Updated to include zinc
if 'pb_history' not in st.session_state:
    st.session_state.pb_history = []

# Only add to history when parameters change
current_params = (collector, air_rate, smbs, ph, luproset, fe_feed_grade, carbon_feed_grade, zn_feed_grade)
if 'pb_last_params' not in st.session_state or st.session_state.pb_last_params != current_params:
    st.session_state.pb_history.append({
        'time': len(st.session_state.pb_history),
        'recovery': recovery,
        'grade': grade,
        'iron_grade': iron_grade,
        'carbon_grade': carbon_grade,
        'zinc_grade': zinc_grade,
        'smbs': smbs,
        'luproset': luproset,
        'selectivity': grade/zinc_grade if zinc_grade > 0 else 100
    })
    st.session_state.pb_last_params = current_params

# Keep only last 50 points
if len(st.session_state.pb_history) > 50:
    st.session_state.pb_history = st.session_state.pb_history[-50:]

# Trends plot - Updated to include zinc
if len(st.session_state.pb_history) > 1:
    df_history = pd.DataFrame(st.session_state.pb_history)
    
    fig4 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Recovery Trend', 'Grade Trends', 'Contaminants (Fe, C, Zn)', 'Selectivity Index'),
        specs=[[{"secondary_y": False}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": False}]]
    )
    
    # Recovery trend
    fig4.add_trace(
        go.Scatter(x=df_history['time'], y=df_history['recovery'], 
                  name='Recovery', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Grade trends
    fig4.add_trace(
        go.Scatter(x=df_history['time'], y=df_history['grade'], 
                  name='Pb Grade', line=dict(color='red')),
        row=1, col=2, secondary_y=False
    )
    fig4.add_trace(
        go.Scatter(x=df_history['time'], y=df_history['zinc_grade'], 
                  name='Zn Grade', line=dict(color='orange')),
        row=1, col=2, secondary_y=True
    )
    
    # Contaminants trends
    fig4.add_trace(
        go.Scatter(x=df_history['time'], y=df_history['iron_grade'], 
                  name='Iron', line=dict(color='brown')),
        row=2, col=1, secondary_y=False
    )
    fig4.add_trace(
        go.Scatter(x=df_history['time'], y=df_history['carbon_grade'], 
                  name='Carbon', line=dict(color='black')),
        row=2, col=1, secondary_y=True
    )
    fig4.add_trace(
        go.Scatter(x=df_history['time'], y=df_history['zinc_grade'], 
                  name='Zinc', line=dict(color='orange', dash='dot')),
        row=2, col=1, secondary_y=True
    )
    
    # Selectivity index
    fig4.add_trace(
        go.Scatter(x=df_history['time'], y=df_history['selectivity'], 
                  name='Pb/Zn Selectivity', line=dict(color='purple')),
        row=2, col=2
    )
    
    fig4.update_layout(height=500, title_text="Process Trends")
    fig4.update_yaxes(title_text="Recovery (%)", row=1, col=1)
    fig4.update_yaxes(title_text="Pb Grade (%)", secondary_y=False, row=1, col=2)
    fig4.update_yaxes(title_text="Zn Grade (%)", secondary_y=True, row=1, col=2)
    fig4.update_yaxes(title_text="Fe Grade (%)", secondary_y=False, row=2, col=1)
    fig4.update_yaxes(title_text="C & Zn Grade (%)", secondary_y=True, row=2, col=1)
    fig4.update_yaxes(title_text="Selectivity Ratio", row=2, col=2)
    
    st.plotly_chart(fig4, use_container_width=True)

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
        **Zinc Contamination Control:**
        
        - **Source**: Sphalerite (ZnS) activated by xanthate collectors
        - **pH Control**: Higher pH improves Pb/Zn selectivity
        - **SMBS Effect**: Depresses zinc minerals effectively
        - **Collector Control**: Lower collector reduces zinc activation
        - **Air Rate**: Higher air increases zinc flotation
        """)

    st.write("""
    **Optimization Strategy for Pb/Zn Selectivity:**
    1. **pH Optimization**: Operate at pH 8.5-9.5 for better Pb/Zn selectivity
    2. **SMBS Control**: Use sufficient SMBS to depress both iron and zinc
    3. **Collector Management**: Optimize collector to minimize zinc activation while maintaining lead recovery
    4. **Air Rate Balance**: Control air rate to avoid excessive zinc flotation
    5. **Monitor Selectivity Index**: Target Pb/Zn ratio > 15 for good selectivity
    """)

# Reset button
if st.button("Reset Trends"):
    st.session_state.pb_history = []
    st.rerun()