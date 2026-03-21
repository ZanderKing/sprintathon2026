"""
Fresh Streamlit Dashboard - Using Data Visualisation.py Algorithms Only
Shows 3-phase analysis: Macro-Movement (Swings), Micro-Acoustic (Cracks), and Spectrogram
Multi-page dashboard with summary and detailed analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt, find_peaks, spectrogram
from pathlib import Path
from datetime import datetime

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="KneeSound - Clinical Analysis Dashboard",
    page_icon="🦵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# ALGORITHM PARAMETERS (From Data Visualisation.py)
# ==========================================
SAMPLE_RATE = 5000

# Filter Settings for High-Pitch Snaps/Cracks
CRACK_LOW = 100
CRACK_HIGH = 2400

# Filter Settings for Low-Pitch Leg Swings
SWING_LOW = 20
SWING_HIGH = 400
ENVELOPE_CUTOFF = 2

# Noise Floor Gates
MIN_SWING_VOLUME = 50
MIN_CRACK_VOLUME = 500

# ==========================================
# STYLING
# ==========================================
st.markdown("""
    <style>
    .title {
        font-size: 2.8rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #f5f5f5;
        border-radius: 0.5rem;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-box {
        background-color: #f0f4f8;
        border-radius: 0.5rem;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .health-healthy {
        color: #4caf50;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .health-at-risk {
        color: #ff9800;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .health-koa {
        color: #f44336;
        font-weight: bold;
        font-size: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# INITIALIZE SESSION STATE
# ==========================================
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = {}  # {filename: {'crepitus_events': int, 'swing_count': int, 'timestamp': datetime}}

def get_health_status(crepitus_events, swing_count):
    """Determine knee health status based on crepitus events and swing count"""
    if swing_count == 0:
        if crepitus_events == 0:
            return "Healthy", "🟢"
        else:
            return "At Risk", "🟠"
    
    crepitus_per_swing = crepitus_events / swing_count
    
    if crepitus_events == 0:
        return "Healthy", "🟢"
    elif crepitus_per_swing < 0.2:
        return "Healthy", "🟢"
    elif crepitus_per_swing <= 0.7:
        return "At Risk", "🟠"
    else:
        return "Possible KOA", "🔴"

def get_recommendations(status):
    """Get recommended actions based on health status"""
    if status == "Healthy":
        return [
            "Continue regular monitoring",
            "Maintain current exercise routine",
            "No medical intervention needed",
            "Schedule routine check-ups annually"
        ]
    elif status == "At Risk":
        return [
            "Consult a healthcare provider",
            "Increase physical therapy sessions",
            "Reduce high-impact activities",
            "Monitor symptoms closely",
            "Consider anti-inflammatory interventions"
        ]
    else:  # Possible KOA
        return [
            "Seek immediate medical attention",
            "Consider advanced diagnostic imaging",
            "Begin formal treatment program",
            "Consult orthopedic specialist",
            "Explore treatment options including physical therapy or medication"
        ]

# ==========================================
# TITLE & DESCRIPTION
# ==========================================
st.markdown('<div class="title">KneeSound Clinical Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Three-Phase Movement & Acoustic Analysis</div>', unsafe_allow_html=True)

# ==========================================
# SIDEBAR - FILE SELECTION
# ==========================================
st.sidebar.header("📁 Data Selection")

# Set paths
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
data_folder = project_root / "data"

# Get available CSV files
csv_files = sorted(list(data_folder.glob("*.csv")))
file_names = [f.name for f in csv_files]

if file_names:
    selected_file = st.sidebar.selectbox("Choose a data file:", file_names)
    file_path = data_folder / selected_file
else:
    st.error("❌ No CSV files found in data folder")
    st.stop()

# ==========================================
# ALGORITHM FUNCTIONS
# ==========================================
def analyze_data(csv_path):
    """
    Run the three-phase analysis algorithm from Data Visualisation.py
    Returns all computed data for visualization
    """
    # Load data
    df = pd.read_csv(csv_path).dropna()
    
    if 'Signal' not in df.columns or 'Timestamp' not in df.columns:
        raise ValueError("CSV must contain 'Timestamp' and 'Signal' columns.")
    
    df['Time_sec'] = df['Timestamp'] / 1000.0
    raw_signal = df['Signal'] - df['Signal'].mean()
    nyquist = 0.5 * SAMPLE_RATE
    
    # ==========================================
    # ALGORITHM 1: CRACK & SNAP DETECTION
    # ==========================================
    b_crack, a_crack = butter(5, [CRACK_LOW / nyquist, CRACK_HIGH / nyquist], btype='bandpass')
    crack_signal = filtfilt(b_crack, a_crack, raw_signal)
    crack_signal_gated = np.where(np.abs(crack_signal) > MIN_CRACK_VOLUME, crack_signal, 0)
    
    # Detect crepitus events (bone pops) as distinct peaks in the crack signal
    crack_peaks, _ = find_peaks(np.abs(crack_signal), height=MIN_CRACK_VOLUME, distance=int(SAMPLE_RATE * 0.05))
    crepitus_event_count = len(crack_peaks)
    
    # ==========================================
    # ALGORITHM 2: SWING TRACKING (ENVELOPE)
    # ==========================================
    b_swing, a_swing = butter(3, [SWING_LOW / nyquist, SWING_HIGH / nyquist], btype='bandpass')
    swing_friction = filtfilt(b_swing, a_swing, raw_signal)
    
    b_env, a_env = butter(3, ENVELOPE_CUTOFF / nyquist, btype='low')
    envelope = filtfilt(b_env, a_env, np.abs(swing_friction))
    
    # Peak detection
    min_swing_distance = int(SAMPLE_RATE * 1.2)
    dynamic_threshold = np.mean(envelope) + (np.std(envelope) * 0.2)
    final_swing_threshold = max(dynamic_threshold, MIN_SWING_VOLUME)
    
    peaks, _ = find_peaks(envelope, distance=min_swing_distance, height=final_swing_threshold)
    swing_count = len(peaks)
    
    # ==========================================
    # SPECTROGRAM DATA
    # ==========================================
    f, t, Sxx = spectrogram(crack_signal, SAMPLE_RATE, nperseg=1024, noverlap=512)
    
    return {
        'df': df,
        'raw_signal': raw_signal,
        'swing_friction': swing_friction,
        'envelope': envelope,
        'peaks': peaks,
        'swing_count': swing_count,
        'crack_signal_gated': crack_signal_gated,
        'crepitus_event_count': crepitus_event_count,
        'crack_peaks': crack_peaks,
        'f': f,
        't': t,
        'Sxx': Sxx,
        'final_swing_threshold': final_swing_threshold
    }

def create_visualizations(data):
    """
    Create interactive 3-panel visualization using Plotly
    Matching Data Visualisation.py analysis structure
    """
    df = data['df']
    swing_friction = data['swing_friction']
    envelope = data['envelope']
    peaks = data['peaks']
    swing_count = data['swing_count']
    crack_signal_gated = data['crack_signal_gated']
    f = data['f']
    t = data['t']
    Sxx = data['Sxx']
    final_swing_threshold = data['final_swing_threshold']
    
    # Create subplots with Plotly
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f'Phase 1: Macro-Movement Tracking ({swing_count} Swings Detected)',
            'Phase 2: Micro-Acoustic Tracking (Looking for Bone Pops)',
            'Phase 3: Frequency Spectrogram'
        ),
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}]
        ],
        vertical_spacing=0.12,
        row_heights=[0.33, 0.33, 0.34]
    )
    
    # ==========================================
    # PHASE 1: MACRO-MOVEMENT TRACKING
    # ==========================================
    # Joint friction signal
    fig.add_trace(
        go.Scatter(
            x=df['Time_sec'],
            y=swing_friction,
            mode='lines',
            name='Joint Friction (20-400Hz)',
            line=dict(color='lightgray', width=1),
            opacity=0.7,
            hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Friction:</b> %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Movement envelope
    fig.add_trace(
        go.Scatter(
            x=df['Time_sec'],
            y=envelope,
            mode='lines',
            name='Movement Envelope',
            line=dict(color='dodgerblue', width=2),
            hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Envelope:</b> %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Detected swings (peaks)
    fig.add_trace(
        go.Scatter(
            x=df['Time_sec'].iloc[peaks],
            y=envelope[peaks],
            mode='markers',
            name=f'Detected Swings ({swing_count})',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            hovertemplate='<b>Swing Detected</b><br><b>Time:</b> %{x:.2f}s<br><b>Amplitude:</b> %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Noise gate line
    fig.add_hline(
        y=MIN_SWING_VOLUME,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Noise Gate ({MIN_SWING_VOLUME})",
        annotation_position="right",
        row=1, col=1
    )
    
    # Vertical lines for peaks
    for peak in peaks:
        fig.add_vline(
            x=df['Time_sec'].iloc[peak],
            line_dash="dash",
            line_color="red",
            opacity=0.3,
            row=1, col=1
        )
    
    # ==========================================
    # PHASE 2: MICRO-ACOUSTIC TRACKING
    # ==========================================
    fig.add_trace(
        go.Scatter(
            x=df['Time_sec'],
            y=crack_signal_gated,
            mode='lines',
            name='Isolated Snaps/Cracks (100-2400Hz)',
            line=dict(color='crimson', width=0.8),
            hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Amplitude:</b> %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Noise gates
    fig.add_hline(y=MIN_CRACK_VOLUME, line_dash="dot", line_color="black", opacity=0.5, row=2, col=1)
    fig.add_hline(y=-MIN_CRACK_VOLUME, line_dash="dot", line_color="black", opacity=0.5, row=2, col=1)
    
    # ==========================================
    # PHASE 3: FREQUENCY SPECTROGRAM
    # ==========================================
    # Convert Sxx to dB scale
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)
    
    fig.add_trace(
        go.Heatmap(
            z=Sxx_dB,
            x=t,
            y=f,
            colorscale='Viridis',
            name='Spectrogram',
            hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Frequency:</b> %{y:.1f} Hz<br><b>Intensity:</b> %{z:.1f} dB<extra></extra>',
            colorbar=dict(title="Intensity [dB]", x=1.02)
        ),
        row=3, col=1
    )
    
    # Update y-axis for spectrogram
    fig.update_yaxes(range=[0, 2500], row=3, col=1)
    
    # ==========================================
    # UPDATE LAYOUT & AXES
    # ==========================================
    fig.update_yaxes(title_text="Friction Volume", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (ADC)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency [Hz]", row=3, col=1)
    
    fig.update_xaxes(title_text="Time [Seconds]", row=3, col=1)
    
    # Enable grid for all traces
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=1, col=1)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=2, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=1, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=2, col=1)
    
    # Set x-axis range consistency
    x_min = df['Time_sec'].min()
    x_max = df['Time_sec'].max()
    fig.update_xaxes(range=[x_min, x_max], row=1, col=1)
    fig.update_xaxes(range=[x_min, x_max], row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=1000,
        title_text="<b>Three-Phase Clinical Analysis</b>",
        title_font_size=16,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
    )
    
    return fig

# ==========================================
# MAIN DASHBOARD - MULTI-PAGE STRUCTURE
# ==========================================

# Display file info
st.markdown(f'<div class="info-box"><b>📂 Selected File:</b> {selected_file}</div>', unsafe_allow_html=True)

# Process button
if st.button("🔍 Analyze Data", type="primary", use_container_width=True):
    with st.spinner("Processing data... Running 3-phase analysis..."):
        try:
            # Run analysis
            data = analyze_data(file_path)
            swing_count = data['swing_count']
            crack_events = data['crepitus_event_count']
            
            # Update session state history
            st.session_state.analysis_history[selected_file] = {
                'swing_count': swing_count,
                'crepitus_events': crack_events,
                'data': data,
                'timestamp': datetime.now()
            }
            
            st.success("✅ Analysis Complete! Go to Summary to view results.")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# ==========================================
# PAGE SELECTION
# ==========================================
st.divider()

page = st.radio(
    "📱 Select Page:",
    ["Summary", "Detailed Statistics"],
    horizontal=True
)

# ==========================================
# PAGE 1: SUMMARY
# ==========================================
if page == "Summary":
    st.subheader("📊 Summary Dashboard")
    
    if len(st.session_state.analysis_history) == 0:
        st.info("📌 Analyze at least one CSV file to see summary statistics.")
    else:
        # Dropdown to select which analysis to display
        analyzed_files = sorted(list(st.session_state.analysis_history.keys()))
        
        # Calculate weekly average
        total_crepitus = sum([v['crepitus_events'] for v in st.session_state.analysis_history.values()])
        total_swings = sum([v['swing_count'] for v in st.session_state.analysis_history.values()])
        weekly_avg_crepitus = total_crepitus / len(analyzed_files) if len(analyzed_files) > 0 else 0
        weekly_avg_swings = total_swings / len(analyzed_files) if len(analyzed_files) > 0 else 0
        
        # Add "Weekly Average" option at the beginning
        display_options = ["📊 Weekly Average"] + analyzed_files
        selected_display = st.selectbox(
            "Select which data to display:",
            display_options,
            index=0
        )
        
        # Determine which data to show
        is_weekly_average = selected_display == "📊 Weekly Average"
        
        if is_weekly_average:
            display_crepitus = weekly_avg_crepitus
            display_swings = weekly_avg_swings
            display_status = "Weekly Average"
            health_status, health_emoji = get_health_status(display_crepitus, display_swings)
        else:
            file_data = st.session_state.analysis_history[selected_display]
            display_crepitus = file_data['crepitus_events']
            display_swings = file_data['swing_count']
            display_status = selected_display
            health_status, health_emoji = get_health_status(display_crepitus, display_swings)
        
        # Display status boxes - adjusted columns based on view type
        if is_weekly_average:
            col1, col2 = st.columns(2)
        else:
            col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div style="font-size: 0.9rem; color: #666;">Crepitus Events</div>
                <div style="font-size: 2.5rem; font-weight: bold; color: #1f77b4; margin: 0.5rem 0;">
                    {display_crepitus:.1f}
                </div>
                <div style="font-size: 0.85rem; color: #666;">({display_status})</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            status_color = "#4caf50" if health_status == "Healthy" else ("#ff9800" if health_status == "At Risk" else "#f44336")
            st.markdown(f"""
            <div class="status-box">
                <div style="font-size: 0.9rem; color: #666;">Knee Health Status</div>
                <div style="font-size: 1.8rem; margin: 0.5rem 0;">{health_emoji}</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: {status_color};">
                    {health_status}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if not is_weekly_average:
            with col3:
                crepitus_per_swing = display_crepitus / display_swings if display_swings > 0 else 0
                st.markdown(f"""
                <div class="metric-box">
                    <div style="font-size: 0.9rem; color: #666;">Events per Swing</div>
                    <div style="font-size: 2.5rem; font-weight: bold; color: #1f77b4; margin: 0.5rem 0;">
                        {crepitus_per_swing:.2f}
                    </div>
                    <div style="font-size: 0.85rem; color: #666;">Crepitus Ratio</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # Show time series graph for weekly average
        if is_weekly_average:
            st.subheader("📈 Crepitus Events Timeline")
            
            if len(analyzed_files) > 1:
                # Create weekly monitoring chart
                timeline_data = {
                    'File': analyzed_files,
                    'Crepitus Events': [st.session_state.analysis_history[f]['crepitus_events'] for f in analyzed_files],
                }
                
                fig_weekly = go.Figure()
                
                fig_weekly.add_trace(go.Scatter(
                    x=timeline_data['File'],
                    y=timeline_data['Crepitus Events'],
                    mode='lines+markers',
                    name='Crepitus Events',
                    line=dict(color='#f44336', width=3),
                    marker=dict(size=10),
                    hovertemplate='<b>%{x}</b><br>Crepitus Events: %{y}<extra></extra>'
                ))
                
                fig_weekly.update_layout(
                    title="<b>Crepitus Events Trend Over Time</b>",
                    xaxis_title="CSV File (Chronological Order)",
                    yaxis_title="Number of Crepitus Events",
                    height=400,
                    hovermode='closest',
                    plot_bgcolor='rgba(240,240,240,0.5)',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_weekly, use_container_width=True, config={"displayModeBar": True})
                
                st.info("📌 This chart shows the trend of crepitus events across all analyzed measurements. An increasing trend may indicate joint deterioration.")
            else:
                st.info("📌 Analyze multiple CSV files to see the weekly monitoring chart.")
            
            st.divider()
        
        # Display recommendations
        recommendations = get_recommendations(health_status)
        
        if health_status == "Healthy":
            st.markdown(f"""
            <div class="success-box">
                <div style="font-size: 1.3rem; margin-bottom: 1rem;">
                    <b>✅ Healthy Knee Joint</b>
                </div>
                <div style="margin-bottom: 1rem;">
                    <b>Clinical Assessment:</b> Your knee shows healthy acoustic characteristics with no significant crepitus detected.
                </div>
                <div>
                    <b>Recommended Actions:</b>
                    <ul>
            """, unsafe_allow_html=True)
        elif health_status == "At Risk":
            st.markdown(f"""
            <div class="warning-box">
                <div style="font-size: 1.3rem; margin-bottom: 1rem;">
                    <b>⚠️ At Risk</b>
                </div>
                <div style="margin-bottom: 1rem;">
                    <b>Clinical Assessment:</b> Mild crepitus detected. Early intervention may help prevent further joint deterioration.
                </div>
                <div>
                    <b>Recommended Actions:</b>
                    <ul>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="danger-box">
                <div style="font-size: 1.3rem; margin-bottom: 1rem;">
                    <b>🔴 Possible KOA</b>
                </div>
                <div style="margin-bottom: 1rem;">
                    <b>Clinical Assessment:</b> Significant crepitus pattern detected. This may indicate knee osteoarthritis. Professional evaluation recommended.
                </div>
                <div>
                    <b>Recommended Actions:</b>
                    <ul>
            """, unsafe_allow_html=True)
        
        for rec in recommendations:
            st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
        st.markdown("</ul></div></div>", unsafe_allow_html=True)

# ==========================================
# PAGE 2: DETAILED STATISTICS
# ==========================================
elif page == "Detailed Statistics":
    st.subheader("📈 Detailed Analysis")
    
    if len(st.session_state.analysis_history) == 0:
        st.info("📌 Analyze at least one CSV file to view detailed statistics.")
    else:
        analyzed_files = sorted(list(st.session_state.analysis_history.keys()))
        
        # Dropdown to select which analysis to display
        selected_file_detail = st.selectbox(
            "Select which CSV file to analyze:",
            analyzed_files,
            key="detail_selectbox"
        )
        
        file_data = st.session_state.analysis_history[selected_file_detail]
        file_swing_count = file_data['swing_count']
        file_crack_events = file_data['crepitus_events']
        
        # Get health status for this file
        file_status, file_emoji = get_health_status(file_crack_events, file_swing_count)
        recommendations = get_recommendations(file_status)
        
        # Status callout
        if file_status == "Healthy":
            st.markdown(f"""
            <div class="success-box">
                <div style="font-size: 1.3rem; margin-bottom: 1rem;">
                    <b>{file_emoji} Knee Health Status: {file_status}</b>
                </div>
                <div style="margin-bottom: 1rem;">
                    <b>Clinical Assessment:</b> No crepitus events detected. Your knee joint shows healthy acoustic characteristics.
                </div>
                <div>
                    <b>Recommended Actions:</b>
                    <ul>
            """, unsafe_allow_html=True)
            for rec in recommendations:
                st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
            st.markdown("</ul></div></div>", unsafe_allow_html=True)
        
        elif file_status == "At Risk":
            crepitus_per_swing = file_crack_events / file_swing_count if file_swing_count > 0 else 0
            st.markdown(f"""
            <div class="warning-box">
                <div style="font-size: 1.3rem; margin-bottom: 1rem;">
                    <b>{file_emoji} Knee Health Status: {file_status}</b>
                </div>
                <div style="margin-bottom: 1rem;">
                    <b>Clinical Assessment:</b> Mild crepitus detected ({file_crack_events} events, {crepitus_per_swing:.2f} per swing). This may indicate early joint involvement.
                </div>
                <div>
                    <b>Recommended Actions:</b>
                    <ul>
            """, unsafe_allow_html=True)
            for rec in recommendations:
                st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
            st.markdown("</ul></div></div>", unsafe_allow_html=True)
        
        else:  # Possible KOA
            crepitus_per_swing = file_crack_events / file_swing_count if file_swing_count > 0 else 0
            st.markdown(f"""
            <div class="danger-box">
                <div style="font-size: 1.3rem; margin-bottom: 1rem;">
                    <b>{file_emoji} Knee Health Status: {file_status}</b>
                </div>
                <div style="margin-bottom: 1rem;">
                    <b>Clinical Assessment:</b> Significant crepitus detected ({file_crack_events} events, {crepitus_per_swing:.2f} per swing). This pattern is consistent with possible knee osteoarthritis.
                </div>
                <div>
                    <b>Recommended Actions:</b>
                    <ul>
            """, unsafe_allow_html=True)
            for rec in recommendations:
                st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
            st.markdown("</ul></div></div>", unsafe_allow_html=True)
        
        st.divider()
        
        # Display the 3-phase visualization
        st.subheader("📊 Three-Phase Clinical Analysis")
        fig = create_visualizations(file_data['data'])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "responsive": True})
        
        # Summary statistics
        st.divider()
        st.subheader("📋 Measurement Details")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔴 Leg Swings Detected", file_swing_count)
        with col2:
            st.metric("⚡ Bone Pops Detected", file_crack_events)
        with col3:
            crepitus_ratio = file_crack_events / file_swing_count if file_swing_count > 0 else 0
            st.metric("📊 Pops per Swing", f"{crepitus_ratio:.2f}")
        with col4:
            st.metric("⏱️ Recording Duration", f"{file_data['data']['df']['Time_sec'].max():.1f}s")

# ==========================================
# FOOTER
# ==========================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85rem;'>
<b>KneeSound Clinical Analysis Dashboard</b> | Three-Phase Movement & Acoustic Analysis<br>
Using algorithms from Data Visualisation.py | MedTech Sprintathon 2026
</div>
""", unsafe_allow_html=True)
