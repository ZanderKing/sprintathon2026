import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks
import os
import subprocess
from pathlib import Path
from datetime import datetime

# Crepitus detection parameters
THRESHOLD_MEAN_MAGNITUDE = 5705.60  # Optimal threshold determined from 4-sample analysis

# Set page configuration
st.set_page_config(
    page_title="KneeSound User Dashboard",
    page_icon="🦵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .title-style {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .subheading-style {
        font-size: 1.5rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .stats-callout {
        background-color: #f0f2f6;
        border-left: 5px solid #1f77b4;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 2rem;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Subheading
st.markdown('<div class="title-style">KneeSound User Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subheading-style">Knee Health Monitoring</div>', unsafe_allow_html=True)

# Sidebar for data file selection
st.sidebar.header("📁 Data Selection")
data_folder = Path("../data")
output_folder = Path("../output")

# Get available data files
csv_files = sorted(list(data_folder.glob("*.csv")))
file_names = [f.name for f in csv_files]

if file_names:
    selected_file_name = st.sidebar.selectbox("Select Data File:", file_names)
    selected_file_path = data_folder / selected_file_name
    st.sidebar.info(f"Selected: `{selected_file_name}`")
else:
    st.sidebar.warning("No CSV files found in data folder")
    selected_file_path = None

# Function to get latest CSV file from output folder
def get_latest_csv_file(folder_path):
    """Get the most recently modified CSV file from the specified folder."""
    try:
        csv_files = list(Path(folder_path).glob("*.csv"))
        if not csv_files:
            return None
        latest_file = max(csv_files, key=os.path.getctime)
        return latest_file
    except Exception as e:
        st.error(f"Error finding CSV file: {e}")
        return None

# Function to process data file using main.py
def process_data_file(csv_path):
    """Process a raw data CSV file using main.py"""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from main import process_data
        
        result = process_data(str(csv_path), generate_plots=True)
        return result['output_path']
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return None

# Function to load and process data
def load_data(file_path):
    """Load CSV file and prepare data for visualization."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

# Function to create magnitude visualization
def create_magnitude_visualization(df, threshold=THRESHOLD_MEAN_MAGNITUDE):
    """Create interactive time-series plot with threshold line, crepitus highlighting, and peak detection."""
    
    # Ensure we have the required columns
    if 'Time_ms' not in df.columns or 'Magnitude_1025Hz' not in df.columns:
        st.error("CSV file missing required columns: 'Time_ms' or 'Magnitude_1025Hz'")
        return None
    
    # Convert time to seconds for better readability
    df['Time_s'] = df['Time_ms'] / 1000
    magnitudes = df['Magnitude_1025Hz'].values
    
    # Identify crepitus regions (where magnitude > threshold)
    df['is_crepitus'] = magnitudes > threshold
    
    # Detect peaks in magnitude (for crepitus signal visualization)
    # Only detect peaks where signal exceeds threshold
    peak_indices, peak_properties = find_peaks(magnitudes, height=threshold, distance=10, prominence=threshold*0.1)
    
    # Create figure
    fig = go.Figure()
    
    # Add baseline and crepitus regions with different colors
    # Find contiguous regions where crepitus is detected
    crepitus_start = None
    for idx in range(len(df)):
        if df['is_crepitus'].iloc[idx] and crepitus_start is None:
            crepitus_start = idx
        elif not df['is_crepitus'].iloc[idx] and crepitus_start is not None:
            # Add shaded region for crepitus
            fig.add_vrect(
                x0=df['Time_s'].iloc[crepitus_start],
                x1=df['Time_s'].iloc[idx - 1],
                fillcolor="red",
                opacity=0.15,
                layer="below",
                line_width=0,
            )
            crepitus_start = None
    
    # Handle case where crepitus extends to end of data
    if crepitus_start is not None:
        fig.add_vrect(
            x0=df['Time_s'].iloc[crepitus_start],
            x1=df['Time_s'].iloc[-1],
            fillcolor="red",
            opacity=0.15,
            layer="below",
            line_width=0,
        )
    
    # Add magnitude line
    fig.add_trace(go.Scatter(
        x=df['Time_s'],
        y=magnitudes,
        mode='lines',
        name='Magnitude (1025 Hz)',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Magnitude:</b> %{y:.2f}<extra></extra>'
    ))
    
    # Add peak markers if crepitus is detected
    if len(peak_indices) > 0:
        peak_times = df['Time_s'].iloc[peak_indices].values
        peak_values = magnitudes[peak_indices]
        
        fig.add_trace(go.Scatter(
            x=peak_times,
            y=peak_values,
            mode='markers',
            name=f'Crepitus Peaks ({len(peak_indices)})',
            marker=dict(
                size=8,
                color='red',
                symbol='diamond',
                line=dict(color='darkred', width=2)
            ),
            hovertemplate='<b>Peak at:</b> %{x:.2f}s<br><b>Magnitude:</b> %{y:.2f}<extra></extra>'
        ))
    
    # Add threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="orange",
        annotation_text="Crepitus Threshold",
        annotation_position="right",
        annotation_font_size=11,
        annotation_font_color="orange"
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Magnitude of Target Frequency Over Time",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Time (seconds)",
        yaxis_title="Magnitude",
        hovermode='x unified',
        height=500,
        template='plotly_white',
        plot_bgcolor='rgba(240, 242, 246, 0.5)',
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig, df, peak_indices

# Function to calculate session statistics
def calculate_session_stats(df, threshold=THRESHOLD_MEAN_MAGNITUDE):
    """Calculate statistics for the current session."""
    
    if df is None or df.empty:
        return None
    
    # Count crepitus detections
    if 'Crepitus_Detected' in df.columns:
        # Count transitions from False to True (actual crepitus events)
        crepitus_count = (df['Crepitus_Detected'].astype(int).diff() == 1).sum()
    else:
        # Fallback: count samples above threshold
        above_threshold = df['Magnitude_1025Hz'] > threshold
        # Count transitions
        crepitus_count = (above_threshold.astype(int).diff() == 1).sum()
    
    # If no transitions detected, check if any crepitus detected at all
    if crepitus_count == 0 and 'Crepitus_Detected' in df.columns:
        if df['Crepitus_Detected'].any():
            crepitus_count = 1
    elif crepitus_count == 0:
        if (df['Magnitude_1025Hz'] > threshold).any():
            crepitus_count = 1
    
    # Calculate session duration
    if 'Time_ms' in df.columns:
        session_duration_s = (df['Time_ms'].max() - df['Time_ms'].min()) / 1000
    else:
        session_duration_s = 0
    
    # Calculate max magnitude
    max_magnitude = df['Magnitude_1025Hz'].max()
    
    # Calculate mean magnitude
    mean_magnitude = df['Magnitude_1025Hz'].mean()
    
    # Get severity if available
    severity = "Unknown"
    if 'Severity' in df.columns:
        severity_counts = df['Severity'].value_counts()
        severity = severity_counts.index[0] if len(severity_counts) > 0 else "Unknown"
    
    return {
        'crepitus_count': int(crepitus_count),
        'session_duration_s': session_duration_s,
        'max_magnitude': max_magnitude,
        'mean_magnitude': mean_magnitude,
        'severity': severity
    }

# Function to display statistics callout
def display_statistics_callout(stats):
    """Display session statistics in a formatted callout block."""
    
    if stats is None:
        st.warning("Unable to calculate statistics")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="stats-callout">
                <div class="stat-value">{stats['crepitus_count']}</div>
                <div class="stat-label">Crepitus Events Detected</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="stats-callout">
                <div class="stat-value">{stats['session_duration_s']:.1f}s</div>
                <div class="stat-label">Session Duration</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="stats-callout">
                <div class="stat-value">{stats['max_magnitude']:,.0f}</div>
                <div class="stat-label">Peak Magnitude</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="stats-callout">
                <div class="stat-value">{stats['severity'].upper()}</div>
                <div class="stat-label">Severity Level</div>
            </div>
            """, unsafe_allow_html=True)

# Main application logic
def main():
    # Define paths
    data_folder = Path("../data")
    output_folder = Path("../output")
    
    # Ensure output folder exists
    if not output_folder.exists():
        st.error("Output folder not found. Please run main.py first to generate processed data.")
        return
    
    # Check if user has selected a data file from sidebar
    if selected_file_path and selected_file_path.exists():
        st.info(f"📂 Processing file: `{selected_file_path.name}`")
        
        # Process the selected data file
        with st.spinner("Processing data... This may take a minute."):
            processed_csv = process_data_file(selected_file_path)
        
        if processed_csv and Path(processed_csv).exists():
            st.success("✅ Data processed successfully!")
            latest_csv = processed_csv
        else:
            st.error("Failed to process data file.")
            return
    else:
        # Fallback to latest CSV from output folder
        latest_csv = get_latest_csv_file(output_folder)
        if latest_csv is None:
            st.error("No processed data available. Please select a data file or run main.py first.")
            return
        st.info(f"📁 Using latest data file: `{latest_csv.name}`")
    
    # Create tab selector
    tab_selection = st.selectbox(
        "Select View:",
        options=["Current Session", "Statistics Summary"],
        index=0,
        label_visibility="collapsed"
    )
    
    # Load data
    df = load_data(latest_csv)
    
    if df is None or df.empty:
        st.error("Failed to load data from CSV file.")
        return
    
    # Display based on tab selection
    if tab_selection == "Current Session":
        st.markdown("---")
        
        # Create and display visualization
        result = create_magnitude_visualization(df)
        
        if result is not None:
            fig, df_processed, peak_indices = result
            st.plotly_chart(fig, use_container_width=True)
            
            # Display peak information if crepitus detected
            if len(peak_indices) > 0:
                st.success(f"🎯 **Crepitus Detected!** {len(peak_indices)} peaks identified")
                st.markdown(f"<div style='background-color: #fff3cd; padding: 15px; border-radius: 5px;'>"
                           f"<b>Peak Detection Summary:</b><br>"
                           f"• Total peaks detected: <b>{len(peak_indices)}</b><br>"
                           f"• Peak markers shown as red diamonds on the graph<br>"
                           f"• Red shaded regions indicate crepitus activity"
                           f"</div>", unsafe_allow_html=True)
            else:
                st.info("✓ No significant crepitus peaks detected in this session.")
            
            # Calculate and display statistics
            stats = calculate_session_stats(df_processed)
            st.markdown("---")
            st.markdown("### Session Summary")
            display_statistics_callout(stats)
        else:
            st.error("Failed to create visualization.")
    
    elif tab_selection == "Statistics Summary":
        st.markdown("---")
        
        # Calculate statistics
        stats = calculate_session_stats(df)
        
        if stats is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Crepitus Events", stats['crepitus_count'])
                st.metric("Session Duration", f"{stats['session_duration_s']:.1f}s")
            
            with col2:
                st.metric("Peak Magnitude", f"{stats['max_magnitude']:,.0f}")
                st.metric("Mean Magnitude", f"{stats['mean_magnitude']:,.0f}")
            
            st.metric("Severity Classification", stats['severity'].upper())
            
            # Show raw data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
        else:
            st.error("Failed to calculate statistics.")

# Run the app
if __name__ == "__main__":
    main()
