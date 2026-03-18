# KneeSound User Dashboard - Usage Guide

## Overview

The **KneeSound User Dashboard** is a Streamlit-based application for visualizing knee crepitus detection results from accelerometer data.

## How to Run

From the project root directory:

```bash
cd src
source venv-sprint/bin/activate
streamlit run dashboard.py
```

The dashboard will open in your default browser at `http://localhost:8501`

## Features

### 1. Data File Selection
- **Sidebar Menu**: Browse and select any CSV file from the `data/` folder
- **Automatic Processing**: Selected files are automatically processed using the main.py pipeline
- **Status Indicators**: Shows which file is currently being analyzed

### 2. Current Session Tab (Default)
Displays a time-series visualization with:

#### Enhanced Peak Detection
- **Red Diamond Markers**: Clearly mark all detected crepitus peaks above the threshold
- **Red Shaded Regions**: Background highlighting shows zones where crepitus was detected
- **Orange Dashed Line**: Threshold value (5,705.60) for crepitus detection
- **Blue Solid Line**: Magnitude of target frequency over time (1025 Hz)

#### Peak Summary
- **Peak Count**: Shows total number of crepitus peaks detected
- **Success Badge**: Green indicator when crepitus is found ("Crepitus Detected!")
- **Info Message**: Explains what each visual element means

#### Session Statistics
Below the graph, displays:
- **Total Crepitus Events**: Number of distinct crepitus detections
- **Session Duration**: Total time of recording
- **Peak Magnitude**: Highest magnitude value recorded
- **Severity Level**: Classification (baseline, borderline, moderate, strong)

### 3. Statistics Summary Tab
Displays:
- Crepitus event count
- Session duration
- Peak magnitude value
- Mean magnitude across session
- Overall severity classification
- Data preview table (first 10 rows)

## Understanding the Visualization

### When Crepitus is Detected:
✅ Red diamond markers appear on peaks above the threshold  
✅ Red shaded regions highlight active crepitus zones  
✅ Success message confirms detection  
✅ Peak count shows how many events occurred  

### When No Crepitus is Detected:
✓ Clean blue line shows magnitude baseline  
✓ No peaks above threshold line  
✓ Info message indicates clean session  
✓ Statistics show baseline severity level  

## Data Processing Pipeline

When you select a data file:

1. **Raw Data Loading**: CSV from data/ folder is loaded
2. **Processing**: main.py runs with algorithms for:
   - Crack & snap detection (100-2400 Hz)
   - Swing tracking (20-400 Hz)
   - Target frequency analysis (1000-1050 Hz)
3. **Output Generation**: Creates processed CSV with detection results
4. **Visualization**: Results displayed in dashboard

## Supported Data Formats

Input CSV files must have columns:
- `Timestamp` - Time in milliseconds
- `Signal` - Raw accelerometer signal data

Example:
```
Timestamp,Signal
0.0,1500.5
20.2,1510.3
40.4,1505.8
...
```

## Algorithm Parameters

**Crepitus Detection Threshold**: 5,705.60  
**Target Frequency**: 1,025 Hz  
**Frequency Band**: 1,000 - 1,050 Hz  
**Filter Type**: 5th Order Butterworth Bandpass  

## Troubleshooting

### Dashboard won't start
```bash
# Ensure Streamlit is installed
pip install streamlit plotly pandas numpy scipy

# Try again
streamlit run dashboard.py
```

### No data files visible
- Check that CSV files exist in `data/` folder
- Verify file format has `Timestamp` and `Signal` columns

### Processing takes too long
- Large files (>1 minute of data) may take 30+ seconds to process
- Wait for the spinner to complete

## Tips for Best Results

1. **Use Consistent Data**: All test samples should have 5000 Hz sampling rate
2. **Check Peaks**: Look for red diamond markers to quickly spot crepitus events
3. **Review Statistics**: Use severity level to assess joint health
4. **Compare Samples**: Load different files to compare detection results

## Key Features

🎯 **Peak Detection**: Automatically identifies and marks crepitus peaks  
📊 **Interactive Graphs**: Hover to see exact values and timestamps  
📈 **Multi-Tab Interface**: Switch between visualization and statistics  
🚀 **Real-time Processing**: Select a file and see results instantly  
📱 **Responsive Design**: Works on desktop and tablet browsers  

---

**Version**: 1.0  
**Last Updated**: March 18, 2026  
**Status**: Production Ready ✅
