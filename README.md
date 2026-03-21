# KneeSound Clinical Analysis Dashboard

A real-time knee joint acoustic analysis system for detecting crepitus (bone pops) and monitoring osteoarthritis progression.

**James Dyson MedTech Sprintathon 2026**

Team: Jeremy Chen Yun Ze, Alvin Ong Zhao Wei, Remy Ng Hao Yuan, Baranikumar Saritha Padmanaban

## Overview

KneeSound uses a three-phase acoustic analysis pipeline to assess knee joint health:

1. **Phase 1: Macro-Movement Tracking** - Detects leg swings (20-400 Hz)
2. **Phase 2: Micro-Acoustic Tracking** - Identifies bone pops/crepitus (100-2400 Hz)
3. **Phase 3: Frequency Spectrogram** - Visualizes frequency content over time

## Repository Structure

```
sprintathon2026/
├── src/
│   ├── app.py                 # Main Streamlit dashboard (PRODUCTION)
│   ├── requirements.txt       # Python dependencies
│   └── venv-sprint/          # Virtual environment
├── data/
│   └── *.csv                 # Sample measurement data
├── main/
│   ├── Data Visualisation.py # Reference implementation
│   └── ESP32 Code.ino        # Hardware firmware
└── output/                   # Generated analysis outputs
```

## Quick Start

### 1. Setup Environment

```bash
cd src
source venv-sprint/bin/activate
pip install -r requirements.txt
```

### 2. Run Dashboard

```bash
cd src
streamlit run app.py
```

The dashboard will be available on your local host for this version release.

## Dashboard Features

### Summary Page (Default)
- View crepitus events and knee health status
- Select between **Weekly Average** or individual measurements
- Weekly Average includes trend visualization across all analyses
- Personalized health recommendations

### Detailed Statistics Page
- Full three-phase clinical analysis visualization
- Interactive plots with hover tooltips and zoom
- Detailed measurement metrics
- Phase-by-phase signal analysis

## Knee Health Classification

- **🟢 Healthy**: 0 events or < 0.2 crepitus per swing
- **🟠 At Risk**: 0.2 - 0.7 crepitus per swing  
- **🔴 Possible KOA**: > 0.7 crepitus per swing

## Algorithm Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Sampling Rate | 5000 Hz | Signal acquisition rate |
| Crack Detection Band | 100-2400 Hz | High-frequency crepitus |
| Swing Detection Band | 20-400 Hz | Low-frequency movement |
| Crepitus Threshold | 500 ADC | Amplitude threshold for bone pop detection |
| Min Peak Distance | 50 ms | Separation between distinct pops |

## Data Format

CSV files must contain:
- `Timestamp` - Time in milliseconds
- `Signal` - Raw sensor signal amplitude

## Dependencies

- **streamlit** - Web dashboard framework
- **plotly** - Interactive visualizations
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scipy** - Signal processing (Butterworth filters, peak detection, spectrograms)

## Output

Analysis generates:
- Interactive 3-phase visualization
- Crepitus event count
- Health status classification
- Recommended clinical actions
- Trend analysis across multiple measurements

## Technical Details

The pipeline uses:
- **5th-order Butterworth filters** for frequency band isolation
- **Envelope detection** with adaptive thresholding for swing identification
- **Peak detection** algorithm for individual crepitus event counting
- **STFT spectrogram** for time-frequency analysis

## References

- Data Visualisation.py - Original three-phase algorithm implementation
- ESP32 Code.ino - Hardware sensor interface code