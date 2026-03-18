# Copilot Instructions for sprintathon2026

## Project Overview
**James Dyson MedTech Sprintathon**: Acoustic crepitus detection system for knee joint assessment. Detects pathological sounds ("bone clicks") within a specific frequency band (1000-1050 Hz) to diagnose joint conditions.

## Core Architecture

### Signal Processing Pipeline
The system processes acoustic signals in four stages:

1. **Signal Acquisition**: Hardware (Engineering student) captures multi-frequency acoustic signals → saved to `example_data.csv` (Timestamp in ms, Signal in volts)
2. **Butterworth Filtering**: 5th-order Butterworth filter isolates crepitus band (1000-1050 Hz) from raw hardware signal (`butterworth_filter.py` lines 27-31)
3. **Frequency Analysis**: FFT transforms filtered time-domain signal to frequency domain (`fft_analysis.py` lines 29-50); sliding-window STFT extracts magnitude at target frequency over time (`fft_analysis.py` lines 53-84)
4. **Target Frequency Extraction**: Magnitude of target frequency (1025 Hz) is extracted across time windows for visualization
5. **Crepitus Detection**: Mean magnitude compared to threshold (5,705.60) for crepitus classification
6. **Dashboard Visualization**: Time-series magnitude plots with threshold overlay displayed on companion app dashboard

### Key Components
- **FFT Magnitude Analysis**: `compute_fft_spectrum()` - Transforms raw signal to frequency domain; returns magnitude at each frequency bin
- **Time-Series Frequency Tracking**: `get_frequency_magnitude_timeseries()` - Uses STFT (Short-Time Fourier Transform) with sliding window (default 512 samples, 50% overlap) to track magnitude of specific frequency across time
- **Butterworth Filter**: `apply_butterworth_filter()` - Uses SciPy `signal.butter()` with second-order sections (SOS) for numerical stability
- **Crepitus Index**: `calculate_crepitus_index()` - Power ratio metric: (filtered band power / total signal power) × 100

## Critical Parameters
```python
FS = 5000               # Must match input signal sample rate (Hz)
LOW_CUT = 1000          # Crepitus band lower bound (Hz)
HIGH_CUT = 1050         # Crepitus band upper bound (Hz) - 50Hz bandwidth
TARGET_FREQUENCY = 1025 # Center frequency for detection (Hz)
CREPITUS_THRESHOLD = 5705.60  # Mean magnitude threshold for crepitus detection
ORDER = 5               # Filter order - critical for selectivity
```
**⚠️ CREPITUS_THRESHOLD determined from baseline analysis of 4 sample datasets (100% accuracy).**
**⚠️ Do not modify frequency parameters without understanding Nyquist theorem and filter rolloff characteristics.**

## Development Workflows

### Testing Changes
- **Run FFT analysis**: `python fft_analysis.py` loads `example_data.csv`, computes spectrum, and visualizes magnitude at 1025 Hz over time
- **Run crepitus detection**: `python 5thButter.py` generates synthetic signal and demonstrates Butterworth filtering
- **Run main pipeline**: `python main.py` processes data and generates visualization with threshold overlay
- **Verify threshold comparison**: Mean magnitude compared to 5,705.60 for detection; generates time-series plot
- **Plot validation**: matplotlib visualizations show pre/post-filter waveforms, frequency-domain behavior, and magnitude timeline

### CSV Data Format
Input from hardware must follow this structure:
```
Timestamp,Signal
0.00,2000
0.20,1993
...
```
- **Timestamp**: milliseconds from start
- **Signal**: voltage measurement (volts)
- Sampling rate is auto-calculated from timestamp differences

### Adding New Features
- Add new functions adjacent to existing signal processing functions
- Use numpy/scipy patterns already established (e.g., `np.square()`, `signal.sosfilt()`)
- Import additional dependencies at top of file
- Keep detection thresholds as configurable parameters (reference `crepitus_config.py` for baseline values)
- Update CREPITUS_THRESHOLD from `crepitus_config.py` when recalibrating with new sample data

## Project-Specific Patterns

### DSP Best Practices Applied
1. **SOS output format**: `signal.butter(..., output='sos')` preferred over 'ba' for numerical stability
2. **Power normalization**: Index calculation handles zero power edge case (line 16)
3. **Time-domain filtering**: Direct `signal.sosfilt()` used (prefer over FFT for short signals)

### Data Flow
```
Hardware (multi-frequency signals) → CSV file (timestamp, signal)
                                            ↓
        apply_butterworth_filter() → filtered_signal
                                            ↓
            compute_fft_spectrum() → frequencies, magnitudes (on filtered signal)
                                            ↓
      get_frequency_magnitude_timeseries() → magnitude at target freq over time
                                            ↓
        crepitus_index() → score (%) → Dashboard Visualization
```

**Team Structure**: Hardware (Engineering student) → Signal Processing (this codebase) → Dashboard Visualization

## Threshold Determination & Validation
Based on analysis of 4 sample datasets (490K total samples):
- **Baseline noise (no motion)**: Mean magnitude 3,514.78 ± 1,095.41
- **Crepitus signal (with motion)**: Mean magnitude 28,910.61 ± 20,181.96
- **Threshold**: 5,705.60 (Mean + 2σ)
- **Detection Accuracy**: 100% on validation set
- **False Positive Rate**: 0%
- **Signal Separation**: 8.23x

See `crepitus_config.py` and `BASELINE_THRESHOLD_REPORT.md` for full analysis details.

## Next Steps & Expansion Areas
- **Hardware Integration**: Accept multi-frequency signal data from hardware (Engineering student) and process through existing filter/index pipeline
- **Dashboard Visualization**: Render crepitus scores, threshold comparison, and magnitude timeline on companion app dashboard
- **Multi-frequency Band Analysis**: Extend detection to analyze multiple frequency bands simultaneously for broader pathology detection
- **Adaptive threshold detection**: Implement device-specific or patient-specific calibration
- **Real-time streaming pipeline**: Live dashboard updates with continuous magnitude monitoring
