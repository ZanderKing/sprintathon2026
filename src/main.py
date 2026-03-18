"""
Main processing pipeline for MedTech Sprintathon
Pipeline: Hardware data → Butterworth 5th Order filter → FFT analysis → Target frequency extraction → CSV output
Outputs processed results with magnitude of target frequency over time for visualization with crepitus threshold overlay
Also includes crack/snap detection and swing tracking algorithms from Data Visualisation.py
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks, spectrogram
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Import functions from existing modules
from fft_analysis import (
    load_hardware_data, 
    compute_fft_spectrum, 
    get_frequency_magnitude_timeseries
)
from butterworth_filter import apply_butterworth_filter, calculate_crepitus_index, FS
from crepitus_config import (
    THRESHOLD_MEAN_MAGNITUDE,
    TARGET_FREQUENCY_HZ,
    FREQUENCY_BAND_LOW_HZ,
    FREQUENCY_BAND_HIGH_HZ,
    SEVERITY_THRESHOLDS
)

# ==========================================
# ALGORITHM PARAMETERS (From Data Visualisation.py)
# ==========================================
# Filter Settings for High-Pitch Snaps/Cracks
CRACK_LOW = 100
CRACK_HIGH = 2400
MIN_CRACK_VOLUME = 100  # If a snap is smaller than 100 ADC units, ignore it

# Filter Settings for Low-Pitch Leg Swings
SWING_LOW = 20
SWING_HIGH = 400
ENVELOPE_CUTOFF = 2
MIN_SWING_VOLUME = 50  # If the envelope is below 50, it is just skin friction/handling noise


def process_data(csv_input_path, output_dir=None, target_frequency=None, 
                 low_cut=None, high_cut=None, order=5, generate_plots=True):
    """
    Main processing pipeline: Load CSV -> Apply Butterworth filter -> FFT analysis -> Output CSV + Visualization
    Also performs crack/snap detection and swing tracking algorithms
    
    Args:
        csv_input_path: Path to input CSV file
        output_dir: Directory to save output CSV (default: project_root/output)
        target_frequency: Frequency of interest for magnitude timeseries (Hz)
        low_cut: Lower cutoff frequency for bandpass filter (Hz)
        high_cut: Upper cutoff frequency for bandpass filter (Hz)
        order: Filter order (default 5)
        generate_plots: Whether to generate visualization plots (default True)
    
    Returns:
        dict with output_path, detection_result, severity, swing_count, and crack_analysis
    """
    
    # Set default output directory if not provided
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, 'output')
    
    print("=" * 60)
    print("MedTech Sprintathon Data Processing Pipeline")
    print("=" * 60)
    
    # 1. Load data from CSV
    print(f"\n[1] Loading data from: {csv_input_path}")
    timestamps, signal_data = load_hardware_data(csv_input_path)
    
    # Remove NaN values if present
    valid_idx = ~np.isnan(signal_data)
    if not np.all(valid_idx):
        print(f"    Warning: Found {np.sum(~valid_idx)} NaN values, removing...")
        signal_data = signal_data[valid_idx]
        timestamps = timestamps[valid_idx]
    
    print(f"    Loaded {len(signal_data)} samples")
    print(f"    Time range: {timestamps[0]:.2f} to {timestamps[-1]:.2f} ms")
    
    # 2. Compute FFT spectrum to get sampling rate
    print(f"\n[2] Computing FFT spectrum...")
    frequencies, magnitudes, sampling_rate = compute_fft_spectrum(signal_data, timestamps)
    print(f"    Sampling rate: {sampling_rate:.2f} Hz")
    print(f"    Frequency range: {frequencies[0]:.2f} to {frequencies[-1]:.2f} Hz")
    
    # Set default frequency parameters
    if target_frequency is None:
        target_frequency = TARGET_FREQUENCY_HZ
    if low_cut is None:
        low_cut = FREQUENCY_BAND_LOW_HZ
    if high_cut is None:
        high_cut = FREQUENCY_BAND_HIGH_HZ
    
    print(f"    Using frequencies - Target: {target_frequency:.0f} Hz, Band: {low_cut:.0f}-{high_cut:.0f} Hz")
    
    # 3. Apply Butterworth 5th Order bandpass filter FIRST
    print(f"\n[3] Applying Butterworth {order}th Order bandpass filter")
    print(f"    Filter range: {low_cut:.0f} - {high_cut:.0f} Hz")
    filtered_signal = apply_butterworth_filter(signal_data, low_cut, high_cut, 
                                                sampling_rate, order=order)
    
    # ==========================================
    # ALGORITHM 1: CRACK & SNAP DETECTION
    # ==========================================
    print(f"\n[3A] Running Crack & Snap Detection Algorithm")
    nyquist = 0.5 * sampling_rate
    b_crack, a_crack = butter(5, [CRACK_LOW / nyquist, CRACK_HIGH / nyquist], btype='bandpass')
    crack_signal = filtfilt(b_crack, a_crack, signal_data)
    # Apply noise gate to remove handling noise
    crack_signal_gated = np.where(np.abs(crack_signal) > MIN_CRACK_VOLUME, crack_signal, 0)
    crack_peaks = np.sum(np.abs(crack_signal_gated) > MIN_CRACK_VOLUME)
    print(f"    Detected {crack_peaks} potential snap/crack events (>{MIN_CRACK_VOLUME} ADC units)")
    
    # ==========================================
    # ALGORITHM 2: SWING TRACKING (ENVELOPE)
    # ==========================================
    print(f"\n[3B] Running Swing Tracking Algorithm")
    b_swing, a_swing = butter(3, [SWING_LOW / nyquist, SWING_HIGH / nyquist], btype='bandpass')
    swing_friction = filtfilt(b_swing, a_swing, signal_data)
    
    b_env, a_env = butter(3, ENVELOPE_CUTOFF / nyquist, btype='low')
    envelope = filtfilt(b_env, a_env, np.abs(swing_friction))
    
    # Peak detection using dynamic threshold
    min_swing_distance = int(sampling_rate * 1.2)
    dynamic_threshold = np.mean(envelope) + (np.std(envelope) * 0.2)
    final_swing_threshold = max(dynamic_threshold, MIN_SWING_VOLUME)
    
    peaks, _ = find_peaks(envelope, distance=min_swing_distance, height=final_swing_threshold)
    swing_count = len(peaks)
    print(f"    Detected {swing_count} full leg swings")
    print(f"    Dynamic threshold: {dynamic_threshold:.2f}, Final threshold: {final_swing_threshold:.2f}")
    
    # Store swing and crack detection results
    swing_detection = {
        'swing_count': swing_count,
        'swing_peaks': peaks,
        'envelope': envelope,
        'swing_friction': swing_friction,
    }
    
    crack_detection = {
        'crack_peaks': crack_peaks,
        'crack_signal_gated': crack_signal_gated,
        'crack_signal': crack_signal,
    }
    
    # 4. Calculate crepitus index
    print(f"\n[4] Calculating crepitus index...")
    crepitus_score = calculate_crepitus_index(signal_data, filtered_signal)
    print(f"    Crepitus Index Score: {crepitus_score:.4f}%")
    
    # 5. Compute FFT on the FILTERED signal to break down to individual frequencies
    print(f"\n[5] Computing FFT on filtered signal...")
    frequencies_filtered, magnitudes_filtered, _ = compute_fft_spectrum(filtered_signal, timestamps)
    print(f"    FFT spectrum computed on filtered signal")
    
    # Calculate mean magnitude in frequency band for threshold comparison
    band_mask = (frequencies_filtered >= low_cut) & (frequencies_filtered <= high_cut)
    band_magnitudes = magnitudes_filtered[band_mask]
    # Filter out any NaN or inf values
    valid_magnitudes = band_magnitudes[~(np.isnan(band_magnitudes) | np.isinf(band_magnitudes))]
    mean_magnitude = np.mean(valid_magnitudes) if len(valid_magnitudes) > 0 else 0
    is_crepitus = mean_magnitude > THRESHOLD_MEAN_MAGNITUDE
    
    # 6. Extract magnitude at target frequency over time from filtered signal (for visualization)
    print(f"\n[6] Extracting magnitude timeseries at {target_frequency} Hz from filtered signal")
    time_windows, magnitudes_at_freq = get_frequency_magnitude_timeseries(
        filtered_signal, timestamps, target_frequency, window_size=512
    )
    print(f"    Generated {len(time_windows)} time windows")
    
    # Determine severity
    severity = 'unknown'
    for sev, (low, high) in SEVERITY_THRESHOLDS.items():
        if low <= mean_magnitude < high:
            severity = sev
            break
    
    # 7. Create output dataframe
    print(f"\n[7] Generating output CSV...")
    output_df = pd.DataFrame({
        'Time_ms': time_windows,
        f'Magnitude_{target_frequency}Hz': magnitudes_at_freq,
        'Filtered_Signal': filtered_signal[:len(time_windows)],  # Align length
        'Crepitus_Score': [crepitus_score] * len(time_windows),
        'Threshold_Check': [THRESHOLD_MEAN_MAGNITUDE] * len(time_windows),
        'Crepitus_Detected': [is_crepitus] * len(time_windows),
        'Severity': [severity] * len(time_windows),
        'Swing_Count': [swing_count] * len(time_windows),
        'Crack_Events': [crack_peaks] * len(time_windows),
    })
    
    # 7. Save output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'processed_data.csv')
    output_df.to_csv(output_path, index=False)
    
    print(f"    Output saved to: {output_path}")
    print(f"    Output shape: {output_df.shape}")
    print(f"    Columns: {', '.join(output_df.columns.tolist())}")
    
    # 8. Summary statistics
    print(f"\n[8] Summary Statistics:")
    print(f"    Mean magnitude at {target_frequency} Hz: {magnitudes_at_freq.mean():.4f}")
    print(f"    Max magnitude at {target_frequency} Hz: {magnitudes_at_freq.max():.4f}")
    print(f"    Min magnitude at {target_frequency} Hz: {magnitudes_at_freq.min():.4f}")
    print(f"    Std Dev: {magnitudes_at_freq.std():.4f}")
    
    # 9. Crepitus Detection
    print(f"\n[9] Crepitus Detection:")
    print(f"    Mean magnitude: {mean_magnitude:.4f}")
    print(f"    Threshold: {THRESHOLD_MEAN_MAGNITUDE:.4f}")
    print(f"    Status: {'✓ CREPITUS DETECTED' if is_crepitus else '✗ BASELINE (No crepitus)'}")
    print(f"    Severity: {severity.upper()}")
    
    # 10. Algorithm Results Summary
    print(f"\n[10] Algorithm Results:")
    print(f"    Swing Detection: {swing_count} swings detected")
    print(f"    Crack Detection: {crack_peaks} snap/crack events detected")
    
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60 + "\n")
    
    # Generate visualization plots
    if generate_plots:
        plot_file = generate_magnitude_visualization(
            time_windows, magnitudes_at_freq, mean_magnitude,
            THRESHOLD_MEAN_MAGNITUDE, is_crepitus, severity, output_dir,
            swing_detection, crack_detection, signal_data, sampling_rate
        )
        print(f"Visualization saved to: {plot_file}\n")
    
    return {
        'output_path': output_path,
        'detection': is_crepitus,
        'severity': severity,
        'mean_magnitude': mean_magnitude,
        'threshold': THRESHOLD_MEAN_MAGNITUDE,
        'swing_count': swing_count,
        'crack_events': crack_peaks,
    }


def generate_magnitude_visualization(time_windows, magnitudes_at_freq, mean_magnitude,
                                      threshold, is_crepitus, severity, output_dir,
                                      swing_detection=None, crack_detection=None, 
                                      signal_data=None, sampling_rate=None):
    """
    Generate visualization of magnitude over time with threshold overlay
    Also includes swing and crack detection visualizations
    
    Args:
        time_windows: Time values for each window (ms)
        magnitudes_at_freq: Magnitude values
        mean_magnitude: Mean magnitude value
        threshold: Detection threshold
        is_crepitus: Boolean detection result
        severity: Severity classification
        output_dir: Directory to save plot
        swing_detection: Dict with swing detection results
        crack_detection: Dict with crack detection results
        signal_data: Raw signal data
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Path to saved plot file
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Magnitude over time with threshold
    ax = axes[0]
    ax.plot(time_windows, magnitudes_at_freq, 'b-', linewidth=2, label='Magnitude at target frequency')
    ax.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}')
    ax.axhline(mean_magnitude, color='green', linestyle=':', linewidth=2, label=f'Mean: {mean_magnitude:.2f}')
    
    # Highlight detection regions
    above_threshold = magnitudes_at_freq > threshold
    if np.any(above_threshold):
        ax.fill_between(time_windows, magnitudes_at_freq, threshold, 
                        where=above_threshold, alpha=0.2, color='red', label='Crepitus detected')
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Magnitude', fontsize=12)
    ax.set_title(f'Phase 1: Target Frequency Analysis\n' + 
                f'Detection: {"✓ CREPITUS" if is_crepitus else "✗ BASELINE"} | ' +
                f'Severity: {severity.upper()}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Swing tracking (Movement Envelope)
    if swing_detection and signal_data is not None and sampling_rate is not None:
        ax = axes[1]
        time_sec = np.arange(len(signal_data)) / sampling_rate
        ax.plot(time_sec, swing_detection['swing_friction'], color='lightgray', alpha=0.7, 
                label='Joint Friction (20-400Hz)', linewidth=0.5)
        ax.plot(time_sec, swing_detection['envelope'], color='dodgerblue', linewidth=2, 
                label='Movement Envelope')
        
        if len(swing_detection['swing_peaks']) > 0:
            peak_times = time_sec[swing_detection['swing_peaks']]
            peak_values = swing_detection['envelope'][swing_detection['swing_peaks']]
            ax.scatter(peak_times, peak_values, color='red', marker='v', s=80, 
                      label=f"Detected Swings ({len(swing_detection['swing_peaks'])})", zorder=5)
            for peak in swing_detection['swing_peaks']:
                ax.axvline(time_sec[peak], color='red', linestyle='--', alpha=0.3)
        
        ax.axhline(MIN_SWING_VOLUME, color='black', linestyle='--', label=f'Noise Gate ({MIN_SWING_VOLUME})')
        ax.set_title(f'Phase 2: Macro-Movement Tracking ({len(swing_detection["swing_peaks"])} Swings Detected)', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Friction Volume')
        ax.set_xlabel('Time [Seconds]')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.7)
    
    # Plot 3: Crack detection
    if crack_detection:
        ax = axes[2]
        time_sec = np.arange(len(crack_detection['crack_signal_gated'])) / sampling_rate if sampling_rate else np.arange(len(crack_detection['crack_signal_gated']))
        ax.plot(time_sec, crack_detection['crack_signal_gated'], color='crimson', linewidth=0.8, 
                label='Isolated Snaps/Cracks (100-2400Hz)')
        ax.axhline(MIN_CRACK_VOLUME, color='black', linestyle=':', alpha=0.5)
        ax.axhline(-MIN_CRACK_VOLUME, color='black', linestyle=':', alpha=0.5)
        ax.set_title(f'Phase 3: Micro-Acoustic Tracking (Looking for Bone Pops) - {np.sum(np.abs(crack_detection["crack_signal_gated"]) > MIN_CRACK_VOLUME)} Events', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Amplitude (ADC)')
        ax.set_xlabel('Time [Seconds]')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.7)
    else:
        # Fallback: Histogram if no swing/crack detection data
        ax = axes[2]
        ax.hist(magnitudes_at_freq, bins=50, alpha=0.7, color='blue', edgecolor='black', label='Magnitude distribution')
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}')
        ax.axvline(mean_magnitude, color='green', linestyle=':', linewidth=2, label=f'Mean: {mean_magnitude:.2f}')
        
        ax.set_xlabel('Magnitude', fontsize=12)
        ax.set_ylabel('Frequency Count', fontsize=12)
        ax.set_title('Distribution of Magnitude Values', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'magnitude_visualization.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n[11] Generated 3-panel visualization plot (Target Frequency + Swing + Crack Analysis)")
    
    plt.close()
    return plot_path


if __name__ == "__main__":
    # Get the data folder path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_folder = os.path.join(project_root, 'data')
    
    # Find the CSV file in data folder
    csv_files = list(Path(data_folder).glob('*.csv'))
    
    if not csv_files:
        print(f"Error: No CSV files found in {data_folder}")
        exit(1)
    
    # Use the first CSV file found
    csv_input = str(csv_files[0])
    print(f"Found data file: {csv_input}\n")
    
    # Process the data
    result = process_data(csv_input)
    print(f"Ready for visualization using: {result['output_path']}")
