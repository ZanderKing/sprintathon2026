"""
Baseline Noise and Crepitus Threshold Analysis
Analyzes all 4 sample datasets to establish:
1. Baseline amplitude of background noise (from "No swings" samples)
2. Ideal threshold frequency amplitude for detecting crepitus
"""

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import os
from fft_analysis import load_hardware_data, compute_fft_spectrum, get_frequency_magnitude_timeseries
from butterworth_filter import apply_butterworth_filter

# Define sample paths
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = {
    'Sample 1 - No swings': os.path.join(BASE_PATH, 'Sample 1 - No swings/download.csv'),
    'Sample 2 - 5 swings': os.path.join(BASE_PATH, 'Sample 2 - 5 swings/download.csv'),
    'Sample 3 - No swings': os.path.join(BASE_PATH, 'Sample 3 - No swings/download.csv'),
    'Sample 4 - 21 taps': os.path.join(BASE_PATH, 'Sample 4 - 21 taps/download.csv'),
}

def analyze_sample(sample_name, csv_path, low_cut=1000, high_cut=1050):
    """
    Analyze a single sample dataset
    
    Returns:
        dict with analysis results
    """
    print(f"\n{'='*70}")
    print(f"Analyzing: {sample_name}")
    print(f"{'='*70}")
    
    # Load data
    timestamps, signal_data = load_hardware_data(csv_path)
    
    # Remove NaN values
    valid_idx = ~np.isnan(signal_data)
    timestamps = timestamps[valid_idx]
    signal_data = signal_data[valid_idx]
    
    print(f"Loaded {len(signal_data)} samples")
    print(f"Time range: {timestamps[0]:.2f} to {timestamps[-1]:.2f} ms")
    
    # Compute FFT on raw signal
    frequencies, magnitudes, sampling_rate = compute_fft_spectrum(signal_data, timestamps)
    print(f"Sampling rate: {sampling_rate:.2f} Hz")
    
    # Apply Butterworth filter
    filtered_signal = apply_butterworth_filter(signal_data, low_cut, high_cut, 
                                                sampling_rate, order=5)
    
    # Compute FFT on filtered signal
    frequencies_filt, magnitudes_filt, _ = compute_fft_spectrum(filtered_signal, timestamps)
    
    # Get peak frequency and magnitude in band of interest
    band_mask = (frequencies_filt >= low_cut) & (frequencies_filt <= high_cut)
    peak_idx = np.argmax(magnitudes_filt[band_mask])
    peak_freq = frequencies_filt[band_mask][peak_idx]
    peak_mag = magnitudes_filt[band_mask][peak_idx]
    
    # Get RMS of raw and filtered signal
    rms_raw = np.sqrt(np.mean(signal_data**2))
    rms_filtered = np.sqrt(np.mean(filtered_signal**2))
    
    # Get mean magnitude in frequency band
    mean_mag_band = np.mean(magnitudes_filt[band_mask])
    median_mag_band = np.median(magnitudes_filt[band_mask])
    
    # Get time-domain statistics
    std_raw = np.std(signal_data)
    std_filtered = np.std(filtered_signal)
    
    results = {
        'sample_name': sample_name,
        'num_samples': len(signal_data),
        'sampling_rate': sampling_rate,
        'rms_raw': rms_raw,
        'rms_filtered': rms_filtered,
        'std_raw': std_raw,
        'std_filtered': std_filtered,
        'peak_freq': peak_freq,
        'peak_mag': peak_mag,
        'mean_mag_band': mean_mag_band,
        'median_mag_band': median_mag_band,
        'frequencies': frequencies_filt,
        'magnitudes': magnitudes_filt,
        'signal_raw': signal_data,
        'signal_filtered': filtered_signal,
        'timestamps': timestamps,
    }
    
    print(f"RMS (Raw): {rms_raw:.4f}")
    print(f"RMS (Filtered): {rms_filtered:.4f}")
    print(f"Std Dev (Raw): {std_raw:.4f}")
    print(f"Std Dev (Filtered): {std_filtered:.4f}")
    print(f"Peak frequency in band: {peak_freq:.2f} Hz")
    print(f"Peak magnitude: {peak_mag:.4f}")
    print(f"Mean magnitude in band: {mean_mag_band:.4f}")
    print(f"Median magnitude in band: {median_mag_band:.4f}")
    
    return results

def compute_baseline_and_threshold(noise_samples, crepitus_samples, low_cut=1000, high_cut=1050):
    """
    Establish baseline noise and determine threshold
    
    Args:
        noise_samples: dict of sample names and results (no motion)
        crepitus_samples: dict of sample names and results (with motion)
        low_cut, high_cut: Frequency band of interest
    
    Returns:
        Analysis summary with recommendations
    """
    print(f"\n{'='*70}")
    print("BASELINE AND THRESHOLD ANALYSIS")
    print(f"{'='*70}\n")
    
    # Extract baseline metrics from no-swing samples
    baseline_peaks = [noise_samples[s]['peak_mag'] for s in noise_samples]
    baseline_means = [noise_samples[s]['mean_mag_band'] for s in noise_samples]
    baseline_medians = [noise_samples[s]['median_mag_band'] for s in noise_samples]
    baseline_rms = [noise_samples[s]['rms_filtered'] for s in noise_samples]
    
    # Extract crepitus metrics
    crepitus_peaks = [crepitus_samples[s]['peak_mag'] for s in crepitus_samples]
    crepitus_means = [crepitus_samples[s]['mean_mag_band'] for s in crepitus_samples]
    crepitus_medians = [crepitus_samples[s]['median_mag_band'] for s in crepitus_samples]
    crepitus_rms = [crepitus_samples[s]['rms_filtered'] for s in crepitus_samples]
    
    # Compute statistics
    baseline_peak_mean = np.mean(baseline_peaks)
    baseline_peak_std = np.std(baseline_peaks)
    baseline_peak_max = np.max(baseline_peaks)
    
    baseline_mean_mag_mean = np.mean(baseline_means)
    baseline_mean_mag_std = np.std(baseline_means)
    
    baseline_median_mag_mean = np.mean(baseline_medians)
    baseline_median_mag_std = np.std(baseline_medians)
    
    baseline_rms_mean = np.mean(baseline_rms)
    baseline_rms_std = np.std(baseline_rms)
    
    crepitus_peak_mean = np.mean(crepitus_peaks)
    crepitus_peak_std = np.std(crepitus_peaks)
    crepitus_peak_min = np.min(crepitus_peaks)
    
    crepitus_mean_mag_mean = np.mean(crepitus_means)
    crepitus_mean_mag_std = np.std(crepitus_means)
    
    crepitus_median_mag_mean = np.mean(crepitus_medians)
    crepitus_median_mag_std = np.std(crepitus_medians)
    
    crepitus_rms_mean = np.mean(crepitus_rms)
    crepitus_rms_std = np.std(crepitus_rms)
    
    # Calculate separation
    peak_separation_ratio = crepitus_peak_mean / baseline_peak_mean if baseline_peak_mean > 0 else 0
    mean_separation_ratio = crepitus_mean_mag_mean / baseline_mean_mag_mean if baseline_mean_mag_mean > 0 else 0
    median_separation_ratio = crepitus_median_mag_mean / baseline_median_mag_mean if baseline_median_mag_mean > 0 else 0
    rms_separation_ratio = crepitus_rms_mean / baseline_rms_mean if baseline_rms_mean > 0 else 0
    
    print("BASELINE (No Motion) Statistics:")
    print(f"  Peak Magnitude:   mean={baseline_peak_mean:.4f}, std={baseline_peak_std:.4f}, max={baseline_peak_max:.4f}")
    print(f"  Mean Magnitude:   mean={baseline_mean_mag_mean:.4f}, std={baseline_mean_mag_std:.4f}")
    print(f"  Median Magnitude: mean={baseline_median_mag_mean:.4f}, std={baseline_median_mag_std:.4f}")
    print(f"  RMS (Filtered):   mean={baseline_rms_mean:.4f}, std={baseline_rms_std:.4f}")
    
    print("\nCREPITUS (With Motion) Statistics:")
    print(f"  Peak Magnitude:   mean={crepitus_peak_mean:.4f}, std={crepitus_peak_std:.4f}, min={crepitus_peak_min:.4f}")
    print(f"  Mean Magnitude:   mean={crepitus_mean_mag_mean:.4f}, std={crepitus_mean_mag_std:.4f}")
    print(f"  Median Magnitude: mean={crepitus_median_mag_mean:.4f}, std={crepitus_median_mag_std:.4f}")
    print(f"  RMS (Filtered):   mean={crepitus_rms_mean:.4f}, std={crepitus_rms_std:.4f}")
    
    print("\nSeparation Ratios (Crepitus / Baseline):")
    print(f"  Peak Magnitude Ratio:   {peak_separation_ratio:.2f}x")
    print(f"  Mean Magnitude Ratio:   {mean_separation_ratio:.2f}x")
    print(f"  Median Magnitude Ratio: {median_separation_ratio:.2f}x")
    print(f"  RMS Ratio:              {rms_separation_ratio:.2f}x")
    
    # Recommend thresholds (conservative approach: baseline mean + 2*std)
    print("\nRECOMMENDED THRESHOLDS:")
    threshold_peak_2std = baseline_peak_mean + 2 * baseline_peak_std
    threshold_peak_3std = baseline_peak_mean + 3 * baseline_peak_std
    threshold_mean_2std = baseline_mean_mag_mean + 2 * baseline_mean_mag_std
    threshold_mean_3std = baseline_mean_mag_mean + 3 * baseline_mean_mag_std
    threshold_rms_2std = baseline_rms_mean + 2 * baseline_rms_std
    threshold_rms_3std = baseline_rms_mean + 3 * baseline_rms_std
    
    print(f"  Peak Magnitude (Mean + 2σ): {threshold_peak_2std:.4f}")
    print(f"  Peak Magnitude (Mean + 3σ): {threshold_peak_3std:.4f}")
    print(f"  Mean Magnitude (Mean + 2σ): {threshold_mean_2std:.4f}")
    print(f"  Mean Magnitude (Mean + 3σ): {threshold_mean_3std:.4f}")
    print(f"  RMS Filtered (Mean + 2σ):   {threshold_rms_2std:.4f}")
    print(f"  RMS Filtered (Mean + 3σ):   {threshold_rms_3std:.4f}")
    
    # Check detection capability
    print("\nDETECTION CAPABILITY:")
    detected_peak_2std = np.sum(np.array(crepitus_peaks) > threshold_peak_2std) / len(crepitus_peaks)
    detected_peak_3std = np.sum(np.array(crepitus_peaks) > threshold_peak_3std) / len(crepitus_peaks)
    detected_mean_2std = np.sum(np.array(crepitus_means) > threshold_mean_2std) / len(crepitus_means)
    detected_mean_3std = np.sum(np.array(crepitus_means) > threshold_mean_3std) / len(crepitus_means)
    detected_rms_2std = np.sum(np.array(crepitus_rms) > threshold_rms_2std) / len(crepitus_rms)
    detected_rms_3std = np.sum(np.array(crepitus_rms) > threshold_rms_3std) / len(crepitus_rms)
    
    false_positives_peak_2std = np.sum(np.array(baseline_peaks) > threshold_peak_2std) / len(baseline_peaks)
    false_positives_peak_3std = np.sum(np.array(baseline_peaks) > threshold_peak_3std) / len(baseline_peaks)
    false_positives_mean_2std = np.sum(np.array(baseline_means) > threshold_mean_2std) / len(baseline_means)
    false_positives_mean_3std = np.sum(np.array(baseline_means) > threshold_mean_3std) / len(baseline_means)
    
    print(f"  Peak (Mean + 2σ): {detected_peak_2std*100:.1f}% detection, {false_positives_peak_2std*100:.1f}% false positives")
    print(f"  Peak (Mean + 3σ): {detected_peak_3std*100:.1f}% detection, {false_positives_peak_3std*100:.1f}% false positives")
    print(f"  Mean (Mean + 2σ): {detected_mean_2std*100:.1f}% detection, {false_positives_mean_2std*100:.1f}% false positives")
    print(f"  Mean (Mean + 3σ): {detected_mean_3std*100:.1f}% detection, {false_positives_mean_3std*100:.1f}% false positives")
    print(f"  RMS (Mean + 2σ):  {detected_rms_2std*100:.1f}% detection")
    print(f"  RMS (Mean + 3σ):  {detected_rms_3std*100:.1f}% detection")
    
    return {
        'baseline_peak_mean': baseline_peak_mean,
        'baseline_peak_std': baseline_peak_std,
        'baseline_mean_mag': baseline_mean_mag_mean,
        'baseline_mean_mag_std': baseline_mean_mag_std,
        'baseline_median_mag': baseline_median_mag_mean,
        'baseline_rms': baseline_rms_mean,
        'baseline_rms_std': baseline_rms_std,
        'crepitus_peak_mean': crepitus_peak_mean,
        'crepitus_mean_mag': crepitus_mean_mag_mean,
        'threshold_peak_2std': threshold_peak_2std,
        'threshold_peak_3std': threshold_peak_3std,
        'threshold_mean_2std': threshold_mean_2std,
        'threshold_mean_3std': threshold_mean_3std,
        'threshold_rms_2std': threshold_rms_2std,
        'threshold_rms_3std': threshold_rms_3std,
        'peak_separation_ratio': peak_separation_ratio,
        'mean_separation_ratio': mean_separation_ratio,
        'rms_separation_ratio': rms_separation_ratio,
    }

def plot_frequency_comparisons(results_dict, low_cut=1000, high_cut=1050):
    """Create comparison plots for baseline vs crepitus"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: FFT comparison
    ax = axes[0, 0]
    for sample_name, results in results_dict.items():
        freq_band = (results['frequencies'] >= low_cut - 100) & (results['frequencies'] <= high_cut + 100)
        ax.semilogy(results['frequencies'][freq_band], results['magnitudes'][freq_band], 
                   label=sample_name, alpha=0.7)
    ax.axvline(low_cut, color='red', linestyle='--', alpha=0.5, label='Filter band')
    ax.axvline(high_cut, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (log scale)')
    ax.set_title('FFT Magnitude Spectrum Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Time-domain signal comparison
    ax = axes[0, 1]
    for sample_name, results in results_dict.items():
        time_subset = results['timestamps'][:500]  # First 500 samples
        signal_subset = results['signal_filtered'][:500]
        ax.plot(time_subset, signal_subset, label=sample_name, alpha=0.7)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Filtered Signal')
    ax.set_title('Filtered Signal Comparison (First 500 samples)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Peak magnitude comparison
    ax = axes[1, 0]
    sample_names = list(results_dict.keys())
    peak_mags = [results_dict[s]['peak_mag'] for s in sample_names]
    colors = ['green' if 'No swings' in s else 'red' for s in sample_names]
    bars = ax.bar(range(len(sample_names)), peak_mags, color=colors, alpha=0.7)
    ax.set_ylabel('Peak Magnitude')
    ax.set_title('Peak Magnitude in Frequency Band')
    ax.set_xticks(range(len(sample_names)))
    ax.set_xticklabels(sample_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{peak_mags[i]:.4f}', ha='center', va='bottom')
    
    # Plot 4: RMS comparison
    ax = axes[1, 1]
    rms_values = [results_dict[s]['rms_filtered'] for s in sample_names]
    bars = ax.bar(range(len(sample_names)), rms_values, color=colors, alpha=0.7)
    ax.set_ylabel('RMS (Filtered Signal)')
    ax.set_title('RMS of Filtered Signal')
    ax.set_xticks(range(len(sample_names)))
    ax.set_xticklabels(sample_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rms_values[i]:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BASELINE NOISE AND CREPITUS THRESHOLD ANALYSIS")
    print("="*70)
    
    # Analyze all samples
    results = {}
    for sample_name, csv_path in SAMPLES.items():
        if os.path.exists(csv_path):
            results[sample_name] = analyze_sample(sample_name, csv_path)
        else:
            print(f"WARNING: File not found: {csv_path}")
    
    # Separate baseline and crepitus samples
    baseline_samples = {k: v for k, v in results.items() if 'No swings' in k}
    crepitus_samples = {k: v for k, v in results.items() if 'No swings' not in k}
    
    # Compute baseline and threshold analysis
    analysis_results = compute_baseline_and_threshold(baseline_samples, crepitus_samples)
    
    # Generate plots
    if len(results) > 0:
        fig = plot_frequency_comparisons(results)
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   'output/baseline_analysis.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlots saved to: {output_path}")
        plt.show()
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
