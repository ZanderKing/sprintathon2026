import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

# Load hardware data from CSV
def load_hardware_data(csv_file):
    """Load timestamp and signal strength from hardware CSV"""
    df = pd.read_csv(csv_file)
    timestamps = df['Timestamp'].values  # milliseconds
    signal_data = df['Signal'].values     # volts
    return timestamps, signal_data

# Compute FFT magnitude spectrum across the entire signal
def compute_fft_spectrum(signal_data, timestamps):
    """
    Compute FFT to get magnitude at each frequency
    
    Args:
        signal_data: Raw signal values (volts)
        timestamps: Time values in milliseconds
    
    Returns:
        frequencies: Frequency bins (Hz)
        magnitudes: Magnitude spectrum
        sampling_rate: Sampling rate in Hz
    """
    # Calculate sampling rate from timestamps
    time_diff_ms = np.diff(timestamps)
    avg_time_diff_ms = np.mean(time_diff_ms)
    sampling_rate = 1000 / avg_time_diff_ms  # Convert to Hz
    
    # Perform FFT
    fft_values = np.fft.fft(signal_data)
    magnitudes = np.abs(fft_values)
    
    # Calculate frequency bins
    n = len(signal_data)
    frequencies = np.fft.fftfreq(n, 1/sampling_rate)
    
    # Return only positive frequencies
    positive_freq_idx = frequencies >= 0
    frequencies = frequencies[positive_freq_idx]
    magnitudes = magnitudes[positive_freq_idx]
    
    return frequencies, magnitudes, sampling_rate

# Extract magnitude at a specific frequency over time using windowed FFT
def get_frequency_magnitude_timeseries(signal_data, timestamps, target_frequency, window_size=512):
    """
    Compute magnitude of a specific frequency across time using sliding window FFT
    
    Args:
        signal_data: Raw signal values (volts)
        timestamps: Time values in milliseconds
        target_frequency: Frequency of interest (Hz)
        window_size: Samples per window for STFT
    
    Returns:
        time_windows: Time values for each window (milliseconds)
        magnitudes_at_freq: Magnitude at target frequency for each window
    """
    # Calculate sampling rate
    time_diff_ms = np.diff(timestamps)
    avg_time_diff_ms = np.mean(time_diff_ms)
    sampling_rate = 1000 / avg_time_diff_ms
    
    # Compute STFT (Short-Time Fourier Transform)
    f, t, Sxx = signal.spectrogram(
        signal_data,
        fs=sampling_rate,
        nperseg=window_size,
        noverlap=window_size//2  # 50% overlap
    )
    
    # Find frequency bin closest to target frequency
    freq_idx = np.argmin(np.abs(f - target_frequency))
    
    # Extract magnitude at that frequency across time
    magnitudes_at_freq = np.abs(Sxx[freq_idx, :])
    
    # Convert time indices to milliseconds
    time_windows = timestamps[0] + t * 1000  # Convert seconds to milliseconds
    
    return time_windows, magnitudes_at_freq

# Visualization functions
def plot_frequency_spectrum(frequencies, magnitudes, sampling_rate):
    """Plot FFT magnitude spectrum"""
    plt.figure(figsize=(12, 5))
    plt.plot(frequencies, magnitudes)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT Magnitude Spectrum')
    plt.grid(True)
    plt.xlim(0, min(500, sampling_rate/2))  # Focus on relevant frequencies
    plt.tight_layout()
    plt.show()

def plot_frequency_timeseries(time_windows, magnitudes_at_freq, target_frequency):
    """Plot magnitude of specific frequency across time"""
    plt.figure(figsize=(12, 5))
    plt.plot(time_windows, magnitudes_at_freq, linewidth=1.5)
    plt.xlabel('Time (ms)')
    plt.ylabel('Magnitude')
    plt.title(f'Magnitude at {target_frequency} Hz Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load data
    timestamps, signal_data = load_hardware_data('example_data.csv')
    print(f"Loaded {len(signal_data)} samples")
    print(f"Time range: {timestamps[0]:.2f} to {timestamps[-1]:.2f} ms")
    
    # 1. Compute full FFT spectrum
    frequencies, magnitudes, fs = compute_fft_spectrum(signal_data, timestamps)
    print(f"Sampling rate: {fs:.2f} Hz")
    
    # Plot full spectrum
    plot_frequency_spectrum(frequencies, magnitudes, fs)
    
    # 2. Extract and visualize magnitude at specific frequency (3025 Hz - crepitus band center)
    target_freq = 3025  # Hz
    time_windows, freq_magnitudes = get_frequency_magnitude_timeseries(
        signal_data, timestamps, target_freq, window_size=512
    )
    plot_frequency_timeseries(time_windows, freq_magnitudes, target_freq)
    
    # 3. Optional: Also apply Butterworth filter from existing pipeline
    # (Requires importing from 5thButter.py after refactoring it into modules)
    # filtered_signal = apply_butterworth_filter(signal_data, 3000, 3050, fs, order=5)
    # crepitus_score = calculate_crepitus_index(signal_data, filtered_signal)
    # print(f"\nCrepitus Index Score: {crepitus_score:.4f}%")
