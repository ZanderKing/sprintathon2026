import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 1. System Parameters
FS = 10000              # Sampling frequency (Hz)
LOW_CUT = 3000          # Lower bound of crepitus band (Hz)
HIGH_CUT = 3050         # Upper bound of crepitus band (Hz)
ORDER = 5               # 5th Order as per design

def apply_butterworth_filter(data, lowcut, highcut, fs, order=5):
    sos = signal.butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    filtered_data = signal.sosfilt(sos, data)
    return filtered_data

def calculate_crepitus_index(raw_data, filtered_data):
    total_power = np.sum(np.square(raw_data - np.mean(raw_data)))
    band_power = np.sum(np.square(filtered_data))
    if total_power == 0: return 0
    return (band_power / total_power) * 100
t = np.linspace(0, 2, 2 * FS)  # 2 seconds of time
noise = np.random.normal(0, 0.5, len(t))
simulated_click = 2.0 * np.sin(2 * np.pi * 3025 * t)  # A 3025Hz "bone click"
raw_signal = noise + simulated_click

# Apply Filter
filtered_signal = apply_butterworth_filter(raw_signal, LOW_CUT, HIGH_CUT, FS, ORDER)

# Calculate Score
score = calculate_crepitus_index(raw_signal, filtered_signal)
print(f"Detection Score: {score:.4f}%")

# Visualization
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t[:500], raw_signal[:500], label="Raw Acoustic Signal")
plt.title("Time Domain: Pre-Filter")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(t[:500], filtered_signal[:500], color='red', label="3000Hz Bandpass Output")
plt.title("Time Domain: Post-Filter (Crepitus Extraction)")
plt.tight_layout()
plt.show()