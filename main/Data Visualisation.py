import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, spectrogram

# ==========================================
# 1. CONFIGURATION
# ==========================================
FILE_PATH = 'C:/Users/User/Documents/Sprintathon/download.csv'  # Change back to your real CSV
SAMPLE_RATE = 5000                   

# Filter Settings for High-Pitch Snaps/Cracks
CRACK_LOW = 100
CRACK_HIGH = 2400

# Filter Settings for Low-Pitch Leg Swings
SWING_LOW = 20
SWING_HIGH = 400
ENVELOPE_CUTOFF = 2

# --- NOISE FLOOR GATES (The Fix for Air/Handling Noise) ---
MIN_SWING_VOLUME = 50   # If the envelope is below 50, it is just skin friction/handling noise.
MIN_CRACK_VOLUME = 100  # If a snap is smaller than 100 ADC units, ignore it.

# ==========================================
# 2. DATA LOADING & PREP
# ==========================================
print(f"Loading data from: {FILE_PATH}...")
df = pd.read_csv(FILE_PATH).dropna()

if 'Signal' not in df.columns or 'Timestamp' not in df.columns:
    raise ValueError("CSV must contain 'Timestamp' and 'Signal' columns.")

df['Time_sec'] = df['Timestamp'] / 1000.0
raw_signal = df['Signal'] - df['Signal'].mean() # Remove DC Baseline
nyquist = 0.5 * SAMPLE_RATE

# ==========================================
# 3. ALGORITHM 1: CRACK & SNAP DETECTION
# ==========================================
b_crack, a_crack = butter(5, [CRACK_LOW / nyquist, CRACK_HIGH / nyquist], btype='bandpass')
crack_signal = filtfilt(b_crack, a_crack, raw_signal)

# Apply the Absolute Noise Gate to Cracks
# This mutes the signal to 0 if it doesn't break the MIN_CRACK_VOLUME threshold
crack_signal_gated = np.where(np.abs(crack_signal) > MIN_CRACK_VOLUME, crack_signal, 0)

# ==========================================
# 4. ALGORITHM 2: SWING TRACKING (ENVELOPE)
# ==========================================
b_swing, a_swing = butter(3, [SWING_LOW / nyquist, SWING_HIGH / nyquist], btype='bandpass')
swing_friction = filtfilt(b_swing, a_swing, raw_signal)

b_env, a_env = butter(3, ENVELOPE_CUTOFF / nyquist, btype='low')
envelope = filtfilt(b_env, a_env, np.abs(swing_friction))

# Peak detection using BOTH the dynamic threshold and the absolute Noise Gate
min_swing_distance = int(SAMPLE_RATE * 1.2) 
dynamic_threshold = np.mean(envelope) + (np.std(envelope) * 0.2)

# Pick whichever threshold is HIGHER (The sliding one, or our hard floor)
final_swing_threshold = max(dynamic_threshold, MIN_SWING_VOLUME)

peaks, _ = find_peaks(envelope, distance=min_swing_distance, height=final_swing_threshold)
swing_count = len(peaks)
print(f"Algorithm detected {swing_count} full leg swings.")

# ==========================================
# 5. VISUALIZATION (THE CLINICAL DASHBOARD)
# ==========================================
fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Plot A: Kinematic Tracking
axs[0].plot(df['Time_sec'], swing_friction, color='lightgray', alpha=0.7, label='Joint Friction (20-400Hz)', linewidth=0.5)
axs[0].plot(df['Time_sec'], envelope, color='dodgerblue', linewidth=2, label='Movement Envelope')
axs[0].scatter(df['Time_sec'].iloc[peaks], envelope[peaks], color='red', marker='v', s=80, label=f'Detected Swings ({swing_count})', zorder=5)
for peak in peaks:
    axs[0].axvline(df['Time_sec'].iloc[peak], color='red', linestyle='--', alpha=0.3)
axs[0].axhline(MIN_SWING_VOLUME, color='black', linestyle='--', label=f'Noise Gate ({MIN_SWING_VOLUME})')
axs[0].set_title(f'Phase 1: Macro-Movement Tracking ({swing_count} Swings Detected)')
axs[0].set_ylabel('Friction Volume')
axs[0].legend(loc='upper right')
axs[0].grid(True, linestyle=':', alpha=0.7)

# Plot B: Acoustic Tracking (Using the GATED crack signal)
axs[1].plot(df['Time_sec'], crack_signal_gated, color='crimson', linewidth=0.8, label='Isolated Snaps/Cracks (100-2400Hz)')
axs[1].axhline(MIN_CRACK_VOLUME, color='black', linestyle=':', alpha=0.5)
axs[1].axhline(-MIN_CRACK_VOLUME, color='black', linestyle=':', alpha=0.5)
axs[1].set_title('Phase 2: Micro-Acoustic Tracking (Looking for Bone Pops)')
axs[1].set_ylabel('Amplitude (ADC)')
axs[1].legend(loc='upper right')
axs[1].grid(True, linestyle=':', alpha=0.7)

# Plot C: The Wide Spectrogram
f, t, Sxx = spectrogram(crack_signal, SAMPLE_RATE, nperseg=1024, noverlap=512)
cax = axs[2].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
axs[2].set_title('Phase 3: Frequency Spectrogram')
axs[2].set_ylabel('Frequency [Hz]')
axs[2].set_xlabel('Time [Seconds]')
axs[2].set_ylim([0, 2500])
fig.colorbar(cax, ax=axs[2], label='Intensity [dB]')

plt.tight_layout()
output_filename = FILE_PATH.replace('.csv', '_master_analysis.png')
plt.savefig(output_filename, dpi=150)
plt.close()

print(f"=========================================")
print(f"Analysis Complete! Dashboard saved as: {output_filename}")
print(f"=========================================")