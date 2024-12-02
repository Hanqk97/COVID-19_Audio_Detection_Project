import numpy as np
import librosa

# Generate a sine wave
sr = 44100
duration = 1.0  # 1 second
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)

# Apply time stretching
try:
    stretched = librosa.effects.time_stretch(sine_wave, rate=1.2)
    print(f"Time-stretch successful. Output shape: {stretched.shape}")
except Exception as e:
    print(f"Error in time-stretch: {e}")
