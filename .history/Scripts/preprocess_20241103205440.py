import os
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter

# Paths
AUDIO_PATH = 'AUDIO'  # Path to the audio directory
OUTPUT_PATH = 'Data/Processed'  # Path where processed audio files will be saved
SEGMENT_LENGTH = 3  # Length of each segment in seconds

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Butterworth band-pass filter for denoising
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Process audio files
def process_audio(file_path, output_dir, segment_length=SEGMENT_LENGTH):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=44100)  # Load with original sample rate
    
    # Denoise using band-pass filter (remove frequencies below 50Hz and above 15kHz)
    y_denoised = bandpass_filter(y, lowcut=50, highcut=15000, fs=sr)
    
    # Segment the audio file into smaller clips
    num_segments = int(len(y_denoised) / (segment_length * sr))
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    for i in range(num_segments):
        start = i * segment_length * sr
        end = start + segment_length * sr
        segment = y_denoised[start:end]
        
        # Save each segment as a new file
        segment_path = os.path.join(output_dir, f"{file_name}_seg_{i+1}.flac")
        sf.write(segment_path, segment, sr)
        print(f"Saved {segment_path}")

# Iterate over each audio type (breathing, cough, speech) in AUDIO folder
for sound_category in os.listdir(AUDIO_PATH):
    category_path = os.path.join(AUDIO_PATH, sound_category)
    if os.path.isdir(category_path):
        output_category_dir = os.path.join(OUTPUT_PATH, sound_category)
        os.makedirs(output_category_dir, exist_ok=True)
        
        for audio_file in os.listdir(category_path):
            file_path = os.path.join(category_path, audio_file)
            if file_path.endswith('.flac'):
                print(f"Processing {file_path} in category {sound_category}...")
                process_audio(file_path, output_category_dir)
