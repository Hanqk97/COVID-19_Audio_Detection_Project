import os
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter

# Paths
AUDIO_PATH = 'Data/AUDIO'  # Updated path to the audio directory
OUTPUT_PATH = 'Data/Processed'  # Path where processed audio files will be saved
SEGMENT_LENGTH = 3  # Length of each segment in seconds

# Ensure output directory exists
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

# Function to process individual audio files
def process_audio(file_path, output_dir, segment_length=SEGMENT_LENGTH):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=44100)  # Using the sample rate of 44.1 kHz
    
    # Denoise using band-pass filter (cut off frequencies < 50Hz and > 15kHz)
    y_denoised = bandpass_filter(y, lowcut=50, highcut=15000, fs=sr)
    
    # Segment the audio file into smaller clips of segment_length seconds
    num_segments = int(len(y_denoised) / (segment_length * sr))
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    for i in range(num_segments):
        start = i * segment_length * sr
        end = start + segment_length * sr
        segment = y_denoised[start:end]
        
        # Save each segment as a new file
        segment_path = os.path.join(output_dir, f"{file_name}_seg_{i+1}.flac")
        sf.write(segment_path, segment, sr)

# Iterate over each sound category (breathing, cough, speech) in AUDIO folder
processed_count = 0
total_files = sum([len(files) for _, _, files in os.walk(AUDIO_PATH)])

for sound_category in os.listdir(AUDIO_PATH):
    category_path = os.path.join(AUDIO_PATH, sound_category)
    if os.path.isdir(category_path):
        output_category_dir = os.path.join(OUTPUT_PATH, sound_category)
        os.makedirs(output_category_dir, exist_ok=True)
        
        for audio_file in os.listdir(category_path):
            file_path = os.path.join(category_path, audio_file)
            if file_path.endswith('.flac'):
                process_audio(file_path, output_category_dir)
                processed_count += 1
                print(f"Processed {processed_count}/{total_files} files")

print("Processing complete.")
