import os
import librosa

def count_audio_lengths_by_type(input_dir, sample_rate=44100):
    """
    Counts the lengths of audio files by type (cough, breathing, speech) in given intervals.

    Args:
        input_dir (str): Directory containing audio files.
        sample_rate (int): Sample rate for loading audio files.

    Returns:
        dict: Length counts categorized by type and interval.
    """
    # Initialize counts for each type and interval
    audio_types = ['cough', 'breathing', 'speech']
    counts = {
        'cough': {"<=3s": 0, "(3,6]": 0, "(6,9]": 0, ">9s": 0},
        'speech': {"<=3s": 0, "(3,6]": 0, "(6,9]": 0, ">9s": 0},
        'breathing': {"[0,5)": 0, "[5,10)": 0, "[10,inf)": 0}
    }

    # Traverse the directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.flac'):
                file_path = os.path.join(root, file)
                
                # Determine the audio type based on the filename
                for audio_type in audio_types:
                    if audio_type in file.lower():
                        # Load audio and calculate duration
                        y, sr = librosa.load(file_path, sr=sample_rate)
                        duration = librosa.get_duration(y=y, sr=sr)

                        # Increment appropriate interval
                        if audio_type == 'breathing':
                            if 0 <= duration < 5:
                                counts[audio_type]["[0,5)"] += 1
                            elif 5 <= duration < 10:
                                counts[audio_type]["[5,10)"] += 1
                            else:
                                counts[audio_type]["[10,inf)"] += 1
                        else:
                            if duration <= 3:
                                counts[audio_type]["<=3s"] += 1
                            elif 3 < duration <= 6:
                                counts[audio_type]["(3,6]"] += 1
                            elif 6 < duration <= 9:
                                counts[audio_type]["(6,9]"] += 1
                            else:
                                counts[audio_type][">9s"] += 1
                        break  # Stop checking other types once matched

    return counts

# Define the input directory
INPUT_DIR = 'data/reclassified'

# Count audio lengths by type
length_counts_by_type = count_audio_lengths_by_type(INPUT_DIR)

# Print the results
print("Audio Length Distribution by Type:")
for audio_type, intervals in length_counts_by_type.items():
    print(f"\n{audio_type.capitalize()}:")
    for interval, count in intervals.items():
        print(f"  {interval}: {count} files")
