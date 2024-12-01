import os
import pandas as pd
import shutil

# Paths
CSV_FILE = 'metadata.csv'  # Path to the CSV file
RAW_AUDIO_PATH = 'data/raw_data/AUDIO'  # Path to the original audio directory
OUTPUT_PATH = 'data/reclassified'  # Path for reclassified data

# Create the output path if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Read the CSV file
metadata = pd.read_csv(CSV_FILE)
metadata_fixed = metadata['SUB_ID COVID_STATUS GENDER'].str.split(expand=True)
metadata_fixed.columns = ['SUB_ID', 'COVID_STATUS', 'GENDER']

# Process each entry in the metadata
for _, row in metadata_fixed.iterrows():
    subject_id = row['SUB_ID']
    covid_status = 'positive' if row['COVID_STATUS'].lower() == 'p' else 'negative'
    gender = 'male' if row['GENDER'].lower() == 'm' else 'female'

    # Define output subdirectory
    subject_output_dir = os.path.join(OUTPUT_PATH, covid_status, gender, subject_id)
    os.makedirs(subject_output_dir, exist_ok=True)

    # Process all file types
    for file_type in ['breathing', 'cough', 'speech']:
        original_file = os.path.join(RAW_AUDIO_PATH, file_type, f"{subject_id}.flac")
        if os.path.exists(original_file):
            # Define the new file name and path
            new_file_name = f"{subject_id}_{file_type}.flac"
            new_file_path = os.path.join(subject_output_dir, new_file_name)
            
            # Copy and rename the file
            shutil.copy(original_file, new_file_path)
            print(f"Copied and renamed: {original_file} -> {new_file_path}")
        else:
            print(f"File not found: {original_file}")

print("Reclassification complete!")
