import os
import pandas as pd
import shutil

# Paths
CSV_FILE = 'data/raw_data/metadata.csv'  # Path to the CSV file
RAW_AUDIO_PATH = 'data/raw_data/AUDIO'  # Path to the original audio directory
OUTPUT_PATH = 'data/reclassified'  # Path for reclassified data

# Create the output path if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Read the CSV file
metadata = pd.read_csv(CSV_FILE)
metadata_fixed = metadata['SUB_ID COVID_STATUS GENDER'].str.split(expand=True)
metadata_fixed.columns = ['SUB_ID', 'COVID_STATUS', 'GENDER']

# Initialize a set to ensure unique IDs are processed
processed_subjects = set()

# Total subjects for progress tracking
total_subjects = metadata_fixed['SUB_ID'].nunique()
processed_count = 0

# Process each entry in the metadata
for _, row in metadata_fixed.iterrows():
    subject_id = row['SUB_ID']
    covid_status = 'positive' if row['COVID_STATUS'].lower() == 'p' else 'negative'
    gender = 'male' if row['GENDER'].lower() == 'm' else 'female'

    # Ensure only one folder per subject
    if subject_id not in processed_subjects:
        processed_count += 1  # Increment processed count

        # Define output subdirectory for the subject
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
                print(f"Copied: {original_file} -> {new_file_path}")
            else:
                print(f"File not found: {original_file}")

        # Mark subject ID as processed
        processed_subjects.add(subject_id)

        # Print progress
        print(f"Processed {processed_count}/{total_subjects} subjects.")

print("Reclassification complete!")
