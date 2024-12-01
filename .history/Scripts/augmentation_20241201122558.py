import os

# Define the base path for the reclassified data
BASE_PATH = 'data/reclassified'

# Initialize counters
total_subjects = 0
counts = {
    'positive': {'male': 0, 'female': 0},
    'negative': {'male': 0, 'female': 0},
}

# Traverse the directory structure
for covid_status in ['positive', 'negative']:
    status_path = os.path.join(BASE_PATH, covid_status)
    if os.path.exists(status_path):
        for gender in ['male', 'female']:
            gender_path = os.path.join(status_path, gender)
            if os.path.exists(gender_path):
                # Count subfolders (unique IDs)
                subject_folders = [f for f in os.listdir(gender_path) if os.path.isdir(os.path.join(gender_path, f))]
                counts[covid_status][gender] += len(subject_folders)
                total_subjects += len(subject_folders)

# Print the results
print(f"Total subjects: {total_subjects}")
print(f"Positive male: {counts['positive']['male']}")
print(f"Positive female: {counts['positive']['female']}")
print(f"Negative male: {counts['negative']['male']}")
print(f"Negative female: {counts['negative']['female']}")
