import pandas as pd

# Load the metadata file
file_path = 'Data/metadata.csv'  # Replace with the correct path to your file
data = pd.read_csv(file_path, header=None)

# Rename the single column for easier manipulation
data.columns = ['info']

# Skip the header row and split the remaining rows by spaces to create separate columns
data_transformed = data[1:].copy()  # Exclude header
data_transformed[['SUB_ID', 'COVID_STATUS', 'GENDER']] = data_transformed['info'].str.split(expand=True)

# Drop the original 'info' column as it's no longer needed
data_transformed = data_transformed[['SUB_ID', 'COVID_STATUS', 'GENDER']]

# Save the transformed data to a new CSV file
data_transformed.to_csv('metadata_update.csv', index=False)

print("Data has been transformed and saved to metadata_update.csv.")
