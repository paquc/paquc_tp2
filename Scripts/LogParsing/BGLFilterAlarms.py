import pandas as pd

print("Filtering BGL log file to remove duplicate alarms...")

# Load the CSV file into a pandas DataFrame
output_logs_file = f"./BGL_Brain_results/BGL.log_structured_full_content_cleaned_3.csv"
logs_file = f"./BGL_Brain_results/BGL.log_structured_full_content_cleaned_2.csv"

bgl_log_data_df = pd.read_csv(logs_file)

# *************************************************
# 1. Keep only logs with FATAL and FAILURE severity
# *************************************************
# Filter the DataFrame to keep only rows with 'FATAL' or 'FAILURE' severity
bgl_log_data_df = bgl_log_data_df[bgl_log_data_df['Severity'].isin(['FATAL', 'FAILURE'])]


# 2. Discard REPEATED messages:
#   C1: M1 has the same text description than M2
#   C2: M1 and M2 comes from the same source
#   C3: M1 and M2 sent i less than a time T or M2 repeats M* and M* repeats M1

# Set a time threshold (e.g., 3600 seconds for 1 hour)
time_threshold = 5

# Create an empty list to store the indices of rows to keep
keep_indices = []

# Create a dictionary to store the latest occurrence of (AlertFlagLabel, NodeLoc)
last_occurrence = {}

# Initialize the 'ToDrop' column with 0
bgl_log_data_df['ToDrop'] = 0

# Iterate through each row in the DataFrame
for idx, row in bgl_log_data_df.iterrows():
    key = (row['NodeLoc'], row['SubSys'], row['EventId'])
    current_time = row['EpochTime']
    
    # Get the next row if it exists
    if idx + 2 < len(bgl_log_data_df):
        next_row = bgl_log_data_df.iloc[idx + 1]
        key_next_row = (next_row['NodeLoc'], next_row['SubSys'], next_row['EventId'])
        next_time = next_row['EpochTime']

        if (key == key_next_row) and (next_time - current_time <= time_threshold):
            bgl_log_data_df.at[idx + 1, 'ToDrop'] = 1
        

# Drop rows where 'ToDrop' is 1
bgl_log_data_df = bgl_log_data_df[bgl_log_data_df['ToDrop'] != 1]

# Drop the 'ToDrop' column
bgl_log_data_df = bgl_log_data_df.drop(columns=['ToDrop'])

# Save the filtered DataFrame to a new CSV file if needed
bgl_log_data_df.to_csv(output_logs_file, index=False)

print(f"Filtered log file saved to: {output_logs_file}")

