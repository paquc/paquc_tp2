import pandas as pd

# Load the parsed logs from the CSV file - STRUCTED log file
df = pd.read_csv('./Thunderbird_Brain_results/Thunderbird_10M.log_structured.csv')
#df = pd.read_csv('./Thunderbird_Brain_results/Thunderbird.log_structured_Samples.csv')
#df = pd.read_csv('parsed-toy-data.csv')

# Combine Date and Time into a single datetime column for time-based splitting
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Hour'])

# Sort the DataFrame by datetime to ensure it's ordered correctly
# df = df.sort_values(by='Datetime')

# Function to split DataFrame based on time intervals and aggregate by Cluster ID counts
def split_and_aggregate_by_cluster(df, time_interval='30T', error_threshold=5, anomaly_clusters=None):
    
    # Set 'Datetime' as index for time-based resampling
    df.set_index('Datetime', inplace=True)

    # Resample based on time intervals and count occurrences of each Cluster ID
    cluster_counts = df.groupby([pd.Grouper(freq = time_interval), 'EventId']).size().unstack(fill_value=0)
    # cluster_counts = df.groupby([pd.Grouper(freq=time_interval), 'Cluster ID']).size().unstack(fill_value=0)

    # Initialize an empty column for the anomaly label
    cluster_counts['IsAlarm'] = '0'
    # cluster_counts['Anomaly'] = '0'

    # Label the chunk as 'anomaly' based on the specified threshold and specific Cluster IDs
    for index, row in cluster_counts.iterrows():
        if anomaly_clusters:
            # Check if any of the anomaly clusters exceed the error threshold
            if row[anomaly_clusters].sum() >= error_threshold:
                cluster_counts.at[index, 'IsAlarm'] = '1'
        else:
            # Check if any cluster exceeds the threshold
            if row.sum() >= error_threshold:
                cluster_counts.at[index, 'IsAlarm'] = '1'

    # ******************************************
    # IMPORTANT!!!!!!!!!
    # Drop the columns corresponding to the Cluster IDs in anomaly_clusters
    if anomaly_clusters:
        print("Dropping columns: ", anomaly_clusters)
        cluster_counts.drop(columns = anomaly_clusters, inplace=True)
    # ******************************************
   
    return cluster_counts

# Set the time interval to split the logs (e.g., '30T' for 30 minutes)
#time_interval = '3600'  # 1 hour
time_interval = '30min'  # Change to your preferred interval

# Define the anomaly conditions (specific clusters or thresholds)
anomaly_clusters = ['E1119']  # Define the Cluster IDs that you consider anomalous
error_threshold = 3  # If the sum of occurrences of these clusters in a chunk exceeds this, label as anomaly

# Split and aggregate the DataFrame into chunks based on the specified time interval
result_df = split_and_aggregate_by_cluster(df, time_interval, error_threshold, anomaly_clusters)

# Reset the index to flatten the DataFrame, with time intervals as rows
result_df.reset_index(inplace=True)

# Display the resulting DataFrame
print(result_df)

#Drop DateTime column
#cluster_counts.reset_index(drop=True, inplace=True)
del result_df['Datetime']

# Display the resulting DataFrame
print(result_df)

# Optionally, save the result to a new CSV file
#result_df.to_csv('./Thunderbird_Brain_results/Thunderbird.log_structured_Preprocess_Samples.csv', index=False)
result_df.to_csv('./Thunderbird_Brain_results/VAPI_alarm_occurences_matrix_preprocessed.csv', index=False)