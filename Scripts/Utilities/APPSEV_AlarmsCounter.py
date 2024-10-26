import pandas as pd

# Load the CSV file, assuming comma-separated values with no header
input_file = './BGL_Brain_results/BGL.log_structured.csv'  # Replace with the path to your input file

df = pd.read_csv(input_file, header=None, names=["ID", "Label", "Timestamp", "Date", "Location", "Datetime", "RAS", "Source", "Type", "Severity", "Message", "EventID", "Template"])

# Group by 'Label' and 'EventID', then count occurrences
event_counts = df.groupby(['Label', 'EventID']).size().reset_index(name='Count')

# Filter to include only rows where 'Label' is 'APPSEV' and 'EventID' is 'E375'
filtered_counts = event_counts[(event_counts['Label'] == 'APPSEV') & (event_counts['EventID'] == 'E375')]

# Export the results to a new file, with tabs as the separator
output_file = './BGL_Brain_results/APPSEV_label_event_counts.csv'
filtered_counts.to_csv(output_file, sep='\t', index=False)

# Display the result
print(filtered_counts)