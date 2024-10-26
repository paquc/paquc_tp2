import pandas as pd

# Load the CSV file, assuming comma-separated values with no header
input_file = './BGL_Brain_results/BGL.log_structured.csv'  # Replace with the path to your input file

df = pd.read_csv(input_file, header=None, names=["ID", "Label", "Timestamp", "Date", "Location", "Datetime", "RAS", "Source", "Type", "Severity", "Message", "EventID", "Template"])

# Filter for alarm labels (where Label is not '-')
alarm_df = df[df['Label'] != '-']

# Group by 'Label' and count occurrences
alarm_counts = alarm_df.groupby('Label').size().reset_index(name='Count')

# Export the result to a new file in the same format as the input, with commas as the separator
output_file = './BGL_Brain_results/label_event_counts.csv'

alarm_counts.to_csv(output_file, sep=',', index=False, header=False)

# Display the result
print(alarm_counts)