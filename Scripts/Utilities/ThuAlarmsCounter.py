import sys
sys.path.append('../../')

import pandas as pd

# Load the CSV file, assuming comma-separated values with no header
input_file = '../../data/Thunderbird/Thunderbird_Brain_results/Thunderbird_10M.log_structured.csv'  # Replace with the path to your input file

df = pd.read_csv(input_file, header=None, names=["LineID", "AlertFlagLabel", "EpochTime", "Date", "Noeud", "Month", "Day", "Hour", "Content", "EventId", "EventTemplate"])

# Filter for alarm labels (where Label is not '-')
alarm_df = df[df['AlertFlagLabel'] != '-']

# Group by 'Label' and count occurrences
alarm_counts = alarm_df.groupby('AlertFlagLabel').size().reset_index(name='Count')

# Export the result to a new file in the same format as the input, with commas as the separator
output_file = './Thunderbird_Brain_results/Thu_alarms_counts.csv'

alarm_counts.to_csv(output_file, sep=',', index=False, header=False)

# Display the result
print(alarm_counts)