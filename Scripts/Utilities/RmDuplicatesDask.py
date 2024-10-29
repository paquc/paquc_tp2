import dask.dataframe as dd

# Path to your input CSV file
input_file = "./Thunderbird_Brain_results/VAPI_alarm_occurences_matrix_V4.csv"
# Path to the output CSV file without duplicates
output_file = "./Thunderbird_Brain_results/VAPI_alarm_occurences_matrix_V4_dedup.csv"

# Read the CSV file using Dask
df = dd.read_csv(input_file)

# Drop duplicate rows (by default, it considers all columns)
df_no_duplicates = df.drop_duplicates()

# Save the DataFrame back to a CSV file
df_no_duplicates.to_csv(output_file, index=False, single_file=True)
