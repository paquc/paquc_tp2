import pandas as pd

# Define the file paths
file1 = './Thunderbird_Brain_results/VAPI_alarm_occurences_matrix_V4_part_1_dedup.csv'
file2 = './Thunderbird_Brain_results/VAPI_alarm_occurences_matrix_V4_part_2_2_dedup.csv'

output_file = './Thunderbird_Brain_results/VAPI_alarm_occurences_matrix_V4_full_dedup.csv'

# Read both CSV files
#df1 = pd.read_csv(file1)
#df2 = pd.read_csv(file2)

print("Merging files...")

# Open file1 in append mode ('a') and file2 in read mode ('r')
with open(file1, 'a') as f1, open(file2, 'r') as f2:
    # Skip the header of file2 if both files have headers and you want to avoid duplication
    # next(f2)
    
    # Write each line from file2 to file1
    for line in f2:
        f1.write(line)


print("Done!")

