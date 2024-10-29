import fileinput

print("Removing duplicates from large_file.csv...")

# Path to your input CSV file
input_file = "./Thunderbird_Brain_results/VAPI_alarm_occurences_matrix_V4_dedup.csv"
#input_file = "./Thunderbird_Brain_results/Samples.csv"
# Path to the output CSV file without duplicates
output_file = "./Thunderbird_Brain_results/VAPI_alarm_occurences_matrix_V4_dedup_2.csv"
#output_file = "./Thunderbird_Brain_results/Samples_dedup.csv"


unique_lines = set()

# Use fileinput.input() to iterate over each line in the file
with fileinput.input(files=input_file) as infile:
    with open(output_file, "w") as outfile:
        for line in infile:
            # Process the line (print in this case)
            if line not in unique_lines:
                unique_lines.add(line)
                outfile.write(line)
                print(line[:10])


print("Done!")
