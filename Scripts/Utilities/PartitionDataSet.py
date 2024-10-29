import fileinput
import sys

n_data_set_id = 2   # >= 1

if len(sys.argv) >= 3:
    n_data_set_id_min = int(sys.argv[1])
    n_data_set_id_max = int(sys.argv[2])
    n_nb_Samples = int(sys.argv[3])
else:
    print("Usage: PartitionDataSet.py <data_set_id>")
    sys.exit(1)

print("Partition data set...")

# The number of lines ending with ',1': 296385 [713035 : 6860422]

window_size = 100

#n_nb_Samples = 800000

input_file = f"./Thunderbird_Brain_results/VAPI_alarm_occurences_matrix_V4_dedup.csv"
output_file = f"./Thunderbird_Brain_results/VAPI_alarm_occurences_matrix_V4_data_set_{n_data_set_id}.csv"


#input_file = "./Thunderbird_Brain_results/Samples.csv"
#output_file = "./Thunderbird_Brain_results/Samples_set.csv"
#n_start_line = 2
#n_nb_Samples = 3


# Get the header of the input CSV file
with open(input_file, "r") as infile:
    header = infile.readline()

print("Header:", header[:20])

n_start_line_alarms = 713035 - (2 * window_size)

for n_data_set_id in range(n_data_set_id_min, n_data_set_id_max+1):
    n_start_line_set = n_data_set_id * n_start_line_alarms  # Data set start position
    print(f"Data set {n_data_set_id} start line: {n_start_line_set}")
    # Use fileinput.input() to iterate over each line in the file
    with fileinput.input(files=input_file) as infile:
        output_file = f"./Thunderbird_Brain_results/VAPI_alarm_occurences_matrix_V4_data_set_{n_data_set_id}.csv"
        with open(output_file, "w") as outfile:
            outfile.write(header)
            line_count = 0
            samples_count = 0
            for line in infile:
                if line_count != 0:
                    if line_count >= n_start_line_set:
                        if samples_count < n_nb_Samples:
                            samples_count += 1
                            outfile.write(line)
                            print(line[:10])
                            if samples_count >= n_nb_Samples:
                                break
                line_count += 1


print("Done!")
