import fileinput
import sys


if len(sys.argv) > 1:
    n_data_set_id = int(sys.argv[1])
else:
    print("Usage: PartitionDataSet.py <data_set_id>")
    sys.exit(1)

#input_file = f"./Thunderbird_Brain_results/VAPI_alarm_occurences_matrix_V4_data_set_{n_data_set_id}.csv"
input_file = f"./Thunderbird_Brain_results/VAPI_alarm_occurences_matrix_V4_dedup.csv"

def count_lines_ending_with(file_path, ending_string):
    count = 0
    with fileinput.input(files=input_file) as infile:
        for line in infile:
            # Remove any trailing whitespace characters (e.g., \n) and check if it ends with the specified string
            if line.rstrip().endswith(ending_string):
                count += 1
                print(f"Line {infile.filelineno()}: {count}")
                if count == 1:
                    first_line = infile.filelineno()
                last_line = infile.filelineno()


    print(f"The number of lines ending with '{ending_string}': {count} [{first_line} : {last_line}]")
    return count

count_lines_ending_with(input_file, ",1") 

print("Counting is completed.")


