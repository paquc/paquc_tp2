def count_lines_ending_with(file_path, ending_string):
    """
    Count the number of lines in a file that end with a given string.

    Parameters:
    - file_path (str): Path to the input file.
    - ending_string (str): The string to match at the end of each line.

    Returns:
    - int: The number of lines that end with the given string.
    """
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            # Remove any trailing whitespace characters (e.g., \n) and check if it ends with the specified string
            if line.rstrip().endswith(ending_string):
                count += 1

    print(f"The number of lines ending with '{ending_string}': {count}")

    return count

# Example usage:
#file_path = 'sample_file.txt'  # Replace with your file path
#ending_string = 'error'        # Replace with the string you are looking for
#result = count_lines_ending_with(file_path, ending_string)


print("Counting in events sequences ...")
for i in range(5):  
    count_lines_ending_with(f"./BGL_Brain_results/KERNDTLB_alarm_sequences_V1_part{i+1}.csv", f",1")

print("Counting in original occurences matrices...")
for i in range(5):  
    count_lines_ending_with(f"./BGL_Brain_results/KERNDTLB_alarm_occurences_matrix_V1_part{i+1}.csv", f",1")

print("Counting in deduped occurences matrices...")
for i in range(5):  
    count_lines_ending_with(f"./BGL_Brain_results/KERNDTLB_alarm_occurences_matrix_V1_part{i+1}_dedup.csv", f",1")

print("Counting is completed.")


