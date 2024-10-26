import pandas as pd

def remove_duplicates(file_path, output_path=None):
    """
    Remove duplicate rows from a CSV file and keep only one of each type.
    
    Parameters:
    - file_path (str): Path to the input CSV file.
    - output_path (str, optional): Path to save the deduplicated file. If None, overwrites the input file.
    
    Returns:
    - DataFrame: The deduplicated DataFrame.
    """
    # Load the file
    df = pd.read_csv(file_path)
    
    # Drop duplicate rows and keep the first occurrence
    df_dedup = df.drop_duplicates(keep='first')
    
    # Define the output path
    if output_path is None:
        output_path = file_path  # Overwrite the original file if no output path is provided
    
    # Save the deduplicated data
    df_dedup.to_csv(output_path, index=False)
    
    print(f"Duplicates removed. Deduplicated file saved to {output_path}")
    return df_dedup

# Example usage:
#file_path = 'large_file.csv'         # Path to your CSV file
#output_path = 'deduplicated_file.csv'  # Optional: Path to save the deduplicated file
#remove_duplicates(file_path, output_path)

for i in range(1):  # Loops from i = 0 to i = 4
    print(f"Dedup part {i+1}...")
    remove_duplicates(f"./BGL_Brain_results/KERNDTLB_alarm_occurences_matrix_V1_part{i+1}.csv", f"./BGL_Brain_results/KERNDTLB_alarm_occurences_matrix_V1_part{i+1}_dedup.csv")


print("Matrices V1 generated successfully!")


