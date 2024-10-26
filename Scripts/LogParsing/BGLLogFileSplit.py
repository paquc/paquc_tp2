import pandas as pd
import os

def split_csv(file_path, num_splits):
    # Load the large CSV file
    df = pd.read_csv(file_path)
    
    # Calculate the number of rows per split
    rows_per_split = len(df) // num_splits
    
    for i in range(num_splits):
        # Determine the start and end row for each split
        start_row = i * rows_per_split
        # If it's the last split, go till the end of the DataFrame
        end_row = None if i == num_splits - 1 else (i + 1) * rows_per_split
        
        # Get the slice of DataFrame for this split
        df_split = df.iloc[start_row:end_row]
        
        # Define the output filename with a suffix
        output_file = f"./BGL_Brain_results/BGL.log_structured_part{i+1}.csv"
        
        # Save the split to a new CSV file
        df_split.to_csv(output_file, index=False)
        
        print(f"Saved {output_file}")

# Example usage:
file_path = './BGL_Brain_results/BGL.log_structured.csv'  # Path to your large CSV file
num_splits = 5                                  # Number of smaller files you want

split_csv(file_path, num_splits)
