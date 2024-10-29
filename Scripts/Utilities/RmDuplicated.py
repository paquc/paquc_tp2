import pandas as pd

def remove_duplicates(file_path, output_path=None):
   
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


print("Deduplicates removal...")
remove_duplicates(f"./Thunderbird_Brain_results/VAPI_alarm_occurences_matrix_V4.csv", f"./Thunderbird_Brain_results/VAPI_alarm_occurences_matrix_V4_dedup.csv")
print("Deduplicated sequences saved successfully!")
