import pandas as pd
from collections import Counter

def GenMatrices(part_nb):
    #logs_file = './BGL_Brain_results/BGL.log_structured.csv'
    #logs_file = './BGL_Brain_results/BGL.Alarms_Samples.csv'
    logs_file = f"./BGL_Brain_results/BGL.log_structured_part{part_nb}.csv"

    df_logs = pd.read_csv(logs_file)

    # Définir les paramètres
    window_size = 100  # Nombre d'événements avant chaque alarme
    sequences = []
    labels = []

    for i in range(len(df_logs) - window_size):
        # Sélectionner une fenêtre de 'window_size' événements
        window = df_logs.iloc[i:i + window_size]
        # Générer une séquence d'EventIds
        sequence = ','.join(window['EventId'].tolist())
        # Vérifier si cette séquence est suivie d'une alarme
        is_alarm = int('KERNDTLB' in df_logs.iloc[i + window_size]['AlertFlagLabel'])
        sequences.append(sequence)
        labels.append(is_alarm)

    # Créer un DataFrame avec les séquences et les étiquettes
    df_sequences = pd.DataFrame({'EventSequence': sequences, 'IsAlarm': labels})

    df_sequences.to_csv(f"./BGL_Brain_results/KERNDTLB_alarm_sequences_V1_part{part_nb}.csv", index=False)

    print(df_sequences.head())

    #********************************************************************************************************************
    # Identifier tous les événements uniques
    unique_events = set()
    for sequence in df_sequences['EventSequence']:
        unique_events.update(sequence.split(','))
    unique_events = sorted(unique_events)

    # Construire la matrice d'occurrences
    matrix_data = []
    for sequence in df_sequences['EventSequence']:
        events = sequence.split(',')
        event_count = Counter(events)
        row = [event_count.get(event, 0) for event in unique_events]
        matrix_data.append(row)

    # Ajouter la colonne d'étiquette pour les alarmes
    df_matrix = pd.DataFrame(matrix_data, columns=unique_events)
    df_matrix['IsAlarm'] = df_sequences['IsAlarm']

    # Sauvegarder la matrice pour l'apprentissage
    df_matrix.to_csv(f"./BGL_Brain_results/KERNDTLB_alarm_occurences_matrix_V1_part{part_nb}.csv", index=False)

    print(df_matrix.head())


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


def merge_csv(num_splits, merged_file_name):
    # Collect all the smaller files with the given prefix and suffix pattern
    file_list = [f"./BGL_Brain_results/KERNDTLB_alarm_occurences_matrix_V1_part{i+1}_dedup.csv" for i in range(num_splits)]
    
    print(file_list)

    # Initialize an empty DataFrame
    merged_df = pd.DataFrame()
    
    # Read and concatenate each file
    for file in file_list:
        df = pd.read_csv(file)
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    
    # Save the concatenated DataFrame to a single large file
    merged_df.to_csv(merged_file_name, index=False)
    print(f"Merged file saved as {merged_file_name}")


for i in range(5):  
    print(f"Generating matrices V1 for part {i+1}")
    GenMatrices(i+1)
    print(f"Remove duplicates part {i+1}")
    remove_duplicates(f"./BGL_Brain_results/KERNDTLB_alarm_occurences_matrix_V1_part{i+1}.csv", f"./BGL_Brain_results/KERNDTLB_alarm_occurences_matrix_V1_part{i+1}_dedup.csv")


print("Matrices V1 generated successfully!")
