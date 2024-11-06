import pandas as pd
import sys

from collections import Counter


if len(sys.argv) >= 3:
    node_name = str(sys.argv[1])
    window_size = int(sys.argv[2])
    alarms_THRESHOLD = int(sys.argv[3])
else:
    print("Usage: script time_wnd_hours")
    sys.exit(1)

node_name_formated = node_name.replace(":", "_").replace("-", "_").replace(" ", "_").replace("'", "")

slide_window_suffix = f"{node_name_formated}_{window_size}"

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


def GenMatrices():
    logs_file = './BGL_Brain_results/BGL.log_structured_V4.csv'
    #logs_file = f"./BGL_Brain_results/BGL.log_structured_full_content_filtered.csv"  

    #logs_file = './BGL_Brain_results/BGL.Alarms_Samples.csv'
    df_logs = pd.read_csv(logs_file)
    
    # Filter the dataframe to keep only lines where NodeLoc is equal to node_name
    # df_logs = df_logs[df_logs['NodeLoc'] == node_name]

    # Définir les paramètres
    # window_size = 500  # Nombre d'événements avant chaque alarme

    # Open the output file in write mode
    with open(f"./BGL_Brain_results/{slide_window_suffix}_alarm_sequences_FixWindow.csv", "w") as sequences_output_file:
        print(f"Generating sequences of {window_size} events before each alarm...")
        
        # Write the header to the CSV file
        sequences_output_file.write("EventSequence,IsAlarm\n")

        alarm_count = 0

        # Iterate over the logs with a sliding window
        for i in range(0, len(df_logs) - window_size + 1, window_size):
            # Select a window of 'window_size' events
            window_df = df_logs.iloc[i : i + window_size - 1]
            # Generate a sequence of EventIds
            sequence = ','.join(window_df['EventId'].tolist())

            # Count the number of lines where NodeLoc is node_name and AlertFlagLabel is different than '-'
            nb_alarms_found = (window_df['NodeLoc'].str.contains(node_name) & (window_df['AlertFlagLabel'] != '-')).sum()

            if nb_alarms_found >= alarms_THRESHOLD:
                sequences_output_file.write(f'"{sequence}",1\n')
                print(f"Alarms found in window = {nb_alarms_found}")
                alarm_count += 1
                print(f"Alarms detected in nb windows = {alarm_count}")  
            else:
                sequences_output_file.write(f'"{sequence}",0\n')
           
            # Check if the sequence is followed by an alarm
            #is_alarm = window_df['AlertFlagLabel'].ne('-').sum() >= alarms_THRESHOLD
            # is_alarm = int(alarm_name in df_logs.iloc[i + window_size]['AlertFlagLabel'])
            # is_alarm = int(df_logs.iloc[i + window_size]['AlertFlagLabel'] != '-')

            # Write the sequence and label to the file (ensure to insert "" around the sequence!!)
            # sequences_output_file.write(f'"{sequence}",{is_alarm}\n')
      

    sequences_file = f"./BGL_Brain_results/{slide_window_suffix}_alarm_sequences_FixWindow.csv"
    sequences_file_dedup = f"./BGL_Brain_results/{slide_window_suffix}_alarm_sequences_FixWindow_dedup.csv"

    # Count the number of alarms in df_sequences
    #df_sequences_original = pd.read_csv(sequences_file, header=0)
    #num_alarms = df_sequences_original['IsAlarm'].sum()
    #print(f"Total number of alarms in sequences: {num_alarms}")

    print("Sequences generated successfully!")
    remove_duplicates(sequences_file, sequences_file_dedup)
    print("Deduplicated sequences saved successfully!")
    
    # Charger le fichier de séquences d'événements générées précédemment
    # Ce fichier contient les séquences d'événements et leurs étiquettes (indicateur d'alarme)
    df_sequences = pd.read_csv(sequences_file_dedup, header=0)

    # Count the number of alarms in df_sequences
    num_alarms = df_sequences['IsAlarm'].sum()
    print(f"Total number of alarms in sequences deduplicated: {num_alarms}")

    #********************************************************************************************************************
    # Identifier tous les événements uniques à travers toutes les séquences d'événements
    print("Identifying unique events...")
    unique_events = set()  # Initialiser un ensemble vide pour collecter tous les événements uniques

    # Parcourir chaque séquence d'événements dans la colonne 'EventSequence' du DataFrame
    for sequence in df_sequences['EventSequence']:
        # Diviser chaque séquence en événements individuels et les ajouter à l'ensemble unique_events
        unique_events.update(sequence.split(','))

    # Trier les événements uniques pour garantir un ordre cohérent des colonnes dans la matrice d'occurrences
    unique_events = sorted(unique_events)
    print(f"Found {len(unique_events)} unique events.")

    #********************************************************************************************************************
    # Écrire les données de la matrice d'occurrences ligne par ligne dans le fichier CSV
    matrix_output_file_path = f"./BGL_Brain_results/{slide_window_suffix}_alarm_occurences_matrix_FixWindow.csv"

    # Ouvrir le fichier de sortie en mode écriture
    with open(matrix_output_file_path, 'w') as matrix_output_file:
        print(f"Generating occurrence matrix at {matrix_output_file_path}...")
        # Écrire l'en-tête (les événements uniques comme colonnes et 'IsAlarm' comme dernière colonne)
        header = ','.join(unique_events) + ',IsAlarm\n'
        matrix_output_file.write(header)

        # Parcourir chaque séquence d'événements dans la colonne 'EventSequence' du DataFrame
        for idx, row in df_sequences.iterrows():
            # Diviser la séquence en événements individuels
            events = row['EventSequence'].split(',')
            # Compter le nombre d'occurrences de chaque événement dans la séquence actuelle
            event_count = Counter(events)
            # Créer une ligne pour la matrice d'occurrences, avec chaque colonne représentant un événement unique
            # La valeur de chaque cellule est le nombre d'occurrences de cet événement dans la séquence actuelle
            matrix_row = [event_count.get(event, 0) for event in unique_events]
            # Ajouter l'étiquette 'IsAlarm' à la fin de la ligne
            matrix_row.append(row['IsAlarm'])
            # Écrire la ligne dans le fichier CSV, avec les valeurs séparées par des virgules
            matrix_output_file.write(','.join(map(str, matrix_row)) + '\n')
            #print(f"Row {idx + 1} written to the occurrence matrix...") 

    print(f"Occurrence matrix generated successfully at {matrix_output_file_path}!")
    remove_duplicates(matrix_output_file_path, matrix_output_file_path.replace(".csv", "_dedup.csv"))
    print(f"Deduplicated occurrence matrix saved successfully at {matrix_output_file_path.replace('.csv', '_dedup.csv')}")

GenMatrices()

print("Matrices FixWindow generated successfully!")
