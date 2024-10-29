import pandas as pd
from collections import Counter

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
    logs_file = './Thunderbird_Brain_results/Thunderbird_10M.log_structured.csv'
    #logs_file = './BGL_Brain_results/BGL.Alarms_Samples.csv'
    df_logs = pd.read_csv(logs_file)

    # Définir les paramètres
    window_size = 100  # Nombre d'événements avant chaque alarme

    # Open the output file in write mode
    with open(f"./Thunderbird_Brain_results/VAPI_alarm_sequences_V4.csv", "w") as sequences_output_file:
        print(f"Generating sequences of {window_size} events before each alarm...")
        # Write the header to the CSV file
        sequences_output_file.write("EventSequence,IsAlarm\n")

        # Iterate over the logs with a sliding window
        for i in range(len(df_logs) - window_size):
            # Select a window of 'window_size' events
            window = df_logs.iloc[i:i + window_size]
            # Generate a sequence of EventIds
            sequence = ','.join(window['EventId'].tolist())
            # Check if the sequence is followed by an alarm
            is_alarm = int('VAPI' in df_logs.iloc[i + window_size]['AlertFlagLabel'])
            # Write the sequence and label to the file (ensure to insert "" around the sequence!!)
            sequences_output_file.write(f'"{sequence}",{is_alarm}\n')

    print("Sequences generated successfully!")
    remove_duplicates("./Thunderbird_Brain_results/VAPI_alarm_sequences_V4.csv", "./Thunderbird_Brain_results/VAPI_alarm_sequences_V4_dedup.csv")
    print("Deduplicated sequences saved successfully!")
    
    # Charger le fichier de séquences d'événements générées précédemment
    # Ce fichier contient les séquences d'événements et leurs étiquettes (indicateur d'alarme)
    sequences_file = "./Thunderbird_Brain_results/VAPI_alarm_sequences_V4_dedup.csv"
    df_sequences = pd.read_csv(sequences_file, header=0)

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
    matrix_output_file_path = "./Thunderbird_Brain_results/VAPI_alarm_occurences_matrix_V4.csv"

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

    print(f"Occurrence matrix generated successfully at {matrix_output_file_path}!")
    remove_duplicates(matrix_output_file_path, matrix_output_file_path.replace(".csv", "_dedup.csv"))
    print(f"Deduplicated occurrence matrix saved successfully at {matrix_output_file_path.replace('.csv', '_dedup.csv')}")

GenMatrices()

print("Thunderbird matrices V4 generated successfully!")
