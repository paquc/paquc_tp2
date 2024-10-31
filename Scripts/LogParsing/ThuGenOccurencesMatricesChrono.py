import pandas as pd
from collections import Counter
import fileinput
import sys

# if len(sys.argv) >= 1:
#     hours = int(sys.argv[1])
# else:
#     print("Usage: script time_wnd_hours")
#     sys.exit(1)

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


def find_event(df_logs, current_index, time_gap_seconds, direction='future'):
    # Get the EpochTime value at index 0
    start_epoch_time = df_logs.iloc[current_index]['EpochTime']

    # Calculate the end epoch time for the desired time interval
    end_epoch_time = start_epoch_time + time_gap_seconds
    
    end_epoch_time_df = df_logs[df_logs['EpochTime'] >= end_epoch_time]
    if end_epoch_time_df.empty:
        return None

    end_index = end_epoch_time_df.index[0]

    # Filter the DataFrame to include only rows from index start until the specified time interval
    sequence_df = df_logs[(df_logs['EpochTime'] >= start_epoch_time) & (df_logs['EpochTime'] <= end_epoch_time)]

    return [sequence_df, end_index]


def GenMatrices():

    #real_data = True
    real_data = False


    if real_data:
        suffix = "10M"
        time_window_epoch = 15
        prediction_window_epoch = 10  
        moving_window_epoch = 10
        logs_file = f"./Thunderbird_Brain_results/Thunderbird_{suffix}.log_structured.csv"
    else:   
        suffix = "Samples"
        time_window_epoch = 60 * 30  
        prediction_window_epoch = 60 + 10
        moving_window_epoch = 60 + 60
        logs_file = f"./Thunderbird_Brain_results/Thunderbird.log_structured_{suffix}.csv"


    print(f"Processing log file: {logs_file}")
    df_logs = pd.read_csv(logs_file)

    next_start_log_time_epoch = df_logs.iloc[0]['EpochTime']

    # Open the output file in write mode
    with open(f"./Thunderbird_Brain_results/VAPI_alarm_sequences_{suffix}_chrono.csv", "w") as sequences_output_file:

        print(f"Generating sequences of events within {time_window_epoch} seconds for each entry...")
        # Write the header to the CSV file
        sequences_output_file.write("EventSequence,IsAlarm\n")
       
        aggregated_alarms_TH = 5
        window_box_start_index = 0

        while True:
                      
            # Get the window box
            window_box_event_data = find_event(df_logs, window_box_start_index, time_window_epoch, 'future')
            if window_box_event_data is None:
                break
            window_box_sequences_events_df = window_box_event_data[0]
            window_box_tail_event_index = window_box_event_data[1]
            #print(f"Window box start index: {window_box_start_index}, tail index: {window_box_tail_event_index}")
                 
            # Get the window after for prediction
            prediction_box_data = find_event(df_logs, window_box_tail_event_index + 1, prediction_window_epoch, 'future')
            if prediction_box_data is None:
                break
            prediction_df = prediction_box_data[0]
            #print(f"Prediction box start index: {window_box_tail_event_index + 1}, tail index: {prediction_box_data[1]}")

            # Filter the prediction DataFrame to keep only rows where 'AlertFlagLabel' is 'VAPI'
            prediction_VAPI_df = prediction_df.loc[prediction_df['AlertFlagLabel'] == 'VAPI']
            prediction_VAPI_bn257_df = prediction_VAPI_df.loc[prediction_VAPI_df['Noeud'] == 'bn257']

            num_VAPI_bn257_alarms = prediction_VAPI_bn257_df.shape[0]

            if num_VAPI_bn257_alarms >= aggregated_alarms_TH:
                is_alarm = 1
                # Generate a sequence of EventIds within the time window
                sequence = ','.join(window_box_sequences_events_df['EventId'].tolist())
                # Check if the sequence is not empty
                if sequence:
                    # Write the sequence and label to the file (*** ensure to insert "" around the sequence ***)
                    sequences_output_file.write(f'"{sequence}",{is_alarm}\n')
                    print(f"ALRM - Number of VAPI alarms in prediction window for bn257: {num_VAPI_bn257_alarms} > {aggregated_alarms_TH})")


            # Update the start index for the next window box
             # Get info for NEXT window box
            next_window_box_event_data = find_event(df_logs, window_box_start_index, moving_window_epoch, 'future')
            if next_window_box_event_data is None:
                break
            
            # Move the window box start index to the next window box
            window_box_start_index = next_window_box_event_data[1]
                       

    # Remove diplicates from the sequences
    print("Sequences generated successfully!")
    remove_duplicates(f"./Thunderbird_Brain_results/VAPI_alarm_sequences_{suffix}_chrono.csv", f"./Thunderbird_Brain_results/VAPI_alarm_sequences_{suffix}_chrono_dedup.csv")
    print("Deduplicated sequences saved successfully!")
    
    # Charger le fichier de séquences d'événements générées précédemment
    # Ce fichier contient les séquences d'événements et leurs étiquettes (indicateur d'alarme)
    sequences_file = f"./Thunderbird_Brain_results/VAPI_alarm_sequences_{suffix}_chrono_dedup.csv"
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
    matrix_output_file_path = f"./Thunderbird_Brain_results/VAPI_alarm_occurences_matrix_{suffix}_chrono.csv"

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
