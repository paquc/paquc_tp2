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


# def find_future_event(df_logs, current_index, time_gap_seconds):
#     #print(f"Finding future event within {time_gap_seconds} seconds from index {current_index}...")
#     current_time_epoch = df_logs.iloc[current_index]['EpochTime']
#     future_time_epoch = current_time_epoch + time_gap_seconds

#     # Get the time of the last item in the log
#     if future_time_epoch >= df_logs.iloc[-1]['EpochTime']:
#         return None

#     # Find the first event that occurs after the future_time_epoch
#     for i in range(current_index + 1, len(df_logs)):
#         if df_logs.iloc[i]['EpochTime'] - df_logs.iloc[current_index]['EpochTime'] >= time_gap_seconds:
#             return [df_logs.iloc[i], i]
    
#     # If no future event is found within the time gap, return None
#     return None

def find_event(df_logs, current_index, time_gap_seconds, direction='future'):
   
    current_time_epoch = df_logs.iloc[current_index]['EpochTime']

    if direction == 'future':
        target_time_epoch = current_time_epoch + time_gap_seconds

        # Ensure the target time is within the log range
        if target_time_epoch > df_logs.iloc[-1]['EpochTime']:
            return None

        # Find the first event that occurs after the target_time_epoch
        for i in range(current_index + 1, len(df_logs), 1):
            if df_logs.iloc[i]['EpochTime'] - df_logs.iloc[current_index]['EpochTime'] >= time_gap_seconds:
                return [df_logs.iloc[i], i]

    elif direction == 'past':
        target_time_epoch = current_time_epoch - time_gap_seconds

        # Ensure the target time is within the log range
        if target_time_epoch < df_logs.iloc[0]['EpochTime']:
            return None

        # Find the first event that occurs before the target_time_epoch
        for i in range(current_index - 1, -1, -1):
           if df_logs.iloc[current_index]['EpochTime'] - df_logs.iloc[i]['EpochTime'] >= time_gap_seconds:
                return [df_logs.iloc[i], i]

    # If no event is found within the time gap, return None
    return None


def GenMatrices():

    #real_data = True
    real_data = False

    if real_data:
        suffix = "10M"
        logs_file = f"./Thunderbird_Brain_results/Thunderbird_{suffix}.log_structured.csv"
        time_window_epoch = 30
        prediction_window_epoch = 30  
        moving_window_epoch = -1         # -1 to diable
        moving_windows_index = 100      # -1 to diable
        log_index=False
    else:   
        suffix = "Samples"
        logs_file = f"./Thunderbird_Brain_results/Thunderbird.log_structured_{suffix}.csv"
        time_window_epoch = 3600  
        prediction_window_epoch = 60 * 5
        moving_window_epoch = -1        # -1 to diable
        moving_windows_index = 5        # -1 to diable
        log_index=True


    print(f"Processing log file: {logs_file}")
    df_logs = pd.read_csv(logs_file)

    start_time_epoch = 0

    # Open the output file in write mode
    with open(f"./Thunderbird_Brain_results/VAPI_alarm_sequences_{suffix}_chrono.csv", "w") as sequences_output_file:

        print(f"Generating sequences of events within {time_window_epoch} seconds for each entry...")
        # Write the header to the CSV file
        sequences_output_file.write("EventSequence,IsAlarm\n")
       
        init = True
        current_tail_wnd_event_index = 0
        current_start_wnd_event_index = 0

        while True:
            
            if(init):
                # Get the first event and index
                init = False
                current_start_wnd_event_index = 0
                next_tail_event_data = find_event(df_logs, current_start_wnd_event_index, time_window_epoch, 'future')
                if next_tail_event_data is None:
                    break
            else:
                # Find the next event to start the sequence
                if moving_window_epoch > 0:
                    next_start_event_data = find_event(df_logs, current_start_wnd_event_index, moving_window_epoch, 'future')
                    if next_start_event_data is None:
                        break   # Stop if no future event is found within the moving window
                    current_start_wnd_event_index = next_start_event_data[1]
                
                if moving_windows_index > 0:
                    current_start_wnd_event_index = current_start_wnd_event_index + moving_windows_index
                    
                # Find the next event to end the sequence    
                next_tail_event_data = find_event(df_logs, current_start_wnd_event_index, time_window_epoch, 'future')
                if next_tail_event_data is None:
                    break   # Stop if no future event is found within the moving window


            current_tail_wnd_event_index = next_tail_event_data[1]

            # Get the window of events using an index interval
            log_entries_window = df_logs.iloc[current_start_wnd_event_index : current_tail_wnd_event_index]

            # Generate a sequence of EventIds within the time window
            sequence = ','.join(log_entries_window['EventId'].tolist())

            # Check if the sequence is not empty
            if sequence:
                # Find the next event within the prediction window
                prediction_event = find_event(df_logs, current_tail_wnd_event_index, prediction_window_epoch, 'future')
                if prediction_event is None:
                    break   # Stop if no future event is found within the prediction window

                # Check if there is an alarm ('VAPI') 
                is_alarm = int('VAPI' in prediction_event[0]['AlertFlagLabel'])
                # Write the sequence and label to the file (ensure to insert "" around the sequence!!)
                if log_index:
                    sequences_output_file.write(f'{current_start_wnd_event_index}:{current_tail_wnd_event_index},"{sequence}",{is_alarm}\n')
                else:
                    sequences_output_file.write(f'"{sequence}",{is_alarm}\n')
                print(f"Sequence: {sequence} - IsAlarm: {is_alarm}")


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
