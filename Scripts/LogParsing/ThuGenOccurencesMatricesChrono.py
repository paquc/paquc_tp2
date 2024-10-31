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

# Function to find the event sequence within a specified time window
def find_event(df_logs, window_box_start_time_epoch, window_box_time_width_epoch, direction='future'):
    
    # Calculate the end epoch time for the desired time interval
    end_epoch_time = window_box_start_time_epoch + window_box_time_width_epoch

    # Filter the DataFrame to include only rows from start until the specified time interval
    sequence_df = df_logs[(df_logs['EpochTime'] >= window_box_start_time_epoch) & (df_logs['EpochTime'] <= end_epoch_time)]
    if sequence_df.empty:
        return None

    # Get the last row of the sequence to get the end time of the sequence
    window_box_last_row_epochtime = sequence_df.iloc[-1]['EpochTime']

    return [sequence_df, window_box_last_row_epochtime]

def GenMatrices():

    real_data = True
    #real_data = False

    if real_data:
        suffix = "10M"
        time_window_epoch = 60*30
        prediction_window_epoch = 60*10   
        moving_window_epoch = 60*15
        logs_file = f"./Thunderbird_Brain_results/Thunderbird_{suffix}.log_structured.csv"
    else:   
        suffix = "Samples"
        time_window_epoch = 60*1 
        prediction_window_epoch = 60*20
        moving_window_epoch = 60*1
        logs_file = f"./Thunderbird_Brain_results/Thunderbird.log_structured_{suffix}.csv"

    # Load the log data from the CSV file
    print(f"Processing log file: {logs_file}")
    df_logs = pd.read_csv(logs_file)
    
    # Open the output file in write mode
    with open(f"./Thunderbird_Brain_results/VAPI_alarm_sequences_{suffix}_chrono.csv", "w") as sequences_output_file:

        print(f"Generating sequences of events within {time_window_epoch} seconds for each entry...")
        sequences_output_file.write("EventSequence,IsAlarm\n")
       
        VAPI_bn257_df = df_logs.loc[(df_logs['AlertFlagLabel'] == 'VAPI') & (df_logs['Noeud'] == 'bn257')]
        print(VAPI_bn257_df.head(10))
        print(f"Number of VAPI alarms for node 'bn257' in full log file: {VAPI_bn257_df.shape[0]}")
        #sys.exit(0)

        aggregated_alarms_TH = 5
        print_zeros = 1
        
        window_box_sequence_start_time_epoch = df_logs.iloc[0]['EpochTime']

        while True:
                      
            # Get the window box
            window_box_sequence_data = find_event(df_logs, window_box_sequence_start_time_epoch, time_window_epoch, 'future')
            if window_box_sequence_data is None:
                break
            #print(f"Window box start time: {window_box_sequence_start_time_epoch}, tail time: {window_box_sequence_data[1]}, Nb rows: {window_box_sequence_data[0].shape[0]}, Total time: {window_box_sequence_data[0].iloc[-1]['EpochTime'] - window_box_sequence_start_time_epoch}")

            # Get the prediction window
            window_box_sequence_tail_time_epoch = window_box_sequence_data[1]
            prediction_box_data = find_event(df_logs, window_box_sequence_tail_time_epoch, prediction_window_epoch, 'future')
            if prediction_box_data is None:
                break
            prediction_df = prediction_box_data[0]
            #print(prediction_df.head(10))
            #print(f"Prediction box start time: {window_box_sequence_tail_time_epoch + 1}, tail time: {prediction_box_data[1]}, Total time: {prediction_df.iloc[-1]['EpochTime'] - window_box_sequence_tail_time_epoch}")

            # Filter the prediction DataFrame to keep only rows where 'AlertFlagLabel' is 'VAPI' for node 'bn257'
            prediction_VAPI_bn257_df = prediction_df.loc[(prediction_df['AlertFlagLabel'] == 'VAPI') & (prediction_df['Noeud'] == 'bn257')]
            #print(f"Number of VAPI alarms for node 'bn257' in prediction box: {prediction_VAPI_bn257_df.shape[0]}")

            # Count the number of alarms in the prediction window (shape() returns a tuple of (num_rows, num_columns))
            num_VAPI_bn257_alarms = prediction_VAPI_bn257_df.shape[0]

            # There is an alram if the number of alarms is greater than the threshold
            if num_VAPI_bn257_alarms >= aggregated_alarms_TH:
                is_alarm = 1
                print(f"ALARM: Alarms detected: {num_VAPI_bn257_alarms} alarms.")
                print_zeros = 1
            else:
                is_alarm = 0
                if print_zeros:
                    print(f"NO ALARM: alarms detected: {num_VAPI_bn257_alarms} alarms. Searching...")
                    print_zeros = 0

            # Filter out all events corresponfig to node 'bn257' in window box for sequence
            window_box_sequences_events_df = window_box_sequence_data[0]
            window_box_sequences_events_df = window_box_sequences_events_df.loc[window_box_sequences_events_df['Noeud'] == 'bn257']   

            # Generate a sequence of EventIds within the sequence window box
            sequence = ','.join(window_box_sequences_events_df['EventId'].tolist())

            # Check if the sequence is not empty and generate the sequence
            if sequence:
                # Write the sequence and label to the file (*** ensure to insert "" around the sequence ***)
                sequences_output_file.write(f'"{sequence}",{is_alarm}\n')

            # Get info for NEXT window box
            next_window_box_event_data = find_event(df_logs, window_box_sequence_start_time_epoch, moving_window_epoch, 'future')
            if next_window_box_event_data is None:
                break
            
            # Get start time of the next window box
            window_box_sequence_start_time_epoch = next_window_box_event_data[1] + 1
        
        # sys.exit(0)

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
