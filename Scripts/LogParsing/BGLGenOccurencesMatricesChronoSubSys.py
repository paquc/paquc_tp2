import pandas as pd
from collections import Counter
import fileinput
import sys

if len(sys.argv) >= 4:
    alarm_name = str(sys.argv[1])
    node_name = str(sys.argv[2])
    time_window_epoch = int(sys.argv[3])
    prediction_window_epoch = int(sys.argv[4])
    moving_window_epoch = int(sys.argv[5])
else:
    print("Usage: script time_wnd_hours")
    sys.exit(1)

node_name_formated = node_name.replace(":", "_").replace("-", "_").replace(" ", "_").replace("'", "")


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

def find_next_start(df_logs, window_box_start_time_epoch, window_box_time_width_epoch, direction='future'):
    
    # Calculate the end epoch time for the desired time interval
    end_epoch_time = window_box_start_time_epoch + window_box_time_width_epoch

    # Filter rows where 'EpochTime' is greater than the given value
    filtered_df = df_logs[df_logs['EpochTime'] >= end_epoch_time]
    if filtered_df.empty:
        return None

    next_start_epoch_time = filtered_df.iloc[0]['EpochTime']
    
    return [filtered_df, next_start_epoch_time]


def print_alarm_types(df_logs, node_name = 'R30-M0-N9-C:J16-U01', alarm_name = 'KERNDTLB'):
             
    kernel_alarms_count = df_logs[df_logs['AlertFlagLabel'] == alarm_name].groupby('NodeLoc').size()
    print(kernel_alarms_count)
    
    filtered_df = df_logs[df_logs['AlertFlagLabel'] == alarm_name].groupby('NodeLoc')

    output_file = f"./BGL_Brain_results/{alarm_name}_{node_name_formated}_alarms_types.csv"
    output_file = f"./BGL_Brain_results/{alarm_name}_{node_name_formated}_alarms_types.csv".replace("'", "")

    with open(output_file, "w") as alarm_tpes_output_file:
        KERNEL_node_df = df_logs.loc[(df_logs['AlertFlagLabel'] == alarm_name) & (df_logs['NodeLoc'] == node_name)]
        alarm_tpes_output_file.write(f"Number of {node_name} alarms for node {node_name} in full log file: {KERNEL_node_df.shape[0]}")
        alarm_tpes_output_file.write(f"\n\n")

        for noeud, group in filtered_df:
            alarm_tpes_output_file.write(f"Group for NodeLoc: {noeud}\n")
            alarm_tpes_output_file.write(f"{group}\n")
            alarm_tpes_output_file.write(f"\n")
            alarm_tpes_output_file.write(f"Number of alarms for NodeLoc {noeud}: {group.shape[0]}\n")

        top_10_filtered_df = filtered_df.apply(lambda x: x.shape[0]).nlargest(10)
        for node, count in top_10_filtered_df.items():
            alarm_tpes_output_file.write(f"NodeLoc: {node}, Number of alarms: {count}\n")

    print(f"Alarm types saved to {output_file}")


def GenMatrices():

    #logs_file = f"./BGL_Brain_results/BGL.log_structured_V4.csv"
    logs_file = f"./BGL_Brain_results/BGL.log_structured_full_content_filtered.csv"  

    # Load the log data from the CSV file
    print(f"Processing log file: {logs_file}")

    df_logs = pd.read_csv(logs_file)

    print(f"Number of log entries: {df_logs.shape[0]}")
    print(f"Last epoch time: {df_logs.iloc[-1]['EpochTime']}")

    # Open the output file in write mode
    with open(f"./BGL_Brain_results/{alarm_name}_alarm_sequences_{node_name_formated}_{time_window_epoch}_{prediction_window_epoch}_{moving_window_epoch}_chrono.csv", "w") as sequences_output_file:

        print(f"Generating sequences of events within {time_window_epoch} seconds for each entry...")
        sequences_output_file.write("EventSequence,IsAlarm\n")
       
        # Number of alarms for NodeLoc R30-M0-N9-C:J16-U01: 152182
        #print_alarm_types(df_logs, node_name, alarm_name)
        #sys.exit(0)
        
        aggregated_alarms_TH = 1
        search_counter = 0  
        
        window_box_sequence_start_time_epoch = df_logs.iloc[0]['EpochTime']

        while True:
                      
            # Get the window box
            window_box_sequence_data = find_event(df_logs, window_box_sequence_start_time_epoch, time_window_epoch, 'future')
            if window_box_sequence_data is None:
                print("No more events to process (window_box_sequence_data). Exiting...")
                break
            #print(f"Window box start time: {window_box_sequence_start_time_epoch}, tail time: {window_box_sequence_data[1]}, Nb rows: {window_box_sequence_data[0].shape[0]}, Total time: {window_box_sequence_data[0].iloc[-1]['EpochTime'] - window_box_sequence_start_time_epoch}")

            # Get the prediction window
            window_box_sequence_tail_time_epoch = window_box_sequence_data[1]
            prediction_box_data = find_event(df_logs, window_box_sequence_tail_time_epoch, prediction_window_epoch, 'future')
            if prediction_box_data is None:
                print("No more events to process (prediction_box_data). Exiting...")
                break

            prediction_df = prediction_box_data[0]
            #print(prediction_df.head(10))
            #print(f"Prediction box start time: {window_box_sequence_tail_time_epoch + 1}, tail time: {prediction_box_data[1]}, Total time: {prediction_df.iloc[-1]['EpochTime'] - window_box_sequence_tail_time_epoch}")

            # Filter the prediction DataFrame to keep only rows where 'AlertFlagLabel' is KERNELDTLB for node 'bn257'
            # print(prediction_df.head())
            prediction_KERNEL_node_df = prediction_df.loc[(prediction_df['AlertFlagLabel'] == alarm_name) & (prediction_df['SubSys'] == 'KERNEL')]
            # print(f"Number of KERNELDTLB alarms for node {node_name} in prediction box: {prediction_KERNEL_node_df.shape[0]}")

            # Count the number of alarms in the prediction window (shape() returns a tuple of (num_rows, num_columns))
            num_KERNEL_node_alarms = prediction_KERNEL_node_df.shape[0]

            # There is an alram if the number of alarms is greater than the threshold
            if num_KERNEL_node_alarms >= aggregated_alarms_TH:
                is_alarm = 1
                print(f"ALARM: Alarms detected: {num_KERNEL_node_alarms} alarms.")
                search_counter = 0
            else:
                is_alarm = 0
                if search_counter == 0:
                    print(f"NO ALARM: alarms detected: {num_KERNEL_node_alarms} alarms. Searching...")
                search_counter += 1
                print(f"ALARMS searching --> {search_counter}..... ")
                if search_counter > 10:
                    search_counter = 0

            # Filter out all events corresponfig to node 'node_name' in window box for sequence
            window_box_sequences_events_df = window_box_sequence_data[0]
            # window_box_sequences_node_events_df = window_box_sequences_events_df.loc[window_box_sequences_events_df['NodeLoc'] == node_name]   
            # window_box_sequences_node_events_df = window_box_sequences_events_df.loc[(window_box_sequences_events_df['AlertFlagLabel'] != '-') & (window_box_sequences_events_df['SubSys'] == 'KERNEL')]
            window_box_sequences_node_events_df = window_box_sequences_events_df.loc[(window_box_sequences_events_df['SubSys'] == 'KERNEL') & (window_box_sequences_events_df['Severity'] == 'FATAL')]

            # Generate a sequence of EventIds within the sequence window box
            sequence_events = ','.join(window_box_sequences_node_events_df['EventId'].tolist())

            # Check if the sequence is not empty and generate the sequence
            if sequence_events:
                # Write the sequence and label to the file (*** ensure to insert "" around the sequence ***)
                sequences_output_file.write(f'"{sequence_events}",{is_alarm}\n')
                #print(f"Sequence: {sequence_events}, IsAlarm: {is_alarm}")

            # Get info for NEXT window box
            df_logs_data = find_next_start(df_logs, window_box_sequence_start_time_epoch, moving_window_epoch, 'future')
            if df_logs_data is None:
                print("No more events to process (next_window_box_event_data). Exiting...")
                break
            
            df_logs = df_logs_data[0]
            window_box_sequence_start_time_epoch = df_logs_data[1]
            
        
        # sys.exit(0)

    # Remove diplicates from the sequences
    print("Sequences generated successfully!")
    remove_duplicates(f"./BGL_Brain_results/{alarm_name}_alarm_sequences_{node_name_formated}_{time_window_epoch}_{prediction_window_epoch}_{moving_window_epoch}_chrono.csv", f"./BGL_Brain_results/{alarm_name}_alarm_sequences_{node_name_formated}_{time_window_epoch}_{prediction_window_epoch}_{moving_window_epoch}_chrono_dedup.csv")
    print("Deduplicated sequences saved successfully!")
    
    # Charger le fichier de séquences d'événements générées précédemment
    # Ce fichier contient les séquences d'événements et leurs étiquettes (indicateur d'alarme)
    sequences_file = f"./BGL_Brain_results/{alarm_name}_alarm_sequences_{node_name_formated}_{time_window_epoch}_{prediction_window_epoch}_{moving_window_epoch}_chrono_dedup.csv"
    df_sequences = pd.read_csv(sequences_file, header=0)

    #********************************************************************************************************************
    # Identifier tous les événements uniques à travers toutes les séquences d'événements
    print("Identifying unique events...")
    unique_events = set()  # Initialiser un ensemble vide pour collecter tous les événements uniques

    # Parcourir chaque séquence d'événements dans la colonne 'EventSequence' du DataFrame
    for sequence_events in df_sequences['EventSequence']:
        # Diviser chaque séquence en événements individuels et les ajouter à l'ensemble unique_events (set ignore the duplicates!!)
        unique_events.update(sequence_events.split(','))

    # Trier les événements uniques pour garantir un ordre cohérent des colonnes dans la matrice d'occurrences
    unique_events = sorted(unique_events)
    print(f"Found {len(unique_events)} unique events.")

    #********************************************************************************************************************
    # Écrire les données de la matrice d'occurrences ligne par ligne dans le fichier CSV
    matrix_output_file_path = f"./BGL_Brain_results/{alarm_name}_alarm_occurences_matrix_{node_name_formated}_{time_window_epoch}_{prediction_window_epoch}_{moving_window_epoch}_chrono.csv"

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
print("COMPLETED - BGL matrices CHRONO generated successfully!")
