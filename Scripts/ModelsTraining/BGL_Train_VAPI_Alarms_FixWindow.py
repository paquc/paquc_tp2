import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, accuracy_score, mean_absolute_error, r2_score
from sklearn.metrics import roc_auc_score, roc_curve
from collections import Counter
from sklearn.datasets import make_regression
from sklearn.utils import resample
import os


#Exemple: python Train.py 1 1 V1_part1_dedup KERNDTLB 5 1 1 1 10 60 20

if len(sys.argv) >= 7:
    train_LR = int(sys.argv[1])
    train_RF = int(sys.argv[2])
    node_name = str(sys.argv[3])
    window_size = int(sys.argv[4])
    use_bootstrap = int(sys.argv[5])
    n_bootstrap_samples = int(sys.argv[6])
    train_data_size = int(sys.argv[7])
    test_data_size = int(sys.argv[8])
    val_data_size = int(sys.argv[9])
    alarm_name = str(sys.argv[10])
else:
    print("Usage: script time_wnd_hours")
    sys.exit(1)

# Evaluation function
def get_model_evaluation(y_test, y_pred, model_name, log_file, estimators, randomize_val, model = None, X_train = None, RF = True):
    log_file.write("------------------------------------\n")
    log_file.write(f"--- {model_name} ---\n\n")

    log_file.write(f"Size of test set: {y_test.shape[0]}\n")
    log_file.write(f"Size of predicted set: {y_pred.shape[0]}\n\n")
    log_file.write(f"Number of estimators: {estimators}\n")
    log_file.write(f"Randomize value: {randomize_val}\n\n")

    unique_classes_tests = np.unique(y_test)
    unique_classes_pred = np.unique(y_pred)
    log_file.write(f"Unique classes in TESTS\VALIDATION set: {unique_classes_tests}\n")
    log_file.write(f"Unique classes in PREDICTED set: {unique_classes_pred}\n\n")

    compute_AUC = True
    if len(unique_classes_tests) <= 1:
        log_file.write(f'WARNING - Only 1 class present in the TESTS\VALIDATION set: {unique_classes_tests}\n\n')
        compute_AUC = False

    if len(unique_classes_pred) <= 1:
        log_file.write(f'WARNING - Only 1 class present in the predicted set: {unique_classes_pred}\n\n')    
        compute_AUC = False

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    msg=f"Accuracy: {accuracy:.4f}"
    print(msg)
    log_file.write(msg+"\n")
        
    # Calculate recall
    recall = recall_score(y_test, y_pred)
    msg=f"Recall: {recall:.2f}"
    print(msg)
    log_file.write(msg+"\n")

    # Calculate AUC    
    if compute_AUC:
        AUC = roc_auc_score(y_test, y_pred)
        msg=f"AUC: {AUC:.2f}"
        print(msg)
        log_file.write(msg+"\n")
    else:
        msg=f"AUC: N/A"
        print(msg)
        log_file.write(msg+"\n")

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    msg=f"Mean Absolute Error (MAE): {mae:.2f}"
    print(msg)
    log_file.write(msg+"\n")
    
    # R-squared (R²)
    r2 = r2_score(y_test, y_pred)
    msg=f"R-squared: {r2:.2f}"
    print(msg)
    log_file.write(msg+"\n")

    # Alternatively, print the classification report
    msg=classification_report(y_test, y_pred)
    print(msg)
    log_file.write(msg+"\n")

    if not RF:
        # Get the coefficients and intercept
        coefficients = model.coef_[0]  # array of coefficients for each feature
        intercept = model.intercept_[0]  # intercept

        # Exponentiate coefficients to get odds ratios
        odds_ratios = np.exp(coefficients)

        # Display the coefficients and odds ratios
        print("Coefficients:")

        feature_names = X_train.columns  # X_train is your training data in DataFrame format

        # Print feature name, feature number, and coefficient
        for feature_num, (feature_name, coef) in enumerate(zip(feature_names, coefficients)):
            LR_log_file.write(f"Feature {feature_num} ({feature_name}): Coefficient = {coef:.4f}\n")


    if model is not None and RF:
        # Obtenir l'importance des caractéristiques
        feature_importances = model.feature_importances_

        # Créer un DataFrame pour associer les caractéristiques à leur importance
        feature_importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': feature_importances
        })

        # Trier les caractéristiques par importance décroissante
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Afficher les caractéristiques les plus importantes
        print(feature_importance_df)

        # Imprimer les 10 valeurs les plus importantes
        log_file.write("Caracteristiques les plus importantes:\n")
        log_file.write(feature_importance_df.to_string(index=False) + "\n")

        # Visualiser les caractéristiques les plus importantes
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature (Event)')
        plt.title('Importance des Caractéristiques pour la Détection d’Anomalies')
        plt.gca().invert_yaxis()
        #plt.show()

    log_file.write("------------------------------------\n\n")


print("************************************")
print("Start of script")


node_name_formated = node_name.replace(":", "_").replace("-", "_").replace(" ", "_").replace("'", "")

slide_window_suffix = f"{alarm_name}_{node_name_formated}_{window_size}"

# E255_15_alarm_occurences_matrix_preprocessed.csv

# Ex. KERNDTLB_alarm_occurences_matrix_R30-M0-N9-C-J16-U01_1800_600_350_chrono_dedup
# Charger la matrice d'occurrence
input_file_path = f"./BGL_Brain_results/{slide_window_suffix}_alarm_occurences_matrix_FixWindow_dedup.csv"


full_data = pd.read_csv(input_file_path)

# Split the full_data into two halves
mid_index = len(full_data) // 2
first_half = full_data.iloc[:mid_index]
second_half = full_data.iloc[mid_index:]

# Use the first half for training and the second half for testing
train_data = first_half
test_data = second_half

sampling_data_desc = f'{train_data_size}_{test_data_size}_{val_data_size}'

sub_folder = f"BGL_Train_Fix_Window_{n_bootstrap_samples}_{alarm_name}_{node_name_formated}"

# Create the sub_folder if it doesn't exist
if not os.path.exists(f"BGL_Brain_results/{sub_folder}"):
    os.makedirs(f"BGL_Brain_results/{sub_folder}")

with open(f"./BGL_Brain_results/{sub_folder}/{slide_window_suffix}_Training_Set_RF_{sampling_data_desc}_FixWindow_Output.log", "w") as RF_log_file, open(f"./BGL_Brain_results/{sub_folder}/{slide_window_suffix}_Training_Set_LR_{sampling_data_desc}_FixWindow_Output.log", "w") as LR_log_file:
    for bs_index in range(n_bootstrap_samples):

        RF_log_file.write(f"Train data size: {train_data_size}, Validation size: {val_data_size}, Test size: {test_data_size}\n\n")
        LR_log_file.write(f"Train data size: {train_data_size}, Validation size: {val_data_size}, Test size: {test_data_size}\n\n")

        RF_log_file.write(input_file_path)
        RF_log_file.write("\n\n")
        LR_log_file.write(input_file_path)
        LR_log_file.write("\n\n")

        # data_bootstrap = resample(full_data, replace=True, n_samples=len(full_data))
        # Generate a bootstrap sample from the original dataset and allow different random_state (random_state used with same value gives same result!!)
        
        #if use_bootstrap == 1:
        #    data_bootstrap = resample(full_data, replace=False, n_samples=len(full_data))
        #    data = data_bootstrap
        #else:
        data = full_data

        #X_data = data.drop(columns=['IsAlarm'])
        #y_data = data['IsAlarm']
        # Simply split the bootstrap without randomness using 80-20 split NO RANDOMNESS to keep chronological order
        #X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state = 0)

        # Split the data into chunks based on bs_index
        chunk_size = len(data) // n_bootstrap_samples
        start_index = bs_index * chunk_size
        end_index = start_index + chunk_size if bs_index < n_bootstrap_samples - 1 else len(data)
        data_chunk = data.iloc[start_index:end_index]
        print(f"Bootstrap sample {bs_index} - Start index: {start_index}, End index: {end_index}, Chunk size: {data_chunk.shape[0]}")

        data = data_chunk

        # Split the data into 60% training, 20% validation, and 20% testing
        train_size = int((train_data_size / 100) * len(data))
        test_size = int((test_data_size / 100) * len(data))
        val_size = int((val_data_size / 100) * len(data))       # can be 0
        print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")
        
        train_df = data.iloc[:train_size]
        test_df = data.iloc[train_size:train_size + test_size]
        val_df = data.iloc[train_size + test_size:train_size + test_size + val_size]
        print(f"Train size: {train_df.shape[0]}, Validation size: {val_df.shape[0]}, Test size: {test_df.shape[0]}")

        # Separate features and labels
        X_train = train_df.drop(columns=['IsAlarm'])
        y_train = train_df['IsAlarm'].astype(int)   

        X_test = test_df.drop(columns=['IsAlarm'])
        y_test = test_df['IsAlarm'].astype(int)
        
        X_val = val_df.drop(columns=['IsAlarm'])
        y_val = val_df['IsAlarm'].astype(int)

        estimators = 10  # (bs_index + 1) * 2
        randomize_val= 55  # (bs_index + 5) * 11

        if train_RF == 1:
            RF_log_file.write(f"Data sample {bs_index} - Start index: {start_index}, End index: {end_index}, Chunk size: {data_chunk.shape[0]}\n\n")
            # Entraîner le modèle Random Forest Classifier
            model = RandomForestClassifier() # (n_estimators=estimators, random_state=randomize_val)         # class_weight='balanced', n_estimators=(bs_index+1)*2, warm_start=False, random_state=(bs_index+1)*10)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            get_model_evaluation(y_test, y_test_pred, f'Random Forest Classifier - TEST DATA - {bs_index}', RF_log_file, estimators, randomize_val, model, X_train, True)

            if val_size > 0:
                y_val_pred = model.predict(X_val)
                get_model_evaluation(y_val, y_val_pred, f'Random Forest Classifier - VALIDATION DATA - {bs_index}', RF_log_file, estimators, randomize_val, model, X_train, True)

            y_train_pred = model.predict(X_train)
            get_model_evaluation(y_train, y_train_pred, f'Random Forest Classifier - TRAIN DATA - {bs_index}', RF_log_file, estimators, randomize_val, model, X_train, True)


            # Assume `model` is your trained model
            #with open(f'random_forest_model_{suffix}_{node_name}_{bs_index}.pkl', 'wb') as file:
            #    pickle.dump(model, file)


        if train_LR == 1:
            LR_log_file.write(f"Data sample {bs_index} - Start index: {start_index}, End index: {end_index}, Chunk size: {data_chunk.shape[0]}\n\n")
            # Model de regression lineaire
            model = LogisticRegression() # (random_state=randomize_val)      # random_state=(bs_index+1)*10, solver='liblinear')
            
            unique_classes_tests = np.unique(y_train)
            if len(unique_classes_tests) == 2:
                model.fit(X_train, y_train)

                y_test_pred = model.predict(X_test)
                get_model_evaluation(y_test, y_test_pred, f'Linear Regression Classifier - TEST DATA', LR_log_file, -1, randomize_val, model, X_train, False)

                if val_size > 0:
                    y_val_pred = model.predict(X_val)
                    get_model_evaluation(y_val, y_val_pred, f'Linear Regression Classifier - VALIDATION DATA', LR_log_file, -1, randomize_val, model, X_train, False)

                y_train_pred = model.predict(X_train)
                get_model_evaluation(y_train, y_train_pred, f'Linear Regression Classifier - TRAIN DATA', LR_log_file, -1, randomize_val, model, X_train, False)
            else:
                LR_log_file.write(f'WARNING - Only 1 class present in the TRAINING set: {unique_classes_tests}\n\n')

          
            # Assume `model` is your trained model
            #with open(f'linear_regression_model_{suffix}_{node_name}_{bs_index}.pkl', 'wb') as file:
            #    pickle.dump(model, file)



print(f"RF log file: {RF_log_file.name}")
print(f"LR log file: {LR_log_file.name}")

print("End of script")
print("************************************")
