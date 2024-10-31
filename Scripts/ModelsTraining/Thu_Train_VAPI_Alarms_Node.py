import sys
import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, accuracy_score, mean_absolute_error, r2_score
from sklearn.metrics import roc_auc_score, roc_curve
from collections import Counter
from sklearn.datasets import make_regression
from sklearn.utils import resample


if len(sys.argv) > 2:
    train_LR = int(sys.argv[1])
    train_RF = int(sys.argv[2])
    suffix = sys.argv[3]
    node_name = sys.argv[4]
else:
    print("Usage: Train.py <data_set_id>")
    sys.exit(1)

# Charger la matrice d'occurrence
input_file_path = f"./Thunderbird_Brain_results/VAPI_alarm_occurences_matrix_{suffix}_{node_name}_chrono_dedup.csv"
full_data = pd.read_csv(input_file_path)
#data = pd.read_csv(f"./Thunderbird_Brain_results/VAPI_alarm_occurences_matrix_preprocessed_5min.csv")
#data = pd.read_csv('./BGL_Brain_results/KERNDTLB_alarm_occurences_matrix_V1_part1_dedup.csv')  # for testing only!

#n_data_set_samples = 5000   # len(data)

# Evaluation function
def get_model_evaluation(y_test, y_pred, model_name, log_file, estimators, randomize_val):
    log_file.write("------------------------------------\n")
    log_file.write(f"--- {model_name} ---\n\n")

    log_file.write(f"Size of test set: {y_test.shape[0]}\n")
    log_file.write(f"Size of predicted set: {y_pred.shape[0]}\n\n")
    log_file.write(f"Number of estimators: {estimators}\n")
    log_file.write(f"Randomize value: {randomize_val}\n\n")

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
    
    # Handle invalid data set to compute AUC
    if recall == 0.0:
        print("Recall is 0.0, skipping machine learning!!!! Need 2 classes (0, 1)")
        return

    # AUC score
    # Get the predicted probabilities for the positive class (class 1)
    #y_prob = model.predict_proba(X_test)[:, 1]
    # Compute AUC score
    AUC = roc_auc_score(y_test, y_pred)
    msg=f"AUC: {AUC:.2f}"
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

    log_file.write("------------------------------------\n\n")


print("************************************")
print("Start of script")

n_bootstrap_samples = 10
use_bootstrap = 0

with open(f"./Thunderbird_Brain_results/Thu_VAPI_Training_Set_RF_60_20_20_{suffix}_{node_name}_Output.log", "w") as RF_log_file, open(f"./Thunderbird_Brain_results/Thu_VAPI_Training_Set_LR_60_20_20_{suffix}_{node_name}_Output.log", "w") as LR_log_file:
    for bs_index in range(n_bootstrap_samples):

        # data_bootstrap = resample(full_data, replace=True, n_samples=len(full_data))
        # Generate a bootstrap sample from the original dataset and allow different random_state (random_state used with same value gives same result!!)
        
        if use_bootstrap == 1:
            data_bootstrap = resample(full_data, replace=False, n_samples=len(full_data))
            data = data_bootstrap
        else:
            data = full_data

        #X_data = data.drop(columns=['IsAlarm'])
        #y_data = data['IsAlarm']
        # Simply split the bootstrap without randomness using 80-20 split NO RANDOMNESS to keep chronological order
        #X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state = 0)
               
        # Split the data into 60% training, 20% validation, and 20% testing
        train_size = int(0.60 * len(data))
        val_size = int(0.20 * len(data)) - 1
        test_size = val_size
        print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

        train_df = data.iloc[:train_size]
        val_df = data.iloc[train_size + 1 : train_size + val_size + 1]
        test_df = data.iloc[ train_size + val_size + 1 :  train_size + val_size + val_size + 1]
        print(f"Train size: {train_df.shape[0]}, Validation size: {val_df.shape[0]}, Test size: {test_df.shape[0]}")

        # Separate features and labels
        X_train = train_df.drop(columns=['IsAlarm'])
        y_train = train_df['IsAlarm'].astype(int)   

        X_val = val_df.drop(columns=['IsAlarm'])
        y_val = val_df['IsAlarm'].astype(int)

        X_test = test_df.drop(columns=['IsAlarm'])
        y_test = test_df['IsAlarm'].astype(int)

        estimators = 10 #(bs_index+1)*2
        randomize_val= (bs_index+1)*10

        if train_RF == 1:
            # Entraîner le modèle Random Forest Classifier
            model = RandomForestClassifier(n_estimators=estimators, random_state=randomize_val)         # class_weight='balanced', n_estimators=(bs_index+1)*2, warm_start=False, random_state=(bs_index+1)*10)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Assume `model` is your trained model
            with open(f'random_forest_model_{suffix}_{node_name}_{bs_index}.pkl', 'wb') as file:
                pickle.dump(model, file)

            get_model_evaluation(y_test, y_pred, f'Random Forest Classifier - TEST DATA - {bs_index}', RF_log_file, estimators, randomize_val)
            get_model_evaluation(y_val, y_pred, f'Random Forest Classifier - VALIDATION DATA - {bs_index}', RF_log_file, estimators, randomize_val)

        if train_LR == 1:
            # Model de regression lineaire
            randomize_val= (bs_index+1)*10
            model = LogisticRegression(random_state=randomize_val)      # random_state=(bs_index+1)*10, solver='liblinear')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Assume `model` is your trained model
            with open(f'linear_regression_model_{suffix}_{node_name}_{bs_index}.pkl', 'wb') as file:
                pickle.dump(model, file)

            get_model_evaluation(y_test, y_pred, f'Linear Regression Classifier - TEST DATA', LR_log_file, -1, randomize_val)
            get_model_evaluation(y_val, y_pred, f'Linear Regression Classifier - VALIDATION DATA', LR_log_file, -1, randomize_val)


print("End of script")
print("************************************")
