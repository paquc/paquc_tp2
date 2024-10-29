import sys
import pandas as pd
import numpy as np
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
else:
    print("Usage: Train.py <data_set_id>")
    sys.exit(1)

# Charger la matrice d'occurrence
data = pd.read_csv(f"./Thunderbird_Brain_results/VAPI_alarm_occurences_matrix_preprocessed.csv")
#data = pd.read_csv('./BGL_Brain_results/KERNDTLB_alarm_occurences_matrix_V1_part1_dedup.csv')  # for testing only!

n_bootstrap_samples = 1
n_data_set_samples = 5000   # len(data)


print("************************************")
print("Start of script")

with open(f"./Thunderbird_Brain_results/Thu_VAPI_Training_Set_RF_60_20_20_Preprocessed_Output.log", "w") as RF_log_file, open(f"./Thunderbird_Brain_results/Thu_VAPI_Training_Set_LR_60_20_20_Preprocessed_Output.log", "w") as LR_log_file:
    for bs_index in range(n_bootstrap_samples):

        # Split the data into 60% training, 20% validation, and 20% testing
        train_size = int(0.60 * len(data))
        val_size = int(0.20 * len(data))
        test_size = len(data) - train_size - val_size

        train_df = data.iloc[:train_size]
        val_df = data.iloc[train_size:train_size + val_size]
        test_df = data.iloc[train_size + val_size:]

        # Separate features and labels
        X_train = train_df.drop(columns=['IsAlarm'])
        y_train = train_df['IsAlarm'].astype(int)   

        X_val = val_df.drop(columns=['IsAlarm'])
        y_val = val_df['IsAlarm'].astype(int)

        X_test = test_df.drop(columns=['IsAlarm'])
        y_test = test_df['IsAlarm'].astype(int)

        # Generate a bootstrap sample from the original dataset and allow different random_state (random_state used with same value gives same result!!)
        #data_bootstrap = resample(data, replace=True, n_samples=len(data))
        #X_data = data.drop(columns=['IsAlarm'])
        #y_data = data['IsAlarm']
        # Simply split the bootstrap without randomness using 80-20 split NO RANDOMNESS to keep chronological order
        #X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state = 0)
       

        if train_RF == 1:
            print("------------------------------------")
            msg=f"--- Random Forest Classifier Bootstrap sample {bs_index+1}/{n_bootstrap_samples} ---"
            print(msg)
            RF_log_file.write(msg+"\n")

            # Entraîner le modèle Random Forest Classifier
            model = RandomForestClassifier(class_weight='balanced', n_estimators=(bs_index+1)*2, warm_start=False, random_state=(bs_index+1)*10)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            msg=f"Accuracy: {accuracy:.4f}"
            print(msg)
            RF_log_file.write(msg+"\n")

            # Calculate recall
            recall = recall_score(y_test, y_pred)
            msg=f"Recall: {recall:.2f}"
            print(msg)
            RF_log_file.write(msg+"\n")
            
            # Handle invalid data set to compute AUC
            if recall == 0.0:
                print("Recall is 0.0, skipping machine learning!!!! Need 2 classes (0, 1)")
                break

            # AUC score
            # Get the predicted probabilities for the positive class (class 1)
            y_prob = model.predict_proba(X_test)[:, 1]
            # Compute AUC score
            AUC = roc_auc_score(y_test, y_prob)
            msg=f"AUC: {AUC:.2f}"
            print(msg)
            RF_log_file.write(msg+"\n")

            # Mean Absolute Error (MAE)
            mae = mean_absolute_error(y_test, y_pred)
            msg=f"Mean Absolute Error: {mae:.2f}"
            print(msg)
            RF_log_file.write(msg+"\n")
            
            # R-squared (R²)
            r2 = r2_score(y_test, y_pred)
            msg=f"R-squared: {r2:.2f}"
            print(msg)
            RF_log_file.write(msg+"\n")

            # Alternatively, print the classification report
            msg=classification_report(y_test, y_pred)
            print(msg)
            RF_log_file.write(msg+"\n")

            print("------------------------------------")


        if train_LR == 1:
            msg=f"--- Linear Regression Bootstrap sample {bs_index+1}/{n_bootstrap_samples} ---"
            print(msg)
            LR_log_file.write(msg+"\n")

            # -------------------------------------------------------------------------
            # Model de regression lineaire
            model = LogisticRegression(random_state=42, solver='liblinear')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            msg=f"Accuracy: {accuracy:.4f}"
            print(msg)
            LR_log_file.write(msg+"\n")

            # Calculate recall
            recall = recall_score(y_test, y_pred)
            msg=f"Recall: {recall:.2f}"
            print(msg)
            LR_log_file.write(msg+"\n")

            # Handle invalid data set to compute AUC
            if recall != 0.0:
                # AUC score
                # Get the predicted probabilities for the positive class (class 1)
                y_prob = model.predict_proba(X_test)[:, 1]
                # Compute AUC score
                AUC = roc_auc_score(y_test, y_prob)
                msg=f"AUC: {AUC:.2f}"
                print(msg)
                LR_log_file.write(msg+"\n")

            # Mean Absolute Error (MAE)
            mae = mean_absolute_error(y_test, y_pred)
            msg=f"Mean Absolute Error: {mae:.2f}"
            print(msg)
            LR_log_file.write(msg+"\n")
            
            # R-squared (R²)
            r2 = r2_score(y_test, y_pred)
            msg=f"R-squared: {r2:.2f}"
            print(msg)
            LR_log_file.write(msg+"\n")

            msg=classification_report(y_test, y_pred)
            print(msg)
            LR_log_file.write(msg+"\n")

            print("------------------------------------")


print("End of script")
print("************************************")