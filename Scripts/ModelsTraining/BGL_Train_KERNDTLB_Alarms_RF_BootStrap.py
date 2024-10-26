import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, accuracy_score, mean_absolute_error, r2_score
from collections import Counter
from sklearn.datasets import make_regression
from sklearn.utils import resample

# Charger la matrice d'occurrence
data1 = pd.read_csv('./BGL_Brain_results/KERNDTLB_alarm_occurences_matrix_V1_part1_dedup.csv')
data2 = pd.read_csv('./BGL_Brain_results/KERNDTLB_alarm_occurences_matrix_V1_part2_dedup.csv')
data3 = pd.read_csv('./BGL_Brain_results/KERNDTLB_alarm_occurences_matrix_V1_part3_dedup.csv')
data4 = pd.read_csv('./BGL_Brain_results/KERNDTLB_alarm_occurences_matrix_V1_part4_dedup.csv')
data5 = pd.read_csv('./BGL_Brain_results/KERNDTLB_alarm_occurences_matrix_V1_part5_dedup.csv')

n_bootstrap_samples = 10

print("************************************")
print("Start of script")

with open("./BGL_Brain_results/BGL_KERNDTLB_Trainig_Output.log", "w") as log_file:
    for bs_index in range(n_bootstrap_samples):
        print("------------------------------------")
        msg=f"--- Bootstrap sample {bs_index+1}/{n_bootstrap_samples} ---"
        print(msg)
        log_file.write(msg+"\n")

        # Generate a bootstrap sample from the original dataset and allow different random_state (random_state used with same value gives same result!!)
        data_bootstrap = resample(data1, replace=True, n_samples=len(data1))
        X_bootstrap = data_bootstrap.drop(columns=['IsAlarm'])
        y_bootstrap = data_bootstrap['IsAlarm']
        # Simply split the bootstrap without randomness using 10% for testing
        X_train, X_test, y_train, y_test = train_test_split(X_bootstrap, y_bootstrap, test_size=0.1, random_state=0)

        # Entraîner le modèle
        model = RandomForestClassifier(n_estimators=(bs_index+1)*2, warm_start=False, random_state=(bs_index+1)*10)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        msg=f"Accuracy: {accuracy:.4f}"
        print(msg)
        log_file.write(msg+"\n")

        # Calculate recall
        recall = recall_score(y_test, y_pred)
        #print(f"Recall: {recall:.2f}")
        msg=f"Recall: {recall:.2f}"
        print(msg)
        log_file.write(msg+"\n")

        # Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_test, y_pred)
        #print(f"Mean Absolute Error: {mae:.2f}")
        msg=f"Mean Absolute Error: {mae:.2f}"
        print(msg)
        log_file.write(msg+"\n")
        
        # R-squared (R²)
        r2 = r2_score(y_test, y_pred)
        #print(f"R-squared: {r2:.2f}")
        msg=f"R-squared: {r2:.2f}"
        print(msg)
        log_file.write(msg+"\n")

        # Alternatively, print the classification report
        #print(classification_report(y_test, y_pred))
        msg=classification_report(y_test, y_pred)
        print(msg)
        log_file.write(msg+"\n")
        print("------------------------------------")


print("End of script")
print("************************************")