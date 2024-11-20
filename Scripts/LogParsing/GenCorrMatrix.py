import pandas as pd
import numpy as np
import sys

if len(sys.argv) >=  1:
    input_file = str(sys.argv[1])
else:
    print("Usage: script time_wnd_hours")
    sys.exit(1)


# df_sequences = pd.read_csv(f"./BGL_Brain_results/KERNDTLB_ALL_1000_alarm_occurences_matrix_FixWindow_dedup.csv", header=0)

df_sequences = pd.read_csv(input_file, header=0)

corr_matrix = df_sequences.corr()

print(corr_matrix.head())

threshold = 0.70

corr_matrix = corr_matrix.abs()  # Take absolute values to check both positive and negative correlations
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)  # Extract upper triangle of the correlation matrix

print("Upper triangle of the correlation matrix:")
print(upper_triangle)

# Initialize a set to keep track of features to drop
features_to_drop = set()

# Iterate over the columns of the upper triangle
for column in upper_triangle.columns:
    for row in upper_triangle.index:
        if upper_triangle.loc[row, column] > threshold:
            features_to_drop.add(column)
            break

print("Features to drop based on correlation threshold:")
print(features_to_drop)

# Find features with correlation greater than the threshold
to_drop = [
    column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)
]

print("Features to drop:")
print(to_drop)

# Compare features_to_drop and to_drop
if features_to_drop == set(to_drop):
    print("Both methods identified the same features to drop.")
else:
    print("The methods identified different features to drop.")

df_copy = df_sequences
df_copy_reduced = df_copy.drop(columns=features_to_drop)
print(f"Features kept: {df_copy_reduced.columns.tolist()}")

df_reduced = df_sequences.drop(columns=to_drop)
print(f"Features kept: {df_reduced.columns.tolist()}")

output = input_file.replace(".csv", "_corr.csv")
df_reduced.to_csv(output, index=False)