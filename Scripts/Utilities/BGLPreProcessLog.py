import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np

# Load BGL logs into DataFrame
# logs_file = f"./BGL_Brain_results/BGL.log_structured_full_content_filtered.csv"  
logs_file = './BGL_Brain_results/BGL.log_structured_V4.csv'
bgl_logs = pd.read_csv(logs_file)

# Step 1: Log Parsing (assumed already parsed using a tool like Drain)
# Extracting and using EventTemplate for analysis

# Step 2: Normalize and Remove Identifiers
# bgl_logs['NodeLoc'] = bgl_logs['NodeLoc'].str.extract('(\d+)').fillna(0).astype(int)  # Extracting numeric part of NodeLoc

# Step 3: Encode Categorical Features
encoder = OneHotEncoder(sparse=False)
encoded_severity = encoder.fit_transform(bgl_logs[['Severity']])

# Adding encoded features to the DataFrame
severity_df = pd.DataFrame(encoded_severity, columns=encoder.get_feature_names_out(['Severity']))
bgl_logs = pd.concat([bgl_logs, severity_df], axis=1)

# Step 4: Sliding Window Approach
window_size = 10
occurrence_matrix = []

for i in range(0, len(bgl_logs) - window_size + 1):
    window = bgl_logs.iloc[i:i + window_size]
    occurrence_matrix.append(window['EventTemplate'].value_counts().reindex(bgl_logs['EventTemplate'].unique(), fill_value=0).values)

occurrence_matrix_df = pd.DataFrame(occurrence_matrix)

# Step 5: Labeling the Data
labels = bgl_logs['AlertFlagLabel'][:len(occurrence_matrix_df)]

# Step 6: Feature Engineering - Optional Time Features
bgl_logs['TimeDifference'] = bgl_logs['EpochTime'].diff().fillna(0)

# Step 7: Dimensionality Reduction (Optional)
pca = PCA(n_components=10)
reduced_features = pca.fit_transform(occurrence_matrix_df)

# Step 8: Splitting Data for Training
X_train, X_test, y_train, y_test = train_test_split(reduced_features, labels, test_size=0.2, random_state=42)

# Model Training Example
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
