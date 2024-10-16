import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Function to load CSV files and add labels
def load_data(base_dir):
    data = []
    labels = []
    
    # Iterate over each application folder (FB, Gmail, Instagram, etc.)
    for app_folder in os.listdir(base_dir):
        app_path = os.path.join(base_dir, app_folder)
        
        if os.path.isdir(app_path):
            # Iterate over each activity folder (e.g., Add story-image)
            for activity_folder in os.listdir(app_path):
                activity_path = os.path.join(app_path, activity_folder)
                
                if os.path.isdir(activity_path):
                    # Create a label using the app name and activity name
                    label = f"{app_folder} + {activity_folder}"
                    
                    # Iterate over all CSV files in the folder
                    for file_name in os.listdir(activity_path):
                        if file_name.endswith('.csv'):
                            file_path = os.path.join(activity_path, file_name)
                            # Load the CSV file
                            df = pd.read_csv(file_path)
                            
                            # Ensure the required columns are present
                            if 'frame.time_relative' in df.columns and 'frame.len' in df.columns:
                                # Add a new column for the label, replicated for each row in the DataFrame
                                df['label'] = label
                                # Append the data
                                data.append(df[['frame.time_relative', 'frame.len', 'label']])  # Only append relevant columns
    
    return data

# Preprocess the data
def preprocess_data(data_list):
    # Concatenate all data into a single DataFrame
    combined_data = pd.concat(data_list, ignore_index=True)
    
    # Select features and labels
    X = combined_data[['frame.time_relative', 'frame.len']].values
    y = combined_data['label'].values  # Ensure y has the same length as X
    return X, y

# Load data
base_dir = "D:/Dataset/D-OutCsv"
data = load_data(base_dir)

# Preprocess data
X, y = preprocess_data(data)

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
