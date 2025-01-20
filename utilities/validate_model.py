import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from scipy.spatial import KDTree

# Results directory is one level up from this script. We need to use os.path.join to correctly resolve the path.
PACKAGE_DIR = os.path.join(os.path.dirname(__file__), '..', 'package')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

def process_original_data():
    """
    Load and process the original Iris dataset in the same way as the test dataset.

    Returns:
    pd.DataFrame: The processed original data with labels.
    np.ndarray: The scaler used for the transformations.
    """
    # Load the Iris dataset
    data = load_iris()
    X, y = data['data'], data['target']

    # Convert the target labels to categorical values and one-hot encode them
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y, 3)  # Iris dataset has 3 classes

    # Scale the features in the dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Combine the scaled features and labels into a DataFrame for easier comparison
    df_original = pd.DataFrame(X_scaled, columns=data['feature_names'])
    df_original['Species'] = le.inverse_transform(np.argmax(y, axis=1))

    return df_original, scaler

def validate_model():
    """
    Validate the hardware-predicted outputs against the HLS model outputs.

    This function loads the test data and predictions from both the HLS model (CPU) and the hardware (pynq-z2).
    It calculates and prints the accuracy for each set of predictions and generates a DataFrame for visual comparison.

    Returns:
    None
    """
    # Load the processed original Iris dataset
    print("Loading the original Iris dataset...")
    df_original, scaler = process_original_data()

    # Load test data and predictions
    print("Loading test data and predictions...")
    X_test = np.load(os.path.join(PACKAGE_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(PACKAGE_DIR, 'y_test.npy'))
    y_hls = np.load(os.path.join(PACKAGE_DIR, 'y_hls.npy'))
    y_hw = np.load(os.path.join(RESULTS_DIR, 'y_hw.npy'))
    classes = np.load(os.path.join(PACKAGE_DIR, 'classes.npy'), allow_pickle=True)

    # Calculate and print accuracy for each set of predictions
    accuracy_hls = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))
    accuracy_hw = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hw, axis=1))

    print(f"Accuracy hls4ml, CPU:     {accuracy_hls * 100:.2f}%")
    print(f"Accuracy hls4ml, pynq-z2: {accuracy_hw * 100:.2f}%")

    # Create KDTree for nearest neighbor search
    tree = KDTree(df_original.iloc[:, :-1].values)

    # Find the nearest neighbor in the original dataset for each test sample
    distances, indices = tree.query(X_test, k=1)
    df_test = pd.DataFrame(X_test, columns=df_original.columns[:-1])
    df_test['Expected Data species'] = df_original.iloc[indices]['Species'].values
    df_test['CPU model actual'] = np.argmax(y_hls, axis=1)
    df_test['FPGA model actual'] = np.argmax(y_hw, axis=1)

    # Print the comparison table
    print(df_test.to_string(index=False))

    # Check for mismatches and print the rows with mismatches
    mismatches = df_test[(df_test['Expected Data species'] != df_test['CPU model actual']) |
                         (df_test['Expected Data species'] != df_test['FPGA model actual'])]
    if not mismatches.empty:
        print("\nRows with mismatches:")
        print(mismatches.to_string(index=False))
    else:
        print("\nAll rows match.")

if __name__ == '__main__':
    validate_model()
