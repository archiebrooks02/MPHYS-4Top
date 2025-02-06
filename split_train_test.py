import h5py
import numpy as np
from sklearn.model_selection import train_test_split
directory = "C:/Users/matis/OneDrive/Documents/Y4/Project/MPhys-4Top/H5 Files/"
def read_h5_files(file_list):
    """Reads multiple H5 files and combines their data."""
    combined_inputs = []
    combined_labels = []
    
    for file_name in file_list:
        with h5py.File(file_name, 'r') as f:
            inputs = f['INPUTS'][:]
            labels = f['LABELS'][:]
            combined_inputs.append(inputs)
            combined_labels.append(labels)

    combined_inputs = np.concatenate(combined_inputs, axis=0)
    print(np.shape(combined_inputs))
    combined_labels = np.concatenate(combined_labels, axis=0)
    return combined_inputs, combined_labels

def split_and_save_data(inputs, labels, train_file, test_file, test_size=0.2):
    """Splits the data into training and testing sets and saves them into H5 files."""
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=test_size, random_state=42)
    print(np.shape(X_train))
    print(np.shape(X_test))
    print(np.shape(y_train))
    print(np.shape(y_test))
    with h5py.File(train_file, 'w') as f:
        f.create_dataset('INPUTS', data=X_train)
        f.create_dataset('LABELS', data=y_train)
    
    with h5py.File(test_file, 'w') as f:
        f.create_dataset('INPUTS', data=X_test)
        f.create_dataset('LABELS', data=y_test)

import os

def main():
    # Directory where the H5 files are located
    read_directory = "C:/Users/matis/OneDrive/Documents/Y4/Project/MPhys-4Top/H5_Files/"
    write_directory = "C:/Users/matis/OneDrive/Documents/Y4/Project/MPhys-4Top/Training_Testing_Split/" 

    # List of H5 files to read
    file_list = ['reco_tm_3tj_1L_02Dec.h5', 'reco_tm_3tW_1L_02Dec.h5', 'reco_tm_4t_1L_02Dec.h5']

    # Full paths for the output training and testing files
    train_file = os.path.join(write_directory, 'reco_tm_1L_training.h5')
    test_file = os.path.join(write_directory, 'reco_tm_1L_testing.h5')
    
    # Read and combine data from H5 files
    inputs, labels = read_h5_files([os.path.join(read_directory, f) for f in file_list])
    
    # Split data and save to files
    split_and_save_data(inputs, labels, train_file, test_file)

if __name__ == "__main__":
    main()
