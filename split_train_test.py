import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse

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
    combined_labels = np.concatenate(combined_labels, axis=0)
    combined_inputs = np.concatenate(combined_inputs, axis=0)
    return combined_inputs, combined_labels

def split_and_save_data(inputs, labels, train_file, test_file, test_size=0.1):
    """Splits the data into training and testing sets and saves them into H5 files."""
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=test_size, random_state=42)
    with h5py.File(train_file, 'w') as f:
        f.create_dataset('INPUTS', data=X_train)
        f.create_dataset('LABELS', data=y_train)
    
    with h5py.File(test_file, 'w') as f:
        f.create_dataset('INPUTS', data=X_test)
        f.create_dataset('LABELS', data=y_test)

def parse_args():
    parser = argparse.ArgumentParser(description="Process and split H5 data files")
    parser.add_argument("variable_type", choices=["combined", "reco", "tops"], help="Type of data to process (combined, reco, or tops)")
    parser.add_argument("lepton_channel", choices=["0L","1L"],help="Lepton channel to process (0L, 1L)")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Directory where the H5 files are located
    read_directory = "C:/Users/matis/OneDrive/Documents/Y4/Project/MPhys-4Top/H5_Files/"
    write_directory = "C:/Users/matis/OneDrive/Documents/Y4/Project/MPhys-4Top/Training_Testing_Split/"
    
    # List of channels to process
    channels = ['3tj', '3tW', '4t']
    file_list = []
    for channel in channels:
        # Dynamically create the file list based on the input data type
        file = os.path.join(read_directory, f'tm_{args.lepton_channel}_{channel}_{args.variable_type}.h5')
        file_list.append(file)

    # Full paths for the output training and testing files
    train_file = os.path.join(write_directory, f'tm_{args.lepton_channel}_{args.variable_type}_training.h5')
    test_file = os.path.join(write_directory, f'tm_{args.lepton_channel}_{args.variable_type}_testing.h5')
        
    # Read and combine data from H5 files
    inputs, labels = read_h5_files(file_list)
    # Split data and save to files
    split_and_save_data(inputs, labels, train_file, test_file)

if __name__ == "__main__":
    main()
