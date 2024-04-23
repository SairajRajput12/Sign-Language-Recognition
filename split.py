import os
import shutil
from sklearn.model_selection import train_test_split

def split_data_into_train_test(source_folder, train_folder, test_folder, test_size=0.2, random_state=None):
    # Create train and test directories if they don't exist
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    
    # Get list of files in the source folder
    files = os.listdir(source_folder)
    
    # Split files into train and test sets
    train_files, test_files = train_test_split(files, test_size=test_size, random_state=random_state)
    
    # Copy files to train folder
    for file in train_files:
        shutil.copy(os.path.join(source_folder, file), os.path.join(train_folder, file))
    
    # Copy files to test folder
    for file in test_files:
        shutil.copy(os.path.join(source_folder, file), os.path.join(test_folder, file))

# Example usage:
source_folder = 'Images'
train_folder = 'train_data'
test_folder = 'test_data'
split_data_into_train_test(source_folder, train_folder, test_folder, test_size=0.2, random_state=42)
