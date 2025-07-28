import os
import re
import numpy as np
import pefile
import datetime

# Define the feature names (order must match what your model expects)
TRAINED_FEATURES = [
    'general_size',
    'general_exports',
    'general_imports',
    'strings_numstrings',
    'year',
    'month'
]

def extract_general_size(file_path):
    """Returns the file size in bytes."""
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        print(f"[ERROR] Unable to get file size: {e}")
        return 0

def extract_general_exports(file_path):
    """Extracts and returns the number of exported symbols from a PE file."""
    try:
        pe = pefile.PE(file_path)
        if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
            num_exports = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
        else:
            num_exports = 0
        pe.close()
        return num_exports
    except Exception as e:
        print(f"[ERROR] Unable to extract exports: {e}")
        return 0

def extract_general_imports(file_path):
    """Extracts and returns the total number of imported symbols from a PE file."""
    try:
        pe = pefile.PE(file_path)
        num_imports = 0
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                num_imports += len(entry.imports)
        pe.close()
        return num_imports
    except Exception as e:
        print(f"[ERROR] Unable to extract imports: {e}")
        return 0

def extract_strings_numstrings(file_path, min_length=4):
    """Extracts all printable strings from the file and returns the count of strings with at least min_length characters."""
    try:
        with open(file_path, 'rb') as f:
            data = f.read().decode('latin-1', errors='ignore')
        pattern = r'[\x20-\x7E]{' + str(min_length) + r',}'
        strings = re.findall(pattern, data)
        return len(strings)
    except Exception as e:
        print(f"[ERROR] Unable to extract strings: {e}")
        return 0

def extract_date_features(file_path):
    """
    Uses the file's last modification time to derive 'year' and 'month'.
    You can adjust this to use another date if available.
    """
    try:
        timestamp = os.path.getmtime(file_path)
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt.year, dt.month
    except Exception as e:
        print(f"[ERROR] Unable to extract date features: {e}")
        return 0, 0

def extract_features(file_path):
    """
    Extracts the features used during training from a PE file.
    
    The features (in order) are:
        [general_size, general_exports, general_imports, strings_numstrings, year, month]
    Returns:
        A NumPy array of shape (1, 6) containing the extracted features.
    """
    # Extract each feature:
    size = extract_general_size(file_path)
    num_exports = extract_general_exports(file_path)
    num_imports = extract_general_imports(file_path)
    num_strings = extract_strings_numstrings(file_path, min_length=4)
    year, month = extract_date_features(file_path)
    
    features = [size, num_exports, num_imports, num_strings, year, month]
    
    # Convert features into a NumPy array and reshape to (1, number_of_features)
    feature_vector = np.array(features, dtype=np.float32).reshape(1, -1)
    return feature_vector

if __name__ == "__main__":
    # Test the feature extraction with a sample PE file.
    # Replace 'sample_pe_file.exe' with an actual file path.
    test_file_path = "sample_pe_file.exe"
    feature_vector = extract_features(test_file_path)
    if feature_vector is not None:
        print("Extracted Feature Vector:")
        print(feature_vector)
    else:
        print("Feature extraction failed.")
