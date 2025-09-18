import pandas as pd
import numpy as np
import os
from pathlib import Path
import re

def parse_file(file_path):
    """
    Parse a txt file and return a structured DataFrame.
    
    Args:
        file_path (str): Path to the text file containing coordinates
    
    Returns:
        pd.DataFrame: Pandas dataframe with columns for each joint coordinate
    """
    
    # Define joint names based on Kinect structure (20 joints)
    joint_names = [
        'head', 'shoulder_center', 'shoulder_left', 'shoulder_right',
        'elbow_left', 'elbow_right', 'wrist_left', 'wrist_right',
        'hand_left', 'hand_right', 'spine', 'hip_center',
        'hip_left', 'hip_right', 'knee_left', 'knee_right',
        'ankle_left', 'ankle_right', 'foot_left', 'foot_right'
    ]
    
    # Create column names for x, y, z coordinates of each joint
    columns = []
    for joint in joint_names:
        columns.extend([f'{joint}_x', f'{joint}_y', f'{joint}_z'])
    
    data_rows = []
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Split the line by whitespace and convert to float
            coordinates = line.split()
            
            # Check if we have exactly 60 coordinates (20 joints × 3)
            if len(coordinates) != 60:
                print(f"Warning: Line {line_num + 1} has {len(coordinates)} coordinates instead of 60")
                continue
            
            try:
                coord_values = [float(coord) for coord in coordinates]
                data_rows.append(coord_values)
            except ValueError as e:
                print(f"Error parsing line {line_num + 1}: {e}")
                continue
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=columns)
    
    return df

def parse_all_files(directory_path):
    """
    Parse all files in a directory and combine into a single DataFrame.
    
    Args:
        directory_path (str): Path to directory containing files
        
    Returns:
        pd.DataFrame: Combined data from all files with sequence identifiers
    """
    
    file_pattern = "*.txt"
    directory = Path(directory_path)
    if not directory.exists():
        print(f"Directory not found: {directory_path}")
        return pd.DataFrame()
    
    # Find all matching files
    files = list(directory.glob(file_pattern))
    if not files:
        print(f"No files found matching pattern {file_pattern} in {directory_path}")
        return pd.DataFrame()
    
    all_dataframes = []
    
    for i, file_path in enumerate(files):
        print(f"Processing file {i+1}/{len(files)}: {file_path.name}")
        
        # Parse the file
        df = parse_file(file_path)
        
        if not df.empty:
            # Add sequence identifier to distinguish between different gesture sequences
            df['sequence_id'] = i
            df['gesture_file'] = file_path.stem
            all_dataframes.append(df)
        else:
            print(f"Failed to parse {file_path.name}")
    
    if not all_dataframes:
        print("No data was successfully parsed from any files")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    return combined_df

def extract_gesture_metadata(filename):
    """
    Extract metadata from filename for a schema containing 3 items
    
    Args:
        filename (str): Name of the file
    
    Returns:
        dict: Metadata extracted from filename
    """
    
    metadata = {
        'gesture_type': None,
        'performer': None,
        'session': None,
    }
    
    # Split filename to input metadata
    filename_file = filename.lower().split('_')
    gesture_type_file = filename_file[0]
    performer_file = filename_file[1]
    session_file = filename_file[2]

    # Assign correct data to metadata dictionary
    metadata['gesture_type'] = gesture_type_file
    metadata['performer'] = performer_file
    metadata['session'] = session_file
    
    return metadata

def create_gesture_database(directory_path, outputFilename):
    """
    Create a full database from files
    
    Args:
        directory_path (str): Path to directory containing files
    
    Returns:
        pd.DataFrame: Complete gesture database
    """
    
    print("Parsing coordinate files...")
    df = parse_all_files(directory_path)
    
    if df.empty:
        return df
    
    print(f"Successfully parsed {len(df)} frames from {df['sequence_id'].nunique()} sequences")
    
    # Add metadata for each sequence
    metadata_list = []
    for _, group in df.groupby('sequence_id'):
        filename = group['gesture_file'].iloc[0]
        metadata = extract_gesture_metadata(filename)
        metadata['sequence_id'] = group['sequence_id'].iloc[0]
        metadata_list.append(metadata)
    
    metadata_df = pd.DataFrame(metadata_list)
    
    # Merge metadata back with main dataframe
    df = df.merge(metadata_df[['sequence_id', 'performer', 'gesture_type', 'session']], 
                  on='sequence_id', how='left')
    
    # Calculate basic statistics
    print("\nDataset Statistics:")
    print(f"Total sequences: {df['sequence_id'].nunique()}")
    print(f"Total frames: {len(df)}")
    print(f"Average frames per sequence: {len(df) / df['sequence_id'].nunique():.1f}")
    print(f"Frame length range: {df.groupby('sequence_id').size().min()} - {df.groupby('sequence_id').size().max()}")
    
    # Show coordinate ranges for quality check
    coord_columns = [col for col in df.columns if col.endswith(('_x', '_y', '_z'))]
    print(f"\nCoordinate ranges:")
    print(f"X coordinates: {df[[col for col in coord_columns if col.endswith('_x')]].min().min():.3f} to {df[[col for col in coord_columns if col.endswith('_x')]].max().max():.3f}")
    print(f"Y coordinates: {df[[col for col in coord_columns if col.endswith('_y')]].min().min():.3f} to {df[[col for col in coord_columns if col.endswith('_y')]].max().max():.3f}")
    print(f"Z coordinates: {df[[col for col in coord_columns if col.endswith('_z')]].min().min():.3f} to {df[[col for col in coord_columns if col.endswith('_z')]].max().max():.3f}")
    
    #Remove unnecessary Metadata
    df = df.drop('gesture_file', axis=1)

    # Save to csv file
    df.to_csv(outputFilename, index=False)
    print(f"\nDatabase saved to: {outputFilename}")
    
    return df



if __name__ == "__main__":
    # Set to the directory containing your data files
    data_directory = "combined"
    
    # Create the gesture database
    gesture_db = create_gesture_database(data_directory, "gesture_database.csv")
    
    if not gesture_db.empty:
        # Display basic info
        print("\nDataFrame Info:")
        print(gesture_db.info())
        
        # Show first few rows
        print("\nFirst 5 rows:")
        print(gesture_db.head())
    else:
        print("No data was successfully parsed.")