import os
import pickle
import csv
import pandas as pd
import gc
from pathlib import Path

def count_graphs_in_pkl(file_path):
    """
    Count the number of items in a pickle file using len().
    Explicitly manages memory by deleting data and calling garbage collection.
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            length = len(data)
            del data  # Explicitly delete the data
            gc.collect()  # Force garbage collection
            return length
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

def process_folders_and_save_csv(base_path, output_csv='pkl_file_lengths.csv'):
    """
    Process all folders and their pkl files, then save results to CSV.
    """
    results = []
    
    # Define the folders to process
    folders = ['Delta', 'kT', 'mSquare', 'Z']
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist, skipping...")
            continue
            
        print(f"Processing folder: {folder}")
        
        # Get all .pkl files in the folder
        pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
        
        for pkl_file in pkl_files:
            file_path = os.path.join(folder_path, pkl_file)
            graph_count = count_graphs_in_pkl(file_path)
            
            results.append({
                'filename': pkl_file,
                'length': graph_count
            })
            
            print(f"  {pkl_file}: {graph_count} graphs")
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    print(f"Total files processed: {len(results)}")
    
    return df

# Alternative method without pandas if you prefer basic CSV writing
def process_folders_and_save_csv_basic(base_path, output_csv='pkl_file_lengths.csv'):
    """
    Same as above but using basic CSV writer instead of pandas.
    """
    results = []
    folders = ['Delta', 'kT', 'mSquare', 'Z']
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist, skipping...")
            continue
            
        print(f"Processing folder: {folder}")
        
        pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
        
        for pkl_file in pkl_files:
            file_path = os.path.join(folder_path, pkl_file)
            graph_count = count_graphs_in_pkl(file_path)
            
            results.append([pkl_file, graph_count])
            print(f"  {pkl_file}: {graph_count} graphs")
    
    # Write to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'length'])  # Header
        writer.writerows(results)
    
    print(f"\nResults saved to {output_csv}")
    print(f"Total files processed: {len(results)}")

# Usage example:
if __name__ == "__main__":
    # Set your base path where the folders (Delta, KT, mSquare, Z) are located
    base_path = "../data/Multi Level Jet Tagging/"  # Current directory, change this to your actual path
    
    # Option 1: Using pandas (recommended)
    df = process_folders_and_save_csv(base_path, 'pkl_graph_counts.csv')
    
    # Option 2: Using basic CSV writer (uncomment if you prefer this)
    # process_folders_and_save_csv_basic(base_path, 'pkl_graph_counts.csv')
    
    # Display first few rows
    print("\nFirst 10 rows of results:")
    print(df.head(10))