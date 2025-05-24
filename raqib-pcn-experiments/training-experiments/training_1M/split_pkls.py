import pickle
import os
import numpy as np
from tqdm import tqdm
import argparse

def split_pkl_file(input_file, output_dir, num_chunks=10):
    """
    Split a large pickle file into multiple smaller files.
    
    Args:
        input_file: Path to the input pickle file
        output_dir: Directory to save the split files
        num_chunks: Number of chunks to split the file into
    """
    print(f"Loading {input_file}...")
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    # Get the base filename without the extension
    base_name = os.path.basename(input_file).replace('.pkl', '')
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split the data into chunks
    chunk_size = len(data) // num_chunks
    remainder = len(data) % num_chunks
    
    start_idx = 0
    for i in range(num_chunks):
        # Add an extra item to some chunks if data doesn't divide evenly
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        chunk_data = data[start_idx:end_idx]
        output_file = os.path.join(output_dir, f"{base_name}_{i}.pkl")
        
        print(f"Saving chunk {i+1}/{num_chunks} with {len(chunk_data)} items to {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(chunk_data, f)
        
        start_idx = end_idx
    
    print(f"Successfully split {input_file} into {num_chunks} chunks in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Split large pickle files into smaller chunks')
    parser.add_argument('--input_dir', type=str, default='data/', help='Directory containing the input pickle files')
    parser.add_argument('--output_dir', type=str, default='data-splitted/', help='Directory to save the split files')
    parser.add_argument('--num_chunks', type=int, default=10, help='Number of chunks to split each file into')
    
    args = parser.parse_args()
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all pickle files in the input directory
    pkl_files = [f for f in os.listdir(args.input_dir) if f.endswith('.pkl')]
    
    for pkl_file in tqdm(pkl_files, desc="Processing files"):
        input_path = os.path.join(args.input_dir, pkl_file)
        try:
            split_pkl_file(input_path, args.output_dir, args.num_chunks)
        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")

if __name__ == "__main__":
    main()