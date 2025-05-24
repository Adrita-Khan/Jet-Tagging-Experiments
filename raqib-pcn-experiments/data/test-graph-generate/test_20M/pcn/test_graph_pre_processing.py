import numpy as np
import pandas as pd
import awkward as ak
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import uproot
import torch
from tqdm import tqdm
import timeit
import os
import dill
import scipy.sparse as sp
import dgl
import pickle
import gc
from scipy.spatial import cKDTree

# Input features remain the same
input_features = ["part_px", "part_py", "part_pz", "part_energy",
                  "part_deta", "part_dphi", "part_d0val", "part_d0err", 
                  "part_dzval", "part_dzerr", "part_isChargedHadron", "part_isNeutralHadron", 
                  "part_isPhoton", "part_isElectron", "part_isMuon"]

def process_in_batches(filepath, jetType, batch_size=100_000, k=3):
    """Process root file in batches and save graphs incrementally"""
    print(f'Starting batch processing on {jetType} jets')
    
    # Create directory if it doesn't exist
    save_dir = f'../data/Multi Level Jet Tagging/batches/{jetType}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Open the file and get the tree
    with uproot.open(filepath) as file:
        tree = file['tree']
        num_entries = tree.num_entries
        
        # Process in batches
        for start in tqdm(range(0, num_entries, batch_size), desc=f"Processing {jetType} batches"):
            end = min(start + batch_size, num_entries)
            batch_num = start // batch_size
            
            # Skip if this batch already exists
            batch_save_path = f'{save_dir}/{jetType}_{batch_num}.pkl'
            if os.path.exists(batch_save_path):
                print(f"Batch {batch_num} already exists, skipping...")
                continue
            
            # Read a batch of entries
            awk_batch = tree.arrays(tree.keys(), entry_start=start, entry_stop=end)
            
            # Convert to point clouds
            point_clouds = awkToPointCloud(awk_batch, input_features)
            
            # Convert to graphs and save immediately
            batch_graphs = []
            for idx, point_cloud in enumerate(point_clouds):
                try:
                    if len(point_cloud) > 0:  # Check if point cloud is not empty
                        adj_matrix = buildKNNGraph(point_cloud, k)
                        graph = adjacencyToDGL(adj_matrix)
                        graph.ndata['feat'] = torch.tensor(point_cloud, dtype=torch.float32)
                        batch_graphs.append(graph)
                        
                        # Clean up memory
                        del adj_matrix
                except Exception as e:
                    print(f"Error processing jet {idx} in batch {batch_num}: {e}")
            
            # Save this batch of graphs
            with open(batch_save_path, 'wb') as f:
                pickle.dump(batch_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Clean up memory after saving
            del awk_batch, point_clouds, batch_graphs
            gc.collect()
            
    print(f'Batch processing for {jetType} complete!')
    
    # Combine all batches into a single file (optional)
    combine_batches(jetType, save_dir)

def combine_batches(jetType, batch_dir):
    """Combine all batch files into a single file (if needed)"""
    # Check if combined file already exists
    combined_save_path = f'../data/Multi Level Jet Tagging/{jetType}.pkl'
    if os.path.exists(combined_save_path):
        print(f"Combined file for {jetType} already exists at {combined_save_path}, skipping combination step...")
        return
    
    print(f"Combining batches for {jetType}...")
    all_graphs = []
    
    # Get all batch files
    batch_files = sorted([f for f in os.listdir(batch_dir) if f.startswith('batch_') and f.endswith('.pkl')])
    
    for batch_file in tqdm(batch_files, desc="Combining batches"):
        try:
            with open(os.path.join(batch_dir, batch_file), 'rb') as f:
                graphs = pickle.load(f)
                all_graphs.extend(graphs)
                del graphs
                gc.collect()
        except Exception as e:
            print(f"Error loading batch {batch_file}: {e}")
    
    # Save combined file
    with open(combined_save_path, 'wb') as f:
        pickle.dump(all_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Combined file saved to {combined_save_path}")

# Original functions that we keep unchanged
def fileToAwk(path):
    file = uproot.open(path)
    tree = file['tree']
    
    awk = tree.arrays(tree.keys())
    return awk

def awkToPointCloud(awkDict, input_features):
    featureVector = []
    for jet in tqdm(range(len(awkDict)), total=len(awkDict), leave=False):
        currJet = awkDict[jet][input_features]
        try:
            pT = np.sqrt(ak.to_numpy(currJet['part_px']) ** 2 + ak.to_numpy(currJet['part_py']) ** 2)
            # Create numpy array to represent the 4-momenta of all particles in a jet
            currJet = np.column_stack((
                ak.to_numpy(currJet['part_px']),
                ak.to_numpy(currJet['part_py']),
                ak.to_numpy(currJet['part_pz']),
                ak.to_numpy(currJet['part_energy']),
                pT,
                ak.to_numpy(currJet['part_deta']),
                ak.to_numpy(currJet['part_dphi']),
                ak.to_numpy(currJet["part_d0val"]),
                ak.to_numpy(currJet["part_d0err"]),
                ak.to_numpy(currJet["part_dzval"]),
                ak.to_numpy(currJet["part_dzerr"]),
                ak.to_numpy(currJet["part_isChargedHadron"]),
                ak.to_numpy(currJet["part_isNeutralHadron"]),
                ak.to_numpy(currJet["part_isPhoton"]),
                ak.to_numpy(currJet["part_isElectron"]),
                ak.to_numpy(currJet["part_isMuon"])
            ))
            featureVector.append(currJet)
        except Exception as e:
            print(f"Error processing jet {jet}: {e}")
            featureVector.append(np.empty((0, len(input_features) + 1)))  # Add an empty array for failed jets
    return featureVector

def buildKNNGraph(points, k):
    # Compute k-nearest neighbors
    tree = cKDTree(points)
    dists, indices = tree.query(points, k+1)  # +1 to exclude self
    
    # Build adjacency matrix
    num_points = len(points)
    adj_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in indices[i, 1:]:  # exclude self
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
    return adj_matrix

def adjacencyToDGL(adj_matrix):
    adj_matrix = sp.coo_matrix(adj_matrix)
    g_dgl = dgl.from_scipy(adj_matrix)
    return g_dgl

# Main execution function
def process_all_jet_types(batch_size=100_000):
    Higgs = ['HToBB', 'HToCC', 'HToGG', 'HToWW2Q1L', 'HToWW4Q']
    Vector = ['WToQQ', 'ZToQQ']
    Top = ['TTBar', 'TTBarLep']
    QCD = ['ZJetsToNuNu']
    allJets = Higgs + Vector + Top + QCD
    
    for jetType in allJets:
        # Check if final combined file already exists
        final_path = f'../data/Multi Level Jet Tagging/{jetType}.pkl'
        if os.path.exists(final_path):
            print(f"Final file for {jetType} already exists at {final_path}, skipping processing...")
            continue
            
        filepath = f'../data/JetClass/JetRoots/{jetType}/{jetType}_2M.root'  # Adjusted for 2M jets
        process_in_batches(filepath, jetType, batch_size=batch_size)
        
        # Force garbage collection between jet types
        gc.collect()

# Process emitter types separately to avoid memory issues
def process_emitter_types(batch_size=100_000):
    Emitter = ['Emitter-Vector', 'Emitter-Top', 'Emitter-Higgs', 'Emitter-QCD']
    
    for jetType in Emitter:
        # Check if final combined file already exists
        final_path = f'../data/Multi Level Jet Tagging/{jetType}.pkl'
        if os.path.exists(final_path):
            print(f"Final file for {jetType} already exists at {final_path}, skipping processing...")
            continue
            
        filepath = f'../data/JetClass/JetRoots/{jetType}/{jetType}_2M.root'
        process_in_batches(filepath, jetType, batch_size=batch_size)
        gc.collect()

if __name__ == "__main__":
    BATCH_SIZE = 100_000
    
    # Process normal jet types
    process_all_jet_types(batch_size=BATCH_SIZE)
    
    # Process emitter types
    # process_emitter_types(batch_size=BATCH_SIZE)