import numpy as np
import pandas as pd
from operator import truth
import awkward as ak
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import uproot
import torch
import math
from tqdm import tqdm
import timeit
import os
import dill
import scipy.sparse as sp
from scipy.spatial import cKDTree
import dgl
import pickle
import gc
import sys
import csv

INTERACTION_VECTOR = "Delta"


# For memory monitoring
try:
    import psutil
    def print_memory_usage(label=""):
        """Print current memory usage."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_usage = mem_info.rss / (1024 * 1024)  # Convert to MB
        print(f"Memory usage {label}: {mem_usage:.2f} MB")
except ImportError:
    def print_memory_usage(label=""):
        print(f"Memory usage {label}: [psutil not installed]")

def force_garbage_collection():
    """Force garbage collection to free memory."""
    gc.collect()

# take ROOT file and convert to an awkward array
def fileToAwk(path):
    file = uproot.open(path)
    tree = file['tree']
    
    awk = tree.arrays(tree.keys())
    return awk

input_features = ["part_px", "part_py", "part_pz", "part_energy",
                  "part_deta", "part_dphi", "part_d0val", "part_d0err", 
                  "part_dzval", "part_dzerr", "part_isChargedHadron", "part_isNeutralHadron", 
                  "part_isPhoton", "part_isElectron", "part_isMuon" ] # features used to train the model

def to_pt2(part_px, part_py, eps=1e-8):
    pt2 = part_px ** 2 + part_py ** 2
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2

# Take AWK dict and convert to a point cloud with memory optimization
def awkToPointCloud(awkDict, input_features, eps=1e-8):
    featureVector = []
    
    # Process in chunks to reduce peak memory usage
    chunk_size = 100_000
    n_jets = len(awkDict)
    
    for chunk_start in range(0, n_jets, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_jets)
        print(f"Processing jets {chunk_start}-{chunk_end} out of {n_jets}")
        
        chunk_features = []
        for jet in tqdm(range(chunk_start, chunk_end), total=chunk_end-chunk_start):
            currJet = awkDict[jet][input_features]
            try:
                # Convert to numpy once to avoid repeated conversions
                part_px = ak.to_numpy(currJet['part_px'])
                part_py = ak.to_numpy(currJet['part_py'])
                
                part_px = torch.from_numpy(part_px)
                part_py = torch.from_numpy(part_py)

                part_pz = ak.to_numpy(currJet['part_pz'])
                energy = ak.to_numpy(currJet['part_energy'])
                
                # Calculate pT
                pT = torch.sqrt(to_pt2(part_px, part_py, eps=eps))
                pT = pT.numpy()
                
                # Stack columns directly
                currJet_array = np.column_stack((
                    part_px, part_py, part_pz, energy, pT,
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
                
                # Optimize memory usage - convert to float32 if needed
                if currJet_array.dtype == np.float64:
                    currJet_array = currJet_array.astype(np.float32)
                
                chunk_features.append(currJet_array)
                
                # Free memory
                del part_px, part_py, part_pz, energy, pT, currJet_array
                
            except Exception as e:
                print(f"Error processing jet {jet}: {e}")
                chunk_features.append(np.empty((0, len(input_features) + 1)))
        
        # Extend the main list
        featureVector.extend(chunk_features)
        
        # Clean up to free memory
        del chunk_features
        force_garbage_collection()
    
    return featureVector


def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi

def rapidity(part_n):
    energy = part_n[3]
    pz = part_n[2]
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    return rapidity

def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2


def delta(part_i, part_j, eps=1e-8):
    part_i = torch.from_numpy(part_i)
    part_j = torch.from_numpy(part_j)

    rap_i = rapidity(part_i)
    rap_j = rapidity(part_j)

    phi_i = part_i[6]
    phi_j = part_j[6]

    delta = delta_r2(rap_i, phi_i, rap_j, phi_j).sqrt()
    
    lndelta = torch.log(delta.clamp(min=eps))
    lndelta = lndelta.numpy()

    return lndelta

#take point cloud and build KNN graph
def buildKNNGraph(points, k):
    
    # Compute k-nearest neighbors
    tree = cKDTree(points)
    dists, indices = tree.query(points, k+1)  # +1 to exclude self
    
    # Build adjacency matrix
    num_points = len(points)
    adj_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in indices[i, 1:]:  # exclude self
            weight_delta = delta(points[i], points[j])
            adj_matrix[i, j] = weight_delta
            adj_matrix[j, i] = weight_delta
    return adj_matrix

# take adjacency matrix and turn it into a DGL graph
def adjacencyToDGL(adj_matrix):
    adj_matrix = sp.coo_matrix(adj_matrix)
    g_dgl = dgl.from_scipy(adj_matrix)
        
    return g_dgl

# wrap the functionality of fileToAwk and awkToPointCloud in a function to return a point cloud numpy array
def fileToPointCloudArray(jetType, input_features):
    filepath = f'../data/JetClass/JetRoots/{jetType}_2M.root' # original root file
    savepath = f'../data/JetClass/PointClouds/{INTERACTION_VECTOR}_{jetType}.npy' # save file

    print_memory_usage("before loading file")
    awk = fileToAwk(filepath)
    print_memory_usage("after loading file")
    
    nparr = awkToPointCloud(awk, input_features)
    
    # Free memory
    del awk
    force_garbage_collection()
    print_memory_usage("after point cloud conversion")
    
    return nparr

# Check if directory exists, if not create it
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Find the last successfully processed batch for recovery
def find_last_batch(jetType):
    batch_files = []
    dir_path = f'{INTERACTION_VECTOR} Multi Level Jet Tagging'
    if os.path.exists(dir_path):
        batch_files = [f for f in os.listdir(dir_path) if f.startswith(f"{INTERACTION_VECTOR}_{jetType}_batch_") and f.endswith(".pkl")]
    
    if not batch_files:
        return 0
    
    # Extract batch numbers
    batch_numbers = []
    for file in batch_files:
        parts = file.replace('.pkl', '').split('_')
        if len(parts) >= 4:
            batch_numbers.append(int(parts[-2]))
    
    return max(batch_numbers) + 1 if batch_numbers else 0

# Improved fileToGraph with batch processing and recovery
def fileToGraph(jetType, k=3, save=True, batch_size=100_000, start_from=None):
    print(f'Starting processing on {jetType} jets')
    
    # Ensure directory exists
    ensure_dir(f'../data/{INTERACTION_VECTOR} Multi Level Jet Tagging')
    
    # Set up final save path
    saveFilePath = f'../data/{INTERACTION_VECTOR} Multi Level Jet Tagging/{INTERACTION_VECTOR}_{jetType}.pkl'
    
    # Check if final file already exists
    if os.path.exists(saveFilePath):
        print(f"File {saveFilePath} already exists. Skipping...")
        # Load and return the existing graphs if needed
        with open(saveFilePath, 'rb') as f:
            return pickle.load(f)
    
    # Determine where to start processing from
    if start_from is None:
        start_from = find_last_batch(jetType) * batch_size
        print(f"Resuming from jet {start_from} based on found batches")
    
    # Load point cloud data
    pointCloudArr = fileToPointCloudArray(jetType, input_features)
    
    total_jets = len(pointCloudArr)
    print(f"Total jets to process: {total_jets}")
    
    # Process in batches
    savedGraphs = []
    for batch_start in range(start_from, total_jets, batch_size):
        batch_end = min(batch_start + batch_size, total_jets)
        print(f"Processing batch {batch_start}-{batch_end} out of {total_jets}")
        
        batch_graphs = []
        for idx in tqdm(range(batch_start, batch_end), leave=False, total=batch_end-batch_start):
            try:
                pointCloud = pointCloudArr[idx]
                
                # Skip empty point clouds
                if pointCloud.shape[0] == 0:
                    print(f"Skipping empty point cloud at index {idx}")
                    continue
                
                adj_matrix = buildKNNGraph(pointCloud, k)
                graph = adjacencyToDGL(adj_matrix)
                
                graph.ndata['feat'] = torch.tensor(pointCloud, dtype=torch.float32)
                
                batch_graphs.append(graph)
                
                # Free memory
                del adj_matrix, graph
                
            except Exception as e:
                print(f"Error processing jet {idx}: {e}")
        
        # Save intermediate batch
        if save and batch_graphs:  # Only save if there are graphs in the batch
            temp_save_path = f'../data/{INTERACTION_VECTOR} Multi Level Jet Tagging/{INTERACTION_VECTOR}_{jetType}_batch_{batch_start}_{batch_end}.pkl'
            with open(temp_save_path, 'wb') as f:
                pickle.dump(batch_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved batch to {temp_save_path}")
        
        # Add to saved graphs
        savedGraphs.extend(batch_graphs)
        
        # Free memory
        del batch_graphs
        force_garbage_collection()
        print_memory_usage(f"after batch {batch_start}-{batch_end}")
    
    # Save complete dataset
    if save and savedGraphs:
        with open(saveFilePath, 'wb') as f:
            pickle.dump(savedGraphs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved complete dataset to {saveFilePath}")
    
    print(f'Graphs for {jetType} processing complete!')
    return savedGraphs

def groupToGraph(jetTypeList, groupName):
    allGraphs = []
    for jetType in jetTypeList:
        graphs = fileToGraph(jetType, save=False)
        allGraphs.extend(graphs)
        
        # Free memory after processing each jet type
        force_garbage_collection()
    
    saveFilePath = f'../data/{INTERACTION_VECTOR} Multi Level Jet Tagging/{INTERACTION_VECTOR}_{groupName}.pkl'
    
    # Save the combined graphs
    with open(saveFilePath, 'wb') as f:
        pickle.dump(allGraphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return allGraphs

def merge_batches(jetType):
    """Merge all batch files for a given jet type into one final file."""
    print(f"Merging batches for {jetType}")
    
    # Find all batch files
    dir_path = f'../data/{INTERACTION_VECTOR} Multi Level Jet Tagging'
    batch_files = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                         if f.startswith(f"{INTERACTION_VECTOR}_{jetType}_batch_") and f.endswith(".pkl")])
    
    if not batch_files:
        print(f"No batch files found for {jetType}")
        return
    
    # Merge all graphs
    all_graphs = []
    for batch_file in tqdm(batch_files):
        with open(batch_file, 'rb') as f:
            batch_graphs = pickle.load(f)
            all_graphs.extend(batch_graphs)
        
        # Free memory
        force_garbage_collection()
    
    # Save merged file
    final_path = f'../data/{INTERACTION_VECTOR} Multi Level Jet Tagging/{INTERACTION_VECTOR}_{jetType}-Testing.pkl'
    with open(final_path, 'wb') as f:
        pickle.dump(all_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Successfully merged {len(all_graphs)} graphs for {jetType}")
    
    # Ask if we should remove the batch files
    if input("Remove batch files? (y/n): ").lower() == 'y':
        for batch_file in batch_files:
            os.remove(batch_file)
            print(f"Removed {batch_file}")

# Define jet types
Higgs = ['HToBB', 'HToCC', 'HToGG', 'HToWW2Q1L', 'HToWW4Q']
Vector = ['WToQQ', 'ZToQQ']
Top = ['TTBar', 'TTBarLep']
QCD = ['ZJetsToNuNu']
Emitter = ['Emitter-Vector', 'Emitter-Top', 'Emitter-Higgs', 'Emitter-QCD']
allJets = Higgs + Vector + Top + QCD

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process jet data with recovery capabilities')
    parser.add_argument('--jet_type', type=str, help='Specific jet type to process')
    parser.add_argument('--batch_size', type=int, default=100_000, help='Batch size for processing')
    parser.add_argument('--merge_only', action='store_true', help='Only merge existing batches')
    parser.add_argument('--start_from', type=int, help='Start processing from this jet index')
    args = parser.parse_args()
    
    # Process specific jet type if provided
    if args.jet_type:
        if args.merge_only:
            merge_batches(args.jet_type)
        else:
            fileToGraph(args.jet_type, batch_size=args.batch_size, start_from=args.start_from)
    else:
        # Process all jet types one by one
        for jetType in allJets:
            try:
                if args.merge_only:
                    merge_batches(jetType)
                else:
                    fileToGraph(jetType, batch_size=args.batch_size, start_from=args.start_from)
            except KeyboardInterrupt:
                print(f"Processing interrupted for {jetType}. Moving to next jet type.")
                continue
            except Exception as e:
                print(f"Error processing {jetType}: {e}")
                continue