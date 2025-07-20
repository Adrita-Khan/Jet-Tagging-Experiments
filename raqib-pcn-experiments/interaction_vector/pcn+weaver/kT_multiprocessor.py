import numpy as np

import pandas as pd

from operator import truth
import pandas as pd
import numpy as np
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
import csv

from multiprocessing import Pool, cpu_count
from functools import partial



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

# Take AWK dict and convert to a point cloud
def awkToPointCloud(awkDict, input_features, eps=1e-8):
    featureVector = []
    for jet in tqdm(range(len(awkDict)), total=len(awkDict)):
        currJet = awkDict[jet][input_features]
        try:
            part_px = ak.to_numpy(currJet['part_px'])
            part_py = ak.to_numpy(currJet['part_py'])

            part_px = torch.from_numpy(part_px)
            part_py = torch.from_numpy(part_py)

            pT = torch.sqrt(to_pt2(part_px, part_py, eps=eps))
            pT = pT.numpy()
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
    return featureVector  # Return a list of arrays instead of a single numpy array


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
    rap_i = rapidity(part_i)
    rap_j = rapidity(part_j)

    phi_i = part_i[6]
    phi_j = part_j[6]

    delta = delta_r2(rap_i, phi_i, rap_j, phi_j).sqrt()

    return delta

def kT(part_i, part_j, eps=1e-8):
    part_i = torch.from_numpy(part_i)
    part_j = torch.from_numpy(part_j)

    pti = part_i[4]
    ptj = part_j[4]

    ptmin = torch.minimum(pti, ptj)

    delta_ij = delta(part_i, part_j)

    lnkt = torch.log((ptmin * delta_ij).clamp(min=eps))
    lnkt = lnkt.numpy()

    return lnkt


# Build a KNN graph from a point cloud
def buildKNNGraph(points, k):
    tree = cKDTree(points)
    dists, indices = tree.query(points, k+1)  # +1 to exclude self

    num_points = len(points)
    adj_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in indices[i, 1:]:  # exclude self
            weight_kT = kT(points[i], points[j])
            adj_matrix[i, j] = weight_kT
            adj_matrix[j, i] = weight_kT
            
    return adj_matrix

# take adjacency matrix and turn it into a DGL graph
def adjacencyToDGL(adj_matrix):
    adj_matrix = sp.coo_matrix(adj_matrix)
    g_dgl = dgl.from_scipy(adj_matrix)
        
    return g_dgl


# wrap the functionality of fileToAwk and awkToPointCloud in a function to return a point cloud numpy array
def fileToPointCloudArray(jetType, input_features):
    # filepath = f'/Volumes/Yash SSD/JetClass/JetRoots/{jetType}_000.root' # original root file
    # savepath = f'/Volumes/Yash SSD/JetClass/PointClouds/{jetType}.npy' # save file
    filepath = f'../data/JetClass/JetRoots/{jetType}/{jetType}_100K.root' # original root file
    savepath = f'../data/JetClass/PointClouds/{jetType}.npy' # save file

    awk = fileToAwk(filepath)
    nparr = awkToPointCloud(awk, input_features)
    
    return nparr

# NEW: Function to process a single jet (this will run in parallel)
def process_single_jet(args):
    """
    Process a single jet to create a graph.
    This function will be called by each worker process.
    """
    idx, pointCloud, k, jetType, adj_csv_dir = args
    
    try:
        adj_matrix = buildKNNGraph(pointCloud, k)

        # Save adjacency matrix as CSV with particle indices
        adj_csv_path = f'{adj_csv_dir}/{jetType}_{idx}.csv'
        # Create DataFrame with row and column indices for particle identification
        df = pd.DataFrame(adj_matrix, 
                        index=[f'{i}' for i in range(adj_matrix.shape[0])],
                        columns=[f'{j}' for j in range(adj_matrix.shape[1])])
        df.to_csv(adj_csv_path)

        graph = adjacencyToDGL(adj_matrix)
        
        graph.ndata['feat'] = torch.tensor(pointCloud, dtype=torch.float32)
        
        return graph, None  # Return graph and no error
        
    except Exception as e:
        # Return None for graph and the error info
        error_info = {
            'graph_type': jetType,
            'weight_type': 'kT',
            'index_number': idx,
            'error': str(e)
        }
        return None, error_info

missing_graphs = []

# MODIFIED: Updated fileToGraph function to use multiprocessing
def fileToGraph(jetType, k=3, save=True, num_processes=4):
    """
    Modified version that uses multiprocessing for faster graph creation.
    
    Parameters:
    - num_processes: Number of processes to use. If None, uses all available CPUs.
    """
    print(f'Starting processing on {jetType} jets')
    
    # If num_processes is None, use all available CPUs
    if num_processes is None:
        num_processes = cpu_count()
    
    print(f'Using {num_processes} processes for parallel processing')
    
    pointCloudArr = fileToPointCloudArray(jetType, input_features)
    
    saveFilePath = f'../data/Multi Level Jet Tagging/kT/kT_{jetType}.pkl'
    # Create directory for adjacency matrices CSV files
    adj_csv_dir = f'adj_matrices/kT/{jetType}'
    os.makedirs(adj_csv_dir, exist_ok=True)
    
    # Prepare arguments for parallel processing
    # Each worker will get: (index, pointCloud, k, jetType, adj_csv_dir)
    process_args = [(idx, pointCloud, k, jetType, adj_csv_dir) 
                   for idx, pointCloud in enumerate(pointCloudArr)]
    
    savedGraphs = []
    
    # Use multiprocessing to process jets in parallel
    with Pool(processes=num_processes) as pool:
        # Use tqdm to show progress bar
        results = list(tqdm(
            pool.imap(process_single_jet, process_args),
            total=len(process_args),
            desc=f'Processing {jetType} jets'
        ))
    
    # Collect results and handle errors
    for graph, error_info in results:
        if graph is not None:
            savedGraphs.append(graph)
        else:
            if error_info:
                print(f"Error processing jet {error_info['index_number']}: {error_info['error']}")
                missing_graphs.append(error_info)
    
    if save:
        with open(saveFilePath, 'wb') as f:
            pickle.dump(savedGraphs, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        del pointCloudArr
        
    print(f'Graphs for {jetType} processing complete! Processed {len(savedGraphs)} graphs successfully.')
        
    return savedGraphs

def groupToGraph(jetTypeList, groupName):
    allGraphs = []
    for jetType in jetTypeList:
        allGraphs += fileToGraph(jetType, save=False)
    
    saveFilePath = f'../data/Multi Level Jet Tagging/kT/kT_{groupName}.pkl' 
    return allGraphs

# Main execution with error handling
if __name__ == '__main__':
    # process all jetTypes
    Higgs = ['HToBB', 'HToCC', 'HToGG', 'HToWW2Q1L', 'HToWW4Q']
    Vector = ['WToQQ', 'ZToQQ']
    Top = ['TTBar', 'TTBarLep']
    QCD = ['ZJetsToNuNu']
    Emitter = ['Emitter-Vector', 'Emitter-Top', 'Emitter-Higgs', 'Emitter-QCD']
    allJets = Higgs + Vector + Top + QCD

    # Optional: You can control the number of processes here
    # For example, use half of your CPUs:
    # num_processes = cpu_count() // 2
    
    for jetType in allJets:
        fileToGraph(jetType)  # Uses all CPUs by default

    # Save the log of missing graphs
    log_dir = 'missing_graphs_logs'
    os.makedirs(log_dir, exist_ok=True)
    pd.DataFrame(missing_graphs).to_csv(os.path.join(log_dir, 'missing_graphs_kT.csv'), index=False)

    # allGraphs = groupToGraph(Higgs, "Emitter-Higgs")