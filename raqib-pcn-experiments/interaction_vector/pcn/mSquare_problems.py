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
from tqdm import tqdm
import timeit
import os
import dill
import scipy.sparse as sp
from scipy.spatial import cKDTree
import dgl
import pickle

# Create directories for saving problem data
os.makedirs('../data/mSquare_problems', exist_ok=True)

# Global lists to track problems
problem_log = []
problem_matrices = []

input_features = ["part_px", "part_py", "part_pz", "part_energy",
                  "part_deta", "part_dphi", "part_d0val", "part_d0err", 
                  "part_dzval", "part_dzerr", "part_isChargedHadron", "part_isNeutralHadron", 
                  "part_isPhoton", "part_isElectron", "part_isMuon"]

def fileToAwk(path):
    file = uproot.open(path)
    tree = file['tree']
    awk = tree.arrays(tree.keys())
    return awk

def awkToPointCloud(awkDict, input_features):
    featureVector = []
    for jet in tqdm(range(len(awkDict)), total=len(awkDict)):
        currJet = awkDict[jet][input_features]
        try:
            pT = np.sqrt(ak.to_numpy(currJet['part_px']) ** 2 + ak.to_numpy(currJet['part_py']) ** 2)
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
            featureVector.append(np.empty((0, len(input_features) + 1)))
    return featureVector

def mSquare_with_logging(part_a, part_b, jetType, graph_idx, particle_i, particle_j):
    """Modified mSquare function that logs problems"""
    global problem_log
    
    e_a = part_a[3]
    e_b = part_b[3]
    p_a = np.array(part_a[0:3])
    p_b = np.array(part_b[0:3])
    
    mSquare_ab = (e_a + e_b)**2 - (np.linalg.norm(p_a + p_b))**2
    
    if mSquare_ab <= 0:
        # Log the problem
        problem_entry = {
            'jetType': jetType,
            'graph_index': graph_idx,
            'particle_i': particle_i,
            'particle_j': particle_j,
            'mSquare_value': mSquare_ab,
            'energy_a': e_a,
            'energy_b': e_b,
            'momentum_a_norm': np.linalg.norm(p_a),
            'momentum_b_norm': np.linalg.norm(p_b),
            'combined_momentum_norm': np.linalg.norm(p_a + p_b)
        }
        problem_log.append(problem_entry)
        
        print(f"Problem detected: {jetType} graph {graph_idx}, particles {particle_i}-{particle_j}, mSquare = {mSquare_ab}")
        
        
        return np.log(mSquare_ab)  
    
    return np.log(mSquare_ab)

def buildKNNGraph_with_logging(points, k, jetType, graph_idx):
    """Modified buildKNNGraph function that tracks problems"""
    tree = cKDTree(points)
    dists, indices = tree.query(points, k+1)
    
    num_points = len(points)
    adj_matrix = np.zeros((num_points, num_points))
    
    has_problem = False
    
    for i in range(num_points):
        for j in indices[i, 1:]:  # exclude self
            weight_m_square = mSquare_with_logging(points[i], points[j], jetType, graph_idx, i, j)
            adj_matrix[i, j] = weight_m_square
            adj_matrix[j, i] = weight_m_square
            
            # Check if this was a problematic calculation
            if len(problem_log) > 0 and problem_log[-1]['graph_index'] == graph_idx:
                has_problem = True
    
    # If this graph had problems, save its adjacency matrix
    if has_problem:
        matrix_filename = f"mSquare_{jetType}_{graph_idx}.csv"
        matrix_path = f"../data/mSquare_problems/{matrix_filename}"
        
        # Save adjacency matrix as CSV
        pd.DataFrame(adj_matrix).to_csv(matrix_path, index=True)
        
        # Track that we saved this matrix
        problem_matrices.append({
            'jetType': jetType,
            'graph_index': graph_idx,
            'filename': matrix_filename,
            'matrix_shape': adj_matrix.shape
        })
        
        print(f"Saved problematic adjacency matrix: {matrix_filename}")
    
    return adj_matrix

def adjacencyToDGL(adj_matrix):
    adj_matrix = sp.coo_matrix(adj_matrix)
    g_dgl = dgl.from_scipy(adj_matrix)
    return g_dgl

def fileToPointCloudArray(jetType, input_features):
    filepath = f'../data/JetClass/JetRoots/{jetType}/{jetType}_100K.root'
    savepath = f'../data/JetClass/PointClouds/{jetType}.npy'
    
    awk = fileToAwk(filepath)
    nparr = awkToPointCloud(awk, input_features)
    
    return nparr

def fileToGraph_with_logging(jetType, k=3, save=True):
    """Modified fileToGraph function that includes problem logging"""
    global problem_log, problem_matrices
    
    print(f'Starting processing on {jetType} jets')
    pointCloudArr = fileToPointCloudArray(jetType, input_features)
    
    saveFilePath = f'../data/Multi Level Jet Tagging/mSquare/mSquare_{jetType}.pkl'
    
    savedGraphs = []
    for idx, pointCloud in tqdm(enumerate(pointCloudArr), leave=False, total=len(pointCloudArr)):
        try:
            adj_matrix = buildKNNGraph_with_logging(pointCloud, k, jetType, idx)
            graph = adjacencyToDGL(adj_matrix)
            
            graph.ndata['feat'] = torch.tensor(pointCloud, dtype=torch.float32)
            
            savedGraphs.append(graph)
            
            del adj_matrix, graph
        except Exception as e:
            print(f"Error processing graph {idx}: {e}")
    
    if save:
        with open(saveFilePath, 'wb') as f:
            pickle.dump(savedGraphs, f, protocol=pickle.HIGHEST_PROTOCOL)
        del pointCloudArr
    
    print(f'Graphs for {jetType} processing complete!')
    return savedGraphs

def save_problem_reports():
    """Save the problem logs to CSV files"""
    global problem_log, problem_matrices
    
    if problem_log:
        # Save detailed problem log
        problem_df = pd.DataFrame(problem_log)
        problem_df.to_csv('../data/mSquare_problems/mSquare_problem_log.csv', index=True)
        print(f"Saved problem log with {len(problem_log)} entries")
    
    if problem_matrices:
        # Save matrix information
        matrices_df = pd.DataFrame(problem_matrices)
        matrices_df.to_csv('../data/mSquare_problems/saved_matrices_info.csv', index=True)
        print(f"Saved matrix info for {len(problem_matrices)} problematic graphs")
    
    # Print summary
    if problem_log:
        print("\n=== PROBLEM SUMMARY ===")
        summary = pd.DataFrame(problem_log).groupby('jetType').agg({
            'graph_index': 'nunique',
            'particle_i': 'count'
        }).rename(columns={'graph_index': 'num_problematic_graphs', 'particle_i': 'total_problematic_pairs'})
        print(summary)

# Process all jet types with logging
Higgs = ['HToBB', 'HToCC', 'HToGG', 'HToWW2Q1L', 'HToWW4Q']
Vector = ['WToQQ', 'ZToQQ']
Top = ['TTBar', 'TTBarLep']
QCD = ['ZJetsToNuNu']
allJets = Higgs + Vector + Top + QCD

# Reset global problem tracking
problem_log = []
problem_matrices = []

# Process each jet type
for jetType in allJets:
    try:
        fileToGraph_with_logging(jetType)
    except Exception as e:
        print(f"Error processing {jetType}: {e}")

# Save all problem reports at the end
save_problem_reports()