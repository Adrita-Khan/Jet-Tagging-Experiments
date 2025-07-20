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
import csv




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


# Build a KNN graph from a point cloud
def buildKNNGraph(points, k):
    tree = cKDTree(points)
    dists, indices = tree.query(points, k+1)  # +1 to exclude self

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
    # filepath = f'/Volumes/Yash SSD/JetClass/JetRoots/{jetType}_000.root' # original root file
    # savepath = f'/Volumes/Yash SSD/JetClass/PointClouds/{jetType}.npy' # save file
    filepath = f'../data/JetClass/JetRoots/{jetType}/{jetType}_100K.root' # original root file
    savepath = f'../data/JetClass/PointClouds/{jetType}.npy' # save file

    awk = fileToAwk(filepath)
    nparr = awkToPointCloud(awk, input_features)
    
    return nparr


missing_graphs = []
# wrap the functionality of fileToPointCloudArray and the 
def fileToGraph(jetType, k=3, save=True):
    print(f'Starting processing on {jetType} jets')
    pointCloudArr = fileToPointCloudArray(jetType, input_features)
    
    # saveFilePath = f'/Volumes/Yash SSD/Multi Level Jet Tagging/{jetType}.pkl'
    saveFilePath = f'../data/Multi Level Jet Tagging/Delta/Delta_{jetType}.pkl'
    
    # Create directory for adjacency matrices CSV files
    adj_csv_dir = f'adj_matrices/delta/{jetType}'
    os.makedirs(adj_csv_dir, exist_ok=True)
    
    savedGraphs = []
    for idx, pointCloud in tqdm(enumerate(pointCloudArr), leave=False, total=len(pointCloudArr)):
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
            
            savedGraphs.append(graph)
            
            del adj_matrix, graph
        except Exception as e:
            print(e)
            # Log the missing graphs
            missing_graphs.append({
                'graph_type': jetType,
                'weight_type': 'Delta',
                'index_number': idx
            })
            
    
    if save:
        with open(saveFilePath, 'wb') as f:
            pickle.dump(savedGraphs, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        del pointCloudArr
        
    print(f'Graphs for {jetType} processing complete!')
        
    return savedGraphs

def groupToGraph(jetTypeList, groupName):
    allGraphs = []
    for jetType in jetTypeList:
        allGraphs += fileToGraph(jetType, save=False)
    
    saveFilePath = f'../data/Multi Level Jet Tagging/Delta/Delta_{groupName}.pkl' 
    return allGraphs

# process all jetTypes
Higgs = ['HToBB', 'HToCC', 'HToGG', 'HToWW2Q1L', 'HToWW4Q']
Vector = ['WToQQ', 'ZToQQ']
Top = ['TTBar', 'TTBarLep']
QCD = ['ZJetsToNuNu']
Emitter = ['Emitter-Vector', 'Emitter-Top', 'Emitter-Higgs', 'Emitter-QCD']
allJets = Higgs + Vector + Top + QCD

for jetType in allJets:
   fileToGraph(jetType)

log_dir = 'missing_graphs_logs'
os.makedirs(log_dir, exist_ok=True)
pd.DataFrame(missing_graphs).to_csv(os.path.join(log_dir, 'missing_graphs_delta.csv'), index=True)


# allGraphs = groupToGraph(Higgs, "Emitter-Higgs")