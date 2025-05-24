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
from scipy.spatial import cKDTree
import scipy.sparse as sp
import dgl
import pickle
# ------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------

# Take ROOT file and convert to an awkward array
def fileToAwk(path):
    file = uproot.open(path)
    tree = file['tree']
    awk = tree.arrays(tree.keys())
    return awk

# Features used to train the model
input_features = [
    "part_px", "part_py", "part_pz", "part_energy",
    "part_deta", "part_dphi", "part_d0val", "part_d0err",
    "part_dzval", "part_dzerr", "part_isChargedHadron", "part_isNeutralHadron",
    "part_isPhoton", "part_isElectron", "part_isMuon"
]

# Take AWK dict and convert to a point cloud
def awkToPointCloud(awkDict, input_features, label):
    featureVector = []
    for jet in tqdm(range(len(awkDict)), total=len(awkDict)):
        if awkDict[jet][label] == 1:  # Filter jets based on the label
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
    return featureVector  # Return a list of arrays instead of a single numpy array


def rapidity(part_n):
    e = part_n[3]
    pz = part_n[2]
    return 0.5 * np.log((e + pz) / (e - pz))


def delta(part_a, part_b):
    y_a = rapidity(part_a)
    y_b = rapidity(part_b)

    phi_a = part_a[6]
    phi_b = part_b[6]

    delta_ab = np.sqrt((y_a - y_b)**2 + (phi_a - phi_b)**2)

    return delta_ab

def kT(part_a, part_b):
    pT_a = part_a[4]
    pT_b = part_b[4]

    delta_ab = delta(part_a, part_b)

    kT_ab = min(pT_a, pT_b) * delta_ab

    return np.log(kT_ab)


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

# Convert adjacency matrix to a DGL graph
def adjacencyToDGL(adj_matrix):
    adj_matrix = sp.coo_matrix(adj_matrix)
    g_dgl = dgl.from_scipy(adj_matrix)
    return g_dgl

# Wrap the functionality of fileToAwk and awkToPointCloud into a single function
def fileToPointCloudArray(jetType, input_features, label):
    filepath = f'../data/JetClass/JetRoots/JetClass_example_100k.root'  # original ROOT file
    savepath = f'../data/JetClass/PointClouds/{jetType}.npy'   # save file
    awk = fileToAwk(filepath)
    nparr = awkToPointCloud(awk, input_features, label)
    return nparr

def saveAdj_Matrices(adj_matrices_list, jetType):
    """
    Save adjacency matrices as CSV and JSON files
    """    
    # Create directory if it doesn't exist
    os.makedirs(f'../adj_matrices/kT/kT_{jetType}', exist_ok=True)
    
    for idx, adj_matrix in enumerate(adj_matrices_list):
        # File paths
        csv_path = f'../adj_matrices/kT/kT_{jetType}/kT_Emitter-{jetType}_jet_{idx}.csv'
        
        # Save as CSV
        pd.DataFrame(adj_matrix).to_csv(csv_path, index=True, header=True)
        

# Combine all steps: read file → build adjacency → build DGL graph → optionally save
def fileToGraph(jetType, label, k=3, save=True):
    print(f'Starting processing on {jetType} jets with label {label}')
    pointCloudArr = fileToPointCloudArray(jetType, input_features, label)
    saveFilePath = f'../data/Multi Level Jet Tagging/kT/kT_{jetType}.pkl'

    savedGraphs = []
    adj_matrices_list = []
    for idx, pointCloud in tqdm(enumerate(pointCloudArr), leave=False, total=len(pointCloudArr)):
        try:
            adj_matrix = buildKNNGraph(pointCloud, k)
            adj_matrices_list.append(adj_matrix)
            graph = adjacencyToDGL(adj_matrix)
            graph.ndata['feat'] = torch.tensor(pointCloud, dtype=torch.float32)
            savedGraphs.append(graph)

            # Cleanup
            del adj_matrix, graph
        except Exception as e:
            print(e)
    
    # Save adjacency matrices as CSV and JSON
    saveAdj_Matrices(adj_matrices_list, jetType)

    if save:
        with open(saveFilePath, 'wb') as f:
            pickle.dump(savedGraphs, f, protocol=pickle.HIGHEST_PROTOCOL)

        del pointCloudArr, savedGraphs

    print(f'Graphs for {jetType} processing complete!')
    return savedGraphs

def groupToGraph(jetTypeList, groupName, label):
    allGraphs = []
    for jetType in jetTypeList:
        allGraphs += fileToGraph(jetType, label, save=False)

    saveFilePath = f'../data/Multi Level Jet Tagging/kT/kT_{groupName}.pkl'
    with open(saveFilePath, 'wb') as f:
        pickle.dump(allGraphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    return allGraphs

# Dictionary of jet types and their respective labels
jet_types = {
    "Higgs_HToBB": ['HToBB', 'label_Hbb'],
    "Higgs_HToCC": ['HToCC', 'label_Hcc'],
    "Higgs_HToGG": ['HToGG', 'label_Hgg'],
    "Higgs_HToWW2Q1L": ['HToWW2Q1L', 'label_Hqql'],
    "Higgs_HToWW4Q": ['HToWW4Q', 'label_H4q'],
    "Vector_WToQQ": ['WToQQ', 'label_Wqq'],
    "Vector_ZToQQ": ['ZToQQ', 'label_Zqq'],
    "Top_TTBar": ['TTBar', 'label_Tbqq'],
    "Top_TTBarLep": ['TTBarLep', 'label_Tbl'],
    "QCD_ZJetsToNuNu": ['ZJetsToNuNu', 'label_QCD']
}

# Loop through each jet type, generate graphs, and save them
for key, value in jet_types.items():
    jetType, label = value
    graphs = groupToGraph([jetType], f"Emitter-{key}", label)
    
    # Save each graph to a file
    with open(f'../data/Multi Level Jet Tagging/kT/kT_Emitter-{key}.pkl', 'wb') as f:
        pickle.dump(graphs, f)
    
    print(f"DONE: {key}")