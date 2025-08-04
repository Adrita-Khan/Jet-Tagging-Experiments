import pickle
import dgl
import torch
import pandas as pd
import numpy as np
import math

# ==========
# CONFIGURATION
# ==========
base_graph_path = 'data/HToBB-Testing.pkl'  # Adjust if needed
output_csv_path = 'delta_adjacency_htobb.csv'         # Output CSV

# ==========
# DELTA CALCULATION FUNCTIONS (YOUR ORIGINAL ONES)
# ==========
def to_pt2(part_px, part_py, eps=1e-8):
    pt2 = part_px ** 2 + part_py ** 2
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2

def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi

def rapidity(part_n):
    energy = part_n[3]
    pz = part_n[2]
    y = 0.5 * torch.log((energy + pz) / (energy - pz + 1e-8))
    return y

def delta_r2(eta1, phi1, eta2, phi2):
    dphi = delta_phi(phi1, phi2)
    return (eta1 - eta2)**2 + dphi**2

def compute_delta_weights(g):
    features = g.ndata['feat']
    src, dst = g.edges()

    # Prepare adjacency list
    num_nodes = g.num_nodes()
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for s, d in zip(src.tolist(), dst.tolist()):
        part_i = features[s]
        part_j = features[d]

        # Extract features: px, py, pz, E, ..., phi is at index 6
        px_i, py_i, pz_i, E_i, *_ = part_i
        px_j, py_j, pz_j, E_j, *_ = part_j

        phi_i = part_i[6]
        phi_j = part_j[6]

        # Convert to rapidity
        rap_i = rapidity(torch.tensor([px_i, py_i, pz_i, E_i]))
        rap_j = rapidity(torch.tensor([px_j, py_j, pz_j, E_j]))

        delta_val = torch.sqrt(delta_r2(rap_i, phi_i, rap_j, phi_j))
        ln_delta = torch.log(delta_val.clamp(min=1e-8)).item()

        adj_matrix[s][d] = ln_delta

    return adj_matrix

# ==========
# LOAD GRAPH
# ==========
with open(base_graph_path, 'rb') as f:
    graphs = pickle.load(f)

# Use only the first graph for verification
g = graphs[0]

# Compute delta adjacency matrix from graph
adj_matrix = compute_delta_weights(g)

# ==========
# SAVE TO CSV WITH HEADERS
# ==========
df = pd.DataFrame(adj_matrix)
df.index = [f"node_{i}" for i in range(g.num_nodes())]
df.columns = [f"node_{i}" for i in range(g.num_nodes())]
df.to_csv(output_csv_path)

print(f"Delta adjacency matrix saved to '{output_csv_path}' with headers for verification.")
