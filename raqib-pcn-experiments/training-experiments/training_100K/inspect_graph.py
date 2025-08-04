import pickle
import torch
import dgl
import math

# Load one sample base graph file (adjust path and file as needed)
with open('data/HToBB-Testing.pkl', 'rb') as f:
    graphs = pickle.load(f)

# Pick one graph
g = graphs[0]

# Check the node features
print("Node feature shape:", g.ndata['feat'].shape)         # (num_nodes, num_features)
print("Example feature vector (node 0):", g.ndata['feat'][0])

# Optional: label the feature dimensions manually if you know the order
feature_names = [
    "px", "py", "pz", "E", "pt", "deta", "phi", "d0val", "d0err",
    "dzval", "dzerr", "isChargedHadron", "isNeutralHadron", "isPhoton", "isElectron", "isMuon"
]

# Show named mapping if feature count matches
if g.ndata['feat'].shape[1] == len(feature_names):
    print("\nNamed features for node 0:")
    for name, value in zip(feature_names, g.ndata['feat'][0]):
        print(f"{name}: {value.item():.4f}")
else:
    print("⚠️ Feature count mismatch with expected physics features.")

print(f"px: {g.ndata['feat'][0][0]}")

part_px, part_py, part_pz, energy,  = g.ndata['feat'][0][:4]
phi = g.ndata['feat'][0][6]


features = [
    ("part_px", part_px),
    ("part_py", part_py),
    ("part_pz", part_pz),
    ("energy", energy),
    ("phi", phi)
]

for name, value in features:
    print(f"{name} = {value}")



def to_pt2(part_px, part_py, eps=1e-8):
    pt2 = part_px ** 2 + part_py ** 2
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2
pT = torch.sqrt(to_pt2(part_px, part_py, eps=1e-8))
pT = pT.numpy()
print(f"pT = {pT}")


