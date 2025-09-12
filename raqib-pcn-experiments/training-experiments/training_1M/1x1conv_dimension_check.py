import numpy as np
import torch
from tqdm import tqdm
import os
import dgl
import pickle
import math
import gc

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

batch_size = 256

# get pTmin
def get_pTmin(part_i, part_j):
    pT_i = part_i[:, 4]
    pT_j = part_j[:, 4]
    pTmin = torch.minimum(pT_i, pT_j)
    return pTmin

# Delta
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi

def rapidity(part_n):
    energy = part_n[:, 3]
    pz = part_n[:, 2]
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    return rapidity

def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2

def get_delta(part_i, part_j, eps=1e-8):
    rap_i = rapidity(part_i)
    rap_j = rapidity(part_j)
    phi_i = part_i[:, 6] # part_dphi
    phi_j = part_j[:, 6]

    delta = delta_r2(rap_i, phi_i, rap_j, phi_j).sqrt()
    return delta

def delta_weight(part_i, part_j, eps=1e-8):
    lndelta = torch.log(get_delta(part_i, part_j, eps))
    return lndelta

# kT
def kT_weight(part_i, part_j, eps=1e-8):
    pTmin = get_pTmin(part_i, part_j)
    delta_ij = get_delta(part_i, part_j)
    lnkT = torch.log((pTmin * delta_ij).clamp(min=eps))
    return lnkT

# Z
def Z_weight(part_i, part_j, eps=1e-8):
    pTi = part_i[:, 4]
    pTj = part_j[:, 4]
    pTmin = get_pTmin(part_i, part_j)
    lnZ = torch.log((pTmin / (pTi + pTj).clamp(min=eps)).clamp(min=eps))
    return lnZ

# mSquare
def to_m2(part_i, part_j, eps=1e-8):
    energy_i = part_i[:, 3]
    energy_j = part_j[:, 3]
    p_i = part_i[:, 0:3]
    p_j = part_j[:, 0:3]
    m2 = (energy_i + energy_j).square() - (p_i + p_j).square().sum(dim=1)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2

def mSquare_weight(part_i, part_j, eps=1e-8):
    lnm2 = torch.log(to_m2(part_i, part_j, eps=eps))
    return lnm2

# Compute edge weight for a given weight type (GPU calculation only, results returned on CPU)
def compute_edge_weights_gpu(base_graph_cpu, weight_type, device):
    # Compute edge weights on GPU using node features and edge indices; keep graphs on CPU
    weight_functions = {
        'delta': delta_weight,
        'kT': kT_weight,
        'Z': Z_weight,
        'mSquare': mSquare_weight,
    }
    weight_func = weight_functions[weight_type]

    # Edge indices on CPU
    src_nodes_cpu, dst_nodes_cpu = base_graph_cpu.edges()
    src_nodes_cpu = src_nodes_cpu.detach().clone()
    dst_nodes_cpu = dst_nodes_cpu.detach().clone()

    # Move only what is required to GPU
    points_cpu = base_graph_cpu.ndata['feat']
    points_gpu = points_cpu.to(device)
    src_gpu = src_nodes_cpu.to(device)
    dst_gpu = dst_nodes_cpu.to(device)

    # Gather endpoint features for all edges
    src_points = points_gpu.index_select(0, src_gpu)
    dst_points = points_gpu.index_select(0, dst_gpu)

    with torch.no_grad():
        edge_weights_gpu = weight_func(src_points, dst_points)

    # Bring weights back to CPU for CPU graphs
    edge_weights_cpu = edge_weights_gpu.detach().to('cpu')

    # Cleanup GPU tensors
    del src_points, dst_points, edge_weights_gpu, points_gpu, src_gpu, dst_gpu
    torch.cuda.empty_cache()

    return edge_weights_cpu

# Create new graph with same structure but different edge weights (graph stays on CPU)
def create_weighted_graphs(base_graph_cpu, weight_type, device):
    # Compute edge weights on GPU, return on CPU
    edge_weights_cpu = compute_edge_weights_gpu(base_graph_cpu, weight_type, device)

    # Create new graph on CPU with same structure
    src_nodes_cpu, dst_nodes_cpu = base_graph_cpu.edges()
    new_graph_cpu = dgl.graph((src_nodes_cpu, dst_nodes_cpu), num_nodes=base_graph_cpu.num_nodes(), device='cpu')

    # Assign edge weights on CPU
    new_graph_cpu.edata['weight'] = edge_weights_cpu.float().contiguous()

    # Copy node features and any other node data (on CPU)
    node_features_cpu = base_graph_cpu.ndata['feat'].detach().to('cpu').clone()
    new_graph_cpu.ndata['feat'] = node_features_cpu
    del node_features_cpu

    for key in base_graph_cpu.ndata.keys():
        if key != 'feat':
            temp_data_cpu = base_graph_cpu.ndata[key].detach().to('cpu').clone()
            new_graph_cpu.ndata[key] = temp_data_cpu
            del temp_data_cpu

    # Sanity checks
    assert new_graph_cpu.num_nodes() == base_graph_cpu.num_nodes(), "Node count mismatch!"
    assert new_graph_cpu.num_edges() == base_graph_cpu.num_edges(), "Edge count mismatch!"

    # Final GPU cleanup after this graph
    torch.cuda.empty_cache()

    return new_graph_cpu

# MODIFIED MultiGraphDataset Class - Only first 10 graphs per jet type
class MultiGraphDataset(dgl.data.DGLDataset):
    def __init__(self, jetNames, k, loadFromDisk=False, device='cuda', use_gpu=True, max_graphs_per_type=10):
        self.jetNames = jetNames
        self.k = k
        self.device = device
        self.use_gpu = use_gpu
        self.max_graphs_per_type = max_graphs_per_type

        # Initialize lists for all graph types
        self.delta = []
        self.kT = []
        self.Z = []
        self.mSquare = []
        self.sampleCountPerClass = []
        self.labels = []

        for jetType in tqdm(jetNames, total=len(jetNames), desc="Processing jet types"):
            if type(jetType) != list:
                if loadFromDisk:
                    base_path = f'pickleFiles/{jetType}.pkl'
                else:
                    base_path = f'data/{jetType}.pkl'
                
                print(f"Loading {jetType}...")
                with open(base_path, 'rb') as f:
                    base_graphs = pickle.load(f)

                # LIMIT TO FIRST max_graphs_per_type GRAPHS ONLY
                base_graphs = base_graphs[:self.max_graphs_per_type]
                print(f"Processing ONLY FIRST {len(base_graphs)} graphs for {jetType} (for dimension checking)")
                
                # Lists to store graphs for this jet class
                jetType_delta = []
                jetType_kT = []
                jetType_Z = []
                jetType_mSquare = []

                for idx, base_graph in tqdm(enumerate(base_graphs),
                                            total=len(base_graphs),
                                            desc=f"Creating weighted graphs for {jetType}",
                                            leave=False):
                    # Create all 4 weighted graph types for this single base graph
                    graph_delta = create_weighted_graphs(base_graph, 'delta', self.device)
                    graph_kT = create_weighted_graphs(base_graph, 'kT', self.device)
                    graph_Z = create_weighted_graphs(base_graph, 'Z', self.device)
                    graph_mSquare = create_weighted_graphs(base_graph, 'mSquare', self.device)

                    # Graphs are already on CPU, just append them
                    jetType_delta.append(graph_delta)
                    jetType_kT.append(graph_kT)
                    jetType_Z.append(graph_Z)
                    jetType_mSquare.append(graph_mSquare)

                    # Clean up references - including the base_graph 
                    del graph_delta, graph_kT, graph_Z, graph_mSquare
                    del base_graph
                        
                print(f"Generated all weighted graphs for {jetType}")

                # Add graphs from this jet class to main dataset
                self.delta.extend(jetType_delta)
                self.kT.extend(jetType_kT)
                self.Z.extend(jetType_Z)
                self.mSquare.extend(jetType_mSquare)
                class_count = len(base_graphs)
                self.sampleCountPerClass.append(class_count)
                # Create labels for this class
                current_label = len(self.sampleCountPerClass) - 1
                self.labels.extend([current_label] * class_count)

                print(f"Added {len(base_graphs)} graphs from {jetType} to dataset")
                
                # Clean up this jet class data
                del jetType_delta, jetType_kT, jetType_Z, jetType_mSquare
                del base_graphs
                torch.cuda.empty_cache()
                gc.collect()

                print(f"{jetType} COMPLETED")
                print("-" * 50)
        
        for label, sampleCount in enumerate(self.sampleCountPerClass):
            print(f"Class {label} ({self.jetNames[label]}) has {sampleCount} samples")
        
        print(f"DATASET CREATION COMPLETED!")
        print(f"Total samples: {len(self.labels)}")
        print(f"Samples per class: {self.sampleCountPerClass}")

    def process(self):
        return
    
    def __getitem__(self, idx):
        # Return CPU graphs - they'll be moved to GPU in collate function
        return {
            'graph_delta': self.delta[idx],
            'graph_kT': self.kT[idx],
            'graph_Z': self.Z[idx],
            'graph_mSquare': self.mSquare[idx],
            'label': self.labels[idx]
        }
    
    def __len__(self):
        return len(self.delta)

# collate function for multiple graphs
def collateFunction(batch):
    graphs_delta = [item['graph_delta'] for item in batch]
    graphs_kT = [item['graph_kT'] for item in batch]
    graphs_Z = [item['graph_Z'] for item in batch]
    graphs_mSquare = [item['graph_mSquare'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # Batch on CPU first, then move to GPU
    batched_graph_delta = dgl.batch(graphs_delta).to(device)
    batched_graph_kT = dgl.batch(graphs_kT).to(device)
    batched_graph_Z = dgl.batch(graphs_Z).to(device)
    batched_graph_mSquare = dgl.batch(graphs_mSquare).to(device)

    # Ensure all node features AND edge weights are detached to prevent gradient tracking
    batched_graph_delta.ndata['feat'] = batched_graph_delta.ndata['feat'].detach()
    batched_graph_kT.ndata['feat'] = batched_graph_kT.ndata['feat'].detach()
    batched_graph_Z.ndata['feat'] = batched_graph_Z.ndata['feat'].detach()
    batched_graph_mSquare.ndata['feat'] = batched_graph_mSquare.ndata['feat'].detach()
    
    # Also detach edge weights to prevent gradient tracking
    if 'weight' in batched_graph_delta.edata:
        batched_graph_delta.edata['weight'] = batched_graph_delta.edata['weight'].detach()
    if 'weight' in batched_graph_kT.edata:
        batched_graph_kT.edata['weight'] = batched_graph_kT.edata['weight'].detach()
    if 'weight' in batched_graph_Z.edata:
        batched_graph_Z.edata['weight'] = batched_graph_Z.edata['weight'].detach()
    if 'weight' in batched_graph_mSquare.edata:
        batched_graph_mSquare.edata['weight'] = batched_graph_mSquare.edata['weight'].detach()

    # Clear CPU graph lists after batching
    del graphs_delta, graphs_kT, graphs_Z, graphs_mSquare
    
    return (batched_graph_delta, batched_graph_kT, batched_graph_Z, batched_graph_mSquare), torch.tensor(labels, device=device)

# GNN Feature Extractor WITH DIMENSION PRINTING
class GNNFeatureExtractor(nn.Module):
    def __init__(self, in_feats, hidden_feats, k):
        super(GNNFeatureExtractor, self).__init__()
        self.conv1 = dgl.nn.ChebConv(in_feats, hidden_feats, k)
        self.conv2 = dgl.nn.ChebConv(hidden_feats, hidden_feats, k)
        self.conv3 = dgl.nn.ChebConv(hidden_feats, hidden_feats, k)

        self.edgeconv1 = dgl.nn.EdgeConv(hidden_feats, hidden_feats)
        self.edgeconv2 = dgl.nn.EdgeConv(hidden_feats, hidden_feats)
    
    def forward(self, g, graph_name="unknown"):
        print(f"\n=== {graph_name.upper()} GRAPH FEATURE EXTRACTOR ===")
        print(f"Input graph nodes: {g.num_nodes()}, edges: {g.num_edges()}")
        print(f"Input node features shape: {g.ndata['feat'].shape}")
        
        # Apply graph convolutional layers
        h = F.relu(self.conv1(g, g.ndata['feat']))
        print(f"After conv1 + ReLU: {h.shape}")
        
        h = F.relu(self.edgeconv1(g, h))
        print(f"After edgeconv1 + ReLU: {h.shape}")
        
        h = F.relu(self.conv2(g, h))
        print(f"After conv2 + ReLU: {h.shape}")
        
        h = F.relu(self.edgeconv2(g, h))
        print(f"After edgeconv2 + ReLU: {h.shape}")
        
        h = F.relu(self.conv3(g, h))
        print(f"After conv3 + ReLU: {h.shape}")

        # Store the node embeddings in the node data directory
        g.ndata['h'] = h

        # Compute graph-level representations by taking global mean pooling
        hg = dgl.mean_nodes(g, 'h')
        print(f"After global mean pooling: {hg.shape}")
        print(f"=== END {graph_name.upper()} GRAPH ===\n")

        return hg

# Classifier class WITH DIMENSION PRINTING - 1x1 CONVOLUTION VERSION
class Classifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        # 1D convolution with kernel_size=1 (1-to-1 convolution)
        # Input: (batch_size, 4, 64) -> Output: (batch_size, 4, 64)
        self.conv1d = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=1)
        
        # Keep the same structure as before
        self.fc1 = torch.nn.Linear(4 * 64, hidden_dim)  # 4*64 = 256, same as before
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        print(f"\n=== CLASSIFIER FORWARD PASS - 1x1 CONVOLUTION ARCHITECTURE ===")
        print(f"Input to classifier: {x.shape}")  # Expected: [batch_size, 4, 64]
        
        # Apply 1D convolution with kernel_size=1 across the 4 graph types
        x = self.conv1d(x)
        print(f"After 1x1 conv1d: {x.shape}")  # Expected: [batch_size, 4, 64]
        
        # Flatten to same dimension as before: (batch_size, 256)
        x = x.view(x.size(0), -1)
        print(f"After flatten: {x.shape}")  # Expected: [batch_size, 256]
        
        # Same structure as before
        x = self.fc1(x)
        print(f"After fc1: {x.shape}")  # Expected: [batch_size, hidden_dim]
        
        x = self.relu(x)
        print(f"After ReLU: {x.shape}")  # Expected: [batch_size, hidden_dim]
        
        x = self.dropout(x)
        print(f"After dropout: {x.shape}")  # Expected: [batch_size, hidden_dim]
        
        x = self.fc2(x)
        print(f"Final output (logits): {x.shape}")  # Expected: [batch_size, output_dim]
        print(f"=== END CLASSIFIER ===\n")
        
        return x

# Main execution
if __name__ == "__main__":
    print("=== 1x1 CONVOLUTION DIMENSION CHECKING SCRIPT ===")
    print("Loading first 10 graphs of each type for dimension analysis...")
    
    # Process all jetTypes
    Higgs = ['HToBB', 'HToCC', 'HToGG', 'HToWW2Q1L', 'HToWW4Q']
    Vector = ['WToQQ', 'ZToQQ']
    Top = ['TTBar', 'TTBarLep']
    QCD = ['ZJetsToNuNu']
    
    testingSet = Top + Vector + QCD + Higgs
    testingSet = [s + "-Testing" for s in testingSet]
    
    jetNames = testingSet
    print(f"Jet types: {jetNames}")
    
    # Create dataset with only first 10 graphs per type
    print("\nCreating dataset with first 10 graphs per type...")
    dataset = MultiGraphDataset(jetNames, 3, loadFromDisk=False, device=device, use_gpu=True, max_graphs_per_type=10)
    dataset.process()
    
    print(f"Total dataset size: {len(dataset)}")
    
    # Create a small dataloader with batch_size=4 for dimension checking
    test_batch_size = 4
    testLoader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collateFunction, drop_last=True, num_workers=0)
    
    # Model parameters
    in_feats = 16
    hidden_feats = 64
    out_feats = len(jetNames)
    chebFilterSize = 16
    
    print(f"\nModel parameters:")
    print(f"in_feats: {in_feats}")
    print(f"hidden_feats: {hidden_feats}")
    print(f"out_feats: {out_feats}")
    print(f"chebFilterSize: {chebFilterSize}")
    
    # Create models
    print("\nCreating models...")
    model_delta = GNNFeatureExtractor(in_feats, hidden_feats, chebFilterSize)
    model_kT = GNNFeatureExtractor(in_feats, hidden_feats, chebFilterSize)
    model_mSquare = GNNFeatureExtractor(in_feats, hidden_feats, chebFilterSize)
    model_Z = GNNFeatureExtractor(in_feats, hidden_feats, chebFilterSize)
    classifier = Classifier(hidden_feats * 4, hidden_feats, out_feats)  # input_dim not used in new architecture
    
    # Move models to device
    model_delta.to(device)
    model_kT.to(device)
    model_mSquare.to(device)
    model_Z.to(device)
    classifier.to(device)
    
    # Set models to eval mode
    all_models = [model_delta, model_kT, model_mSquare, model_Z, classifier]
    for model in all_models:
        model.eval()
    
    print("\n" + "="*80)
    print("STARTING DIMENSION CHECK - 1x1 CONVOLUTION FORWARD PASS")
    print("="*80)
    
    # Process just ONE batch for dimension checking
    with torch.no_grad():
        for batch_idx, (graphs, labels) in enumerate(testLoader):
            print(f"\n>>> PROCESSING BATCH {batch_idx + 1} <<<")
            print(f"Batch labels: {labels}")
            
            # Unpack graphs - Note: Order matters! Match the collateFunction order
            graph_delta, graph_kT, graph_Z, graph_mSquare = graphs
            labels = labels.to(device).long()
            
            print(f"\nBatch size: {labels.shape[0]}")
            
            # Get embeddings from each graph type WITH DIMENSION PRINTING
            hg_delta = model_delta(graph_delta, "delta")
            hg_kT = model_kT(graph_kT, "kT")
            hg_mSquare = model_mSquare(graph_mSquare, "mSquare")
            hg_Z = model_Z(graph_Z, "Z")
            
            print(f"\n=== COMBINING EMBEDDINGS ===")
            print(f"hg_delta shape: {hg_delta.shape}")
            print(f"hg_kT shape: {hg_kT.shape}")
            print(f"hg_mSquare shape: {hg_mSquare.shape}")
            print(f"hg_Z shape: {hg_Z.shape}")
            
            # Stack embeddings to create 4×64 matrix instead of 256-dimensional vector
            stacked_features = torch.stack([hg_delta, hg_kT, hg_mSquare, hg_Z], dim=1)  # (batch_size, 4, 64)
            print(f"After torch.stack (stacked_features): {stacked_features.shape}")
            
            # Get final logits from classifier WITH DIMENSION PRINTING
            logits = classifier(stacked_features)
            
            print(f"\n=== FINAL RESULTS ===")
            print(f"Final logits shape: {logits.shape}")
            print(f"Expected logits shape: [batch_size={test_batch_size}, num_classes={out_feats}]")
            
            # Check if dimensions are as expected
            expected_shape = (test_batch_size, out_feats)
            if logits.shape == expected_shape:
                print(f"✅ SUCCESS: Logits shape matches expected {expected_shape}")
            else:
                print(f"❌ ERROR: Logits shape {logits.shape} does not match expected {expected_shape}")
            
            print(f"\nLogits values (first few):\n{logits[:2]}")  # Print first 2 rows
            
            # Predictions
            predictions = logits.argmax(dim=1)
            print(f"Predictions: {predictions}")
            print(f"Ground truth: {labels}")
            
            # Clean up
            del graphs, labels
            
            # Only process the first batch for dimension checking
            break
    
    print("\n" + "="*80)
    print("1x1 CONVOLUTION DIMENSION CHECK COMPLETED!")
    print("="*80)
    
    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Script finished successfully!")
