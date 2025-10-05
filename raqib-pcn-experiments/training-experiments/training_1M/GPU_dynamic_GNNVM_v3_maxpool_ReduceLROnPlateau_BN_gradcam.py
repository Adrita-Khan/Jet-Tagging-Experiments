import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import seaborn as sns
import os
import dgl
import pickle
import wandb
import matplotlib.pyplot as plt
import argparse
import math
import gc
import threading
import time

from dgllife.utils import RandomSplitter
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

batch_size = 256

# Add argumnet parser
parser = argparse.ArgumentParser(description='Dynamic Multi-Graph PCN Training')
parser.add_argument('--max_epochs', type=int, default=5, help='Maximum number of epochs (default: 5)')
parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size (default: 512)')
parser.add_argument('--device', type=str, default='cuda',choices=['cuda', 'cpu'], help='Device to use (default: cuda)')
parser.add_argument('--classification_level', type=str, default='Dynamic_All_Interactions-max-pooling', help=' (Classification level default: All)')
parser.add_argument('--model_architecture', type=str, default=f'PCN-{batch_size}-ReduceLROnPlateau-BN-gradcamTesting', help='Model architecture name (default: PCN)')
parser.add_argument('--model_type', type=str, default='DGCNN', help='Model type (default: DGCNN)')
parser.add_argument('--load_model', type=str, default='N', help='Load from save file (default: N)')
parser.add_argument('--convergence_threshold', type=float, default=0.0001, help='Convergence threshold (default: 0.0001)')
parser.add_argument('--max_graphs_per_class', type=int, default=1000, help='Maximum graphs per jet class (default: 1000)')

args = parser.parse_args()

# Use argparse values
maxEpochs = args.max_epochs
batchSize = args.batch_size
device = args.device
classificationLevel = args.classification_level
modelArchitecture = args.model_architecture
modelType = args.model_type
load = True if args.load_model == 'Y' else False
convergence_threshold = args.convergence_threshold
max_graphs_per_class = args.max_graphs_per_class


def log_gpu_memory(stage=""):
    """Simple memory logging - continuous monitoring is handled by ContinuousMemoryMonitor"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"{stage} - GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
    else:
        print(f"{stage} - GPU not available")
    
    # Also log CPU memory usage
    import psutil
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / 1024**3

def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    torch.cuda.empty_cache()
    gc.collect()
    # Force multiple garbage collection cycles
    for _ in range(3):
        gc.collect()
    
    # Log memory after cleanup
    log_gpu_memory("after_cleanup")

class ContinuousMemoryMonitor:
    """Continuous memory monitoring"""
    def __init__(self, interval=5):
        self.interval = interval  # seconds
        self.running = False
        self.monitor_thread = None
        self.start_time = None
        self.current_stage = "preprocessing"
        self.current_epoch = 0
    
    def set_stage(self, stage, epoch = 0):
        """Set the current processing stage"""
        self.current_stage = stage
        self.current_epoch = epoch

    def start(self):
        """Start continuous monitoring in background thread"""
        if self.running:
            return
            
        self.running = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"Continuous memory monitoring started (every {self.interval} seconds)")
        
    def stop(self):
        """Stop continuous monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        print("Continuous memory monitoring stopped")
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                self._log_memory()
                time.sleep(self.interval)
            except Exception as e:
                print(f"Memory monitoring error: {e}")
                time.sleep(self.interval)
                
    def _log_memory(self):
        """Log current memory state"""
        # GPU Memory
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_cached = torch.cuda.memory_reserved() / 1024**3
        else:
            gpu_allocated = gpu_cached = 0
            
        # CPU Memory
        import psutil
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024**3
        
        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        # Log to wandb continuously
        wandb.log({
            "Memory_Continuous/GPU_allocated_GB": gpu_allocated,
            "Memory_Continuous/GPU_cached_GB": gpu_cached,
            "Memory_Continuous/CPU_RAM_GB": cpu_memory,
        })

def log_memory_trends():
    """Legacy function - now just logs current state"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
        peak_cached = torch.cuda.max_memory_reserved() / 1024**3
        
        wandb.log({
            "Memory_Trends/GPU_allocated_GB": allocated,
            "Memory_Trends/GPU_cached_GB": cached,
            "Memory_Trends/GPU_peak_allocated_GB": peak_allocated,
            "Memory_Trends/GPU_peak_cached_GB": peak_cached,
            "Memory_Trends/GPU_utilization_percent": (allocated / cached * 100) if cached > 0 else 0,
        })
    
    import psutil
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / 1024**3
    
    wandb.log({
        "Memory_Trends/CPU_RAM_GB": cpu_memory,
        "Memory_Trends/CPU_peak_RAM_GB": process.memory_info().vms / 1024**3,  # Virtual memory size as peak
    })



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


# MultiGraphDataset Class GPU-optimized MultiGraphDataset class
class MultiGraphDataset(dgl.data.DGLDataset):
    def __init__(self, jetNames, k, loadFromDisk=False, device='cuda', use_gpu=True):
        self.jetNames = jetNames
        self.k = k
        self.device = device
        self.use_gpu = use_gpu

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

                # Limit to max_graphs_per_class
                if len(base_graphs) > max_graphs_per_class:
                    print(f"Limiting {jetType} from {len(base_graphs)} to {max_graphs_per_class} graphs")
                    base_graphs = base_graphs[:max_graphs_per_class]

                # Process each graph after loading
                print(f"Processing {len(base_graphs)} graphs for {jetType}... ... ...")
                
                # Lists to store ALL graphs for this ENTIRE jet class
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
                    del base_graph  # Delete base graph immediately after use
                    
                    # Periodic cleanup during processing (reduced frequency for better performance)
                    if idx % 100_000 == 0:
                        force_memory_cleanup()
                        log_gpu_memory(f"processing_{jetType}_{idx}")
                        
                print(f"Generated all weighted graphs for {jetType}")

                # Add ALL graphs from this jet class to main dataset (CPU only)
                self.delta.extend(jetType_delta)
                self.kT.extend(jetType_kT)
                self.Z.extend(jetType_Z)
                self.mSquare.extend(jetType_mSquare)
                class_count = len(base_graphs)
                self.sampleCountPerClass.append(class_count)
                # Create labels for this class immediately to avoid second pass
                current_label = len(self.sampleCountPerClass) - 1
                self.labels.extend([current_label] * class_count)

                print(f"Added {len(base_graphs)} graphs from {jetType} to dataset")
                
                # Clean up this jet class data
                del jetType_delta, jetType_kT, jetType_Z, jetType_mSquare
                del base_graphs

                # COMPLETE GPU AND CPU CLEANUP before moving to next jet class
                force_memory_cleanup()

                print(f"{jetType} COMPLETED - GPU and CPU completely cleared")

                print("-" * 50)         # Visual separator between jet classes
        
        for label, sampleCount in enumerate(self.sampleCountPerClass):
            print(f"Class {label} ({self.jetNames[label]}) has {sampleCount} samples")
        
        print(f"DATASET CREATION COMPLETED!")
        print(f"Total samples: {len(self.labels)}")
        print(f"Samples per class: {self.sampleCountPerClass}")
        
        # Final cleanup
        force_memory_cleanup()
        
        # Extra aggressive cleanup at the end
        for _ in range(2):
            gc.collect()
        
        log_gpu_memory("After final cleanup")
        print("Final GPU and CPU cleanup completed")



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

    # CRITICAL FIX: Clear CPU graph lists after batching
    del graphs_delta, graphs_kT, graphs_Z, graphs_mSquare
    
    return (batched_graph_delta, batched_graph_kT, batched_graph_Z, batched_graph_mSquare), torch.tensor(labels, device=device)

# GNN Feature Extractor
class GNNFeatureExtractor(nn.Module):
    def __init__(self, in_feats, hidden_feats, k):
        super(GNNFeatureExtractor, self).__init__()
        self.conv1 = dgl.nn.ChebConv(in_feats, hidden_feats, k)
        self.bn1 = nn.BatchNorm1d(hidden_feats)
        self.conv2 = dgl.nn.ChebConv(hidden_feats, hidden_feats, k)
        self.bn2 = nn.BatchNorm1d(hidden_feats)
        self.conv3 = dgl.nn.ChebConv(hidden_feats, hidden_feats, k)
        self.bn3 = nn.BatchNorm1d(hidden_feats)

        self.edgeconv1 = dgl.nn.EdgeConv(hidden_feats, hidden_feats)
        self.bn_edge1 = nn.BatchNorm1d(hidden_feats)
        self.edgeconv2 = dgl.nn.EdgeConv(hidden_feats, hidden_feats)
        self.bn_edge2 = nn.BatchNorm1d(hidden_feats)

    def forward(self, g):
        # Apply graph convolutional layers with batch normalization
        h = self.conv1(g, g.ndata['feat'])
        h = self.bn1(h)
        h = F.relu(h)

        h = self.edgeconv1(g, h)
        h = self.bn_edge1(h)
        h = F.relu(h)

        h = self.conv2(g, h)
        h = self.bn2(h)
        h = F.relu(h)

        h = self.edgeconv2(g, h)
        h = self.bn_edge2(h)
        h = F.relu(h)

        h = self.conv3(g, h)
        h = self.bn3(h)
        h = F.relu(h)

        # Store the node embeddings in the node data directory
        g.ndata['h'] = h

        # Compute graph-level representations by taking global mean pooling
        hg = dgl.mean_nodes(g, 'h')

        return hg


# Classifier class
class Classifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        # input_dim is now the feature dimension (64), input shape is [batch_size, 4, 64]
        # 1D convolution along the feature dimension
        self.conv1d = torch.nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.bn_conv1 = nn.BatchNorm1d(4)  # Batch norm for first conv (4 channels)
        self.conv1d_2 = torch.nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.bn_conv2 = nn.BatchNorm1d(4)  # Batch norm for second conv (4 channels)

        # Max pooling across channels (graph types)
        # This will pool across the 4 channels to get [batch_size, 64]

        # Batch norm after max pooling
        self.bn_pool = nn.BatchNorm1d(hidden_dim)

        # Final classification layer - only one FC layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: [batch_size, 4, 64] (4 graph types, 64 features each)

        # Apply 1D convolutions along feature dimension with batch norm
        x = self.conv1d(x)  # [batch_size, 4, 64]
        x = self.bn_conv1(x)
        x = F.relu(x)

        x = self.conv1d_2(x)  # [batch_size, 4, 64]
        x = self.bn_conv2(x)
        x = F.relu(x)

        # Apply max pooling across channels (graph types) to get [batch_size, 64]
        x = torch.max(x, dim=1)[0]  # [batch_size, 64]

        # Apply batch norm after pooling
        x = self.bn_pool(x)

        # Final classification
        x = self.fc(x)
        return x


# Process all jetTypes
Higgs = ['HToBB', 'HToCC', 'HToGG', 'HToWW2Q1L', 'HToWW4Q']
Vector = ['WToQQ', 'ZToQQ']
Top = ['TTBar', 'TTBarLep']
QCD = ['ZJetsToNuNu']
Emitter = ['Emitter-Vector', 'Emitter-Top', 'Emitter-Higgs', 'Emitter-QCD']
allJets = Higgs + Vector + Top + QCD

testingSet = Top + Vector + QCD + Higgs
testingSet = [s + "-Testing" for s in testingSet]

jetNames = testingSet
print(jetNames)

# Generate unique model filename to avoid overwriting - MOVED HERE BEFORE WANDB INIT
os.makedirs("modelSaveFiles", exist_ok=True)
base_filename = classificationLevel + modelArchitecture
base_model_path = f"modelSaveFiles/{base_filename}.pt"

# Check if the base filename exists and increment version if needed
if os.path.exists(base_model_path):
    version = 1
    while True:
        versioned_filename = f"{base_filename}_{version}"
        versioned_model_path = f"modelSaveFiles/{versioned_filename}.pt"
        if not os.path.exists(versioned_model_path):
            modelSaveFile = versioned_model_path
            print(f"Model will be saved as: {versioned_filename}.pt (version {version})")
            break
        version += 1
else:
    modelSaveFile = base_model_path
    print(f"Model will be saved as: {base_filename}.pt (first version)")

# Extract the versioned filename (without path and extension) for consistent naming
versioned_model_name = os.path.splitext(os.path.basename(modelSaveFile))[0]

# Start wandb logging with versioned name
wandb.init(
    project="All Interaction Features (On-the-fly)", 
    name=versioned_model_name,
    config={
        "epochs": maxEpochs,
        "batch_size": batchSize,
        "model": modelArchitecture,
        "model_type": modelType,
        "device": device,
        "convergence_threshold": convergence_threshold,
        "load_model": load,
        "max_graphs_per_class": max_graphs_per_class,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 5,
        "scheduler_min_lr": 1e-7
    }
)

# Define custom metrics to plot against time
wandb.define_metric("Time_Minutes")
wandb.define_metric("Training Loss", step_metric="Time_Minutes")
wandb.define_metric("Validation Loss", step_metric="Time_Minutes")
wandb.define_metric("Training Accuracy", step_metric="Time_Minutes")
wandb.define_metric("Validation Accuracy", step_metric="Time_Minutes")
wandb.define_metric("Gradient Norm", step_metric="Time_Minutes")
wandb.define_metric("Learning Rate", step_metric="Time_Minutes")

# Initialize and start continuous memory monitoring
memory_monitor = ContinuousMemoryMonitor(interval=20)  # Log every 20 seconds 
memory_monitor.start()

# Log initial memory state
log_gpu_memory("initial_state")
print("wandb logging initialized successfully!")
print("Continuous memory monitoring active - check wandb dashboard for real-time charts")

# Create dataset with k=3
print("Creating dataset...")
memory_monitor.set_stage("preprocessing")
dataset = MultiGraphDataset(jetNames, 3, loadFromDisk=False, device=device, use_gpu=True)
dataset.process()

log_gpu_memory()

if maxEpochs != 0:
    print("Creating data splits...")
    train, val, test = RandomSplitter().train_val_test_split(dataset, frac_train=0.8, frac_test=0.1, 
                                                         frac_val=0.1, random_state=42)
else:
    train = dataset

if maxEpochs != 0:
    trainLoader = DataLoader(train, batch_size=batchSize, shuffle=True, collate_fn=collateFunction, drop_last=True, num_workers=0)
    validationLoader = DataLoader(val, batch_size=batchSize, shuffle=True, collate_fn=collateFunction, drop_last=True, num_workers=0)
    testLoader = DataLoader(test, batch_size=batchSize, shuffle=True, collate_fn=collateFunction, drop_last=True, num_workers=0)
else:
    testLoader = DataLoader(train, batch_size=batchSize, shuffle=True, collate_fn=collateFunction, drop_last=True)



in_feats = 16
hidden_feats = 64
out_feats = len(jetNames) # Number of output classes

# Update wandb config with model details
wandb.config.update({
    "in_feats": in_feats,
    "hidden_feats": hidden_feats,
    "out_feats": out_feats,
    "model_save_file": modelSaveFile,
    "architecture_type": "matrix_based_conv_max_pooling_with_scheduler",
    "feature_combination": "vertical_stacking",  # [batch_size, 4, 64] instead of [batch_size, 256]
    "conv1d_channels": "4->4->4",
    "pooling_type": "channel_max_pooling"
})

chebFilterSize = 16

if modelType == "DGCNN":
    # Create 4 feature extractors for each graph type
    # Architecture: Each GNN extracts [batch_size, 64] features per graph type
    # Then stacked vertically to create [batch_size, 4, 64] matrix for conv+pooling
    print("Creating models...")
    model_delta = GNNFeatureExtractor(in_feats, hidden_feats, chebFilterSize)
    model_kT = GNNFeatureExtractor(in_feats, hidden_feats, chebFilterSize)
    model_mSquare = GNNFeatureExtractor(in_feats, hidden_feats, chebFilterSize)
    model_Z = GNNFeatureExtractor(in_feats, hidden_feats, chebFilterSize)
    

    classifier = Classifier(hidden_feats, hidden_feats, out_feats)

else:
    print("Invalid selection. Only DGCNN supported for multi-graph!")
    exit()

# Move models to device
model_delta.to(device)
model_kT.to(device)
model_Z.to(device)
model_mSquare.to(device)
classifier.to(device)

# Create a list of all models for easier handling
all_models = [model_delta, model_kT, model_Z, model_mSquare, classifier]

# Watch only the classifier to reduce memory overhead
wandb.watch(classifier, log='gradients', log_freq=100)

# Define the loss function and optimizer for all models
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW([
    {'params': model_delta.parameters()},
    {'params': model_kT.parameters()},
    {'params': model_Z.parameters()},
    {'params': model_mSquare.parameters()},
    {'params': classifier.parameters()}
], lr=1e-3)

# Add ReduceLROnPlateau scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7
)

trainingLossTracker = []
trainingAccuracyTracker = []
validationLossTracker = []
validationAccuracyTracker = []

bestLoss = float('inf')
epochs_without_improvement = 0
epochsTillQuit = 10

# Memory Fix
def cleanup_tensors(*tensors):
    """Helper function to properly delete tensors and clear cache"""
    for tensor in tensors:
        if tensor is not None and hasattr(tensor, 'data'):
            try:
                # Only set data to None if it's a valid tensor
                if tensor.data is not None:
                    tensor.data = None
            except:
                pass  # Ignore any errors when setting data to None
        del tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# Log training start
if maxEpochs > 0:
    print("Starting training...")
    trainingStartTime = time.time()

    # Train the model
    for epoch in range(maxEpochs):
        epochStartTime = time.time()
        memory_monitor.set_stage("training", epoch + 1)
        runningLoss = 0
        totalCorrectPredictions = 0
        totalSamples = 0
        valTotalCorrectPredictions = 0
        valTotalSamples = 0
    
        # Set all models to training mode
        for model in all_models:
            model.train()
        
        for batchIndex, (graphs, labels) in tqdm(enumerate(trainLoader), total=len(trainLoader), leave=False):
            # Unpack graphs
            graph_delta, graph_kT, graph_Z, graph_mSquare = graphs
            labels = labels.to(device).long()

            # Clear gradients before forward pass
            optimizer.zero_grad()

            # Get embeddings from each graph type
            hg_delta = model_delta(graph_delta)
            hg_kT = model_kT(graph_kT)
            hg_Z = model_Z(graph_Z)
            hg_mSquare = model_mSquare(graph_mSquare)
            
            # Stack embeddings vertically to create matrix [batch_size, 4, 64]
            concatenated_features = torch.stack([hg_delta, hg_kT, hg_Z, hg_mSquare], dim=1)
            
            # Get final logits from classifier
            logits = classifier(concatenated_features)
            
            # Calculate loss and do backpropagation
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            # Update running loss
            runningLoss += loss.item()

            # Compute accuracy
            with torch.no_grad():
                predictions = logits.argmax(dim=1)
                batchCorrectPredictions = (predictions == labels).sum().item()
                batchTotalSamples = labels.numel()

            totalCorrectPredictions += batchCorrectPredictions
            totalSamples += batchTotalSamples

            # Clean up only the unpacked variables
            del graphs

        # Compute epoch statistics
        epochLoss = runningLoss / len(trainLoader)
        trainingLossTracker.append(epochLoss)
    
        epochAccuracy = totalCorrectPredictions / totalSamples
        trainingAccuracyTracker.append(epochAccuracy)

        # COMPLETE TRAINING EPOCH CLEANUP
        torch.cuda.empty_cache()
        gc.collect()
        log_gpu_memory(f"After training epoch {epoch+1}")

        memory_monitor.set_stage("validation", epoch + 1)
        # Validation
        for model in all_models:
            model.eval()
        validationLoss = 0.0

        with torch.no_grad():
            for val_batch_idx, (graphs, labels) in tqdm(enumerate(validationLoader), total=len(validationLoader), leave=False):                
                # Unpack graphs
                graph_delta, graph_kT, graph_Z, graph_mSquare = graphs
                labels = labels.to(device).long()

                # Get embeddings and logits
                hg_delta = model_delta(graph_delta)
                hg_kT = model_kT(graph_kT)
                hg_Z = model_Z(graph_Z)
                hg_mSquare = model_mSquare(graph_mSquare)
                
                concatenated_features = torch.stack([hg_delta, hg_kT, hg_Z, hg_mSquare], dim=1)
                logits = classifier(concatenated_features)
                
                loss = criterion(logits, labels)
                validationLoss += loss.item()
                
                predictions = logits.argmax(dim=1)
                batchCorrectPredictions = (predictions == labels).sum().item()
                batchTotalSamples = labels.numel()
                
                valTotalCorrectPredictions += batchCorrectPredictions
                valTotalSamples += batchTotalSamples
                
                # Clean up only the unpacked variables (no heavy cleanup during validation cycle)
                del graphs, labels
                
        avgValidationLoss = validationLoss / len(validationLoader)
        validationLossTracker.append(avgValidationLoss)
    
        validationAccuracy = valTotalCorrectPredictions / valTotalSamples
        validationAccuracyTracker.append(validationAccuracy)

        # Step the scheduler with validation loss
        scheduler.step(avgValidationLoss)

        # COMPLETE VALIDATION EPOCH CLEANUP
        torch.cuda.empty_cache()
        gc.collect()
        log_gpu_memory(f"After validation epoch {epoch+1}")

        # Check for convergence and ONLY save when improved
        if avgValidationLoss < bestLoss - convergence_threshold:
            bestLoss = avgValidationLoss
            bestStateDict = {
                'model_delta': model_delta.state_dict(),
                'model_kT': model_kT.state_dict(),
                'model_Z': model_Z.state_dict(),
                'model_mSquare': model_mSquare.state_dict(),
                'classifier': classifier.state_dict()
            }
            # Save model checkpoint only when improved
            torch.save({
                'model_delta': model_delta.state_dict(),
                'model_kT': model_kT.state_dict(),
                'model_Z': model_Z.state_dict(),
                'model_mSquare': model_mSquare.state_dict(),
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, modelSaveFile)
            print(f'Saved Models to file {modelSaveFile} at epoch {epoch+1}')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # EPOCH COMPLETE - Final cleanup for this epoch
        torch.cuda.empty_cache()
        gc.collect()

        # Log gradient norm
        grad_norm = 0
        for model in all_models:
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item()**2
        grad_norm = grad_norm ** 0.5
        
        # Print training and validation losses
        epochTime = time.time() - epochStartTime
        totalTime = time.time() - trainingStartTime
        epochTimeMinutes = epochTime / 60.0
        totalTimeMinutes = totalTime / 60.0
        totalTimeHours = totalTime / 3600.0
        print(f"Epoch {epoch + 1} - Training Loss={epochLoss:.4f} - Validation Loss={avgValidationLoss:.4f} - Training Accuracy={epochAccuracy:.4f} - Validation Accuracy={validationAccuracy:.4f} - Time={epochTimeMinutes:.2f}min - Total Time={totalTimeHours:.2f}h")
        
        wandb.log({
            "Epoch": epoch + 1,
            "Training Loss": epochLoss,
            "Validation Loss": avgValidationLoss,
            "Training Accuracy": epochAccuracy,
            "Validation Accuracy": validationAccuracy,
            "Gradient Norm": grad_norm,
            "Learning Rate": optimizer.param_groups[0]['lr'],
            "Time_Minutes": totalTimeMinutes,
        })

        # SET BACK TO TRAINING after validation
        memory_monitor.set_stage("training", epoch + 1)
        
        # Check convergence criteria
        if epochs_without_improvement >= epochsTillQuit:
            print(f'Convergence achieved at epoch {epoch + 1}. Stopping training.')
            break

        

if maxEpochs != 0:
    torch.save(bestStateDict, modelSaveFile)

# Create directory for saving plots using versioned name
# Create plots directory first, then model-specific subdirectory
plots_dir = 'plots'
model_plots_dir = f'{plots_dir}/{versioned_model_name}'
try:
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(model_plots_dir, exist_ok=True)
    imageSavePath = model_plots_dir
    print(f"Created plots directory structure: {imageSavePath}")
except Exception as e:
    print(f"Error creating directories: {e}")

if maxEpochs != 0:
    print("Creating training plots...")
    
    # Plot training loss
    plt.figure()
    plt.plot(range(len(trainingLossTracker)), trainingLossTracker)
    plt.title(f'{versioned_model_name} Training Loss Graph')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(f'{imageSavePath}/Training Loss.png')
    plt.close()

    # Plot training accuracy
    plt.figure()
    plt.plot(range(len(trainingAccuracyTracker)), trainingAccuracyTracker)
    plt.title(f'{versioned_model_name} Training Accuracy Graph')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(f'{imageSavePath}/Training Accuracy.png')
    plt.close()

    # Plot validation loss
    plt.figure()
    plt.plot(range(len(validationLossTracker)), validationLossTracker)
    plt.title(f'{versioned_model_name} Validation Loss Graph')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(f'{imageSavePath}/Validation Loss.png')
    plt.close()

    # Plot validation accuracy
    plt.figure()
    plt.plot(range(len(validationAccuracyTracker)), validationAccuracyTracker)
    plt.title(f'{versioned_model_name} Validation Accuracy Graph')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(f'{imageSavePath}/Validation Accuracy.png')
    plt.close()

# GradCAM for Graph Type Importance
def compute_graph_type_importance(model_delta, model_kT, model_Z, model_mSquare, classifier,
                                   graph_delta, graph_kT, graph_Z, graph_mSquare, target_class=None):
    """
    Compute importance scores for each of the 4 graph types using GradCAM approach.
    Returns importance scores for [delta, kT, Z, mSquare]
    """
    # Enable gradient computation
    model_delta.eval()
    model_kT.eval()
    model_Z.eval()
    model_mSquare.eval()
    classifier.eval()

    # Get embeddings with gradient tracking
    hg_delta = model_delta(graph_delta)
    hg_kT = model_kT(graph_kT)
    hg_Z = model_Z(graph_Z)
    hg_mSquare = model_mSquare(graph_mSquare)

    # Stack embeddings
    concatenated_features = torch.stack([hg_delta, hg_kT, hg_Z, hg_mSquare], dim=1)

    # Get logits
    logits = classifier(concatenated_features)

    # If target_class not specified, use predicted class
    if target_class is None:
        target_class = logits.argmax(dim=1)

    # Compute gradients for each graph type
    importances = []

    for hg in [hg_delta, hg_kT, hg_Z, hg_mSquare]:
        # Zero gradients
        if hg.grad is not None:
            hg.grad.zero_()

        # Compute gradient of target class score w.r.t. this graph type's features
        hg.retain_grad()

    # Recompute forward pass to enable gradient flow
    concatenated_features = torch.stack([hg_delta, hg_kT, hg_Z, hg_mSquare], dim=1)
    logits = classifier(concatenated_features)

    # Get score for target class (average across batch)
    if isinstance(target_class, torch.Tensor):
        score = logits.gather(1, target_class.unsqueeze(1)).mean()
    else:
        score = logits[:, target_class].mean()

    # Backward to compute gradients
    score.backward()

    # Compute importance as gradient * activation (GradCAM style)
    for hg in [hg_delta, hg_kT, hg_Z, hg_mSquare]:
        if hg.grad is not None:
            # Importance = mean(|gradient * activation|) across batch and features
            importance = (hg.grad.abs() * hg.abs()).mean().item()
            importances.append(importance)
        else:
            importances.append(0.0)

    return importances

# Testing and Evaluation
print("Starting testing and evaluation...")
# ADD THIS LINE before testing
memory_monitor.set_stage("testing")

logitsTracker = []
predictionsTracker = []
targetsTracker = []
gradcam_importances = []  # Store GradCAM importance scores

cfs = np.zeros((out_feats, out_feats))

# Set all models to eval mode
for model in all_models:
    model.eval()

import sklearn

if maxEpochs != 0:
    print("Computing GradCAM importances during testing...")
    for batch_idx, (graphs, labels) in enumerate(tqdm(testLoader, total=len(testLoader), leave=False)):
        # Unpack graphs - graphs is already a tuple of 4 graphs
        graph_delta, graph_kT, graph_Z, graph_mSquare = graphs
        labels = labels.to(device)

        # GRADCAM: Compute graph type importance (requires gradients)
        if batch_idx < 100:  # Compute GradCAM for first 100 batches to save time
            importances = compute_graph_type_importance(
                model_delta, model_kT, model_Z, model_mSquare, classifier,
                graph_delta, graph_kT, graph_Z, graph_mSquare, target_class=None
            )
            gradcam_importances.append(importances)

        # Regular inference without gradients
        with torch.no_grad():
            # Get embeddings and logits
            hg_delta = model_delta(graph_delta)
            hg_kT = model_kT(graph_kT)
            hg_Z = model_Z(graph_Z)
            hg_mSquare = model_mSquare(graph_mSquare)

            concatenated_features = torch.stack([hg_delta, hg_kT, hg_Z, hg_mSquare], dim=1)
            logits = classifier(concatenated_features)

            # Convert to numpy immediately to save memory
            logits_np = logits.detach().cpu().numpy()
            targets_np = labels.detach().cpu().numpy()
            logitsTracker.append(logits_np)
            targetsTracker.append(targets_np)

            predictions = logits.argmax(dim=1)
            predictions_np = predictions.detach().cpu().numpy()
            predictionsTracker.append(predictions_np)

            # Update confusion matrix
            for idx, pred in enumerate(predictions):
                cfs[pred.item()][labels[idx].item()] += 1

        # Clean up only the unpacked variables (no heavy cleanup during testing cycle)
        del graphs, labels, logits_np, targets_np, predictions_np
else:
    # Also set testing stage for maxEpochs == 0 case
    memory_monitor.set_stage("testing")
    with torch.no_grad():
        for batch_idx, (graphs, labels) in tqdm(enumerate(testLoader), total=len(testLoader), leave=False):
            # Unpack graphs
            graph_delta, graph_kT, graph_Z, graph_mSquare = graphs
            labels = labels.to(device)

            # Get embeddings and logits
            hg_delta = model_delta(graph_delta)
            hg_kT = model_kT(graph_kT)
            hg_Z = model_Z(graph_Z)
            hg_mSquare = model_mSquare(graph_mSquare)
            
            concatenated_features = torch.stack([hg_delta, hg_kT, hg_Z, hg_mSquare], dim=1)
            logits = classifier(concatenated_features)
            
            # Convert to lists immediately
            logitsTracker.extend(logits.detach().cpu().tolist())
            targetsTracker.extend(labels.detach().cpu().tolist())

            predictions = logits.argmax(dim=1)
            predictionsTracker.extend(predictions.detach().cpu().tolist())
            
            # Update confusion matrix
            for idx, pred in enumerate(predictions):
                cfs[pred.item()][labels[idx].item()] += 1
            
            # Clean up only the unpacked variables (no heavy cleanup during testing cycle)
            del graphs, labels
# COMPLETE TESTING PHASE CLEANUP
torch.cuda.empty_cache()
gc.collect()
log_gpu_memory("After complete testing phase")

# Save metrics using versioned name
os.makedirs('metrics', exist_ok=True)
logitsTrackerFile = f'metrics/{versioned_model_name}-Logits.pkl'
targetsTrackerFile = f'metrics/{versioned_model_name}-Targets.pkl'
predictionsTrackerFile = f'metrics/{versioned_model_name}-Predictions.pkl'

with open(logitsTrackerFile, 'wb') as f:
    pickle.dump(logitsTracker, f)

with open(targetsTrackerFile, 'wb') as f:
    pickle.dump(targetsTracker, f)

with open(predictionsTrackerFile, 'wb') as f:
    pickle.dump(predictionsTracker, f)

# Clear large tracking lists after saving
del logitsTracker, predictionsTracker, targetsTracker
torch.cuda.empty_cache()
gc.collect()

# Force multiple garbage collection cycles to break reference cycles
for _ in range(3):
    gc.collect()

print("Creating evaluation plots...")

# Plot confusion matrix
fig = plt.gcf()
fig.set_size_inches(15, 15)

ax = sns.heatmap(cfs/np.sum(cfs), annot=True, cmap='Blues')
ax.set_title(f'{versioned_model_name} Confusion Matrix')
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')

print(cfs/np.sum(cfs))
plt.savefig(f'{imageSavePath}/Confusion Matrix.png')
plt.close()

# Calculate metrics BEFORE deleting confusion matrix
def calculateConfusionMetrics(confusion_matrix):
    num_classes = len(confusion_matrix)
    metrics = []

    for i in range(num_classes):
        true_positive = confusion_matrix[i][i]
        false_positive = np.sum(confusion_matrix[:, i]) - true_positive
        false_negative = np.sum(confusion_matrix[i, :]) - true_positive
        true_negative = np.sum(confusion_matrix) - true_positive - false_positive - false_negative

        accuracy = (true_positive + true_negative) / np.sum(confusion_matrix)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0

        metrics.append([accuracy, precision, recall, specificity])

    return metrics

metrics = calculateConfusionMetrics(cfs)

classLabels = jetNames
metricsDF = pd.DataFrame(metrics, columns=['Accuracy', 'Precision', 'Recall', 'Specificity'], index=classLabels)

# Calculate micro and macro averages
microAvg = metricsDF.mean(axis=0)
macroAvg = metricsDF.mean(axis=0)

# Add micro and macro averages to the DataFrame
metricsDF.loc['Micro Avg'] = microAvg
metricsDF.loc['Macro Avg'] = macroAvg

# Print the metrics table
print(metricsDF)

# Clear confusion matrix to free memory
del cfs
gc.collect()

# ROC-AUC Curve
from sklearn.metrics import roc_curve, auc
import scikitplot as skplt

# Load data back from files to avoid keeping large arrays in memory
with open(logitsTrackerFile, 'rb') as f:
    logitsTracker = pickle.load(f)

with open(targetsTrackerFile, 'rb') as f:
    targetsTracker = pickle.load(f)

if maxEpochs != 0:
    rocLogits = np.concatenate(logitsTracker, axis=0)
    rocTargets = np.concatenate(targetsTracker, axis=0)
else:
    rocLogits = np.array(logitsTracker)
    rocTargets = np.array(targetsTracker)

skplt.metrics.plot_roc_curve(rocTargets, rocLogits, figsize=(8, 6), title=f'{versioned_model_name} ROC-AUC Curve')
plt.savefig(f'{imageSavePath}/ROC-AUC.png')
plt.close()

# Clear ROC data after use
del rocLogits, rocTargets, logitsTracker, targetsTracker
torch.cuda.empty_cache()
gc.collect()

# Force cleanup to break any remaining reference cycles
for _ in range(2):
    gc.collect()

# Network Flow Diagram Function
def create_network_flow_diagram(importance_percentances, save_path):
    """
    Create a network architecture flow diagram showing gradient flow and importance
    Uses gradient-based coloring where brightness represents importance
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    graph_types = ['Delta', 'kT', 'Z', 'mSquare']

    # Use gradient-based coloring: map importance to color intensity
    # Normalize importances to 0-1 range for color mapping
    norm_importances = importance_percentances / importance_percentances.max()

    # Use a colormap (e.g., 'YlOrRd' for yellow to red, or 'Blues', 'Reds', 'Greens')
    cmap = plt.cm.get_cmap('YlOrRd')  # Yellow (low) to Red (high)
    colors = [cmap(norm_imp) for norm_imp in norm_importances]

    # Column positions
    col1_x = 1.5  # Input graphs
    col2_x = 3.5  # GNN extractors
    col3_x = 5.5  # Features (64-dim)
    col4_x = 7.0  # Concatenated
    col5_x = 8.5  # Classifier

    # Draw input graphs
    y_positions = [7.5, 5.5, 3.5, 1.5]
    for i, (graph_type, color, y_pos, importance) in enumerate(zip(graph_types, colors, y_positions, importance_percentances)):
        # Input box
        rect = plt.Rectangle((col1_x-0.3, y_pos-0.2), 0.6, 0.4,
                             facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        ax.text(col1_x, y_pos, f'{graph_type}\nGraph', ha='center', va='center',
               fontsize=9, fontweight='bold')

        # GNN Feature Extractor
        rect2 = plt.Rectangle((col2_x-0.4, y_pos-0.3), 0.8, 0.6,
                              facecolor=color, edgecolor='black', linewidth=2, alpha=0.5)
        ax.add_patch(rect2)
        ax.text(col2_x, y_pos, f'GNN\nExtractor', ha='center', va='center',
               fontsize=8, fontweight='bold')

        # Feature vector
        rect3 = plt.Rectangle((col3_x-0.3, y_pos-0.2), 0.6, 0.4,
                              facecolor=color, edgecolor='black', linewidth=2, alpha=0.6)
        ax.add_patch(rect3)
        ax.text(col3_x, y_pos, f'hg_{graph_type}\n[64]', ha='center', va='center',
               fontsize=8, fontweight='bold')

        # Arrow from input to GNN
        arrow_width = 0.01 + (importance / 100) * 0.03
        ax.annotate('', xy=(col2_x-0.4, y_pos), xytext=(col1_x+0.3, y_pos),
                   arrowprops=dict(arrowstyle='->', lw=2, color=color, alpha=0.8))

        # Arrow from GNN to features
        ax.annotate('', xy=(col3_x-0.3, y_pos), xytext=(col2_x+0.4, y_pos),
                   arrowprops=dict(arrowstyle='->', lw=2, color=color, alpha=0.8))

        # Arrow from features to concatenation with width based on importance
        ax.annotate('', xy=(col4_x-0.3, 4.5), xytext=(col3_x+0.3, y_pos),
                   arrowprops=dict(arrowstyle='->', lw=1+importance/10, color=color, alpha=0.7))

        # Add importance label on arrow
        mid_x = (col3_x + col4_x) / 2
        mid_y = (y_pos + 4.5) / 2
        ax.text(mid_x, mid_y, f'{importance:.1f}%', fontsize=10,
               fontweight='bold', color=color,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, linewidth=2))

    # Concatenation box
    rect_concat = plt.Rectangle((col4_x-0.3, 4.0), 0.6, 1.0,
                               facecolor='lightgray', edgecolor='black', linewidth=3)
    ax.add_patch(rect_concat)
    ax.text(col4_x, 4.5, 'Stack\n[4×64]', ha='center', va='center',
           fontsize=10, fontweight='bold')

    # Classifier
    rect_classifier = plt.Rectangle((col5_x-0.4, 3.5), 0.8, 2.0,
                                    facecolor='#95E1D3', edgecolor='black', linewidth=3)
    ax.add_patch(rect_classifier)
    ax.text(col5_x, 4.5, 'Classifier\n(Conv1D\n+\nMaxPool\n+\nFC)', ha='center', va='center',
           fontsize=9, fontweight='bold')

    # Arrow to classifier
    ax.annotate('', xy=(col5_x-0.4, 4.5), xytext=(col4_x+0.3, 4.5),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    # Output
    ax.text(col5_x+1.0, 4.5, 'Output\nLogits', ha='center', va='center',
           fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', edgecolor='black', linewidth=2))
    ax.annotate('', xy=(col5_x+0.7, 4.5), xytext=(col5_x+0.4, 4.5),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    # Add gradient flow annotation (backward pass)
    ax.text(5.0, 9.2, 'FORWARD PASS →', fontsize=12, fontweight='bold', color='green')
    ax.text(5.0, 0.3, '← GRADIENT FLOW (∂L/∂W)', fontsize=12, fontweight='bold', color='red')

    # Add gradient arrows (backward)
    for i, (y_pos, importance, color) in enumerate(zip(y_positions, importance_percentages, colors)):
        ax.annotate('', xy=(col3_x+0.3, y_pos-0.3), xytext=(col4_x-0.3, 4.0),
                   arrowprops=dict(arrowstyle='<-', lw=1+importance/20,
                                 color='red', alpha=0.4, linestyle='dashed'))

    # Title
    ax.text(5.0, 9.7, 'Network Architecture & Gradient Flow Diagram',
           fontsize=14, fontweight='bold', ha='center')

    # Add colorbar to show importance scale
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=importance_percentances.min(), vmax=importance_percentances.max()))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Importance (%)', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Sankey-style Flow Diagram
def create_contribution_flow_diagram(importance_percentages, save_path):
    """
    Create a flow diagram showing contribution of each graph type
    Uses gradient-based coloring where brightness represents importance
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    graph_types = ['Delta', 'kT', 'Z', 'mSquare']

    # Use gradient-based coloring: map importance to color intensity
    norm_importances = importance_percentages / importance_percentages.max()
    cmap = plt.cm.get_cmap('YlOrRd')  # Yellow (low) to Red (high)
    colors = [cmap(norm_imp) for norm_imp in norm_importances]

    # Source boxes (left side)
    y_positions = [7.5, 5.5, 3.5, 1.5]
    left_x = 1.5
    right_x = 8.0

    for graph_type, color, y_pos, importance in zip(graph_types, colors, y_positions, importance_percentages):
        # Source box
        height = importance / 25  # Scale based on importance
        rect = plt.Rectangle((left_x, y_pos-height/2), 1.0, height,
                            facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(left_x-0.3, y_pos, f'{graph_type}', ha='right', va='center',
               fontsize=11, fontweight='bold')

        # Flow to prediction (variable width based on importance)
        # Create flowing polygon
        flow_width = importance / 15
        x_coords = [left_x+1.0, right_x-1.0, right_x-1.0, left_x+1.0]
        y_coords = [y_pos, 5.0+flow_width/2, 5.0-flow_width/2, y_pos]

        poly = plt.Polygon(list(zip(x_coords, y_coords)),
                          facecolor=color, edgecolor=color, alpha=0.3, linewidth=1)
        ax.add_patch(poly)

        # Add percentage label
        mid_x = (left_x + right_x) / 2
        ax.text(mid_x, (y_pos + 5.0)/2, f'{importance:.1f}%',
               fontsize=12, fontweight='bold', color=color,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor=color, linewidth=2))

    # Prediction box (right side)
    rect_pred = plt.Rectangle((right_x-1.0, 3.5), 1.5, 3.0,
                             facecolor='#F38181', edgecolor='black', linewidth=3)
    ax.add_patch(rect_pred)
    ax.text(right_x-0.25, 5.0, 'Final\nPrediction', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')

    # Title
    ax.text(5.0, 9.0, 'Graph Type Contribution Flow to Prediction',
           fontsize=14, fontweight='bold', ha='center')

    # Legend
    ax.text(5.0, 0.5, 'Flow width and color intensity represent relative importance',
           fontsize=10, ha='center', style='italic', color='gray')

    # Add colorbar
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=importance_percentages.min(), vmax=importance_percentages.max()))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Importance (%)', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# GradCAM Visualization
if len(gradcam_importances) > 0:
    print("Creating GradCAM visualization...")

    # Convert to numpy array: [num_batches, 4]
    gradcam_array = np.array(gradcam_importances)

    # Compute average importance for each graph type
    avg_importances = gradcam_array.mean(axis=0)

    # Normalize to percentages
    total_importance = avg_importances.sum()
    importance_percentages = (avg_importances / total_importance) * 100

    graph_types = ['Delta', 'kT', 'Z', 'mSquare']

    # Create bar plot with gradient-based colors
    plt.figure(figsize=(10, 6))

    # Map importance to colors using gradient
    norm_importances = importance_percentages / importance_percentages.max()
    cmap = plt.cm.get_cmap('YlOrRd')
    bar_colors = [cmap(norm_imp) for norm_imp in norm_importances]

    bars = plt.bar(graph_types, importance_percentages, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, val in zip(bars, importance_percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title(f'{versioned_model_name} - Graph Type Importance (GradCAM)', fontsize=14, fontweight='bold')
    plt.xlabel('Graph Type', fontsize=12)
    plt.ylabel('Relative Importance (%)', fontsize=12)
    plt.ylim(0, max(importance_percentages) * 1.2)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Add colorbar to bar plot
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=importance_percentages.min(), vmax=importance_percentages.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.02)
    cbar.set_label('Importance (%)', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{imageSavePath}/GradCAM_Graph_Type_Importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create heatmap showing importance across batches
    plt.figure(figsize=(12, 8))

    # Normalize each batch to show relative importance
    normalized_importances = gradcam_array / gradcam_array.sum(axis=1, keepdims=True)

    # Plot heatmap
    sns.heatmap(normalized_importances.T, cmap='YlOrRd', cbar_kws={'label': 'Normalized Importance'},
                yticklabels=graph_types, xticklabels=False, annot=False)
    plt.title(f'{versioned_model_name} - Graph Type Importance Across Batches', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Index', fontsize=12)
    plt.ylabel('Graph Type', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{imageSavePath}/GradCAM_Heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create network flow diagram
    print("Creating network flow diagram...")
    create_network_flow_diagram(importance_percentages, f'{imageSavePath}/Network_Flow_Diagram.png')

    # Create contribution flow diagram
    print("Creating contribution flow diagram...")
    create_contribution_flow_diagram(importance_percentages, f'{imageSavePath}/Contribution_Flow_Diagram.png')

    # Print results
    print("\n" + "="*60)
    print("GradCAM ANALYSIS: Graph Type Importance")
    print("="*60)
    for graph_type, importance, percentage in zip(graph_types, avg_importances, importance_percentages):
        print(f"{graph_type:12s}: {importance:.6f} ({percentage:.2f}%)")
    print("="*60)

    # Log to wandb
    gradcam_table = wandb.Table(
        data=[[gt, imp, pct] for gt, imp, pct in zip(graph_types, avg_importances, importance_percentages)],
        columns=["Graph Type", "Importance Score", "Percentage"]
    )

    wandb.log({
        "GradCAM/Graph_Type_Importance": wandb.Image(f"{imageSavePath}/GradCAM_Graph_Type_Importance.png"),
        "GradCAM/Importance_Heatmap": wandb.Image(f"{imageSavePath}/GradCAM_Heatmap.png"),
        "GradCAM/Network_Flow_Diagram": wandb.Image(f"{imageSavePath}/Network_Flow_Diagram.png"),
        "GradCAM/Contribution_Flow_Diagram": wandb.Image(f"{imageSavePath}/Contribution_Flow_Diagram.png"),
        "GradCAM/Importance_Table": gradcam_table,
        "GradCAM/Delta_Importance": importance_percentages[0],
        "GradCAM/kT_Importance": importance_percentages[1],
        "GradCAM/Z_Importance": importance_percentages[2],
        "GradCAM/mSquare_Importance": importance_percentages[3],
    })

    # Clean up
    del gradcam_array, avg_importances, importance_percentages
    gc.collect()

wandb.log({
    "Results/micro_avg_accuracy": microAvg['Accuracy'],
    "Results/micro_avg_precision": microAvg['Precision'],
    "Results/micro_avg_recall": microAvg['Recall'],
    "Results/micro_avg_specificity": microAvg['Specificity'],
    "Confusion Matrix": wandb.Image(f"{imageSavePath}/Confusion Matrix.png"),
    "ROC-AUC Curve": wandb.Image(f"{imageSavePath}/ROC-AUC.png"),
    "Confusion Matrix Table": wandb.Table(dataframe=metricsDF.reset_index())
})

wandb.save(modelSaveFile)
print("Training completed successfully!")
wandb.finish()

# Stop continuous memory monitoring
memory_monitor.stop()

# Final cleanup
del dataset
if 'train' in locals(): del train
if 'val' in locals(): del val
if 'test' in locals(): del test
if 'trainLoader' in locals(): del trainLoader
if 'validationLoader' in locals(): del validationLoader
if 'testLoader' in locals(): del testLoader

torch.cuda.empty_cache()
gc.collect()

# Final memory summary
log_gpu_memory("final_summary")

print("Memory cleanup completed.")