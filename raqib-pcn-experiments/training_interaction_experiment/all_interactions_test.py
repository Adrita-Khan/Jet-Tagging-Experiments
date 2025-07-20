import numpy as np
import pandas as pd
from operator import truth
import awkward as ak
import torch
from tqdm import tqdm
import os
import dgl
import pickle
import wandb
import GPUtil
import gc
import json
import argparse

import argparse

# Add argument parser
parser = argparse.ArgumentParser(
    description='Multi-Graph Neural Network Testing',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Example usage:
  python multi_graph_testing.py --batch_size 256 --device cuda --classification_level MultiGraph --model_architecture PCN-1024

  python multi_graph_testing.py --batch_size 512 --load_model Y --batch_dir /path/to/batches --output_dir results/

  python multi_graph_testing.py --help  # Show this help message
""")
batch_size = 1024
parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size (default: 512)')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use (default: cuda)')
parser.add_argument('--classification_level', type=str, default='All_Interactions', help='Classification level (default: AllInteractions)')
parser.add_argument('--model_architecture', type=str, default=f'PCN-{batch_size}', help='Model architecture name (default: PCN-512)')
parser.add_argument('--model_type', type=str, default='DGCNN', help='Model type (default: DGCNN)')
parser.add_argument('--load_model', type=str, default='Y', choices=['Y', 'N'], help='Load from save file (default: Y)')
parser.add_argument('--batch_dir', type=str, default='batches', help='Directory containing batch files (default: batches)')
parser.add_argument('--wandb_project', type=str, default='Multi-Graph Testing 20M', help='Wandb project name (default: Multi-Graph Testing 20M)')
parser.add_argument('--checkpoint_freq', type=int, default=5, help='Save intermediate results every N batch sets (default: 5)')
parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results (default: auto-generated)')

args = parser.parse_args()

# Print configuration
print("="*60)
print("MULTI-GRAPH TESTING CONFIGURATION")
print("="*60)
print(f"Batch Size: {args.batch_size}")
print(f"Device: {args.device}")
print(f"Classification Level: {args.classification_level}")
print(f"Model Architecture: {args.model_architecture}")
print(f"Model Type: {args.model_type}")
print(f"Load Model: {args.load_model}")
print(f"Batch Directory: {args.batch_dir}")
print(f"Wandb Project: {args.wandb_project}")
print(f"Checkpoint Frequency: {args.checkpoint_freq}")
print(f"Output Directory: {args.output_dir if args.output_dir else 'auto-generated'}")
print("="*60)

class BatchedMultiGraphDataset(dgl.data.DGLDataset):
    def __init__(self, jetNames, k, batchDir='batches', loadFromDisk=False):
        
        self.jetNames = jetNames
        self.batchDir = batchDir
        
        # Get all batch files for each jet type and each graph type
        self.batch_files = {}
        self.sampleCountPerClass = []
        
        # Graph types we need to load (with correct capitalization)
        self.graph_types = ['delta', 'kT', 'mSquare', 'Z']
        
        for jetType in jetNames:
            if type(jetType) != list:
                # Initialize dictionary for this jet type
                self.batch_files[jetType] = {graph_type: [] for graph_type in self.graph_types}
                
                # Find batch files for each graph type
                for graph_type in self.graph_types:
                    jet_dir = os.path.join(batchDir, graph_type, jetType)
                    if os.path.exists(jet_dir):
                        # Look for files with pattern {jetType}_{index}.pkl
                        batch_files = sorted([f for f in os.listdir(jet_dir) 
                                            if f.endswith('.pkl') and f.startswith(jetType + '_')])
                        self.batch_files[jetType][graph_type] = [os.path.join(jet_dir, f) for f in batch_files]
                        print(f'{jetType} - {graph_type}: Found {len(batch_files)} batch files')
                    else:
                        print(f'Warning: Directory not found: {jet_dir}')
                        self.batch_files[jetType][graph_type] = []
                
                # Use delta files as reference for sample count (assuming all graph types have same number of files)
                self.sampleCountPerClass.append(len(self.batch_files[jetType]['delta']))
            else:
                # Handle list of jet types (if needed)
                combined_files = {graph_type: [] for graph_type in self.graph_types}
                for item in jetType:
                    for graph_type in self.graph_types:
                        jet_dir = os.path.join(batchDir, graph_type, item)
                        if os.path.exists(jet_dir):
                            # Look for files with pattern {item}_{index}.pkl
                            batch_files = sorted([f for f in os.listdir(jet_dir) 
                                                if f.endswith('.pkl') and f.startswith(item + '_')])
                            item_files = [os.path.join(jet_dir, f) for f in batch_files]
                            combined_files[graph_type].extend(item_files)
                
                self.batch_files[str(jetType)] = combined_files
                self.sampleCountPerClass.append(len(combined_files['delta']))  # Use delta as reference
        
        # Create a flat list of all batch files with their labels
        # Each entry contains paths to all 4 graph types for the same batch
        self.all_batch_files = []
        label = 0
        for jetType in jetNames:
            jet_key = jetType if type(jetType) != list else str(jetType)
            
            # Get the number of batch files (using delta as reference)
            num_batches = len(self.batch_files[jet_key]['delta'])
            
            for batch_idx in range(num_batches):
                batch_paths = {}
                for graph_type in self.graph_types:
                    if batch_idx < len(self.batch_files[jet_key][graph_type]):
                        batch_paths[graph_type] = self.batch_files[jet_key][graph_type][batch_idx]
                    else:
                        print(f"Warning: Missing {graph_type} file for {jetType} batch {batch_idx}")
                        batch_paths[graph_type] = None
                
                self.all_batch_files.append((batch_paths, label))
            label += 1
        
        print(f'Total batch sets to process: {len(self.all_batch_files)}')

    def process(self):
        return
    
    def get_all_batch_files(self):
        """Get list of all batch file sets with their labels"""
        return self.all_batch_files
                
    def __getitem__(self, idx):
        # This method is not used in our batch processing approach
        raise NotImplementedError("Use get_all_batch_files() for memory-efficient processing")

    def __len__(self):
        return len(self.all_batch_files)

# Checkpoint management functions
def save_checkpoint(checkpoint_file, processed_files, results):
    """Save checkpoint with processed files and accumulated results"""
    checkpoint_data = {
        'processed_files': processed_files,
        'logitsTracker': results['logitsTracker'],
        'predictionsTracker': results['predictionsTracker'],
        'targetsTracker': results['targetsTracker'],
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'total_processed': results['total_processed']
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f)
    print(f"Checkpoint saved: {len(processed_files)} batch sets processed")

def load_checkpoint(checkpoint_file):
    """Load checkpoint and return processed files and results"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        
        results = {
            'logitsTracker': checkpoint_data['logitsTracker'],
            'predictionsTracker': checkpoint_data['predictionsTracker'],
            'targetsTracker': checkpoint_data['targetsTracker'],
            'confusion_matrix': np.array(checkpoint_data['confusion_matrix']),
            'total_processed': checkpoint_data['total_processed']
        }
        print(f"Checkpoint loaded: {len(checkpoint_data['processed_files'])} batch sets already processed")
        return checkpoint_data['processed_files'], results
    else:
        print("No checkpoint found, starting from beginning")
        return [], None

# Function to calculate and display current metrics
def calculate_and_display_metrics(cfs, targetsTracker, predictionsTracker, jetNames, total_processed, file_count, total_files):
    """Calculate and display current metrics"""
    print("\n" + "="*80)
    print(f"CURRENT RESULTS AFTER {file_count}/{total_files} BATCH SETS ({total_processed} samples)")
    print("="*80)
    
    # Calculate overall accuracy
    if len(targetsTracker) > 0:
        overall_accuracy = sum(1 for t, p in zip(targetsTracker, predictionsTracker) if t == p) / len(targetsTracker)
        print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    # Calculate metrics for each class
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
    metricsDF = pd.DataFrame(metrics, columns=['Accuracy', 'Precision', 'Recall', 'Specificity'], index=jetNames)
    
    # Calculate micro and macro averages
    microAvg = metricsDF.mean(axis=0)
    macroAvg = metricsDF.mean(axis=0)
    
    # Add micro and macro averages to the DataFrame
    metricsDF.loc['Micro Avg'] = microAvg
    metricsDF.loc['Macro Avg'] = macroAvg
    
    print("\nPer-Class Metrics:")
    print(metricsDF.round(4))
    
    # Display confusion matrix (normalized)
    print("\nNormalized Confusion Matrix:")
    normalized_cfs = cfs / (np.sum(cfs) + 1e-8)  # Add small epsilon to avoid division by zero
    cfs_df = pd.DataFrame(normalized_cfs, index=jetNames, columns=jetNames)
    print(cfs_df.round(4))
    
    # Display class distribution
    class_counts = {}
    for i, jet_name in enumerate(jetNames):
        class_counts[jet_name] = targetsTracker.count(i)
    
    print("\nClass Distribution (samples processed so far):")
    for jet_name, count in class_counts.items():
        percentage = (count / len(targetsTracker)) * 100 if len(targetsTracker) > 0 else 0
        print(f"  {jet_name}: {count} samples ({percentage:.1f}%)")
    
    print("="*80)
    print()
    
    return metricsDF

# Function to save intermediate results
def save_intermediate_results(imageSavePath, cfs, jetNames, classificationLevel, modelArchitecture, 
                            logitsTracker, targetsTracker, predictionsTracker, file_count, total_files):
    """Save intermediate visualizations"""
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Save intermediate confusion matrix
    try:
        fig = plt.gcf()
        fig.set_size_inches(15, 15)
        
        ax = sns.heatmap(cfs/np.sum(cfs), annot=True, cmap='Blues')
        ax.set_title(f'{classificationLevel} {modelArchitecture} Multi-Graph Confusion Matrix (Batch Sets: {file_count}/{total_files})')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        
        plt.savefig(f'{imageSavePath}/Confusion Matrix_Intermediate_{file_count}.png')
        plt.clf()
        
        # Also save as the latest
        plt.figure(figsize=(15, 15))
        ax = sns.heatmap(cfs/np.sum(cfs), annot=True, cmap='Blues')
        ax.set_title(f'{classificationLevel} {modelArchitecture} Multi-Graph Confusion Matrix (Batch Sets: {file_count}/{total_files})')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        plt.savefig(f'{imageSavePath}/Confusion Matrix_Latest.png')
        plt.clf()
        
    except Exception as e:
        print(f"Error saving intermediate confusion matrix: {e}")

# Multi-graph model definitions
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dgl.batch import batch

# GNN feature extractor (returns embeddings)
class GNNFeatureExtractor(nn.Module):
    def __init__(self, in_feats, hidden_feats, k):
        super(GNNFeatureExtractor, self).__init__()
        self.conv1 = dgl.nn.ChebConv(in_feats, hidden_feats, k)
        self.conv2 = dgl.nn.ChebConv(hidden_feats, hidden_feats, k)
        self.conv3 = dgl.nn.ChebConv(hidden_feats, hidden_feats, k)
        
        self.edgeconv1 = dgl.nn.EdgeConv(hidden_feats, hidden_feats)
        self.edgeconv2 = dgl.nn.EdgeConv(hidden_feats, hidden_feats)
        
    def forward(self, g):
        # Apply graph convolutional layers
        h = F.relu(self.conv1(g, g.ndata['feat']))
        h = F.relu(self.edgeconv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.edgeconv2(g, h))
        h = F.relu(self.conv3(g, h))
    
        # Store the node embeddings in the node data dictionary
        g.ndata['h'] = h
    
        # Compute graph-level representations by taking global mean pooling
        hg = dgl.mean_nodes(g, 'h')
        
        return hg

# Classifier class
class Classifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Custom collate function for multiple graphs
def collateFunction(batch):
    graphs_delta = [item['graph_delta'] for item in batch]
    graphs_kT = [item['graph_kT'] for item in batch]
    graphs_mSquare = [item['graph_mSquare'] for item in batch]
    graphs_Z = [item['graph_Z'] for item in batch]
    labels = [item['label'] for item in batch]
    
    batched_graph_delta = dgl.batch(graphs_delta)
    batched_graph_kT = dgl.batch(graphs_kT)
    batched_graph_mSquare = dgl.batch(graphs_mSquare)
    batched_graph_Z = dgl.batch(graphs_Z)
    
    return (batched_graph_delta, batched_graph_kT, batched_graph_mSquare, batched_graph_Z), torch.tensor(labels)

# Process all jetTypes
Higgs = ['HToBB', 'HToCC', 'HToGG', 'HToWW2Q1L', 'HToWW4Q']
Vector = ['WToQQ', 'ZToQQ']
Top = ['TTBar', 'TTBarLep']
QCD = ['ZJetsToNuNu']

# For testing, use the original jet names
testingSet = Top + Vector + QCD + Higgs
jetNames = testingSet
print("Jet types to test:", jetNames)

# Create multi-graph dataset object
dataset = BatchedMultiGraphDataset(jetNames, 3, batchDir='batches', loadFromDisk=False)
dataset.process()

# Since we're doing testing (maxEpochs = 0), we'll use the batched approach
maxEpochs = 0  # Set to 0 for testing
batchSize = args.batch_size

# Device and model configuration
device = args.device
classificationLevel = args.classification_level
modelArchitecture = args.model_architecture
modelType = args.model_type
modelSaveFile = "modelSaveFiles/" + classificationLevel + modelArchitecture + ".pt"
load = True if args.load_model == 'Y' else False

# Checkpoint file
checkpoint_file = f"checkpoints/{classificationLevel}-{modelArchitecture}-multigraph-checkpoint.json"
os.makedirs("checkpoints", exist_ok=True)

in_feats = 16
hidden_feats = 64
out_feats = len(jetNames)  # Number of output classes
chebFilterSize = 16

# Start wandb logging
wandb.init(
    project=args.wandb_project, 
    name=f"{classificationLevel}-{modelArchitecture}-MultiGraph-Testing",
    config={
        "epochs": maxEpochs,
        "batch_size": batchSize,
        "model": modelArchitecture,
        "in_feats": in_feats,
        "hidden_feats": hidden_feats,
        "out_feats": out_feats,
        "device": device,
        "testing_mode": True,
        "graph_types": ["delta", "kT", "mSquare", "Z"]
    }
)

# Initialize multi-graph models
if modelType == "DGCNN":
    # Create 4 feature extractors for each graph type
    model_delta = GNNFeatureExtractor(in_feats, hidden_feats, chebFilterSize)
    model_kT = GNNFeatureExtractor(in_feats, hidden_feats, chebFilterSize)
    model_mSquare = GNNFeatureExtractor(in_feats, hidden_feats, chebFilterSize)
    model_Z = GNNFeatureExtractor(in_feats, hidden_feats, chebFilterSize)
    
    # Final classifier that takes concatenated features
    classifier = Classifier(hidden_feats * 4, hidden_feats, out_feats)
else:
    print("Invalid selection. Only DGCNN supported for multi-graph!")
    exit()

# Move models to device
model_delta.to(device)
model_kT.to(device)
model_mSquare.to(device)
model_Z.to(device)
classifier.to(device)

# Create a list of all models for easier handling
all_models = [model_delta, model_kT, model_mSquare, model_Z, classifier]

if load:
    checkpoint = torch.load(modelSaveFile)
    if isinstance(checkpoint, dict) and 'model_delta' in checkpoint:
        # Load multi-graph model
        model_delta.load_state_dict(checkpoint['model_delta'])
        model_kT.load_state_dict(checkpoint['model_kT'])
        model_mSquare.load_state_dict(checkpoint['model_mSquare'])
        model_Z.load_state_dict(checkpoint['model_Z'])
        classifier.load_state_dict(checkpoint['classifier'])
        print(f"Loaded multi-graph model from {modelSaveFile}")
    else:
        print("Error: Model file doesn't contain multi-graph architecture!")
        exit()

# Set all models to eval mode
for model in all_models:
    model.eval()

# Load checkpoint if exists
processed_files, checkpoint_results = load_checkpoint(checkpoint_file)

# Initialize tracking variables for testing
if checkpoint_results:
    logitsTracker = checkpoint_results['logitsTracker']
    predictionsTracker = checkpoint_results['predictionsTracker']
    targetsTracker = checkpoint_results['targetsTracker']
    cfs = checkpoint_results['confusion_matrix']
    total_processed = checkpoint_results['total_processed']
else:
    logitsTracker = []
    predictionsTracker = []
    targetsTracker = []
    cfs = np.zeros((out_feats, out_feats))
    total_processed = 0

# Create results directory early
if args.output_dir:
    imageSavePath = args.output_dir
else:
    imageSavePath = f'{classificationLevel} {modelArchitecture} MultiGraph'
try:
    os.makedirs(imageSavePath, exist_ok=True)
except Exception as e:
    print(e)

print("Starting multi-graph batch-wise testing...")

# Get all batch file sets
all_batch_files = dataset.get_all_batch_files()
total_files = len(all_batch_files)
files_processed_count = len(processed_files)

# Display initial status if resuming
if files_processed_count > 0:
    print(f"\nResuming from checkpoint. Already processed {files_processed_count}/{total_files} batch sets.")
    metricsDF = calculate_and_display_metrics(cfs, targetsTracker, predictionsTracker, jetNames, 
                                            total_processed, files_processed_count, total_files)

# Process each batch file set one at a time
with torch.no_grad():
    for file_idx, (batch_paths, label) in enumerate(tqdm(all_batch_files, desc="Processing batch file sets")):
        
        # Skip if already processed (use a string representation of batch_paths as key)
        batch_key = str(sorted(batch_paths.items()))
        if batch_key in processed_files:
            continue
        
        try:
            print(f"\nLoading batch file set {file_idx + 1}/{total_files}")
            
            # Load all 4 graph types for this batch
            batch_graphs = {}
            batch_size = None
            
            for graph_type in ['delta', 'kT', 'mSquare', 'Z']:
                if batch_paths[graph_type] is not None:
                    print(f"  Loading {graph_type}: {batch_paths[graph_type]}")
                    with open(batch_paths[graph_type], 'rb') as f:
                        batch_graphs[graph_type] = pickle.load(f)
                    
                    if batch_size is None:
                        batch_size = len(batch_graphs[graph_type])
                    elif len(batch_graphs[graph_type]) != batch_size:
                        print(f"Warning: Size mismatch in {graph_type} ({len(batch_graphs[graph_type])} vs {batch_size})")
                else:
                    print(f"Error: Missing {graph_type} file for batch {file_idx}")
                    continue
            
            # Verify all graph types are loaded
            if len(batch_graphs) != 4:
                print(f"Error: Could not load all graph types for batch {file_idx}")
                continue
            
            print(f"Loaded {batch_size} graphs for each type")
            
            # Create labels for this batch
            batch_labels = [label] * batch_size
            
            # Create dataset for this batch
            batch_data = []
            for i in range(batch_size):
                batch_data.append({
                    'graph_delta': batch_graphs['delta'][i],
                    'graph_kT': batch_graphs['kT'][i],
                    'graph_mSquare': batch_graphs['mSquare'][i],
                    'graph_Z': batch_graphs['Z'][i],
                    'label': batch_labels[i]
                })
            
            # Create DataLoader for this batch with specified batch size
            batch_loader = DataLoader(batch_data, batch_size=batchSize, shuffle=False, 
                                    collate_fn=collateFunction, drop_last=False)
            
            # Process this batch file set in mini-batches
            for mini_batch_graphs, mini_batch_labels in tqdm(batch_loader, 
                                                           desc=f"Processing batch set {file_idx + 1}", 
                                                           leave=False):
                # Unpack graphs
                graph_delta, graph_kT, graph_mSquare, graph_Z = mini_batch_graphs
                
                # Move to device
                graph_delta = graph_delta.to(device)
                graph_kT = graph_kT.to(device)
                graph_mSquare = graph_mSquare.to(device)
                graph_Z = graph_Z.to(device)
                mini_batch_labels = mini_batch_labels.to(device)
                
                # Get embeddings from each graph type
                hg_delta = model_delta(graph_delta)
                hg_kT = model_kT(graph_kT)
                hg_mSquare = model_mSquare(graph_mSquare)
                hg_Z = model_Z(graph_Z)
                
                # Concatenate embeddings 
                concatenated_features = torch.cat([hg_delta, hg_kT, hg_mSquare, hg_Z], dim=1)
                
                # Get final logits from classifier
                logits = classifier(concatenated_features)
                predictions = logits.argmax(dim=1)
                
                # Store results
                logitsTracker.extend(logits.cpu().tolist())
                predictionsTracker.extend(predictions.cpu().tolist())
                targetsTracker.extend(mini_batch_labels.cpu().tolist())
                
                # Update confusion matrix
                for idx, pred in enumerate(predictions):
                    cfs[pred][mini_batch_labels[idx]] += 1
                
                # Clean up GPU memory
                del graph_delta, graph_kT, graph_mSquare, graph_Z
                del hg_delta, hg_kT, hg_mSquare, hg_Z, concatenated_features
                del mini_batch_graphs, mini_batch_labels, logits, predictions
                torch.cuda.empty_cache() if device == 'cuda' else None
            
            # Update counters
            total_processed += batch_size
            processed_files.append(batch_key)
            files_processed_count += 1
            
            print(f"Completed batch set {file_idx + 1}. Total processed: {total_processed}")
            
            # Calculate and display current metrics after each batch set
            metricsDF = calculate_and_display_metrics(cfs, targetsTracker, predictionsTracker, jetNames, 
                                                    total_processed, files_processed_count, total_files)
            
            # Save intermediate results (every args.checkpoint_freq batch sets or last batch set)
            if files_processed_count % args.checkpoint_freq == 0 or files_processed_count == total_files:
                save_intermediate_results(imageSavePath, cfs, jetNames, classificationLevel, modelArchitecture,
                                        logitsTracker, targetsTracker, predictionsTracker, 
                                        files_processed_count, total_files)
                
                # Log intermediate results to wandb
                wandb.log({
                    "Current_Overall_Accuracy": metricsDF.loc['Micro Avg', 'Accuracy'],
                    "Current_BatchSets_Processed": files_processed_count,
                    "Current_Samples_Processed": total_processed,
                    "Progress_Percentage": (files_processed_count / total_files) * 100,
                    "Current_Confusion_Matrix": wandb.Image(f"{imageSavePath}/Confusion Matrix_Latest.png") if os.path.exists(f"{imageSavePath}/Confusion Matrix_Latest.png") else None
                })
            
            # Clean up memory
            del batch_graphs, batch_labels, batch_data, batch_loader
            gc.collect()
            
            # Save checkpoint after each batch set
            results = {
                'logitsTracker': logitsTracker,
                'predictionsTracker': predictionsTracker,
                'targetsTracker': targetsTracker,
                'confusion_matrix': cfs,
                'total_processed': total_processed
            }
            save_checkpoint(checkpoint_file, processed_files, results)
            
        except Exception as e:
            print(f"Error processing batch set {file_idx + 1}: {e}")
            continue

print("Multi-graph testing completed!")

# Clean up checkpoint file after successful completion
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)
    print("Checkpoint file cleaned up")

# Save results
try:
    os.makedirs(imageSavePath, exist_ok=True)
except Exception as e:
    print(e)

# Save tracking data
os.makedirs('metrics', exist_ok=True)
logitsTrackerFile = f'metrics/{classificationLevel}-{modelArchitecture}-MultiGraph-Logits.pkl'
targetsTrackerFile = f'metrics/{classificationLevel}-{modelArchitecture}-MultiGraph-Targets.pkl'
predictionsTrackerFile = f'metrics/{classificationLevel}-{modelArchitecture}-MultiGraph-Predictions.pkl'

with open(logitsTrackerFile, 'wb') as f:
    pickle.dump(logitsTracker, f)

with open(targetsTrackerFile, 'wb') as f:
    pickle.dump(targetsTracker, f)

with open(predictionsTrackerFile, 'wb') as f:
    pickle.dump(predictionsTracker, f)

print("Results saved!")

# Generate confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.gcf()
fig.set_size_inches(15, 15)

ax = sns.heatmap(cfs/np.sum(cfs), annot=True, cmap='Blues')
ax.set_title(f'{classificationLevel} {modelArchitecture} Multi-Graph Confusion Matrix')
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')

print(cfs/np.sum(cfs))
plt.savefig(f'{imageSavePath}/Confusion Matrix.png')
plt.clf()

# Generate ROC curve
from sklearn.metrics import roc_curve, auc
import scikitplot as skplt
from sklearn.preprocessing import label_binarize

try:
    # Convert lists to numpy arrays
    logitsTracker_np = np.array(logitsTracker)
    targetsTracker_np = np.array(targetsTracker)
    
    # Get unique classes
    classes = sorted(list(set(targetsTracker_np)))
    # Create one-hot encoding for targets
    rocTargets = label_binarize(targetsTracker_np, classes=classes)
    # Use logitsTracker directly as rocLogits
    rocLogits = logitsTracker_np
    
    # Plot ROC curve
    skplt.metrics.plot_roc_curve(rocTargets, rocLogits, figsize=(8, 6), 
                               title=f'{classificationLevel} {modelArchitecture} Multi-Graph ROC-AUC Curve')
    plt.savefig(f'{imageSavePath}/ROC-AUC.png')
    plt.clf()
    
except Exception as e:
    print(f"Error generating ROC curve: {e}")
    plt.figure(figsize=(8, 6))
    plt.title(f'{classificationLevel} {modelArchitecture} Multi-Graph ROC-AUC Curve (Error)')
    plt.text(0.5, 0.5, f'Error generating ROC curve: {str(e)}', 
             horizontalalignment='center', verticalalignment='center')
    plt.savefig(f'{imageSavePath}/ROC-AUC.png')
    plt.clf()

# Calculate final metrics
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

# Print the final metrics table
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(metricsDF)
print("="*80)

# Log to wandb
wandb.log({
    "Final_Confusion_Matrix": wandb.Image(f"{imageSavePath}/Confusion Matrix.png"),
    "Final_ROC-AUC_Curve": wandb.Image(f"{imageSavePath}/ROC-AUC.png"),
    "Final_Confusion_Matrix_Table": wandb.Table(dataframe=metricsDF.reset_index()),
    "Final_Total_Samples_Processed": total_processed,
    "Final_Overall_Accuracy": metricsDF.loc['Micro Avg', 'Accuracy']
})

wandb.finish()

print("\n" + "="*80)
print("MULTI-GRAPH TESTING COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"Results saved to: {imageSavePath}")
print(f"Metrics saved to: metrics/")
print(f"Model used: {modelSaveFile}")
print(f"Total samples processed: {total_processed}")
print(f"Final accuracy: {metricsDF.loc['Micro Avg', 'Accuracy']:.4f}")
print("="*80)

print("Multi-graph analysis complete!")