import numpy as np
import pandas as pd
from operator import truth
import numpy as np
import awkward as ak
import torch
from tqdm import tqdm
import os
import dgl
import pickle
import wandb
import GPUtil
import argparse

from dgllife.utils import RandomSplitter
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add argument parser
parser = argparse.ArgumentParser(description='Multi-Graph Neural Network Training')
parser.add_argument('--max_epochs', type=int, default=500, help='Maximum number of epochs (default: 500)')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch size (default: 1024)')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use (default: cuda)')
parser.add_argument('--classification_level', type=str, default='AllInteractions', help='Classification level (default: All)')
parser.add_argument('--model_architecture', type=str, default='PCN', help='Model architecture name (default: PCN)')
parser.add_argument('--model_type', type=str, default='DGCNN', help='Model type (default: DGCNN)')
parser.add_argument('--load_model', type=str, default='N', choices=['Y', 'N'], help='Load from save file (default: N)')
parser.add_argument('--convergence_threshold', type=float, default=0.0001, help='Convergence threshold (default: 0.0001)')

args = parser.parse_args()

# Enhanced GPU Memory Monitoring Class
class GPUMemoryTracker:
    def __init__(self):
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.baseline_memory = 0
        if torch.cuda.is_available():
            # Record baseline GPU memory usage before training
            self.baseline_memory = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
    
    def get_memory_stats(self):
        """Get comprehensive memory statistics"""
        if not torch.cuda.is_available():
            return None
            
        # PyTorch memory stats (specific to your process)
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        
        # Total GPU memory info
        gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
        total_memory = gpu.memoryTotal if gpu else 0
        used_memory = gpu.memoryUsed if gpu else 0
        free_memory = gpu.memoryFree if gpu else 0
        
        # Calculate memory specific to your training
        training_memory = allocated - self.baseline_memory
        
        return {
            'allocated_gb': allocated / 1024**3,
            'reserved_gb': reserved / 1024**3,
            'peak_allocated_gb': peak_allocated / 1024**3,
            'peak_reserved_gb': peak_reserved / 1024**3,
            'training_memory_gb': training_memory / 1024**3,
            'total_gpu_memory_gb': total_memory / 1024,  # GPUtil returns MB
            'gpu_utilization_percent': (used_memory / total_memory * 100) if total_memory > 0 else 0,
            'free_memory_gb': free_memory / 1024
        }
    
    def log_memory_stats(self, epoch=None, stage="training"):
        """Log memory stats to console and return dict for wandb"""
        stats = self.get_memory_stats()
        if not stats:
            return {}
            
        prefix = f"Epoch {epoch} - " if epoch is not None else ""
        print(f"{prefix}GPU Memory ({stage}):")
        print(f"  Current Allocated: {stats['allocated_gb']:.2f}GB")
        print(f"  Peak Allocated: {stats['peak_allocated_gb']:.2f}GB")
        print(f"  Training Memory: {stats['training_memory_gb']:.2f}GB")
        print(f"  GPU Utilization: {stats['gpu_utilization_percent']:.1f}%")
        print(f"  Free Memory: {stats['free_memory_gb']:.2f}GB")
        
        # Return formatted dict for wandb logging
        wandb_dict = {
            f"GPU/{stage}_allocated_gb": stats['allocated_gb'],
            f"GPU/{stage}_peak_allocated_gb": stats['peak_allocated_gb'],
            f"GPU/{stage}_training_memory_gb": stats['training_memory_gb'],
            f"GPU/utilization_percent": stats['gpu_utilization_percent'],
            f"GPU/free_memory_gb": stats['free_memory_gb']
        }
        
        return wandb_dict

class MultiGraphDataset(dgl.data.DGLDataset):
    def __init__(self, jetNames, k, loadFromDisk=False):
        
        self.jetNames = jetNames
        
        # Initialize lists for each graph type
        self.graphs_delta = []
        self.graphs_kT = []
        self.graphs_mSquare = []
        self.graphs_Z = []
        self.sampleCountPerClass = []
        
        for jetType in tqdm(jetNames, total=len(jetNames)):
            if type(jetType) != list:
                
                if loadFromDisk:
                    # Load from disk (adjust paths as needed)
                    delta_path = f'pickleFiles/Delta_{jetType}.pkl'
                    kT_path = f'pickleFiles/kT_{jetType}.pkl'
                    mSquare_path = f'pickleFiles/mSquare_{jetType}.pkl'
                    Z_path = f'pickleFiles/Z_{jetType}.pkl'
                else:
                    # Load from your data directory
                    delta_path = f'data/Multi Level Jet Tagging/Delta/Delta_{jetType}.pkl'
                    kT_path = f'data/Multi Level Jet Tagging/kT/kT_{jetType}.pkl'
                    mSquare_path = f'data/Multi Level Jet Tagging/mSquare/mSquare_{jetType}.pkl'
                    Z_path = f'data/Multi Level Jet Tagging/Z/Z_{jetType}.pkl'
                
                # Load each graph type
                with open(delta_path, 'rb') as f:
                    delta_graphs = pickle.load(f)
                    self.graphs_delta += delta_graphs
                
                with open(kT_path, 'rb') as f:
                    kT_graphs = pickle.load(f)
                    self.graphs_kT += kT_graphs
                
                with open(mSquare_path, 'rb') as f:
                    mSquare_graphs = pickle.load(f)
                    self.graphs_mSquare += mSquare_graphs
                
                with open(Z_path, 'rb') as f:
                    Z_graphs = pickle.load(f)
                    self.graphs_Z += Z_graphs
                
                # Assuming all graph types have the same number of samples
                self.sampleCountPerClass.append(len(delta_graphs))
                
                del delta_graphs, kT_graphs, mSquare_graphs, Z_graphs
                
        self.labels = []
        label = 0
        for sampleCount in self.sampleCountPerClass:
            print(f'Class {label} has {sampleCount} samples')
            for _ in range(sampleCount):
                self.labels.append(label)
            label += 1

        print(f'Samples per class: {self.sampleCountPerClass}')
        

    def process(self):
        return
                
    def __getitem__(self, idx):
        return {
            'graph_delta': self.graphs_delta[idx],
            'graph_kT': self.graphs_kT[idx], 
            'graph_mSquare': self.graphs_mSquare[idx],
            'graph_Z': self.graphs_Z[idx],
            'label': self.labels[idx]
        }

    def __len__(self):
        return len(self.graphs_delta)

# collate function for multiple graphs
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

# Initialize GPU memory tracker before dataset loading
print("Initializing GPU Memory Tracker...")
memory_tracker = GPUMemoryTracker()

dataset = MultiGraphDataset(jetNames, 3, loadFromDisk=False)
dataset.process()

# Use argparse values instead of input()
maxEpochs = args.max_epochs
batchSize = args.batch_size
device = args.device
classificationLevel = args.classification_level
modelArchitecture = args.model_architecture
modelType = args.model_type
load = True if args.load_model == 'Y' else False
convergence_threshold = args.convergence_threshold

if maxEpochs != 0:
    train, val, test = RandomSplitter().train_val_test_split(dataset, frac_train=0.8, frac_test=0.1, 
                                                         frac_val=0.1, random_state=42)
else:
    train = dataset # use the full testing dataset if running testing evaluation

if maxEpochs != 0:
    trainLoader = DataLoader(train, batch_size=batchSize, shuffle=True, collate_fn=collateFunction, drop_last=True)
    validationLoader = DataLoader(val, batch_size=batchSize, shuffle=True, collate_fn=collateFunction, drop_last=True)
    testLoader = DataLoader(test, batch_size=batchSize, shuffle=True, collate_fn=collateFunction, drop_last=True)
else:
    testLoader = DataLoader(train, batch_size=batchSize, shuffle=True, collate_fn=collateFunction, drop_last=True)

# Create modelSaveFiles directory if it doesn't exist
os.makedirs("modelSaveFiles", exist_ok=True)
modelSaveFile = "modelSaveFiles/" + classificationLevel + modelArchitecture + ".pt"

in_feats = 16
hidden_feats = 64
out_feats = len(jetNames) # Number of output classes

# Start wandb logging
wandb.init(
    project="All Interaction Vectors", 
    name=f"{classificationLevel}-{modelArchitecture}",
    config={
        "epochs": maxEpochs,
        "batch_size": batchSize,
        "model": modelArchitecture,
        "model_type": modelType,
        "in_feats": in_feats,
        "hidden_feats": hidden_feats,
        "out_feats": out_feats,
        "device": device
    }
)

chebFilterSize = 16

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

# Watch only the classifier to reduce memory overhead
wandb.watch(classifier, log='all', log_freq=200)

# Define the loss function and optimizer for all models
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW([
    {'params': model_delta.parameters()},
    {'params': model_kT.parameters()},
    {'params': model_mSquare.parameters()},
    {'params': model_Z.parameters()},
    {'params': classifier.parameters()}
], lr=1e-3)

trainingLossTracker = []
trainingAccuracyTracker = []
validationLossTracker = []
validationAccuracyTracker = []

bestLoss = float('inf')
epochs_without_improvement = 0
epochsTillQuit = 10

# Log initial memory state
initial_memory = memory_tracker.log_memory_stats(stage="initial")
wandb.log(initial_memory)

# Train the model
for epoch in range(maxEpochs):
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
        graph_delta, graph_kT, graph_mSquare, graph_Z = graphs
        
        # Port data to the device in use
        graph_delta = graph_delta.to(device)
        graph_kT = graph_kT.to(device)
        graph_mSquare = graph_mSquare.to(device)
        graph_Z = graph_Z.to(device)
        labels = labels.to(device).long()

        # Get embeddings from each graph type
        hg_delta = model_delta(graph_delta)
        hg_kT = model_kT(graph_kT)
        hg_mSquare = model_mSquare(graph_mSquare)
        hg_Z = model_Z(graph_Z)
        
        # Concatenate embeddings 
        concatenated_features = torch.cat([hg_delta, hg_kT, hg_mSquare, hg_Z], dim=1)
        
        # Get final logits from classifier
        logits = classifier(concatenated_features)
        
        # Calculate loss and do backpropagation
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update running loss
        runningLoss += loss.item()

        # Compute accuracy
        predictions = logits.argmax(dim=1)
        batchCorrectPredictions = (predictions == labels).sum().item()
        batchTotalSamples = labels.numel()
        totalCorrectPredictions += batchCorrectPredictions
        totalSamples += batchTotalSamples

        # COMPREHENSIVE MEMORY CLEANUP
        del graph_delta, graph_kT, graph_mSquare, graph_Z
        del hg_delta, hg_kT, hg_mSquare, hg_Z, concatenated_features
        del graphs, labels, logits, predictions, loss
        
        # Clear cache periodically
        if batchIndex % 10 == 0:
            torch.cuda.empty_cache()

    # Compute epoch statistics
    epochLoss = runningLoss / len(trainLoader)
    trainingLossTracker.append(epochLoss)
    
    epochAccuracy = totalCorrectPredictions / totalSamples
    trainingAccuracyTracker.append(epochAccuracy)
    
    # Validation
    for model in all_models:
        model.eval()
    validationLoss = 0.0

    with torch.no_grad():
        for graphs, labels in tqdm(validationLoader, total=len(validationLoader), leave=False):
            # Unpack graphs
            graph_delta, graph_kT, graph_mSquare, graph_Z = graphs
            
            graph_delta = graph_delta.to(device)
            graph_kT = graph_kT.to(device)
            graph_mSquare = graph_mSquare.to(device)
            graph_Z = graph_Z.to(device)
            labels = labels.to(device).long()

            # Get embeddings and logits
            hg_delta = model_delta(graph_delta)
            hg_kT = model_kT(graph_kT)
            hg_mSquare = model_mSquare(graph_mSquare)
            hg_Z = model_Z(graph_Z)
            
            concatenated_features = torch.cat([hg_delta, hg_kT, hg_mSquare, hg_Z], dim=1)
            logits = classifier(concatenated_features)
            
            loss = criterion(logits, labels)
            validationLoss += loss.item()
            
            predictions = logits.argmax(dim=1)
            batchCorrectPredictions = (predictions == labels).sum().item()
            batchTotalSamples = labels.numel()
            
            valTotalCorrectPredictions += batchCorrectPredictions
            valTotalSamples += batchTotalSamples
            
            # VALIDATION MEMORY CLEANUP
            del graph_delta, graph_kT, graph_mSquare, graph_Z
            del hg_delta, hg_kT, hg_mSquare, hg_Z, concatenated_features
            del graphs, labels, logits, predictions, loss

    avgValidationLoss = validationLoss / len(validationLoader)
    validationLossTracker.append(avgValidationLoss)
    
    validationAccuracy = valTotalCorrectPredictions / valTotalSamples
    validationAccuracyTracker.append(validationAccuracy)

    # Check for convergence and ONLY save when improved
    if avgValidationLoss < bestLoss - convergence_threshold:
        bestLoss = avgValidationLoss
        bestStateDict = {
            'model_delta': model_delta.state_dict(),
            'model_kT': model_kT.state_dict(),
            'model_mSquare': model_mSquare.state_dict(),
            'model_Z': model_Z.state_dict(),
            'classifier': classifier.state_dict()
        }
        # Save model checkpoint only when improved
        torch.save({
            'model_delta': model_delta.state_dict(),
            'model_kT': model_kT.state_dict(),
            'model_mSquare': model_mSquare.state_dict(),
            'model_Z': model_Z.state_dict(),
            'classifier': classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, modelSaveFile)
        print(f'Saved Models to file {modelSaveFile} at epoch {epoch+1}')
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    # Clear cache after each epoch
    torch.cuda.empty_cache()

    # Log gradient norm
    grad_norm = 0
    for model in all_models:
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item()**2
    grad_norm = grad_norm ** 0.5
    
    # Get comprehensive memory stats
    memory_stats = memory_tracker.log_memory_stats(epoch + 1, "training")
    
    # Print training and validation losses
    print(f"Epoch {epoch + 1} - Training Loss={epochLoss:.4f} - Validation Loss={avgValidationLoss:.4f} - Training Accuracy={epochAccuracy:.4f} - Validation Accuracy={validationAccuracy:.4f}")
    
    wandb.log({
        "Epoch": epoch + 1,
        "Training Loss": epochLoss,
        "Validation Loss": avgValidationLoss,
        "Training Accuracy": epochAccuracy,
        "Validation Accuracy": validationAccuracy,
        "Gradient Norm": grad_norm,
        **memory_stats  # Include all detailed memory stats
    })

    # Check convergence criteria
    if epochs_without_improvement >= epochsTillQuit:
        print(f'Convergence achieved at epoch {epoch + 1}. Stopping training.')
        break

if maxEpochs != 0:
    torch.save(bestStateDict, modelSaveFile)

# Create directory for saving plots
imageSavePath = f'{classificationLevel} {modelArchitecture}'
try:
    os.mkdir(imageSavePath)
except Exception as e:
    print(e)

if maxEpochs != 0:
    import matplotlib.pyplot as plt
    
    # Plot training loss
    plt.figure()
    plt.plot(range(len(trainingLossTracker)), trainingLossTracker)
    plt.title(f'{classificationLevel} {modelArchitecture} Training Loss Graph')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(f'{imageSavePath}/Training Loss.png')
    plt.close()

    # Plot training accuracy
    plt.figure()
    plt.plot(range(len(trainingAccuracyTracker)), trainingAccuracyTracker)
    plt.title(f'{classificationLevel} {modelArchitecture} Training Accuracy Graph')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(f'{imageSavePath}/Training Accuracy.png')
    plt.close()

    # Plot validation loss
    plt.figure()
    plt.plot(range(len(validationLossTracker)), validationLossTracker)
    plt.title(f'{classificationLevel} {modelArchitecture} Validation Loss Graph')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(f'{imageSavePath}/Validation Loss.png')
    plt.close()

    # Plot validation accuracy
    plt.figure()
    plt.plot(range(len(validationAccuracyTracker)), validationAccuracyTracker)
    plt.title(f'{classificationLevel} {modelArchitecture} Validation Accuracy Graph')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(f'{imageSavePath}/Validation Accuracy.png')
    plt.close()

# Testing and Evaluation
logitsTracker = []
predictionsTracker = []
targetsTracker = []

cfs = np.zeros((out_feats, out_feats))

# Set all models to eval mode
for model in all_models:
    model.eval()

import sklearn

if maxEpochs != 0:
    with torch.no_grad():
        for graphs, labels in tqdm(testLoader, total=len(testLoader), leave=False):
            # Unpack graphs
            graph_delta, graph_kT, graph_mSquare, graph_Z = graphs
            
            graph_delta = graph_delta.to(device)
            graph_kT = graph_kT.to(device)
            graph_mSquare = graph_mSquare.to(device)
            graph_Z = graph_Z.to(device)
            labels = labels.to(device)

            # Get embeddings and logits
            hg_delta = model_delta(graph_delta)
            hg_kT = model_kT(graph_kT)
            hg_mSquare = model_mSquare(graph_mSquare)
            hg_Z = model_Z(graph_Z)
            
            concatenated_features = torch.cat([hg_delta, hg_kT, hg_mSquare, hg_Z], dim=1)
            logits = classifier(concatenated_features)
            
            # Convert to numpy immediately to save memory
            logitsTracker.append(logits.detach().cpu().numpy())
            targetsTracker.append(labels.detach().cpu().numpy())

            predictions = logits.argmax(dim=1)
            predictionsTracker.append(predictions.detach().cpu().numpy())
            
            # Update confusion matrix
            for idx, pred in enumerate(predictions):
                cfs[pred][labels[idx]] += 1
            
            # TESTING MEMORY CLEANUP
            del graph_delta, graph_kT, graph_mSquare, graph_Z
            del hg_delta, hg_kT, hg_mSquare, hg_Z, concatenated_features
            del graphs, labels, logits, predictions
else:
    with torch.no_grad():
        for graphs, labels in tqdm(testLoader, total=len(testLoader), leave=False):
            # Unpack graphs
            graph_delta, graph_kT, graph_mSquare, graph_Z = graphs
            
            graph_delta = graph_delta.to(device)
            graph_kT = graph_kT.to(device)
            graph_mSquare = graph_mSquare.to(device)
            graph_Z = graph_Z.to(device)
            labels = labels.to(device)

            # Get embeddings and logits
            hg_delta = model_delta(graph_delta)
            hg_kT = model_kT(graph_kT)
            hg_mSquare = model_mSquare(graph_mSquare)
            hg_Z = model_Z(graph_Z)
            
            concatenated_features = torch.cat([hg_delta, hg_kT, hg_mSquare, hg_Z], dim=1)
            logits = classifier(concatenated_features)
            
            # Convert to lists immediately
            logitsTracker.extend(logits.detach().cpu().tolist())
            targetsTracker.extend(labels.detach().cpu().tolist())

            predictions = logits.argmax(dim=1)
            predictionsTracker.extend(predictions.detach().cpu().tolist())
            
            # Update confusion matrix
            for idx, pred in enumerate(predictions):
                cfs[pred][labels[idx]] += 1
            
            # TESTING MEMORY CLEANUP
            del graph_delta, graph_kT, graph_mSquare, graph_Z
            del hg_delta, hg_kT, hg_mSquare, hg_Z, concatenated_features
            del graphs, labels, logits, predictions

# Clear cache after testing
torch.cuda.empty_cache()

# Log final memory requirements
final_stats = memory_tracker.get_memory_stats()
if final_stats:
    print("\n=== MEMORY REQUIREMENTS SUMMARY ===")
    print(f"Peak Memory Required: {final_stats['peak_allocated_gb']:.2f}GB")
    print(f"Recommended GPU Memory: {final_stats['peak_allocated_gb'] * 1.2:.2f}GB (with 20% buffer)")
    
    # Log final memory requirements to wandb
    wandb.log({
        "Final/peak_memory_required_gb": final_stats['peak_allocated_gb'],
        "Final/recommended_gpu_memory_gb": final_stats['peak_allocated_gb'] * 1.2,
        "Final/training_memory_used_gb": final_stats['training_memory_gb']
    })

# Save metrics
os.makedirs('metrics', exist_ok=True)
logitsTrackerFile = f'metrics/{classificationLevel}-{modelArchitecture}-Logits.pkl'
targetsTrackerFile = f'metrics/{classificationLevel}-{modelArchitecture}-Targets.pkl'
predictionsTrackerFile = f'metrics/{classificationLevel}-{modelArchitecture}-Predictions.pkl'

with open(logitsTrackerFile, 'wb') as f:
    pickle.dump(logitsTracker, f)

with open(targetsTrackerFile, 'wb') as f:
    pickle.dump(targetsTracker, f)

with open(predictionsTrackerFile, 'wb') as f:
    pickle.dump(predictionsTracker, f)

import seaborn as sns
import matplotlib.pyplot as plt

# Plot confusion matrix
fig = plt.gcf()
fig.set_size_inches(15, 15)

ax = sns.heatmap(cfs/np.sum(cfs), annot=True, cmap='Blues')
ax.set_title(f'{classificationLevel} {modelArchitecture} Confusion Matrix')
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')

print(cfs/np.sum(cfs))
plt.savefig(f'{imageSavePath}/Confusion Matrix.png')
plt.close()

# ROC-AUC Curve
from sklearn.metrics import roc_curve, auc
import scikitplot as skplt

if maxEpochs != 0:
    logitsTracker = np.concatenate(logitsTracker, axis=0)
    rocLogits = logitsTracker
    
    targetsTracker = np.concatenate(targetsTracker, axis=0)
    rocTargets = targetsTracker
else:
    rocLogits = np.array(logitsTracker)
    rocTargets = np.array(targetsTracker)

skplt.metrics.plot_roc_curve(rocTargets, rocLogits, figsize=(8, 6), title=f'{classificationLevel} {modelArchitecture} ROC-AUC Curve')
plt.savefig(f'{imageSavePath}/ROC-AUC.png')
plt.close()

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
wandb.log({
    "Confusion Matrix": wandb.Image(f"{imageSavePath}/Confusion Matrix.png"),
    "ROC-AUC Curve": wandb.Image(f"{imageSavePath}/ROC-AUC.png"),
    "Confusion Matrix Table": wandb.Table(dataframe=metricsDF.reset_index())
})

wandb.save(modelSaveFile)
wandb.finish()

print("Training completed successfully!")

# Final memory summary
final_memory = memory_tracker.log_memory_stats(stage="final")
print("\n=== TRAINING COMPLETED ===")