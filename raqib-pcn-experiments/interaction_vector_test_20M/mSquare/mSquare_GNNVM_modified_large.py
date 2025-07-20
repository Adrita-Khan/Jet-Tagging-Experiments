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

class BatchedGraphDataset(dgl.data.DGLDataset):
    def __init__(self, jetNames, k, batchDir='batches', loadFromDisk=False):
        
        self.jetNames = jetNames
        self.batchDir = batchDir
        
        # Get all batch files for each jet type
        self.batch_files = {}
        self.sampleCountPerClass = []
        
        for jetType in jetNames:
            if type(jetType) != list:
                # Find all batch files for this jet type
                jet_dir = os.path.join(batchDir, jetType)
                if os.path.exists(jet_dir):
                    batch_files = sorted([f for f in os.listdir(jet_dir) if f.endswith('.pkl')])
                    self.batch_files[jetType] = [os.path.join(jet_dir, f) for f in batch_files]
                else:
                    self.batch_files[jetType] = []
                
                # Count total samples for this jet type (quick count without loading)
                print(f'{jetType}: Found {len(self.batch_files[jetType])} batch files')
                # We'll get exact count during processing
                self.sampleCountPerClass.append(len(self.batch_files[jetType]))  # placeholder
            else:
                # Handle list of jet types (if needed)
                combined_files = []
                for item in jetType:
                    jet_dir = os.path.join(batchDir, item)
                    if os.path.exists(jet_dir):
                        batch_files = sorted([f for f in os.listdir(jet_dir) if f.endswith('.pkl')])
                        item_files = [os.path.join(jet_dir, f) for f in batch_files]
                        combined_files.extend(item_files)
                
                self.batch_files[str(jetType)] = combined_files
                self.sampleCountPerClass.append(len(combined_files))  # placeholder
        
        # Create a flat list of all batch files with their labels
        self.all_batch_files = []
        label = 0
        for jetType in jetNames:
            jet_key = jetType if type(jetType) != list else str(jetType)
            for batch_file in self.batch_files[jet_key]:
                self.all_batch_files.append((batch_file, label))
            label += 1
        
        print(f'Total batch files to process: {len(self.all_batch_files)}')

    def process(self):
        return
    
    def get_all_batch_files(self):
        """Get list of all batch files with their labels"""
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
    print(f"Checkpoint saved: {len(processed_files)} files processed")

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
        print(f"Checkpoint loaded: {len(checkpoint_data['processed_files'])} files already processed")
        return checkpoint_data['processed_files'], results
    else:
        print("No checkpoint found, starting from beginning")
        return [], None

# NEW: Function to calculate and display current metrics
def calculate_and_display_metrics(cfs, targetsTracker, predictionsTracker, jetNames, total_processed, file_count, total_files):
    """Calculate and display current metrics"""
    print("\n" + "="*80)
    print(f"CURRENT RESULTS AFTER {file_count}/{total_files} FILES ({total_processed} samples)")
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

# NEW: Function to save intermediate results
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
        ax.set_title(f'{classificationLevel} {modelArchitecture} Confusion Matrix (Files: {file_count}/{total_files})')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        
        plt.savefig(f'{imageSavePath}/Confusion Matrix_Intermediate_{file_count}.png')
        plt.clf()
        
        # Also save as the latest
        plt.figure(figsize=(15, 15))
        ax = sns.heatmap(cfs/np.sum(cfs), annot=True, cmap='Blues')
        ax.set_title(f'{classificationLevel} {modelArchitecture} Confusion Matrix (Files: {file_count}/{total_files})')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        plt.savefig(f'{imageSavePath}/Confusion Matrix_Latest.png')
        plt.clf()
        
    except Exception as e:
        print(f"Error saving intermediate confusion matrix: {e}")

# process all jetTypes
Higgs = ['HToBB', 'HToCC', 'HToGG', 'HToWW2Q1L', 'HToWW4Q']
Vector = ['WToQQ', 'ZToQQ']
Top = ['TTBar', 'TTBarLep']
QCD = ['ZJetsToNuNu']

# For testing, use the original jet names (remove "-Testing" suffix if needed)
testingSet = Top + Vector + QCD + Higgs
jetNames = testingSet
print(jetNames)

# Create dataset object
dataset = BatchedGraphDataset(jetNames, 3, batchDir='batches', loadFromDisk=False)
dataset.process()

# Since we're doing testing (maxEpochs = 0), we'll use the batched approach
maxEpochs = 0  # Set to 0 for testing
batchSize = int(input("Batch Size: "))

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dgl.batch import batch

class GNNClassifier(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k):
        super(GNNClassifier, self).__init__()
        self.conv1 = dgl.nn.ChebConv(in_feats, hidden_feats, k)
        self.conv2 = dgl.nn.ChebConv(hidden_feats, hidden_feats, k)
        self.conv3 = dgl.nn.ChebConv(hidden_feats, hidden_feats, k)
        
        self.fc = nn.Linear(hidden_feats, out_feats)
        
    def forward(self, g):
        # Apply graph convolutional layers
        h = F.relu(self.conv1(g, g.ndata['feat']))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
    
        # Store the node embeddings in the node data dictionary
        g.ndata['h'] = h
    
        # Compute graph-level representations by taking global mean pooling
        hg = dgl.mean_nodes(g, 'h')
        
        # Pass the graph-level representation through a fully connected layer
        logits = self.fc(hg)
        
        return logits

class DGCNNClassifier(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k):
        super(DGCNNClassifier, self).__init__()
        self.conv1 = dgl.nn.ChebConv(in_feats, hidden_feats, k)
        self.conv2 = dgl.nn.ChebConv(hidden_feats, hidden_feats, k)
        self.conv3 = dgl.nn.ChebConv(hidden_feats, hidden_feats, k)
        
        self.edgeconv1 = dgl.nn.EdgeConv(hidden_feats, hidden_feats)
        self.edgeconv2 = dgl.nn.EdgeConv(hidden_feats, hidden_feats)
        
        self.fc = nn.Linear(hidden_feats, out_feats)
        
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
        
        # Pass the graph-level representation through a fully connected layer
        logits = self.fc(hg)
        
        return logits

# Custom collate function for batching
def collateFunction(batch):
    graphs = [item['graph'] for item in batch]
    labels = [item['label'] for item in batch]
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

# Device and model configuration
device = input("cuda or cpu: ")
classificationLevel = input("Classification Level: ")
modelArchitecture = input("Model Architecture Name: ")
modelType = input("Model Type:")
modelSaveFile = "modelSaveFiles/" + classificationLevel + modelArchitecture + ".pt"
load = input("Load from a save file (Y or N): ")

# Checkpoint file
checkpoint_file = f"checkpoints/{classificationLevel}-{modelArchitecture}-checkpoint.json"
os.makedirs("checkpoints", exist_ok=True)

in_feats = 16
hidden_feats = 64
out_feats = len(jetNames)  # Number of output classes
chebFilterSize = 16

# Start wandb logging
wandb.init(
    project="mSquare Weight Testing 20M", 
    name=f"{classificationLevel}-{modelArchitecture}-Testing",
    config={
        "epochs": maxEpochs,
        "batch_size": batchSize,
        "model": modelArchitecture,
        "in_feats": in_feats,
        "hidden_feats": hidden_feats,
        "out_feats": out_feats,
        "device": device,
        "testing_mode": True
    }
)

while type(load) != bool:
    if load == "Y":
        load = True
    elif load == "N":
        load = False
    else: 
        print("Invalid Input Please Enter Y or N: ")
        load = input("Load from a save file (Y or N): ")

# Initialize model
if modelType == "GCNN":
    model = GNNClassifier(in_feats, hidden_feats, out_feats, chebFilterSize)
elif modelType == "DGCNN":
    model = DGCNNClassifier(in_feats, hidden_feats, out_feats, chebFilterSize)
else:
    print("Invalid selection. Erroring out!")
    exit()

if load:
    model.load_state_dict(torch.load(modelSaveFile))
    print(f"Loaded model from {modelSaveFile}")

model.to(device)
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

# NEW: Create results directory early
imageSavePath = f'{classificationLevel} {modelArchitecture}'
try:
    os.makedirs(imageSavePath, exist_ok=True)
except Exception as e:
    print(e)

print("Starting batch-wise testing...")

# Get all batch files
all_batch_files = dataset.get_all_batch_files()
total_files = len(all_batch_files)
files_processed_count = len(processed_files)

# NEW: Display initial status if resuming
if files_processed_count > 0:
    print(f"\nResuming from checkpoint. Already processed {files_processed_count}/{total_files} files.")
    metricsDF = calculate_and_display_metrics(cfs, targetsTracker, predictionsTracker, jetNames, 
                                            total_processed, files_processed_count, total_files)

# Process each batch file one at a time
with torch.no_grad():
    for file_idx, (batch_file, label) in enumerate(tqdm(all_batch_files, desc="Processing batch files")):
        
        # Skip if already processed
        if batch_file in processed_files:
            continue
        
        try:
            print(f"\nLoading batch file: {batch_file}")
            
            # Load one batch file at a time
            with open(batch_file, 'rb') as f:
                batch_graphs = pickle.load(f)
            
            print(f"Loaded {len(batch_graphs)} graphs from {batch_file}")
            
            # Create labels for this batch
            batch_labels = [label] * len(batch_graphs)
            
            # Create dataset for this batch
            batch_data = []
            for graph, lbl in zip(batch_graphs, batch_labels):
                batch_data.append({'graph': graph, 'label': lbl})
            
            # Create DataLoader for this batch with specified batch size
            batch_loader = DataLoader(batch_data, batch_size=batchSize, shuffle=False, 
                                    collate_fn=collateFunction, drop_last=False)
            
            # Process this batch file in mini-batches
            for mini_batch_graphs, mini_batch_labels in tqdm(batch_loader, 
                                                           desc=f"Processing {os.path.basename(batch_file)}", 
                                                           leave=False):
                mini_batch_graphs = mini_batch_graphs.to(device)
                mini_batch_labels = mini_batch_labels.to(device)
                
                # Make predictions
                logits = model(mini_batch_graphs)
                predictions = logits.argmax(dim=1)
                
                # Store results
                logitsTracker.extend(logits.cpu().tolist())
                predictionsTracker.extend(predictions.cpu().tolist())
                targetsTracker.extend(mini_batch_labels.cpu().tolist())
                
                # Update confusion matrix
                for idx, pred in enumerate(predictions):
                    cfs[pred][mini_batch_labels[idx]] += 1
                
                # Clean up GPU memory
                del mini_batch_graphs, mini_batch_labels, logits, predictions
                torch.cuda.empty_cache() if device == 'cuda' else None
            
            # Update counters
            total_processed += len(batch_graphs)
            processed_files.append(batch_file)
            files_processed_count += 1
            
            print(f"Completed {batch_file}. Total processed: {total_processed}")
            
            # NEW: Calculate and display current metrics after each file
            metricsDF = calculate_and_display_metrics(cfs, targetsTracker, predictionsTracker, jetNames, 
                                                    total_processed, files_processed_count, total_files)
            
            # NEW: Save intermediate results (every 5 files or last file)
            if files_processed_count % 5 == 0 or files_processed_count == total_files:
                save_intermediate_results(imageSavePath, cfs, jetNames, classificationLevel, modelArchitecture,
                                        logitsTracker, targetsTracker, predictionsTracker, 
                                        files_processed_count, total_files)
                
                # NEW: Log intermediate results to wandb
                wandb.log({
                    "Current_Overall_Accuracy": metricsDF.loc['Micro Avg', 'Accuracy'],
                    "Current_Files_Processed": files_processed_count,
                    "Current_Samples_Processed": total_processed,
                    "Progress_Percentage": (files_processed_count / total_files) * 100,
                    "Current_Confusion_Matrix": wandb.Image(f"{imageSavePath}/Confusion Matrix_Latest.png") if os.path.exists(f"{imageSavePath}/Confusion Matrix_Latest.png") else None
                })
            
            # Clean up memory
            del batch_graphs, batch_labels, batch_data, batch_loader
            gc.collect()
            
            # Save checkpoint after each file
            results = {
                'logitsTracker': logitsTracker,
                'predictionsTracker': predictionsTracker,
                'targetsTracker': targetsTracker,
                'confusion_matrix': cfs,
                'total_processed': total_processed
            }
            save_checkpoint(checkpoint_file, processed_files, results)
            
        except Exception as e:
            print(f"Error processing {batch_file}: {e}")
            continue

print("Testing completed!")

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
logitsTrackerFile = f'metrics/{classificationLevel}-{modelArchitecture}-Logits.pkl'
targetsTrackerFile = f'metrics/{classificationLevel}-{modelArchitecture}-Targets.pkl'
predictionsTrackerFile = f'metrics/{classificationLevel}-{modelArchitecture}-Predictions.pkl'

os.makedirs('metrics', exist_ok=True)

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
ax.set_title(f'{classificationLevel} {modelArchitecture} Confusion Matrix')
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
                               title=f'{classificationLevel} {modelArchitecture} ROC-AUC Curve')
    plt.savefig(f'{imageSavePath}/ROC-AUC.png')
    plt.clf()
    
except Exception as e:
    print(f"Error generating ROC curve: {e}")
    plt.figure(figsize=(8, 6))
    plt.title(f'{classificationLevel} {modelArchitecture} ROC-AUC Curve (Error)')
    plt.text(0.5, 0.5, f'Error generating ROC curve: {str(e)}', 
             horizontalalignment='center', verticalalignment='center')
    plt.savefig(f'{imageSavePath}/ROC-AUC.png')
    plt.clf()

# Calculate metrics
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

# Log to wandb
wandb.log({
    "Confusion Matrix": wandb.Image(f"{imageSavePath}/Confusion Matrix.png"),
    "ROC-AUC Curve": wandb.Image(f"{imageSavePath}/ROC-AUC.png"),
    "Confusion Matrix Table": wandb.Table(dataframe=metricsDF.reset_index()),
    "Total Samples Processed": total_processed
})

wandb.finish()

print("Analysis complete!")