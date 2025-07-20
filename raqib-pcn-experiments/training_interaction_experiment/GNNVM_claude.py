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

class MultiGraphDataset(dgl.data.DGLDataset):
    def __init__(self, jetNames, k, weight_types=['binary', 'delta', 'kT', 'mSquare', 'Z'], loadFromDisk=False):
        self.jetNames = jetNames
        self.weight_types = weight_types
        
        # Store graphs for each weight type
        self.graphs_dict = {wt: [] for wt in weight_types}
        self.sampleCountPerClass = []
        
        for jetType in tqdm(jetNames, total=len(jetNames)):
            if type(jetType) != list:
                # Load graphs for each weight type
                graphs_per_type = {wt: [] for wt in weight_types}
                
                for wt in weight_types:
                    if wt == 'binary':
                        # Original binary weights
                        saveFilePath = f'../data/Multi Level Jet Tagging/{jetType}.pkl'
                    else:
                        # Physics-motivated weights
                        saveFilePath = f'../data/Multi Level Jet Tagging/{wt}/{wt}_{jetType}.pkl'
                    
                    with open(saveFilePath, 'rb') as f:
                        singleJetGraphs = pickle.load(f)
                        graphs_per_type[wt] = singleJetGraphs
                
                # Add graphs to main dictionary
                for wt in weight_types:
                    self.graphs_dict[wt] += graphs_per_type[wt]
                
                # Assuming all weight types have same number of graphs per jet type
                self.sampleCountPerClass.append(len(graphs_per_type[weight_types[0]]))
                
                del graphs_per_type
            else:
                # Handle grouped jet types if needed
                totalCount = 0
                for item in jetType:
                    # Similar loading logic for grouped types
                    pass
                self.sampleCountPerClass.append(totalCount)
        
        # Create labels
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
        # Return all graph types for the same jet
        graphs = {wt: self.graphs_dict[wt][idx] for wt in self.weight_types}
        return {'graphs': graphs, 'label': self.labels[idx]}
    
    def __len__(self):
        return len(self.labels)


class MultiDGCNNClassifier(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k, num_graph_types=4):
        super(MultiDGCNNClassifier, self).__init__()
        
        # Create separate DGCNN for each graph type
        self.dgcnn_models = nn.ModuleList([
            self.create_dgcnn(in_feats, hidden_feats, k) 
            for _ in range(num_graph_types)
        ])
        
        # Final classifier takes concatenated features
        self.classifier = nn.Sequential(
            nn.Linear(hidden_feats * num_graph_types, hidden_feats * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_feats * 2, hidden_feats),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_feats, out_feats)
        )
    
    def create_dgcnn(self, in_feats, hidden_feats, k):
        """Create a single DGCNN module"""
        return nn.ModuleDict({
            'conv1': dgl.nn.ChebConv(in_feats, hidden_feats, k),
            'conv2': dgl.nn.ChebConv(hidden_feats, hidden_feats, k),
            'conv3': dgl.nn.ChebConv(hidden_feats, hidden_feats, k),
            'edgeconv1': dgl.nn.EdgeConv(hidden_feats, hidden_feats),
            'edgeconv2': dgl.nn.EdgeConv(hidden_feats, hidden_feats)
        })
    
    def forward_single_dgcnn(self, g, dgcnn):
        """Forward pass for a single DGCNN"""
        h = F.relu(dgcnn['conv1'](g, g.ndata['feat']))
        h = F.relu(dgcnn['edgeconv1'](g, h))
        h = F.relu(dgcnn['conv2'](g, h))
        h = F.relu(dgcnn['edgeconv2'](g, h))
        h = F.relu(dgcnn['conv3'](g, h))
        
        # Store node embeddings
        g.ndata['h'] = h
        
        # Global mean pooling
        hg = dgl.mean_nodes(g, 'h')
        return hg
    
    def forward(self, graphs_list):
        """
        graphs_list: list of batched graphs, one for each weight type
        """
        # Process each graph type with its corresponding DGCNN
        graph_representations = []
        for i, g in enumerate(graphs_list):
            hg = self.forward_single_dgcnn(g, self.dgcnn_models[i])
            graph_representations.append(hg)
        
        # Concatenate all representations
        combined_features = torch.cat(graph_representations, dim=1)
        
        # Final classification
        logits = self.classifier(combined_features)
        return logits


# Modified collate function for multiple graphs
def collateFunction(batch):
    # Separate graphs by type
    graphs_by_type = {wt: [] for wt in batch[0]['graphs'].keys()}
    labels = []
    
    for item in batch:
        for wt, graph in item['graphs'].items():
            graphs_by_type[wt].append(graph)
        labels.append(item['label'])
    
    # Batch each graph type separately
    batched_graphs = [dgl.batch(graphs_by_type[wt]) for wt in sorted(graphs_by_type.keys())]
    
    return batched_graphs, torch.tensor(labels)


# Main training script
if __name__ == "__main__":
    # Define jet types
    Higgs = ['HToBB', 'HToCC', 'HToGG', 'HToWW2Q1L', 'HToWW4Q']
    Vector = ['WToQQ', 'ZToQQ']
    Top = ['TTBar', 'TTBarLep']
    QCD = ['ZJetsToNuNu']
    allJets = Higgs + Vector + Top + QCD
    
    testingSet = Top + Vector + QCD + Higgs
    testingSet = [s + "-Testing" for s in testingSet]
    
    jetNames = testingSet
    print(jetNames)
    
    # Define weight types to use
    weight_types = ['delta', 'kT', 'mSquare', 'Z']  # Add 'binary' if you have those graphs
    
    # Create dataset
    dataset = MultiGraphDataset(jetNames, 3, weight_types=weight_types, loadFromDisk=False)
    dataset.process()
    
    from dgllife.utils import RandomSplitter
    
    maxEpochs = int(input("Max Epochs: "))
    
    if maxEpochs != 0:
        train, val, test = RandomSplitter().train_val_test_split(
            dataset, frac_train=0.8, frac_test=0.1, frac_val=0.1, random_state=42
        )
    else:
        train = dataset
    
    batchSize = int(input("Batch Size: "))
    
    # Create data loaders
    if maxEpochs != 0:
        trainLoader = DataLoader(train, batch_size=batchSize, shuffle=True, collate_fn=collateFunction, drop_last=True)
        validationLoader = DataLoader(val, batch_size=batchSize, shuffle=True, collate_fn=collateFunction, drop_last=True)
        testLoader = DataLoader(test, batch_size=batchSize, shuffle=True, collate_fn=collateFunction, drop_last=True)
    else:
        testLoader = DataLoader(train, batch_size=batchSize, shuffle=True, collate_fn=collateFunction, drop_last=True)
    
    # Model configuration
    device = input("cuda or cpu: ")
    classificationLevel = input("Classification Level: ")
    modelArchitecture = input("Model Architecture Name: ")
    modelSaveFile = f"modelSaveFiles/{classificationLevel}{modelArchitecture}_multi.pt"
    load = input("Load from a save file (Y or N): ")
    
    in_feats = 16
    hidden_feats = 64
    out_feats = len(jetNames)
    chebFilterSize = 16
    
    # Initialize wandb
    wandb.init(
        project="PCN-MultiGraph",
        name=f"{classificationLevel}-{modelArchitecture}-Multi",
        config={
            "epochs": maxEpochs,
            "batch_size": batchSize,
            "model": f"{modelArchitecture}-Multi",
            "in_feats": in_feats,
            "hidden_feats": hidden_feats,
            "out_feats": out_feats,
            "device": device,
            "weight_types": weight_types
        }
    )
    
    # Create model
    model = MultiDGCNNClassifier(
        in_feats, hidden_feats, out_feats, chebFilterSize, 
        num_graph_types=len(weight_types)
    )
    
    # Load model if requested
    while type(load) != bool:
        if load == "Y":
            load = True
        elif load == "N":
            load = False
        else:
            print("Invalid Input Please Enter Y or N: ")
            load = input("Load from a save file (Y or N): ")
    
    if load:
        model.load_state_dict(torch.load(modelSaveFile))
    
    model.to(device)
    wandb.watch(model, log='all', log_freq=100)
    
    convergence_threshold = float(input("Convergence Threshold: "))
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Training variables
    trainingLossTracker = []
    trainingAccuracyTracker = []
    validationLossTracker = []
    validationAccuracyTracker = []
    bestLoss = float('inf')
    epochs_without_improvement = 0
    epochsTillQuit = 10
    
    # Training loop
    for epoch in range(maxEpochs):
        runningLoss = 0
        totalCorrectPredictions = 0
        totalSamples = 0
        valTotalCorrectPredictions = 0
        valTotalSamples = 0
        model.train()
        
        for batchIndex, (graphs_list, labels) in tqdm(enumerate(trainLoader), total=len(trainLoader), leave=False):
            # Move all graphs and labels to device
            graphs_list = [g.to(device) for g in graphs_list]
            labels = labels.to(device).long()
            
            # Forward pass
            logits = model(graphs_list)
            
            # Calculate loss and backpropagate
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            runningLoss += loss.item()
            predictions = logits.argmax(dim=1)
            batchCorrectPredictions = (predictions == labels).sum().item()
            batchTotalSamples = labels.numel()
            totalCorrectPredictions += batchCorrectPredictions
            totalSamples += batchTotalSamples
            
            del graphs_list, labels, logits, predictions
        
        # Compute epoch statistics
        epochLoss = runningLoss / len(trainLoader)
        trainingLossTracker.append(epochLoss)
        epochAccuracy = totalCorrectPredictions / totalSamples
        trainingAccuracyTracker.append(epochAccuracy)
        
        torch.save(model.state_dict(), modelSaveFile)
        print(f'Saved Model to file {modelSaveFile}')
        
        # Validation
        model.eval()
        validationLoss = 0.0
        
        with torch.no_grad():
            for graphs_list, labels in tqdm(validationLoader, total=len(validationLoader), leave=False):
                graphs_list = [g.to(device) for g in graphs_list]
                labels = labels.to(device)
                
                logits = model(graphs_list)
                loss = criterion(logits, labels)
                
                validationLoss += loss.item()
                predictions = logits.argmax(dim=1)
                batchCorrectPredictions = (predictions == labels).sum().item()
                batchTotalSamples = labels.numel()
                
                valTotalCorrectPredictions += batchCorrectPredictions
                valTotalSamples += batchTotalSamples
                
                del graphs_list, labels, logits, predictions
        
        avgValidationLoss = validationLoss / len(validationLoader)
        validationLossTracker.append(avgValidationLoss)
        validationAccuracy = valTotalCorrectPredictions / valTotalSamples
        validationAccuracyTracker.append(validationAccuracy)
        
        # Check for convergence
        if avgValidationLoss < bestLoss - convergence_threshold:
            bestLoss = avgValidationLoss
            bestStateDict = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Print and log results
        print(f"Epoch {epoch + 1} - Training Loss={epochLoss:.4f} - Validation Loss={avgValidationLoss:.4f} - "
              f"Training Accuracy={epochAccuracy:.4f} - Validation Accuracy={validationAccuracy:.4f}")
        
        # Log to wandb
        grad_norm = sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None) ** 0.5
        
        wandb_log = {
            "Epoch": epoch + 1,
            "Training Loss": epochLoss,
            "Validation Loss": avgValidationLoss,
            "Training Accuracy": epochAccuracy,
            "Validation Accuracy": validationAccuracy,
            "Gradient Norm": grad_norm
        }
        
        if torch.cuda.is_available() and device == "cuda":
            gpu_mem_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            wandb_log["GPU Memory Used (MB)"] = gpu_mem_used
        
        wandb.log(wandb_log)
        
        # Check convergence criteria
        if epochs_without_improvement >= epochsTillQuit:
            print(f'Convergence achieved at epoch {epoch + 1}. Stopping training.')
            break
    
    # Save best model
    if maxEpochs != 0:
        torch.save(bestStateDict, modelSaveFile)
    
    print("Training completed!")