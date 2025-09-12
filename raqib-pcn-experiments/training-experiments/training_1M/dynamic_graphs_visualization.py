print("Starting imports...")
import numpy as np
import pandas as pd
import torch
import dgl
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless server
import matplotlib.pyplot as plt
import networkx as nx
import os
import math
from tqdm import tqdm
print("All imports successful!")

# Create output directory
output_dir = "dynamic_graphs_visualization"
os.makedirs(output_dir, exist_ok=True)
print(f"Created output directory: {output_dir}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Weight computation functions (copied from main script)
def get_pTmin(part_i, part_j):
    pT_i = part_i[:, 4]
    pT_j = part_j[:, 4]
    pTmin = torch.minimum(pT_i, pT_j)
    return pTmin

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

def kT_weight(part_i, part_j, eps=1e-8):
    pTmin = get_pTmin(part_i, part_j)
    delta_ij = get_delta(part_i, part_j)
    lnkT = torch.log((pTmin * delta_ij).clamp(min=eps))
    return lnkT

def Z_weight(part_i, part_j, eps=1e-8):
    pTi = part_i[:, 4]
    pTj = part_j[:, 4]
    pTmin = get_pTmin(part_i, part_j)
    lnZ = torch.log((pTmin / (pTi + pTj).clamp(min=eps)).clamp(min=eps))
    return lnZ

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

# Compute edge weights (same logic as main script)
def compute_edge_weights_gpu(base_graph_cpu, weight_type, device):
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

# Create weighted graph (same logic as main script)
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

    # Final GPU cleanup after this graph
    torch.cuda.empty_cache()

    return new_graph_cpu

def visualize_weighted_graph(weighted_graph, weight_type, jet_type, graph_idx, save_dir):
    """Visualize a weighted graph and save both image and adjacency matrix"""
    
    # Convert DGL graph to NetworkX
    nx_graph = weighted_graph.to_networkx(edge_attrs=['weight'])
    
    # Get edge weights
    edges = list(nx_graph.edges(data=True))
    edge_weights = [d['weight'] for u, v, d in edges]
    
    # Create adjacency matrix
    adj_matrix = nx.adjacency_matrix(nx_graph, weight='weight').todense()
    
    # Save adjacency matrix as CSV
    adj_df = pd.DataFrame(adj_matrix)
    csv_filename = f"dynamic_{weight_type}_{jet_type}_{graph_idx}.csv"
    adj_df.to_csv(os.path.join(save_dir, csv_filename), index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(nx_graph, k=1, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(nx_graph, pos, node_color='lightblue', 
                          node_size=300, alpha=0.8)
    
    # Draw edges with colors based on weights
    if edge_weights:
        # Normalize weights for color mapping
        weight_array = np.array(edge_weights)
        norm_weights = (weight_array - weight_array.min()) / (weight_array.max() - weight_array.min()) if weight_array.max() != weight_array.min() else np.zeros_like(weight_array)
        
        # Draw edges
        edges_list = [(u, v) for u, v, d in edges]
        nx.draw_networkx_edges(nx_graph, pos, edgelist=edges_list, 
                              edge_color=norm_weights, edge_cmap=plt.cm.viridis,
                              alpha=0.6, width=1.5)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                  norm=plt.Normalize(vmin=weight_array.min(), vmax=weight_array.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label(f'{weight_type} weights', rotation=270, labelpad=15)
        
        # Draw edge labels with weight values
        edge_labels = {}
        for u, v, d in edges:
            edge_labels[(u, v)] = f"{d['weight']:.3f}"
        
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels, font_size=6, 
                                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
    
    # Draw node labels
    nx.draw_networkx_labels(nx_graph, pos, font_size=8)
    
    # Set title and save
    plt.title(f'{weight_type} Graph - {jet_type} (Graph #{graph_idx})\n{weighted_graph.num_nodes()} nodes, {weighted_graph.num_edges()} edges', 
              fontsize=14)
    plt.axis('off')
    
    # Save image
    img_filename = f"dynamic_{weight_type}_{jet_type}_{graph_idx}.png"
    plt.savefig(os.path.join(save_dir, img_filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {img_filename} and {csv_filename}")

def main():
    print("Starting HToBB dynamic graphs visualization...")
    print(f"Using device: {device}")
    print(f"Output directory: {output_dir}")
    
    # Load HToBB data
    htobb_path = 'data/HToBB-Testing.pkl'  # Adjust path if needed
    
    try:
        print(f"Loading data from: {htobb_path}")
        with open(htobb_path, 'rb') as f:
            base_graphs = pickle.load(f)
        print(f"Successfully loaded {len(base_graphs)} HToBB graphs")
    except FileNotFoundError:
        print(f"ERROR: File {htobb_path} not found. Please check the path.")
        return
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return
    
    # Find the FIRST graph with 20-30 nodes
    target_graph = None
    target_idx = None
    target_nodes = None
    
    print("Searching for first graph with 20-30 nodes...")
    for idx, graph in enumerate(base_graphs):
        num_nodes = graph.num_nodes()
        if 20 <= num_nodes <= 30:
            target_graph = graph
            target_idx = idx
            target_nodes = num_nodes
            print(f"Found first target graph: Graph {idx} with {num_nodes} nodes")
            break
    
    if target_graph is None:
        print("No graphs found with 20-30 nodes. Trying with different range (15-35)...")
        # Try a broader range if no graphs found
        for idx, graph in enumerate(base_graphs):
            num_nodes = graph.num_nodes()
            if 15 <= num_nodes <= 35:
                target_graph = graph
                target_idx = idx
                target_nodes = num_nodes
                print(f"Found first target graph in broader range: Graph {idx} with {num_nodes} nodes")
                break
    
    if target_graph is None:
        print("No suitable graphs found. Exiting...")
        return
    
    # Process the single target graph
    weight_types = ['delta', 'kT', 'Z', 'mSquare']
    
    print(f"\nProcessing graph {target_idx} ({target_nodes} nodes)...")
    
    for weight_type in weight_types:
        print(f"  Creating {weight_type} weighted graph...")
        
        # Create weighted graph using same logic as main script
        weighted_graph = create_weighted_graphs(target_graph, weight_type, device)
        
        # Visualize and save
        visualize_weighted_graph(weighted_graph, weight_type, 'HToBB', target_idx, output_dir)
        
        # Clean up
        del weighted_graph
        torch.cuda.empty_cache()
    
    print(f"\nVisualization complete! Check the '{output_dir}' folder for results.")

if __name__ == "__main__":
    main()
