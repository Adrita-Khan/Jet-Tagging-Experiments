import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch

# Paths
source_folder = '../data/Multi Level Jet Tagging/Delta'
destination_root = '../graphs_images/'

# Ensure the root destination folder exists
os.makedirs(destination_root, exist_ok=True)

def load_first_10_graphs(file_path):
    """Loads only the first 10 graphs from a given .pkl file."""
    try:
        with open(file_path, 'rb') as f:
            graphs = pickle.load(f)
        print(f"Successfully loaded {len(graphs)} graphs from {file_path}")
        return graphs[:10]  # Load only the first 10 graphs
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None

def extract_weights_from_dgl(dgl_graph):
    """Extract edge weights directly from DGL graph's adjacency matrix."""
    try:
        # Method 1: Check if weights are already in edata
        if 'weight' in dgl_graph.edata:
            weights = dgl_graph.edata['weight'].numpy()
            print(f"  Found weights in edata: {len(weights)} weights")
            return weights
        
        # Method 2: Extract from adjacency matrix
        adj_matrix = dgl_graph.adjacency_matrix(transpose=False, scipy_fmt='coo')
        if hasattr(adj_matrix, 'data') and len(adj_matrix.data) > 0:
            # The adjacency matrix data contains the weights
            weights = adj_matrix.data
            print(f"  Extracted weights from adjacency matrix: {len(weights)} weights")
            print(f"  Weight range: {weights.min():.3f} to {weights.max():.3f}")
            return weights
        
        # Method 3: If adjacency matrix has no weights, try dense format
        adj_dense = dgl_graph.adjacency_matrix(transpose=False).to_dense().numpy()
        
        # Get non-zero elements (excluding diagonal if it's all zeros)
        weights = []
        edges = dgl_graph.edges()
        src, dst = edges[0].numpy(), edges[1].numpy()
        
        for i in range(len(src)):
            weight = adj_dense[src[i], dst[i]]
            if weight != 0:
                weights.append(weight)
        
        if weights:
            weights = np.array(weights)
            print(f"  Extracted weights from dense adjacency: {len(weights)} weights")
            return weights
        
        print("  No weights found")
        return None
        
    except Exception as e:
        print(f"  Error extracting weights: {e}")
        return None

def visualize_dgl_graph_directly(dgl_graph, graph_idx, save_path):
    """Visualize DGL graph directly with edge weights."""
    
    print(f"\nProcessing DGL graph {graph_idx}")
    print(f"  Nodes: {dgl_graph.number_of_nodes()}, Edges: {dgl_graph.number_of_edges()}")
    
    # Skip empty graphs
    if dgl_graph.number_of_nodes() == 0 or dgl_graph.number_of_edges() == 0:
        print(f"  Skipping empty graph {graph_idx}")
        return False
    
    # Extract edge weights
    edge_weights = extract_weights_from_dgl(dgl_graph)
    
    # Get edges from DGL
    edges = dgl_graph.edges()
    src_nodes = edges[0].numpy()
    dst_nodes = edges[1].numpy()
    
    # Create NetworkX graph
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(dgl_graph.number_of_nodes()))
    
    # Add edges with weights
    for i in range(len(src_nodes)):
        u, v = int(src_nodes[i]), int(dst_nodes[i])
        weight = float(edge_weights[i]) if edge_weights is not None else 1.0
        
        # For undirected graph, avoid duplicate edges
        if not nx_graph.has_edge(u, v):
            nx_graph.add_edge(u, v, weight=weight)
    
    print(f"  Created NetworkX graph with {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")
    
    # Create visualization
    plt.figure(figsize=(16, 14))
    
    # Layout
    try:
        if nx_graph.number_of_nodes() > 1:
            pos = nx.spring_layout(nx_graph, k=2, iterations=50)
        else:
            pos = {0: (0, 0)}
    except:
        pos = nx.random_layout(nx_graph)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        nx_graph, pos,
        node_size=800,
        node_color="lightcoral",
        alpha=0.9,
        edgecolors="black",
        linewidths=2
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        nx_graph, pos,
        edge_color="darkgray",
        alpha=0.7,
        width=2
    )
    
    # Draw node labels
    nx.draw_networkx_labels(
        nx_graph, pos,
        font_size=12,
        font_color="black",
        font_weight="bold"
    )
    
    # Draw edge weight labels
    edge_labels = {}
    weights_displayed = 0
    
    for u, v, data in nx_graph.edges(data=True):
        if 'weight' in data:
            weight = round(data['weight'], 3)
            edge_labels[(u, v)] = str(weight)
            weights_displayed += 1
    
    if edge_labels:
        nx.draw_networkx_edge_labels(
            nx_graph, pos, edge_labels,
            font_size=9,
            font_color="red",
            font_weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="red")
        )
        print(f"  ✓ Displayed {weights_displayed} edge weights")
    else:
        print(f"  ✗ No edge weights to display")
    
    # Title and info
    title = f"DGL Graph {graph_idx} - Direct Visualization"
    plt.title(title, fontsize=20, fontweight='bold', pad=25)
    
    info_text = f"Nodes: {nx_graph.number_of_nodes()}, Edges: {nx_graph.number_of_edges()}"
    if edge_weights is not None:
        weight_stats = f"Weight range: [{edge_weights.min():.3f}, {edge_weights.max():.3f}]"
        info_text += f"\n{weight_stats}"
    
    plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    
    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {save_path}")
    return True

def process_dgl_graphs_directly(graphs, file_name):
    """Process DGL graphs directly without conversion issues."""
    
    if not graphs:
        print(f"No graphs found in {file_name}")
        return
    
    print(f"\n=== Processing {len(graphs)} DGL graphs from {file_name} ===")
    
    # Create output directory
    output_dir = os.path.join(destination_root, f'DGL_Direct_{file_name}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    successful_visualizations = 0
    
    for idx, dgl_graph in enumerate(graphs):
        save_path = os.path.join(output_dir, f"DGL_direct_graph_{idx+1}.png")
        
        success = visualize_dgl_graph_directly(dgl_graph, idx+1, save_path)
        if success:
            successful_visualizations += 1
    
    print(f"\n✓ Successfully visualized {successful_visualizations}/{len(graphs)} graphs")

def inspect_graph_structure(graphs, file_name):
    """Inspect the structure of DGL graphs to understand the data."""
    
    if not graphs:
        return
    
    print(f"\n=== Inspecting DGL graphs from {file_name} ===")
    
    # Look at first few graphs
    for i, graph in enumerate(graphs[:3]):
        print(f"\nGraph {i+1}:")
        print(f"  Type: {type(graph)}")
        print(f"  Nodes: {graph.number_of_nodes()}")
        print(f"  Edges: {graph.number_of_edges()}")
        print(f"  Node data keys: {list(graph.ndata.keys())}")
        print(f"  Edge data keys: {list(graph.edata.keys())}")
        
        # Check adjacency matrix
        try:
            adj = graph.adjacency_matrix(scipy_fmt='coo')
            print(f"  Adjacency matrix shape: {adj.shape}")
            print(f"  Non-zero elements: {adj.nnz}")
            if hasattr(adj, 'data') and len(adj.data) > 0:
                print(f"  Weight range in adj matrix: [{adj.data.min():.3f}, {adj.data.max():.3f}]")
            
            # Sample some weights
            if hasattr(adj, 'data') and len(adj.data) >= 5:
                print(f"  Sample weights: {adj.data[:5]}")
                
        except Exception as e:
            print(f"  Error accessing adjacency matrix: {e}")

def process_all_files():
    """Process all pickle files with DGL graphs."""
    
    print(f"Looking for .pkl files in: {os.path.abspath(source_folder)}")
    
    if not os.path.exists(source_folder):
        print(f"ERROR: Source folder does not exist: {source_folder}")
        return
    
    all_files = os.listdir(source_folder)
    pkl_files = [f for f in all_files if f.endswith('.pkl')]
    
    print(f"Found {len(pkl_files)} .pkl files: {pkl_files}")
    
    if not pkl_files:
        print("No .pkl files found!")
        return
    
    for file_name in pkl_files:
        file_path = os.path.join(source_folder, file_name)
        print(f"\n{'='*60}")
        print(f"Processing: {file_name}")
        print(f"{'='*60}")
        
        graphs = load_first_10_graphs(file_path)
        
        if graphs:
            # First inspect the structure
            inspect_graph_structure(graphs, file_name.replace('.pkl', ''))
            
            # Then visualize
            process_dgl_graphs_directly(graphs, file_name.replace('.pkl', ''))
        else:
            print(f"Failed to load graphs from {file_name}")

# Main execution
if __name__ == "__main__":
    print("Starting direct DGL graph visualization...")
    print("This will work with your existing graph files without any modifications!")
    process_all_files()
    print("\nDirect DGL visualization completed!")