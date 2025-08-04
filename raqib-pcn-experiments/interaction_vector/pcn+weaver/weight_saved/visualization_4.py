import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

def load_graphs(file_path):
    try:
        with open(file_path, 'rb') as f:
            graphs = pickle.load(f)
        return graphs
    except Exception as e:
        print(f"Failed to load the file: {e}")
        return None

def visualize_graph_with_edge_values(graph, title="Graph Visualization"):
    """
    Visualize a DGL graph with node numbers and edge values displayed
    """
    # Convert DGL graph to NetworkX
    nx_graph = graph.to_networkx(edge_attrs=list(graph.edata.keys()) if graph.edata else None)
    
    # Create a very large figure for maximum readability
    plt.figure(figsize=(20, 20), dpi=300)  # Much larger figure
    
    # AUTOMATIC LAYOUTS - These algorithms find optimal positions automatically
    
    print("Calculating optimal node positions...")
    
    # Try different automatic layouts (uncomment the one you want):
    
    # Layout 1: Fruchterman-Reingold (Most popular, good for most graphs)
    pos = nx.fruchterman_reingold_layout(nx_graph, k=5, iterations=500, seed=42)
    pos = {node: (coord[0]*20, coord[1]*20) for node, coord in pos.items()}  # Scale up
    layout_name = "Fruchterman-Reingold (Force-directed)"
    
    # Layout 2: Kamada-Kawai (Best spacing, slower but very good results)
    # pos = nx.kamada_kawai_layout(nx_graph, scale=15)
    # layout_name = "Kamada-Kawai (Optimal spacing)"
    
    # Layout 3: Spring layout with optimal parameters
    # pos = nx.spring_layout(nx_graph, k=8, iterations=1000, seed=42)
    # pos = {node: (coord[0]*25, coord[1]*25) for node, coord in pos.items()}
    # layout_name = "Spring (High iterations)"
    
    # Layout 4: SFDP (Good for large dense graphs - requires graphviz)
    # try:
    #     pos = nx.nx_agraph.graphviz_layout(nx_graph, prog='sfdp', args='-Goverlap=false -Gsplines=true')
    #     layout_name = "SFDP (Graphviz)"
    # except:
    #     print("Graphviz not available, using Fruchterman-Reingold instead")
    #     pos = nx.fruchterman_reingold_layout(nx_graph, k=5, iterations=500, seed=42)
    #     pos = {node: (coord[0]*20, coord[1]*20) for node, coord in pos.items()}
    #     layout_name = "Fruchterman-Reingold (Fallback)"
    
    # Layout 5: Spectral layout (Uses graph eigenvalues - very mathematical)
    # try:
    #     pos = nx.spectral_layout(nx_graph, scale=15)
    #     layout_name = "Spectral (Eigenvalue-based)"
    # except:
    #     pos = nx.fruchterman_reingold_layout(nx_graph, k=5, iterations=500, seed=42)
    #     pos = {node: (coord[0]*20, coord[1]*20) for node, coord in pos.items()}
    #     layout_name = "Fruchterman-Reingold (Fallback)"
    
    # Layout 6: Multi-level layout (For very large graphs)
    # try:
    #     pos = nx.nx_agraph.graphviz_layout(nx_graph, prog='neato', args='-Goverlap=false')
    #     layout_name = "Neato (Multi-level)"
    # except:
    #     pos = nx.fruchterman_reingold_layout(nx_graph, k=5, iterations=500, seed=42)
    #     pos = {node: (coord[0]*20, coord[1]*20) for node, coord in pos.items()}
    #     layout_name = "Fruchterman-Reingold (Fallback)"
    
    print(f"Using {layout_name} layout")
    
    # Get edge weights if available
    edge_labels = {}
    edge_widths = []
    
    if graph.edata and 'weight' in graph.edata:
        weights = graph.edata['weight']
        if hasattr(weights, 'numpy'):
            edge_weights = weights.numpy()
        else:
            edge_weights = weights
        
        # Create edge labels with weights
        for i, (u, v) in enumerate(nx_graph.edges()):
            edge_labels[(u, v)] = f'{edge_weights[i]:.3f}'
        
        # Normalize edge weights for line thickness
        min_weight = edge_weights.min()
        max_weight = edge_weights.max()
        if max_weight > min_weight:
            normalized_weights = (edge_weights - min_weight) / (max_weight - min_weight)
            edge_widths = normalized_weights * 3 + 0.5  # Scale to 0.5-3.5 range
        else:
            edge_widths = [1.0] * len(edge_weights)
    else:
        # Default edge labels and widths if no weights
        for u, v in nx_graph.edges():
            edge_labels[(u, v)] = '1.0'
        edge_widths = [1.0] * nx_graph.number_of_edges()
    
    # Draw edges first (so they appear behind nodes)
    nx.draw_networkx_edges(
        nx_graph, pos,
        width=edge_widths,
        alpha=0.6,
        edge_color="gray"
    )
    
    # Draw nodes with larger size for the big image
    nx.draw_networkx_nodes(
        nx_graph, pos,
        node_size=1500,  # Much larger nodes
        node_color="skyblue",
        alpha=0.9,
        edgecolors='black',
        linewidths=2
    )
    
    # Draw node labels (node numbers) with larger font
    nx.draw_networkx_labels(
        nx_graph, pos,
        labels={node: str(node) for node in nx_graph.nodes()},
        font_size=14,  # Larger font
        font_color="black",
        font_weight="bold"
    )
    
    # Draw edge labels (weights) with larger font
    nx.draw_networkx_edge_labels(
        nx_graph, pos,
        edge_labels,
        font_size=12,  # Larger font
        font_color="red",
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9)
    )
    
    # Set title with larger font
    plt.title(f"{title}\nNodes: {nx_graph.number_of_nodes()}, Edges: {nx_graph.number_of_edges()}", 
              fontsize=18, fontweight='bold', pad=30)  # Larger title
    plt.axis('off')
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    return plt

# Load the delta interaction graphs
interaction_weight = 'delta'  # Changed from 'Z' to 'delta'
file_path = f"weighted_graphs/{interaction_weight}/{interaction_weight}_HToBB_first_10.pkl"

print(f"Loading graphs from: {file_path}")
graphs = load_graphs(file_path)

if graphs:
    print(f"Successfully loaded {len(graphs)} graphs")
    
    # Visualize only the first graph
    first_graph = graphs[0]
    
    print(f"\nFirst graph summary:")
    print(f"  Nodes: {first_graph.number_of_nodes()}")
    print(f"  Edges: {first_graph.number_of_edges()}")
    
    # Print node and edge data info
    print(f"\nNode features:")
    for key in first_graph.ndata:
        print(f"  {key}: {first_graph.ndata[key].shape}")
    
    print(f"\nEdge features:")
    if first_graph.edata:
        for key in first_graph.edata:
            print(f"  {key}: {first_graph.edata[key].shape}")
            if key == 'weight':
                weights = first_graph.edata[key]
                if hasattr(weights, 'numpy'):
                    w_array = weights.numpy()
                else:
                    w_array = weights
                print(f"    Weight range: {w_array.min():.3f} to {w_array.max():.3f}")
                print(f"    Weight mean: {w_array.mean():.3f}")
    else:
        print("  No edge features")
    
    # Create the visualization
    plt_obj = visualize_graph_with_edge_values(
        first_graph, 
        title=f"{interaction_weight.upper()} Interaction - HToBB Graph"
    )
    
    # Save as high-quality PNG
    output_filename = f'{interaction_weight}_HToBB_first_graph_with_edge_values.png'
    plt_obj.savefig(output_filename, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
    
    print(f"\nVisualization saved as: {output_filename}")
    
    # Show the plot
    plt_obj.show()
    
else:
    print("Failed to load graphs. Please check the file path.")