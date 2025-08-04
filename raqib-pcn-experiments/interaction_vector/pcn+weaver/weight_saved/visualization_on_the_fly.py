import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import math


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
    Visualize a DGL graph using a spring layout with node numbers and edge weights.
    """
    # Convert DGL graph to NetworkX
    nx_graph = graph.to_networkx(edge_attrs=list(graph.edata.keys()) if graph.edata else None)

    # Create a figure
    plt.figure(figsize=(10, 8))  # Compact and readable
    pos = nx.spring_layout(nx_graph, seed=42, k=0.5)  # Spring (force-directed) layout

    # Initialize edge labels and widths
    edge_labels = {}
    edge_widths = []

    if graph.edata and 'weight' in graph.edata:
        weights = graph.edata['weight']
        edge_weights = weights.numpy() if hasattr(weights, 'numpy') else weights

        for i, (u, v) in enumerate(nx_graph.edges()):
            edge_labels[(u, v)] = f'{edge_weights[i]:.2f}'

        min_weight = edge_weights.min()
        max_weight = edge_weights.max()
        if max_weight > min_weight:
            normalized_weights = (edge_weights - min_weight) / (max_weight - min_weight)
            edge_widths = normalized_weights * 2 + 0.5  # Scale for better visibility
        else:
            edge_widths = [1.0] * len(edge_weights)
    else:
        edge_labels = {(u, v): '1.0' for u, v in nx_graph.edges()}
        edge_widths = [1.0] * nx_graph.number_of_edges()

    # Draw edges
    nx.draw_networkx_edges(
        nx_graph, pos,
        width=edge_widths,
        alpha=0.6,
        edge_color="gray"
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        nx_graph, pos,
        node_size=500,
        node_color="skyblue",
        alpha=0.9,
        edgecolors='black',
        linewidths=1.5
    )

    # Draw node labels
    nx.draw_networkx_labels(
        nx_graph, pos,
        labels={node: str(node) for node in nx_graph.nodes()},
        font_size=10,
        font_color="black"
    )

    # Draw edge labels
    nx.draw_networkx_edge_labels(
        nx_graph, pos,
        edge_labels=edge_labels,
        font_size=8,
        font_color="red"
    )

    # Title and formatting
    plt.title(f"{title}\nNodes: {nx_graph.number_of_nodes()}, Edges: {nx_graph.number_of_edges()}",
              fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()

    return plt

# Load the delta interaction graphs
interaction_weight = 'delta'
file_path = f"weighted_graphs/{interaction_weight}/{interaction_weight}_HToBB_first_10.pkl"

print(f"Loading graphs from: {file_path}")
graphs = load_graphs(file_path)

def to_pt2(part_px, part_py, eps=1e-8):
    pt2 = part_px ** 2 + part_py ** 2
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2

def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi

def delta_weight_calculation(graph):
    part_px, part_py, part_pz, energy,  = graph.ndata['feat'][0][:4]
    phi = graph.ndata['feat'][0][6]

    part_px = torch.tensor(part_px)
    part_py = torch.tensor(part_py)
    part_pz = torch.tensor(part_pz)
    energy = torch.tensor(energy)
    phi = torch.tensor(phi)

    rap_i = rapidity(energy, part_pz)
    rap_j = rapidity(part_j)


if graphs:
    print(f"Successfully loaded {len(graphs)} graphs")

    # Visualize only the first graph
    unweighted_first_graph = graphs[0]

    weight = delta_weight_calculation(unweighted_first_graph)

    first_graph = unweighted_first_graph.edata['weight']

    print(f"\nFirst graph summary:")
    print(f"  Nodes: {first_graph.number_of_nodes()}")
    print(f"  Edges: {first_graph.number_of_edges()}")

    print(f"\nNode features:")
    for key in first_graph.ndata:
        print(f"  {key}: {first_graph.ndata[key].shape}")

    print(f"\nEdge features:")
    if first_graph.edata:
        for key in first_graph.edata:
            print(f"  {key}: {first_graph.edata[key].shape}")
            if key == 'weight':
                weights = first_graph.edata[key]
                w_array = weights.numpy() if hasattr(weights, 'numpy') else weights
                print(f"    Weight range: {w_array.min():.3f} to {w_array.max():.3f}")
                print(f"    Weight mean: {w_array.mean():.3f}")
    else:
        print("  No edge features")

    # Create and show the visualization
    plt_obj = visualize_graph_with_edge_values(
        first_graph,
        title=f"{interaction_weight.upper()} Interaction - HToBB Graph"
    )

    # Save the image
    output_filename = f'{interaction_weight}_HToBB_first_graph_with_edge_values.png'
    plt_obj.savefig(output_filename, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

    print(f"\nVisualization saved as: {output_filename}")
    plt_obj.show()

else:
    print("Failed to load graphs. Please check the file path.")
