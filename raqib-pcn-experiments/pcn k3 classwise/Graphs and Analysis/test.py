import pickle
import dgl
import networkx as nx
import matplotlib.pyplot as plt
import os
import json
import csv
import numpy as np
from dgl import to_networkx

def load_graphs(pkl_path):
    with open(pkl_path, 'rb') as f:
        graphs = pickle.load(f)
    return graphs if graphs else []

def convert_to_simple_graph(graph):
    if isinstance(graph, dgl.DGLGraph):
        nx_graph = to_networkx(graph)
    else:
        nx_graph = graph
    simple_graph = nx.Graph()
    simple_graph.add_edges_from(nx_graph.edges())
    simple_graph.add_nodes_from(nx_graph.nodes())
    return simple_graph

def compute_sparsity(graph):
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    density = nx.density(graph) if num_nodes > 1 else 0
    return {"num_nodes": num_nodes, "num_edges": num_edges, "density": density}

def get_connected_components(graph):
    components = list(nx.connected_components(graph))
    component_sizes = [len(comp) for comp in components]
    largest_component = max(component_sizes) if component_sizes else 0
    return {"num_components": len(components), "largest_component_size": largest_component, "all_component_sizes": component_sizes}

def get_component_node_sequence(graph):
    """
    Computes a dictionary mapping each component index to its sorted node labels (as integers)
    and a boolean flag indicating if all components have nodes in sequential order.
    """
    components = list(nx.connected_components(graph))
    component_sequence = {}
    sequential_order = True
    for idx, comp in enumerate(components):
        # Convert node labels to int and sort them
        comp_sorted = sorted([int(n) for n in comp])
        component_sequence[idx] = comp_sorted
        # Check if the sorted list is sequential
        if comp_sorted != list(range(comp_sorted[0], comp_sorted[-1] + 1)):
            sequential_order = False
    return component_sequence, sequential_order

def compute_centrality(graph):
    return {"degree": {int(k): v for k, v in nx.degree_centrality(graph).items()},
            "betweenness": {int(k): v for k, v in nx.betweenness_centrality(graph).items()},
            "eigenvector": {int(k): v for k, v in nx.eigenvector_centrality(graph, max_iter=100000000).items()}}

def visualize_graph(graph, title, sparsity_info, component_info, save_path):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(graph, pos, node_color="lightblue", edge_color="gray", alpha=0.6, with_labels=True, font_size=8)
    
    info_text = (f"Nodes: {sparsity_info['num_nodes']}\n"
                 f"Edges: {sparsity_info['num_edges']}\n"
                 f"Density: {sparsity_info['density']:.5f}\n"
                 f"Components: {component_info['num_components']}\n"
                 f"Largest Component: {component_info['largest_component_size']}")
    
    plt.text(0.05, 0.05, info_text, transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def save_graph_info(graph_info, output_folder):
    csv_path = os.path.join(output_folder, "Emitter-Vector_ZToQQ.csv")
    json_path = os.path.join(output_folder, "Emitter-Vector_ZToQQ.json")

    # Extract necessary fields
    graph_name = graph_info["graph_name"]
    sparsity = graph_info["sparsity_info"]
    components = graph_info["component_info"]
    sequential_order = graph_info["sequential_order"]
    component_node_sequence = graph_info["component_node_sequence"]
    centrality_degree = graph_info["centrality_degree"]

    # Prepare row data with the new columns added after all_component_sizes
    row_data = {
        "graph_name": graph_name,
        "num_nodes": sparsity["num_nodes"], 
        "num_edges": sparsity["num_edges"], 
        "density": round(sparsity["density"], 5),  # Round density for better readability
        "num_of_connected_components": components["num_components"], 
        "largest_component_size": components["largest_component_size"], 
        "all_component_sizes": components["all_component_sizes"],
        "sequential_order": sequential_order,
        "component_node_sequence": component_node_sequence,
        "centrality_degree (Top 5)": centrality_degree,
        "JetType": "Vector_ZToQQ"
    }

    # Save as JSON
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = []

    existing_data.append(row_data)

    with open(json_path, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)

    # Save as CSV
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(["graph_name", "num_nodes", "num_edges", "density", 
                             "num_of_connected_components", "largest_component_size", 
                             "all_component_sizes", "sequential_order", "component_node_sequence", 
                             "centrality_degree (Top 5)", "JetType"])
        writer.writerow(row_data.values())

def analyze_graph_properties(pkl_path, output_folder):
    graphs = load_graphs(pkl_path)
    if not graphs:
        print("No graphs found in the file!")
        return
    
    os.makedirs(output_folder, exist_ok=True)
    image_folder = os.path.join(output_folder, "Graph Images of Emitter-Vector_ZToQQ")
    os.makedirs(image_folder, exist_ok=True)
    
    for i, graph in enumerate(graphs):
        graph_name = f"Emitter-Vector_ZToQQ_{i+1}"
        print(f"Processing graph {graph_name}...")

        nx_graph = convert_to_simple_graph(graph)
        sparsity_info = compute_sparsity(nx_graph)
        component_info = get_connected_components(nx_graph)
        centrality = compute_centrality(nx_graph)
        
        # Compute the node sequence for each component and check sequential order
        component_node_sequence, sequential_order = get_component_node_sequence(nx_graph)

        graph_info = {
            "graph_name": graph_name,
            "sparsity_info": sparsity_info,
            "component_info": component_info,
            "sequential_order": sequential_order,
            "component_node_sequence": component_node_sequence,
            "centrality_degree": dict(list(sorted(centrality['degree'].items(), key=lambda x: x[1], reverse=True))[:5])
        }

        save_graph_info(graph_info, output_folder)

        save_path = os.path.join(image_folder, f"{graph_name}.png")
        visualize_graph(nx_graph, title=f"Graph {i+1} with Properties Analyzed", 
                        sparsity_info=sparsity_info, component_info=component_info, 
                        save_path=save_path)

pkl_file = '../data/Multi Level Jet Tagging/Emitter-Vector_ZToQQ.pkl'
output_folder = "./graph_analysis_results/Vector_ZToQQ"
analyze_graph_properties(pkl_file, output_folder)
