import pickle
import dgl
import networkx as nx
import os
import json
import csv
import math
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
    components = list(nx.connected_components(graph))
    component_sequence = {}
    sequential_order = True
    for idx, comp in enumerate(components):
        comp_sorted = sorted([int(n) for n in comp])
        component_sequence[idx] = comp_sorted
        if comp_sorted != list(range(comp_sorted[0], comp_sorted[-1] + 1)):
            sequential_order = False
    return component_sequence, sequential_order

def compute_centrality(graph):
    return {
        "degree": {int(k): v for k, v in nx.degree_centrality(graph).items()},
        "betweenness": {int(k): v for k, v in nx.betweenness_centrality(graph).items()},
        "eigenvector": {int(k): v for k, v in nx.eigenvector_centrality(graph, max_iter=100000000).items()}
    }

def compute_entropy_from_centrality(centrality_dict):
    values = list(centrality_dict.values())
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v / total for v in values]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return entropy

def compute_radius_and_diameter(graph):
    if nx.is_connected(graph):
        radius = nx.radius(graph)
        diameter = nx.diameter(graph)
    else:
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc)
        radius = nx.radius(subgraph)
        diameter = nx.diameter(subgraph)
    return {"radius": radius, "diameter": diameter}

def save_graph_info(graph_info, output_folder):
    csv_path = os.path.join(output_folder, "Emitter-Higgs_HToBB.csv")
    json_path = os.path.join(output_folder, "Emitter-Higgs_HToBB.json")

    graph_name = graph_info["graph_name"]
    sparsity = graph_info["sparsity_info"]
    components = graph_info["component_info"]
    sequential_order = graph_info["sequential_order"]
    component_node_sequence = graph_info["component_node_sequence"]
    centrality_degree = graph_info["centrality_degree"]
    centrality_entropy = graph_info["centrality_entropy"]
    radius = graph_info["radius"]
    diameter = graph_info["diameter"]

    row_data = {
        "graph_name": graph_name,
        "num_nodes": sparsity["num_nodes"],
        "num_edges": sparsity["num_edges"],
        "density": round(sparsity["density"], 5),
        "num_of_connected_components": components["num_components"],
        "largest_component_size": components["largest_component_size"],
        "all_component_sizes": components["all_component_sizes"],
        "sequential_order": sequential_order,
        "component_node_sequence": component_node_sequence,
        "centrality_degree (Top 5)": centrality_degree,
        "centrality_entropy": round(centrality_entropy, 5),
        "radius": radius,
        "diameter": diameter,
        "JetType": "HToBB"
    }

    # Save JSON
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = []

    existing_data.append(row_data)

    with open(json_path, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)

    # Save CSV using DictWriter to ensure correct field order
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as csv_file:
        fieldnames = list(row_data.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

def analyze_graph_properties(pkl_path, output_folder):
    graphs = load_graphs(pkl_path)
    if not graphs:
        print("No graphs found in the file!")
        return

    os.makedirs(output_folder, exist_ok=True)

    for i, graph in enumerate(graphs):
        graph_name = f"Emitter-Higgs_HToBB_{i+1}"
        print(f"Processing graph {graph_name}...")

        nx_graph = convert_to_simple_graph(graph)
        sparsity_info = compute_sparsity(nx_graph)
        component_info = get_connected_components(nx_graph)
        centrality = compute_centrality(nx_graph)
        centrality_entropy = compute_entropy_from_centrality(centrality["degree"])
        component_node_sequence, sequential_order = get_component_node_sequence(nx_graph)
        radius_diameter = compute_radius_and_diameter(nx_graph)

        graph_info = {
            "graph_name": graph_name,
            "sparsity_info": sparsity_info,
            "component_info": component_info,
            "sequential_order": sequential_order,
            "component_node_sequence": component_node_sequence,
            "centrality_degree": dict(list(sorted(centrality['degree'].items(), key=lambda x: x[1], reverse=True))[:5]),
            "centrality_entropy": centrality_entropy,
            "radius": radius_diameter["radius"],
            "diameter": radius_diameter["diameter"]
        }

        save_graph_info(graph_info, output_folder)

# File paths
pkl_file = '../data/Multi Level Jet Tagging/Emitter-Higgs_HToBB.pkl'
output_folder = "./graph_analysis_results/HToBB"

# Run analysis
analyze_graph_properties(pkl_file, output_folder)
