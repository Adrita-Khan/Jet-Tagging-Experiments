import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt

# Paths
source_folder = './data/Multi Level Jet Tagging/'
destination_root = './data/Multi Level Jet Tagging/graphs_images/'

# Ensure the root destination folder exists
os.makedirs(destination_root, exist_ok=True)

def load_first_100_graphs(file_path):
    """Loads only the first 100 graphs from a given .pkl file."""
    try:
        with open(file_path, 'rb') as f:
            graphs = pickle.load(f)
        return graphs[:100]  # Load only the first 100 graphs
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None

def save_graph_images(graphs, file_name):
    """Saves graph visualizations as PNG images in a structured directory with clear titles."""
    if not graphs:
        print(f"No graphs found in {file_name}. Skipping...")
        return
    
    # Create destination folder for this particular file's graphs
    graph_folder = os.path.join(destination_root, f'Graphs_of_{file_name}')
    os.makedirs(graph_folder, exist_ok=True)

    for idx, g in enumerate(graphs):
        nx_graph = g.to_networkx()

        # Layout for better graph organization
        pos = nx.spring_layout(nx_graph)

        plt.figure(figsize=(12, 10))  # Increased figure size
        nx.draw(
            nx_graph, pos,
            with_labels=True,
            labels={node: node for node in nx_graph.nodes()},
            node_size=500,  # Increased node size
            node_color="skyblue",
            font_size=10,  # Increased font size
            font_color="black",
            edge_color="gray",
            alpha=0.9
        )
        
        # Add the title at the top in a clear way
        title_text = f"{file_name} Graph {idx+1}"
        plt.title(title_text, fontsize=18, fontweight='bold', pad=20)

        # Ensure the title is fully visible
        plt.tight_layout()

        # Save the graph image
        image_path = os.path.join(graph_folder, f"{title_text}.png")
        plt.savefig(image_path, dpi=300)
        plt.close()

        print(f"Saved: {image_path}")

def process_all_files():
    """Processes all .pkl files in the source folder."""
    for file_name in os.listdir(source_folder):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(source_folder, file_name)
            print(f"Processing {file_name}...")

            graphs = load_first_100_graphs(file_path)
            if graphs:
                save_graph_images(graphs, file_name.replace('.pkl', ''))

# Execute the processing
process_all_files()
