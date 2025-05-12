import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt

# Paths
pkl_folder = '../../data/Multi Level Jet Tagging/'  # <-- CHANGE THIS
visualizations_root = 'visualizations'

os.makedirs(visualizations_root, exist_ok=True)

def load_first_and_last_10_graphs(file_path):
    """Loads first 10 and last 10 graphs from a given .pkl file."""
    try:
        with open(file_path, 'rb') as f:
            graphs = pickle.load(f)
        total_graphs = len(graphs)
        if total_graphs <= 20:
            return graphs, total_graphs  # If not enough graphs, return all
        selected_graphs = graphs[:10] + graphs[-10:]
        return selected_graphs, total_graphs
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None, 0

def save_graph_images(graphs, class_name, total_graphs):
    """Saves graph visualizations as PNG images in a structured directory."""
    if not graphs:
        print(f"No graphs found for {class_name}. Skipping...")
        return
    
    # Create folder for this class
    class_folder = os.path.join(visualizations_root, class_name)
    os.makedirs(class_folder, exist_ok=True)

    for idx, g in enumerate(graphs):
        nx_graph = g.to_networkx()

        # Layout for better graph organization
        pos = nx.spring_layout(nx_graph)

        plt.figure(figsize=(8, 6))
        nx.draw(
            nx_graph, pos,
            with_labels=True,
            labels={node: node for node in nx_graph.nodes()},
            node_size=300,
            node_color="skyblue",
            font_size=8,
            font_color="black",
            edge_color="gray",
            alpha=0.9
        )

        plt.tight_layout()

        # Naming:
        # First 10 -> HToBB-1.png to HToBB-10.png
        # Last 10 -> HToBB-(total_graphs-9).png to HToBB-total_graphs.png
        if idx < 10:
            graph_number = idx + 1
        else:
            graph_number = total_graphs - 10 + (idx - 9)

        image_name = f"{class_name}-{graph_number}.png"
        image_path = os.path.join(class_folder, image_name)
        plt.savefig(image_path, dpi=300)
        plt.close()

        print(f"Saved: {image_path}")

def process_all_files():
    """Processes all .pkl files in the source folder."""
    for file_name in os.listdir(pkl_folder):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(pkl_folder, file_name)
            class_name = file_name.replace('.pkl', '')
            print(f"Processing {class_name}...")

            graphs, total_graphs = load_first_and_last_10_graphs(file_path)
            if graphs:
                save_graph_images(graphs, class_name, total_graphs)

# Execute the processing
if __name__ == "__main__":
    process_all_files()
