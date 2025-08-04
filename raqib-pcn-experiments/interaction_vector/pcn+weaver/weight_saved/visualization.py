import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

# Load your graph
interaction_weight = 'delta'
with open(f"weighted_graphs/{interaction_weight}/{interaction_weight}_HToBB_first_10.pkl", "rb") as f:
    first_graph = pickle.load(f)[0]

g = first_graph

# Convert DGL graph to NetworkX for visualization
G = g.to_networkx(node_attrs=list(g.ndata.keys()), edge_attrs=list(g.edata.keys()) if g.edata else None)

# Create a large figure for better readability
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), dpi=300)

# ============ LEFT PLOT: Overview with node numbers ============
plt.sca(ax1)

# Use circular layout for dense graphs to spread nodes evenly
pos = nx.circular_layout(G)
# Alternative: try spring layout with more space
# pos = nx.spring_layout(G, k=5, iterations=100, seed=42)

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

# Get edge weights
edge_weights = []
edge_labels = {}
if g.edata and 'weight' in g.edata:
    weights = g.edata['weight']
    if hasattr(weights, 'numpy'):
        edge_weights = weights.numpy()
    else:
        edge_weights = weights
    
    # Create edge labels with rounded weights
    for i, (u, v) in enumerate(G.edges()):
        edge_labels[(u, v)] = f'{edge_weights[i]:.2f}'
    
    # Normalize edge weights for line thickness
    normalized_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-8)
    edge_widths = normalized_weights * 2 + 0.3
else:
    edge_widths = [0.5] * len(G.edges())
    for u, v in G.edges():
        edge_labels[(u, v)] = '1.0'

# Node colors based on degree
node_colors = [G.degree(n) for n in G.nodes()]

# Draw edges
nx.draw_networkx_edges(G, pos, 
                      width=edge_widths,
                      alpha=0.4,
                      edge_color='gray',
                      ax=ax1)

# Draw nodes with larger size
scatter1 = nx.draw_networkx_nodes(G, pos,
                                 node_color=node_colors,
                                 node_size=800,  # Larger nodes
                                 cmap=plt.cm.viridis,
                                 alpha=0.8,
                                 ax=ax1)

# Draw node labels (node numbers)
nx.draw_networkx_labels(G, pos, 
                       font_size=8, 
                       font_weight='bold',
                       font_color='white',
                       ax=ax1)

ax1.set_title(f'Graph Overview - Node Numbers\n'
              f'Nodes: {num_nodes}, Edges: {num_edges}', 
              fontsize=14, fontweight='bold')
ax1.axis('off')

# Add colorbar for node colors
cbar1 = plt.colorbar(scatter1, ax=ax1, label='Node Degree', shrink=0.8)

# ============ RIGHT PLOT: Adjacency Matrix Heatmap ============
plt.sca(ax2)

# Create adjacency matrix with weights
# Check if edges have weight attribute
has_weights = False
if G.number_of_edges() > 0:
    first_edge = list(G.edges(data=True))[0]
    has_weights = 'weight' in first_edge[2]

if has_weights:
    adj_matrix = nx.adjacency_matrix(G, weight='weight')
else:
    adj_matrix = nx.adjacency_matrix(G)
    
adj_dense = adj_matrix.toarray()

# Create heatmap
im = ax2.imshow(adj_dense, cmap='Blues', aspect='equal')

# Add text annotations for non-zero values
for i in range(len(adj_dense)):
    for j in range(len(adj_dense[0])):
        if adj_dense[i, j] != 0:
            text = ax2.text(j, i, f'{adj_dense[i, j]:.2f}',
                           ha="center", va="center", 
                           color="red" if adj_dense[i, j] > np.mean(adj_dense[adj_dense > 0]) else "black",
                           fontsize=6, fontweight='bold')

# Set ticks and labels
ax2.set_xticks(range(num_nodes))
ax2.set_yticks(range(num_nodes))
ax2.set_xticklabels(range(num_nodes), fontsize=8)
ax2.set_yticklabels(range(num_nodes), fontsize=8)

# Rotate x-axis labels for better readability
plt.setp(ax2.get_xticklabels(), rotation=90)

ax2.set_title('Adjacency Matrix - Edge Weights\n(Red = Above Average Weight)', 
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Node Index', fontsize=12)
ax2.set_ylabel('Node Index', fontsize=12)

# Add colorbar for adjacency matrix
cbar2 = plt.colorbar(im, ax=ax2, label='Edge Weight', shrink=0.8)

plt.tight_layout()

# Save as high-quality PNG
output_filename = f'{interaction_weight}_HToBB_detailed_visualization.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print(f"Detailed graph visualization saved as: {output_filename}")
print(f"Graph info: {num_nodes} nodes, {num_edges} edges")

# ============ ADDITIONAL: Create a zoomed network plot ============
fig2, ax3 = plt.subplots(1, 1, figsize=(16, 16), dpi=300)

# Use spring layout with more iterations for better spacing
pos2 = nx.spring_layout(G, k=8, iterations=200, seed=42)

# Draw edges with labels
nx.draw_networkx_edges(G, pos2, 
                      width=edge_widths,
                      alpha=0.3,
                      edge_color='gray',
                      ax=ax3)

# Draw nodes
scatter2 = nx.draw_networkx_nodes(G, pos2,
                                 node_color=node_colors,
                                 node_size=1200,  # Even larger nodes
                                 cmap=plt.cm.viridis,
                                 alpha=0.9,
                                 ax=ax3)

# Draw node labels
nx.draw_networkx_labels(G, pos2, 
                       font_size=10, 
                       font_weight='bold',
                       font_color='white',
                       ax=ax3)

# Draw edge labels (showing only a subset to avoid overcrowding)
# Show edge labels for edges with highest weights
if len(edge_weights) > 0:
    # Get top 20% of edges by weight
    weight_threshold = np.percentile(edge_weights, 80)
    filtered_edge_labels = {k: v for k, v in edge_labels.items() 
                           if float(v) >= weight_threshold}
    
    nx.draw_networkx_edge_labels(G, pos2, 
                                filtered_edge_labels,
                                font_size=7,
                                font_color='red',
                                bbox=dict(boxstyle='round,pad=0.2', 
                                         facecolor='white', 
                                         alpha=0.8),
                                ax=ax3)

ax3.set_title(f'Detailed Network - Top 20% Edge Weights Shown\n'
              f'Nodes: {num_nodes}, Edges: {num_edges}', 
              fontsize=16, fontweight='bold')
ax3.axis('off')

# Add colorbar
cbar3 = plt.colorbar(scatter2, ax=ax3, label='Node Degree', shrink=0.6)

plt.tight_layout()

# Save the detailed network plot
output_filename2 = f'{interaction_weight}_HToBB_network_detailed.png'
plt.savefig(output_filename2, dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print(f"Detailed network visualization saved as: {output_filename2}")

# Show both plots
plt.show()

# ============ PRINT SUMMARY STATISTICS ============
print("\n" + "="*50)
print("GRAPH ANALYSIS SUMMARY")
print("="*50)
print(f"Total Nodes: {num_nodes}")
print(f"Total Edges: {num_edges}")
print(f"Graph Density: {nx.density(G):.3f}")
print(f"Average Degree: {np.mean([G.degree(n) for n in G.nodes()]):.2f}")

if len(edge_weights) > 0:
    print(f"\nEdge Weight Statistics:")
    print(f"  Min Weight: {np.min(edge_weights):.3f}")
    print(f"  Max Weight: {np.max(edge_weights):.3f}")
    print(f"  Mean Weight: {np.mean(edge_weights):.3f}")
    print(f"  Std Weight: {np.std(edge_weights):.3f}")

print(f"\nTop 10 nodes by degree:")
degree_dict = dict(G.degree())
sorted_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
for node, degree in sorted_nodes[:10]:
    print(f"  Node {node}: degree {degree}")

print("="*50)