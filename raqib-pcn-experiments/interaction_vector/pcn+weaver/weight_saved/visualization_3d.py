import pickle
import networkx as nx
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def load_graphs(file_path):
    try:
        with open(file_path, 'rb') as f:
            graphs = pickle.load(f)
        return graphs
    except Exception as e:
        print(f"Failed to load the file: {e}")
        return None

def visualize_graph_3d_plotly(graph, title="3D Graph Visualization"):
    """
    Create an interactive 3D visualization of a DGL graph using Plotly.
    """
    # Convert DGL graph to NetworkX
    nx_graph = graph.to_networkx(edge_attrs=list(graph.edata.keys()) if graph.edata else None)
    
    # Generate 3D spring layout
    pos_2d = nx.spring_layout(nx_graph, seed=42, k=0.5)
    
    # Convert to 3D by adding a z-coordinate
    pos_3d = {}
    for node, (x, y) in pos_2d.items():
        # Add some variation in z-coordinate based on node properties
        z = np.sin(x * 2) * np.cos(y * 2) * 0.5  # Creates interesting 3D structure
        pos_3d[node] = (x, y, z)
    
    # Extract node positions
    node_x = [pos_3d[node][0] for node in nx_graph.nodes()]
    node_y = [pos_3d[node][1] for node in nx_graph.nodes()]
    node_z = [pos_3d[node][2] for node in nx_graph.nodes()]
    node_labels = [str(node) for node in nx_graph.nodes()]
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_z = []
    edge_info = []
    edge_weights = []
    
    if graph.edata and 'weight' in graph.edata:
        weights = graph.edata['weight']
        weight_array = weights.numpy() if hasattr(weights, 'numpy') else weights
    else:
        weight_array = np.ones(nx_graph.number_of_edges())
    
    for i, (u, v) in enumerate(nx_graph.edges()):
        x0, y0, z0 = pos_3d[u]
        x1, y1, z1 = pos_3d[v]
        
        # Add edge coordinates (with None to break line segments)
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
        
        # Store edge information
        edge_info.append(f"Edge {u}-{v}: Weight {weight_array[i]:.3f}")
        edge_weights.append(weight_array[i])
    
    # Create edge trace
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(
            color='rgba(125,125,125,0.6)',
            width=2
        ),
        hoverinfo='none',
        name='Edges'
    )
    
    # Create node trace
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        marker=dict(
            size=12,
            color='lightblue',
            colorscale='Viridis',
            line=dict(width=2, color='black'),
            opacity=0.8
        ),
        text=node_labels,
        textposition="middle center",
        textfont=dict(size=10, color='black'),
        hovertemplate='<b>Node %{text}</b><br>' +
                      'x: %{x:.3f}<br>' +
                      'y: %{y:.3f}<br>' +
                      'z: %{z:.3f}<extra></extra>',
        name='Nodes'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>Nodes: {nx_graph.number_of_nodes()}, Edges: {nx_graph.number_of_edges()}</sub>",
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis=dict(
                backgroundcolor="rgb(200, 200, 230)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                title="X"
            ),
            yaxis=dict(
                backgroundcolor="rgb(230, 200, 230)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                title="Y"
            ),
            zaxis=dict(
                backgroundcolor="rgb(230, 230, 200)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                title="Z"
            ),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=900,
        height=700,
        showlegend=True,
        hovermode='closest',
        annotations=[
            dict(
                text="Click and drag to rotate • Scroll to zoom • Double-click to reset view",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(size=12, color='gray')
            )
        ]
    )
    
    return fig

def create_enhanced_3d_visualization(graph, title="Enhanced 3D Graph Visualization"):
    """
    Create an enhanced 3D visualization with edge weights represented by line thickness and color.
    """
    # Convert DGL graph to NetworkX
    nx_graph = graph.to_networkx(edge_attrs=list(graph.edata.keys()) if graph.edata else None)
    
    # Generate 3D positions
    pos_2d = nx.spring_layout(nx_graph, seed=42, k=0.5)
    pos_3d = {}
    for node, (x, y) in pos_2d.items():
        z = np.sin(x * 3) * np.cos(y * 3) * 0.3 + np.random.normal(0, 0.1)
        pos_3d[node] = (x, y, z)
    
    # Node positions
    node_x = [pos_3d[node][0] for node in nx_graph.nodes()]
    node_y = [pos_3d[node][1] for node in nx_graph.nodes()]
    node_z = [pos_3d[node][2] for node in nx_graph.nodes()]
    node_labels = [str(node) for node in nx_graph.nodes()]
    
    # Get edge weights
    if graph.edata and 'weight' in graph.edata:
        weights = graph.edata['weight']
        weight_array = weights.numpy() if hasattr(weights, 'numpy') else weights
        min_weight = weight_array.min()
        max_weight = weight_array.max()
    else:
        weight_array = np.ones(nx_graph.number_of_edges())
        min_weight = max_weight = 1.0
    
    # Create individual edge traces with varying thickness and color
    edge_traces = []
    
    for i, (u, v) in enumerate(nx_graph.edges()):
        x0, y0, z0 = pos_3d[u]
        x1, y1, z1 = pos_3d[v]
        
        # Normalize weight for visualization
        if max_weight > min_weight:
            normalized_weight = (weight_array[i] - min_weight) / (max_weight - min_weight)
        else:
            normalized_weight = 0.5
        
        # Calculate line width and color intensity
        line_width = 1 + normalized_weight * 4  # Width between 1 and 5
        color_intensity = normalized_weight
        
        edge_trace = go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines',
            line=dict(
                color=f'rgba({int(255*color_intensity)}, {int(100*(1-color_intensity))}, {int(150*(1-color_intensity))}, 0.7)',
                width=line_width
            ),
            hovertemplate=f'<b>Edge {u} → {v}</b><br>Weight: {weight_array[i]:.3f}<extra></extra>',
            showlegend=False,
            name=f'Edge {u}-{v}'
        )
        edge_traces.append(edge_trace)
    
    # Node trace with color based on degree
    node_degrees = [nx_graph.degree(node) for node in nx_graph.nodes()]
    
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        marker=dict(
            size=[8 + degree * 2 for degree in node_degrees],  # Size based on degree
            color=node_degrees,
            colorscale='Plasma',
            colorbar=dict(title="Node Degree", thickness=15, len=0.5),
            line=dict(width=2, color='black'),
            opacity=0.8
        ),
        text=node_labels,
        textposition="middle center",
        textfont=dict(size=10, color='white', family="Arial Black"),
        hovertemplate='<b>Node %{text}</b><br>' +
                      'Degree: %{marker.color}<br>' +
                      'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>',
        name='Nodes'
    )
    
    # Combine all traces
    data = edge_traces + [node_trace]
    
    # Create figure
    fig = go.Figure(data=data)
    
    # Update layout with enhanced styling
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>Nodes: {nx_graph.number_of_nodes()}, Edges: {nx_graph.number_of_edges()}</sub><br>" +
                 f"<sub>Weight range: {min_weight:.3f} to {max_weight:.3f}</sub>",
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            bgcolor='rgb(10, 10, 10)',
            xaxis=dict(
                backgroundcolor="rgb(40, 40, 60)",
                gridcolor="rgb(80, 80, 100)",
                showbackground=True,
                zerolinecolor="rgb(100, 100, 120)",
                title="X Axis"
            ),
            yaxis=dict(
                backgroundcolor="rgb(40, 60, 40)",
                gridcolor="rgb(80, 100, 80)",
                showbackground=True,
                zerolinecolor="rgb(100, 120, 100)",
                title="Y Axis"
            ),
            zaxis=dict(
                backgroundcolor="rgb(60, 40, 40)",
                gridcolor="rgb(100, 80, 80)",
                showbackground=True,
                zerolinecolor="rgb(120, 100, 100)",
                title="Z Axis"
            ),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.8, y=1.8, z=1.2)
            )
        ),
        width=1400,
        height=1000,
        showlegend=False,
        hovermode='closest',
        paper_bgcolor='rgb(20, 20, 20)',
        plot_bgcolor='rgb(20, 20, 20)',
        font=dict(color='white')
    )
    
    return fig

# Main execution
if __name__ == "__main__":
    # Load the delta interaction graphs
    interaction_weight = 'delta'
    file_path = f"weighted_graphs/{interaction_weight}/{interaction_weight}_HToBB_first_10.pkl"
    
    print(f"Loading graphs from: {file_path}")
    graphs = load_graphs(file_path)
    
    if graphs:
        print(f"Successfully loaded {len(graphs)} graphs")
        
        # Get the first graph
        first_graph = graphs[0]
        
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
        
        # Create enhanced 3D visualization
        print("Creating enhanced 3D visualization...")
        fig = create_enhanced_3d_visualization(
            first_graph,
            title=f"{interaction_weight.upper()} Interaction - HToBB Graph (Enhanced 3D)"
        )
        
        # Save the visualization
        fig.write_html(f'{interaction_weight}_HToBB_enhanced_3d.html')
        
        print(f"\nVisualization saved as:")
        print(f"  - {interaction_weight}_HToBB_enhanced_3d.html")
        
        # Show the enhanced visualization
        fig.show()
        
    else:
        print("Failed to load graphs. Please check the file path.")