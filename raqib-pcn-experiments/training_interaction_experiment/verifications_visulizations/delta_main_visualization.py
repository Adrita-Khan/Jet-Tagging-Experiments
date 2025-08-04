import pickle

# Open the pkl file and load only the first graph
with open("../data/delta/delta_HToBB-Testing.pkl", "rb") as f:
    first_graph = pickle.load(f)[0]

g = first_graph

# Print graph summary
print("Graph Summary:")
print(g)

# Print node features
print("\nNode features:")
for key in g.ndata:
    print(f"  {key}: {g.ndata[key].shape}")

# Print edge features
print("\nEdge data:")
if g.edata:
    for key in g.edata:
        print(f"  {key}: {g.edata[key].shape}")
else:
    print("  No edge data stored.")
