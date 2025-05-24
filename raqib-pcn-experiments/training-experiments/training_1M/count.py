import pickle
import pandas as pd
import os

pkl_folder = 'data/'  

# Find all .pkl files in that folder
file_list = [f for f in os.listdir(pkl_folder) if f.endswith('.pkl')]

# Dictionary to store counts
graph_counts = {"JetType": [], "Number of graphs": []}

# Loop through each .pkl file
for file in file_list:
    file_path = os.path.join(pkl_folder, file)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)   # load the data
        count = len(data)       # get the length
        jet_type = file.replace(".pkl", "")

        # Save the data
        graph_counts["JetType"].append(jet_type)
        graph_counts["Number of graphs"].append(count)

        # Print progress
        print(f"Processed {jet_type}: {count} graphs")

# Create a DataFrame
df = pd.DataFrame(graph_counts)

# Save to CSV
df.to_csv("graph_counts_test_20M.csv", index=False)

print("\nAll files processed. Saved results to 'graph_counts_test_20M.csv'.")
