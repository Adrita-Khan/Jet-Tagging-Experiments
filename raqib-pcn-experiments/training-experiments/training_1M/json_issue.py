import os
import json
import numpy as np
import pandas as pd
import uproot
import random

# -----------------------------
# CONFIGURATION 
# -----------------------------

BASE_DIR = "data/JetClass_Pythia_100M"
OUTPUT_DIR = "output"

CLASSES = [
    "HToBB", "HToCC", "HToGG",
    "HTolW2QL", "HTolW4Q",
    "TTBar", "TTBarLep",
    "WToQQ", "ZJetsToNuNu", "ZToQQ"
]

JETS_PER_FILE = 1000          # How many jets to sample per root file
TOTAL_JETS = 100_000          # Target total jets per class
RANDOM_SEED = 42              # For reproducibility

# -----------------------------
# MAIN PROCESSING FUNCTIONS
# -----------------------------

def list_root_files(class_dir):
    return sorted([
        f for f in os.listdir(class_dir)
        if f.endswith(".root")
    ])

def sample_indices(total_entries, k):
    return sorted(np.random.choice(total_entries, size=k, replace=False))

def read_jets_from_file(file_path, indices):
    with uproot.open(file_path) as f:
        tree_name = f.keys()[0].split(";")[0]
        tree = f[tree_name]
        all_jets = tree.arrays(library="np")  # read as numpy arrays directly
        jets = {k: all_jets[k][indices] for k in all_jets.keys()}
        return jets

def write_combined_root(data, output_path):
    with uproot.recreate(output_path) as f:
        f["tree"] = data

def write_tracking_files(indices_dict, flat_index_log, output_base):
    # JSON
    with open(output_base + "_indices.json", "w") as f:
        json.dump(indices_dict, f, indent=2)
    # CSV
    df = pd.DataFrame(flat_index_log, columns=["New_Index", "Source_File", "Source_Index"])
    df.to_csv(output_base + "_indices.csv", index=False)

def write_data_preview(data, output_base, limit=100):
    preview_df = pd.DataFrame({k: np.array(v[:limit]) for k, v in data.items()})
    preview_df.to_csv(output_base + ".csv", index=False)
    preview_df.to_json(output_base + ".json", orient="records", indent=2)

def process_class(class_name):
    print(f"\n Processing class: {class_name}")

    class_dir = os.path.join(BASE_DIR, class_name)
    output_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    root_files = list_root_files(class_dir)
    total_collected = 0
    all_data = {}
    source_map = {}
    index_log = []

    for file_name in root_files:
        if total_collected >= TOTAL_JETS:
            break

        file_path = os.path.join(class_dir, file_name)
        with uproot.open(file_path) as f:
            tree_name = f.keys()[0].split(";")[0]
            tree = f[tree_name]
            total_jets_in_file = tree.num_entries

        if total_jets_in_file < JETS_PER_FILE:
            print(f" Skipping {file_name} — not enough jets.")
            continue

        selected_indices = sample_indices(total_jets_in_file, JETS_PER_FILE)
        jets = read_jets_from_file(file_path, selected_indices)

        # Initialize container if needed
        if not all_data:
            for key in jets:
                all_data[key] = []

        # Now append (not extend) — correct way
        for key in jets:
            all_data[key].append(jets[key])

        # Logging
        source_map[file_name] = selected_indices
        for i, idx in enumerate(selected_indices):
            index_log.append([total_collected + i, file_name, idx])

        total_collected += JETS_PER_FILE

    # Correct concatenation after collection
    final_data = {k: np.concatenate(v) for k, v in all_data.items()}
    output_base = os.path.join(output_class_dir, f"{class_name}_100K")

    # Write root file
    write_combined_root(final_data, output_base + ".root")

    # Write tracking files
    write_tracking_files(source_map, index_log, output_base)

    # Write human-readable data preview
    write_data_preview(final_data, output_base)

    print(f" Done: {class_name}")

# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    for class_name in CLASSES:
        process_class(class_name)
