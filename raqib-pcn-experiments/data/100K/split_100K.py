import uproot
import awkward as ak
import os

# --- Configuration ---
input_root_file = "JetClass_example_100k.root"  # Replace with your actual file name
tree_name = "tree"
output_dir = "Jets_100K"
classes = [
    "HToBB", "HToCC", "HToGG", "HToWW2Q1L", "HToWW4Q",
    "TTBarLep", "TTBar", "WToQQ", "ZJetsToNuNu", "ZToQQ"
]
entries_per_class = 10000

# --- Create output folder ---
os.makedirs(output_dir, exist_ok=True)

# --- Load ROOT file and tree ---
with uproot.open(input_root_file) as file:
    tree = file[tree_name]
    full_data = tree.arrays(library="ak")

# --- Loop through classes and save chunks ---
for i, class_name in enumerate(classes):
    start = i * entries_per_class
    end = (i + 1) * entries_per_class
    class_data = full_data[start:end]

    output_path = os.path.join(output_dir, f"{class_name}.root")

    # Infer the branch schema from this subset
    branches = {key: ak.type(class_data[key]) for key in class_data.fields}

    with uproot.recreate(output_path) as fout:
        fout.mktree(tree_name, branches)
        fout[tree_name].extend(class_data)

print("Done! All .root files are saved in 'Jets_100K' with original structure preserved.")
