import pickle
import os
from tqdm import tqdm
import gc

def preprocess_pickle(input_file, output_file):
    print(f"Preprocessing {input_file} to {output_file}")
    try:
        with open(input_file, 'rb') as f:
            graphs = pickle.load(f)
        
        with open(output_file, 'wb') as f:
            for graph in tqdm(graphs, total=len(graphs), desc="Writing graphs"):
                pickle.dump(graph, f)
        
        print(f"Finished preprocessing {input_file}. Output size: {os.path.getsize(output_file) / 1024**3:.2f} GB")
    except Exception as e:
        print(f"Error preprocessing {input_file}: {e}")
    finally:
        del graphs
        gc.collect()

if __name__ == "__main__":
    jetNames = ['TTBar-Testing', 'TTBarLep-Testing', 'WToQQ-Testing', 'ZToQQ-Testing', 'ZJetsToNuNu-Testing',
                'HToBB-Testing', 'HToCC-Testing', 'HToGG-Testing', 'HToWW2Q1L-Testing', 'HToWW4Q-Testing']
    
    for jetType in jetNames:
        input_file = f'data/{jetType}.pkl'
        output_file = f'data/{jetType}_streamed.pkl'
        if os.path.exists(input_file):
            preprocess_pickle(input_file, output_file)
        else:
            print(f"Input file {input_file} not found. Skipping.")