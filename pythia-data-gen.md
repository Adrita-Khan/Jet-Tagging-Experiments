# Complete Python Code to Generate Data Using PYTHIA

This document contains complete, ready-to-run Python scripts for generating particle physics data using PYTHIA and saving it in various formats.

## Installation

First, install the required packages:

```bash
pip install pythia8mc numpy pandas matplotlib h5py
```

## Script 1: Basic Event Generation with CSV Output

```python
"""
Basic PYTHIA event generation script that saves particle data to CSV
"""
import pythia8mc
import csv
import numpy as np

# Configuration
N_EVENTS = 1000
OUTPUT_FILE = "pythia_events.csv"

# Initialize PYTHIA
pythia = pythia8mc.Pythia()

# Configure for QCD jet events at LHC
pythia.readString("Beams:eCM = 13000")  # 13 TeV
pythia.readString("HardQCD:all = on")
pythia.readString("PhaseSpace:pTHatMin = 20.0")  # Minimum pT cut

# Initialize
pythia.init()

# Open CSV file
with open(OUTPUT_FILE, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header
    writer.writerow(['event_id', 'particle_id', 'pdg_id', 'name', 'status', 
                     'px', 'py', 'pz', 'e', 'mass', 'pT', 'eta', 'phi', 
                     'is_final', 'charge', 'mother1', 'mother2'])
    
    # Event generation loop
    for iev in range(N_EVENTS):
        if not pythia.next():
            continue
        
        # Loop over all particles in the event
        for i in range(len(pythia.event)):
            particle = pythia.event[i]
            
            # Extract particle properties
            row = [
                iev,                      # Event ID
                i,                        # Particle ID in event
                particle.id(),            # PDG code
                particle.name(),          # Particle name
                particle.status(),        # Status code
                particle.px(),            # Momentum x
                particle.py(),            # Momentum y
                particle.pz(),            # Momentum z
                particle.e(),             # Energy
                particle.m(),             # Mass
                particle.pT(),            # Transverse momentum
                particle.eta(),           # Pseudorapidity
                particle.phi(),           # Azimuthal angle
                particle.isFinal(),       # Is final state
                particle.charge(),        # Electric charge
                particle.mother1(),       # First mother
                particle.mother2()        # Second mother
            ]
            
            writer.writerow(row)
        
        # Progress indicator
        if (iev + 1) % 100 == 0:
            print(f"Generated {iev + 1} events")

# Print statistics
pythia.stat()

print(f"\nData saved to {OUTPUT_FILE}")
```

## Script 2: Final-State Particles Only with NumPy Arrays

```python
"""
Generate PYTHIA events and save only final-state particles as NumPy arrays
"""
import pythia8mc
import numpy as np

# Configuration
N_EVENTS = 5000
OUTPUT_FILE = "pythia_final_state.npz"

# Initialize PYTHIA
pythia = pythia8mc.Pythia()

# Configure for minimum bias events
pythia.readString("Beams:eCM = 13000")
pythia.readString("SoftQCD:all = on")

pythia.init()

# Lists to store data
event_ids = []
pdg_ids = []
px_list = []
py_list = []
pz_list = []
e_list = []
pt_list = []
eta_list = []
phi_list = []
charges = []

# Event generation loop
for iev in range(N_EVENTS):
    if not pythia.next():
        continue
    
    # Loop over final-state particles only
    for particle in pythia.event:
        if not particle.isFinal():
            continue
        
        event_ids.append(iev)
        pdg_ids.append(particle.id())
        px_list.append(particle.px())
        py_list.append(particle.py())
        pz_list.append(particle.pz())
        e_list.append(particle.e())
        pt_list.append(particle.pT())
        eta_list.append(particle.eta())
        phi_list.append(particle.phi())
        charges.append(particle.charge())
    
    if (iev + 1) % 500 == 0:
        print(f"Generated {iev + 1} events")

pythia.stat()

# Convert to NumPy arrays
data = {
    'event_id': np.array(event_ids, dtype=np.int32),
    'pdg_id': np.array(pdg_ids, dtype=np.int32),
    'px': np.array(px_list, dtype=np.float32),
    'py': np.array(py_list, dtype=np.float32),
    'pz': np.array(pz_list, dtype=np.float32),
    'e': np.array(e_list, dtype=np.float32),
    'pt': np.array(pt_list, dtype=np.float32),
    'eta': np.array(eta_list, dtype=np.float32),
    'phi': np.array(phi_list, dtype=np.float32),
    'charge': np.array(charges, dtype=np.float32)
}

# Save as compressed NumPy archive
np.savez_compressed(OUTPUT_FILE, **data)

print(f"\nData saved to {OUTPUT_FILE}")
print(f"Total particles: {len(event_ids)}")

# To load the data later:
# loaded_data = np.load(OUTPUT_FILE)
# event_ids = loaded_data['event_id']
# pt = loaded_data['pt']
```

## Script 3: Pandas DataFrame with Analysis

```python
"""
Generate PYTHIA events and save as Pandas DataFrame with basic analysis
"""
import pythia8mc
import pandas as pd
import numpy as np

# Configuration
N_EVENTS = 2000
OUTPUT_CSV = "pythia_analysis.csv"
OUTPUT_PARQUET = "pythia_analysis.parquet"

# Initialize PYTHIA
pythia = pythia8mc.Pythia()

# Configure for Z boson production
pythia.readString("Beams:eCM = 13000")
pythia.readString("WeakSingleBoson:ffbar2gmZ = on")
pythia.readString("23:onMode = off")  # Turn off all Z decays
pythia.readString("23:onIfMatch = 11 -11")  # Z -> e+ e- only

pythia.init()

# Store events in a list of dictionaries
events_data = []

for iev in range(N_EVENTS):
    if not pythia.next():
        continue
    
    # Find the Z boson and its decay products
    for i in range(len(pythia.event)):
        particle = pythia.event[i]
        
        # Only store electrons and positrons
        if abs(particle.id()) == 11 and particle.isFinal():
            event_dict = {
                'event_id': iev,
                'pdg_id': particle.id(),
                'px': particle.px(),
                'py': particle.py(),
                'pz': particle.pz(),
                'e': particle.e(),
                'pt': particle.pT(),
                'eta': particle.eta(),
                'phi': particle.phi(),
                'm': particle.m(),
                'charge': particle.charge()
            }
            events_data.append(event_dict)
    
    if (iev + 1) % 200 == 0:
        print(f"Generated {iev + 1} events")

pythia.stat()

# Create DataFrame
df = pd.DataFrame(events_data)

# Basic analysis
print("\n=== Basic Statistics ===")
print(df[['pt', 'eta', 'phi', 'e']].describe())

print("\n=== Mean pT by charge ===")
print(df.groupby('charge')['pt'].mean())

# Save to CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nData saved to {OUTPUT_CSV}")

# Save to Parquet (more efficient for large datasets)
df.to_parquet(OUTPUT_PARQUET, index=False)
print(f"Data saved to {OUTPUT_PARQUET}")
```

## Script 4: HDF5 Format for Large Datasets

```python
"""
Generate PYTHIA events and save to HDF5 format (efficient for large datasets)
"""
import pythia8mc
import h5py
import numpy as np

# Configuration
N_EVENTS = 10000
OUTPUT_FILE = "pythia_events.h5"
CHUNK_SIZE = 1000  # Process events in chunks

# Initialize PYTHIA
pythia = pythia8mc.Pythia()

pythia.readString("Beams:eCM = 13000")
pythia.readString("HardQCD:all = on")
pythia.readString("PhaseSpace:pTHatMin = 50.0")

pythia.init()

# Temporary storage for one chunk
chunk_data = {
    'event_id': [],
    'pdg_id': [],
    'px': [],
    'py': [],
    'pz': [],
    'e': [],
    'pt': [],
    'eta': [],
    'phi': [],
    'is_final': []
}

# Create HDF5 file
with h5py.File(OUTPUT_FILE, 'w') as f:
    # Create extendable datasets
    maxshape = (None,)  # Unlimited dimension
    
    datasets = {}
    for key in chunk_data.keys():
        if key in ['event_id', 'pdg_id']:
            dtype = np.int32
        elif key == 'is_final':
            dtype = np.bool_
        else:
            dtype = np.float32
        
        datasets[key] = f.create_dataset(
            key, 
            shape=(0,), 
            maxshape=maxshape, 
            dtype=dtype,
            compression='gzip'
        )
    
    # Event generation loop
    for iev in range(N_EVENTS):
        if not pythia.next():
            continue
        
        # Extract final-state charged particles
        for particle in pythia.event:
            if particle.isFinal() and particle.isCharged():
                chunk_data['event_id'].append(iev)
                chunk_data['pdg_id'].append(particle.id())
                chunk_data['px'].append(particle.px())
                chunk_data['py'].append(particle.py())
                chunk_data['pz'].append(particle.pz())
                chunk_data['e'].append(particle.e())
                chunk_data['pt'].append(particle.pT())
                chunk_data['eta'].append(particle.eta())
                chunk_data['phi'].append(particle.phi())
                chunk_data['is_final'].append(True)
        
        # Save chunk to file when it reaches CHUNK_SIZE
        if (iev + 1) % CHUNK_SIZE == 0:
            for key, dataset in datasets.items():
                data_array = np.array(chunk_data[key])
                current_size = dataset.shape[0]
                new_size = current_size + len(data_array)
                dataset.resize(new_size, axis=0)
                dataset[current_size:new_size] = data_array
                chunk_data[key] = []  # Clear chunk
            
            print(f"Saved {iev + 1} events to HDF5")
    
    # Save any remaining data
    if len(chunk_data['event_id']) > 0:
        for key, dataset in datasets.items():
            data_array = np.array(chunk_data[key])
            current_size = dataset.shape[0]
            new_size = current_size + len(data_array)
            dataset.resize(new_size, axis=0)
            dataset[current_size:new_size] = data_array
    
    # Store metadata
    f.attrs['n_events'] = N_EVENTS
    f.attrs['center_of_mass_energy'] = 13000
    f.attrs['process'] = "HardQCD:all"

pythia.stat()

print(f"\nData saved to {OUTPUT_FILE}")

# To read the data later:
# with h5py.File(OUTPUT_FILE, 'r') as f:
#     pt = f['pt'][:]
#     eta = f['eta'][:]
#     print(f"Total particles: {len(pt)}")
#     print(f"Metadata: {dict(f.attrs)}")
```

## Script 5: JSON Format with Event Structure

```python
"""
Generate PYTHIA events and save to JSON format with hierarchical structure
"""
import pythia8mc
import json

# Configuration
N_EVENTS = 100  # Fewer events for JSON (can be large)
OUTPUT_FILE = "pythia_events.json"

# Initialize PYTHIA
pythia = pythia8mc.Pythia()

pythia.readString("Beams:eCM = 13000")
pythia.readString("Top:gg2ttbar = on")

pythia.init()

# Store all events
all_events = []

for iev in range(N_EVENTS):
    if not pythia.next():
        continue
    
    # Create event dictionary
    event = {
        'event_id': iev,
        'particles': []
    }
    
    # Store only final-state particles
    for i in range(len(pythia.event)):
        particle = pythia.event[i]
        
        if particle.isFinal():
            particle_dict = {
                'id': i,
                'pdg_id': particle.id(),
                'name': particle.name(),
                'px': particle.px(),
                'py': particle.py(),
                'pz': particle.pz(),
                'e': particle.e(),
                'pt': particle.pT(),
                'eta': particle.eta(),
                'phi': particle.phi(),
                'mass': particle.m(),
                'charge': particle.charge()
            }
            event['particles'].append(particle_dict)
    
    all_events.append(event)
    
    if (iev + 1) % 10 == 0:
        print(f"Generated {iev + 1} events")

pythia.stat()

# Save to JSON
with open(OUTPUT_FILE, 'w') as f:
    json.dump(all_events, f, indent=2)

print(f"\nData saved to {OUTPUT_FILE}")

# To load:
# with open(OUTPUT_FILE, 'r') as f:
#     events = json.load(f)
```

## Script 6: Using Awkward Arrays for Batch Processing

```python
"""
Generate PYTHIA events using batch processing with Awkward arrays
Requires: pip install awkward vector
"""
import pythia8mc
import awkward as ak
import numpy as np

# Configuration
N_EVENTS = 5000
OUTPUT_FILE = "pythia_awkward.parquet"

# Initialize PYTHIA
pythia = pythia8mc.Pythia()

pythia.readString("Beams:eCM = 13000")
pythia.readString("HardQCD:all = on")
pythia.readString("PhaseSpace:pTHatMin = 20.0")

pythia.init()

print(f"Generating {N_EVENTS} events in batch mode...")

# Generate events in batch (much faster than loop)
events = pythia.nextBatch(N_EVENTS)

print(f"Generated {len(events)} events")

# Events is now an Awkward array with all event records
# Access particle properties across all events
print(f"\nTotal particles across all events: {ak.sum(ak.num(events))}")

# Example: Filter for final-state charged particles
final_state = events[events.isFinal]
charged = final_state[final_state.isCharged]

print(f"Final-state particles: {ak.sum(ak.num(final_state))}")
print(f"Charged final-state particles: {ak.sum(ak.num(charged))}")

# Calculate some physics quantities
pt_all = ak.flatten(charged.pT)
eta_all = ak.flatten(charged.eta)
phi_all = ak.flatten(charged.phi)

print(f"\nMean pT: {ak.mean(pt_all):.2f} GeV")
print(f"Max pT: {ak.max(pt_all):.2f} GeV")

# Save to Parquet format (preserves jagged array structure)
ak.to_parquet(charged, OUTPUT_FILE)
print(f"\nData saved to {OUTPUT_FILE}")

pythia.stat()

# To load later:
# loaded_events = ak.from_parquet(OUTPUT_FILE)
```

## Script 7: Complete Analysis Pipeline

```python
"""
Complete PYTHIA data generation and analysis pipeline
"""
import pythia8mc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
N_EVENTS = 5000
OUTPUT_DIR = "./"

# Initialize PYTHIA
pythia = pythia8mc.Pythia()

pythia.readString("Beams:eCM = 13000")
pythia.readString("HardQCD:all = on")
pythia.readString("PhaseSpace:pTHatMin = 20.0")

pythia.init()

# Storage
charged_multiplicity = []
jet_pt = []
event_data = []

print("Generating events...")

for iev in range(N_EVENTS):
    if not pythia.next():
        continue
    
    # Count charged multiplicity
    n_charged = 0
    max_pt_in_event = 0
    
    for particle in pythia.event:
        if particle.isFinal():
            # Store all final state particles
            event_data.append({
                'event_id': iev,
                'pdg_id': particle.id(),
                'pt': particle.pT(),
                'eta': particle.eta(),
                'phi': particle.phi(),
                'is_charged': particle.isCharged()
            })
            
            # Count charged
            if particle.isCharged():
                n_charged += 1
            
            # Track highest pT
            if particle.pT() > max_pt_in_event:
                max_pt_in_event = particle.pT()
    
    charged_multiplicity.append(n_charged)
    jet_pt.append(max_pt_in_event)
    
    if (iev + 1) % 500 == 0:
        print(f"Generated {iev + 1} events")

pythia.stat()

# Convert to DataFrame
df = pd.DataFrame(event_data)

# Save raw data
df.to_csv(f"{OUTPUT_DIR}/pythia_raw_data.csv", index=False)
print(f"\nRaw data saved to {OUTPUT_DIR}/pythia_raw_data.csv")

# Create summary statistics
summary = pd.DataFrame({
    'event_id': range(N_EVENTS),
    'charged_multiplicity': charged_multiplicity,
    'leading_pt': jet_pt
})

summary.to_csv(f"{OUTPUT_DIR}/pythia_summary.csv", index=False)
print(f"Summary saved to {OUTPUT_DIR}/pythia_summary.csv")

# Generate plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Charged multiplicity
axes[0, 0].hist(charged_multiplicity, bins=50, alpha=0.7, color='blue')
axes[0, 0].set_xlabel('Charged Multiplicity')
axes[0, 0].set_ylabel('Events')
axes[0, 0].set_title('Charged Particle Multiplicity Distribution')
axes[0, 0].grid(True, alpha=0.3)

# Leading pT
axes[0, 1].hist(jet_pt, bins=50, alpha=0.7, color='red', range=(0, 200))
axes[0, 1].set_xlabel('Leading pT (GeV)')
axes[0, 1].set_ylabel('Events')
axes[0, 1].set_title('Leading Particle pT Distribution')
axes[0, 1].set_yscale('log')
axes[0, 1].grid(True, alpha=0.3)

# pT distribution (all particles)
charged_df = df[df['is_charged'] == True]
axes[1, 0].hist(charged_df['pt'], bins=100, alpha=0.7, color='green', range=(0, 100))
axes[1, 0].set_xlabel('pT (GeV)')
axes[1, 0].set_ylabel('Particles')
axes[1, 0].set_title('Charged Particle pT Distribution')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# Eta distribution
axes[1, 1].hist(charged_df['eta'], bins=50, alpha=0.7, color='purple', range=(-5, 5))
axes[1, 1].set_xlabel('η (pseudorapidity)')
axes[1, 1].set_ylabel('Particles')
axes[1, 1].set_title('Charged Particle η Distribution')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pythia_analysis.png", dpi=150)
print(f"Plots saved to {OUTPUT_DIR}/pythia_analysis.png")

# Print statistics
print("\n=== Analysis Summary ===")
print(f"Total events: {N_EVENTS}")
print(f"Mean charged multiplicity: {np.mean(charged_multiplicity):.2f}")
print(f"Mean leading pT: {np.mean(jet_pt):.2f} GeV")
print(f"Total particles: {len(df)}")
print(f"Total charged particles: {len(charged_df)}")
```

## Usage Examples

### Running the Scripts

Save any of the above scripts to a `.py` file and run:

```bash
python script_name.py
```

### Loading Saved Data

```python
# CSV
import pandas as pd
df = pd.read_csv("pythia_events.csv")

# NumPy
import numpy as np
data = np.load("pythia_final_state.npz")
pt = data['pt']

# HDF5
import h5py
with h5py.File("pythia_events.h5", 'r') as f:
    pt = f['pt'][:]

# JSON
import json
with open("pythia_events.json", 'r') as f:
    events = json.load(f)

# Parquet (Pandas)
df = pd.read_parquet("pythia_analysis.parquet")

# Parquet (Awkward)
import awkward as ak
events = ak.from_parquet("pythia_awkward.parquet")
```

## Key Points

1. **CSV Format**: Human-readable, good for small datasets, easy to open in Excel
2. **NumPy Format (.npz)**: Efficient for numerical arrays, fast I/O
3. **HDF5 Format**: Best for very large datasets, supports compression and chunking
4. **JSON Format**: Good for hierarchical data, human-readable but large file size
5. **Parquet Format**: Efficient columnar storage, good for DataFrames and Awkward arrays
6. **Awkward Arrays**: Best for batch processing, preserves jagged array structure

## Recommendations by Use Case

- **Small datasets (<1M particles)**: CSV or JSON
- **Medium datasets (1M-100M particles)**: NumPy or Parquet
- **Large datasets (>100M particles)**: HDF5
- **Machine learning pipelines**: Parquet or HDF5
- **Quick analysis**: Pandas DataFrame → Parquet
- **High-performance computing**: Awkward arrays → Parquet
