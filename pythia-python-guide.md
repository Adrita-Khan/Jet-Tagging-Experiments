# Using PYTHIA with Python

## Overview

PYTHIA includes a comprehensive Python interface that allows direct access to the C++ library without requiring knowledge of C++ compilation. The Python interface has been significantly improved since PYTHIA 8.219, with modern versions using PyBind11 for automatic binding generation, making it easy to use in interactive environments like Jupyter notebooks.

## Installation

### Option 1: Using pip (Recommended for Most Users)

The simplest approach is to install via pip. Note that on PyPI, the module name is `pythia8mc` (not `pythia8`) due to naming availability.

```bash
pip install pythia8mc
```

This installs pre-built wheels for most common Python versions (3.6 through 3.12) and platforms (Linux x86_64, macOS ARM64).

### Option 2: Using conda

For conda users, the original module name `pythia8` is available:

```bash
conda install -c conda-forge pythia8
```

Conda also offers related HEP tools in the same ecosystem:

```bash
conda create --name pythia_env
conda activate pythia_env
conda install -c conda-forge pythia8 hepmc2 root
```

### Option 3: Building from Source

For custom builds or to regenerate bindings:

```bash
wget https://pythia.org/download/pythia83/pythia8XXX.tgz
tar xvfz pythia8XXX.tgz
cd pythia8XXX
./configure --with-python
make
```

After building, set the Python path:

```bash
export PYTHONPATH=$HOME/pythia8/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/pythia8/lib:$LD_LIBRARY_PATH
```

## Basic Usage

### Simple Event Generation Example

Here's a minimal example to generate a few events at the LHC:

```python
import pythia8mc

# Create a Pythia instance
pythia = pythia8mc.Pythia()

# Configure for minimum bias events (soft QCD)
pythia.readString("SoftQCD:all = on")

# Initialize
pythia.init()

# Generate an event
pythia.next()

# Explore the event record
for particle in pythia.event:
    print(particle.name())
```

### Generating Multiple Events

```python
import pythia8mc

pythia = pythia8mc.Pythia()

# Settings for hard QCD processes
pythia.readString("HardQCD:all = on")
pythia.readString("PhaseSpace:pTHatMin = 10")  # Minimum pT

pythia.init()

# Generate events
n_events = 1000
for i in range(n_events):
    if not pythia.next():
        continue
    
    # Process event
    for particle in pythia.event:
        if particle.isFinal() and abs(particle.id()) < 30:  # Fundamental particles
            print(f"Particle: {particle.name()}, pT: {particle.pT()}")
```

## Configuration and Initialization

### Configuring PYTHIA

PYTHIA is configured using the `readString()` method with setting strings. Common configuration examples:

```python
# Beam energy (in GeV)
pythia.readString("Beams:eCM = 8000")  # 8 TeV

# Physics processes
pythia.readString("SoftQCD:all = on")  # Soft QCD (minimum bias)
pythia.readString("HardQCD:all = on")  # Hard QCD with jets
pythia.readString("Top:gg2ttbar = on")  # Top pair production
pythia.readString("Higgs:all = on")  # Higgs processes

# Phase space cuts
pythia.readString("PhaseSpace:pTHatMin = 20")  # Minimum transverse momentum
pythia.readString("PhaseSpace:pTHatMax = 100")  # Maximum transverse momentum

# Parton distribution function
pythia.readString("PDF:pSet = 21")  # CTEQ6L

# Tuning
pythia.readString("Tune:preferLHAPDF = 2")  # Use LHAPDF if available
```

### Initialization

After configuration, initialize PYTHIA:

```python
pythia.init()  # Requires parton distributions and other resources
```

**Note:** If using PyPI distribution, XML data files are automatically located. If building from source, ensure `PYTHIA8DATA` environment variable is set or pass the XML path to the constructor:

```python
pythia = pythia8mc.Pythia("/path/to/share/Pythia8/xmldoc")
```

## Event Structure

### Accessing Event Information

The event record is stored in `pythia.event` and contains all particles from the collision:

```python
pythia.next()

# Iterate through all particles
for i in range(len(pythia.event)):
    particle = pythia.event[i]
    
    # Basic properties
    print(f"ID: {particle.id()}")  # PDG ID
    print(f"Name: {particle.name()}")
    print(f"Status: {particle.status()}")
    
    # Kinematic properties
    print(f"pT: {particle.pT()}")  # Transverse momentum
    print(f"p: {particle.p()}")  # Momentum magnitude
    print(f"E: {particle.e()}")  # Energy
    print(f"m: {particle.m()}")  # Mass
    
    # Production/decay properties
    print(f"Mother1: {particle.mother1()}")
    print(f"Mother2: {particle.mother2()}")
    print(f"Daughter1: {particle.daughter1()}")
    print(f"Daughter2: {particle.daughter2()}")
    
    # Status checks
    print(f"Is final: {particle.isFinal()}")
    print(f"Is initial: {particle.isInitial()}")
```

### Filtering Particles

Common particle filtering examples:

```python
pythia.next()

# Get final-state particles only
final_particles = [p for p in pythia.event if p.isFinal()]

# Get only quarks and gluons
quarks_and_gluons = [p for p in pythia.event if abs(p.id()) <= 5 or p.id() == 21]

# Get only leptons (electrons, muons, taus)
leptons = [p for p in pythia.event if abs(p.id()) in [11, 13, 15] and p.isFinal()]

# Get jets (using fundamental particles with small ID)
jet_constituents = [p for p in pythia.event if p.isFinal() and abs(p.id()) < 30]
```

## Advanced Features

### Batch Processing with Awkward Arrays

For efficient processing of large event samples, use the batch interface with Awkward arrays:

```bash
pip install awkward
```

```python
import pythia8mc

pythia = pythia8mc.Pythia()
pythia.readString("HardQCD:all = on")
pythia.init()

# Generate 100 events and return as Awkward array
events = pythia.nextBatch(100)

# Process all events at once in Python
for event in events:
    # Each event is now available as a Python object
    for particle in event:
        if particle.isFinal():
            print(f"Particle: {particle.name()}, pT: {particle.pT()}")
```

Error handling in batch mode:

```python
# Skip failed events (default behavior)
events = pythia.nextBatch(100, errorMode="skip")

# Include failed events as None
events = pythia.nextBatch(100, errorMode="none")

# Fail immediately if event generation fails
events = pythia.nextBatch(100, errorMode="fail")

# Continue until 200 generation attempts (with factor 2)
events = pythia.nextBatch(100, errorMode=2.0)
```

### Using scikit-hep Vector Library

Integrate with vector library for advanced four-vector operations:

```bash
pip install vector
```

```python
import vector
import pythia8mc

vector.register_awkward()

pythia = pythia8mc.Pythia()
pythia.readString("HardQCD:all = on")
pythia.init()

events = pythia.nextBatch(100)

# Now you can use advanced vector operations with slicing
# This is demonstrated in main297.py in the PYTHIA examples
```

### User Hooks and Event Vetoing

For more control over event generation, derive from `UserHooks`:

```python
import pythia8mc

class MyHooks(pythia8mc.UserHooks):
    def canVetoPT(self):
        return True
    
    def vetoPT(self, iSys, eventIn, limitPTmax):
        # Access event and potentially veto based on pT
        return False  # Return False to accept event

pythia = pythia8mc.Pythia()
pythia.setUserHooksPtr(MyHooks())
pythia.readString("HardQCD:all = on")
pythia.init()

pythia.next()
```

See `main293.py` in PYTHIA examples for detailed implementation.

### Parallel Event Generation

For multi-threaded event generation:

```python
import pythia8mc

def analyze_event(event):
    """Analysis function for each event"""
    return len([p for p in event if p.isFinal()])

pythia_parallel = pythia8mc.PythiaParallel()
pythia_parallel.readString("HardQCD:all = on")
pythia_parallel.init()

# Generate events with automatic parallelization
results = pythia_parallel.generate(1000, analyze_event)
```

## Getting Event Information

### General Information

Access event-level information via `pythia.info`:

```python
pythia.next()

# Note: infoPython() creates a new Info instance
info = pythia.infoPython()

print(f"Event weight: {info.weight()}")
print(f"Process ID: {info.code()}")
print(f"Cross section (mb): {info.sigmaGen()}")
```

### Settings and Configuration

Access and query settings:

```python
settings = pythia.settings

# Get all settings as a dictionary
flags = settings.getFlagMap("")
modes = settings.getModeMap("")
parms = settings.getParmMap("")

for key, value in flags.items():
    print(f"{key}: {value}")
```

## Working with External Generators

PYTHIA can process events from external generators using the Les Houches Event Format (LHEF):

```python
pythia = pythia8mc.Pythia()

# Read from LHEF file
pythia.readString("Beams:frameType = 4")
pythia.readString("Beams:LHEF = events.lhe")

pythia.init()

# Process events (parton shower applied)
for i in range(1000):
    if not pythia.next():
        continue
    # Analyze events
```

## Accessing Particle Data

Query the built-in particle database:

```python
pythia = pythia8mc.Pythia()

# Access particle data
particle_data = pythia.particleData

# Get particle by PDG ID
proton = particle_data.findParticle(2212)
print(f"Proton mass: {proton.m0()}")
print(f"Proton spin: {proton.spinType()}")

# Get decay channels
for i in range(proton.sizeChannels()):
    channel = proton.channel(i)
    print(f"Branching ratio: {channel.bRatio()}")
```

## Performance Considerations

### Optimization Tips

1. **Batch Processing:** Use `nextBatch()` for large event samples to reduce Python/C++ call overhead

2. **Direct Event Iteration:** Iterate directly over `pythia.event` to avoid unnecessary copying

3. **Early Filtering:** Apply cuts in Python after generation rather than using complex PYTHIA configurations

4. **Parallelization:** Use `PythiaParallel` for multi-threaded generation on multi-core systems

### Common Issues and Solutions

**Issue:** ImportError when importing pythia8

```python
# Solution: Check PYTHONPATH
import sys
print(sys.path)

# Or set it manually
import sys
sys.path.insert(0, '/path/to/pythia/lib')
```

**Issue:** XML files not found

```python
# Solution: Set environment variable or pass path
import os
os.environ['PYTHIA8DATA'] = '/path/to/share/Pythia8/xmldoc'

import pythia8mc
pythia = pythia8mc.Pythia()
```

**Issue:** Slow event generation in interactive mode

```python
# Solution: Use batch processing
events = pythia.nextBatch(1000)  # Much faster than loop with pythia.next()
```

## Examples from Official Distribution

PYTHIA provides several Python example programs:

- **main292.py:** Python interface equivalent to main222.cc; demonstrates deriving PYTHIA classes in Python
- **main293.py:** Demonstrates usage of PYTHIA plugins (e.g., UserHooks) within Python
- **main294.py:** Standalone script parsing XML particle database
- **main295.py:** Madgraph interface example
- **main297.py:** Batch processing with Awkward arrays
- **main298.py:** Advanced batch processing features

These are located in the `examples/` directory of the PYTHIA distribution.

## Resources

- **Official Manual:** https://pythia.org/latest-manual/Welcome.html
- **Python Interface Documentation:** https://pythia.org/latest-manual/PythonInterface.html
- **PyPI Package:** https://pypi.org/project/pythia8mc/
- **GitHub Tutorials:** https://gitlab.com/Pythia8/tutorials/
- **Main Website:** https://pythia.org/

## Quick Reference

| Task | Code |
|------|------|
| Install | `pip install pythia8mc` |
| Import | `import pythia8mc` |
| Create instance | `pythia = pythia8mc.Pythia()` |
| Configure | `pythia.readString("setting = value")` |
| Initialize | `pythia.init()` |
| Generate event | `pythia.next()` |
| Batch generate | `events = pythia.nextBatch(N)` |
| Iterate particles | `for p in pythia.event:` |
| Access momentum | `p.pT()`, `p.p()`, `p.e()` |
| Get particle name | `p.name()` |
| Check status | `p.isFinal()`, `p.isInitial()` |
| Get PDG ID | `p.id()` |
