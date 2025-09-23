# Pythia Jet Tagging Data Generation Tutorial

## Overview
This tutorial covers how to generate jet tagging data using Pythia8, focusing on creating datasets suitable for machine learning applications in high-energy physics. We'll cover event generation, jet finding, feature extraction, and data preparation for training jet tagging algorithms.

## Prerequisites
- Pythia 8.315 or later installed
- FastJet library (recommended for jet finding)
- ROOT (optional, for data storage)
- Basic knowledge of C++ and high-energy physics concepts

## Table of Contents
1. [Setting up Pythia for Jet Studies](#setting-up-pythia)
2. [Basic Event Generation](#basic-event-generation)
3. [Jet Finding with FastJet](#jet-finding)
4. [Feature Extraction](#feature-extraction)
5. [Creating Training Datasets](#creating-datasets)
6. [Advanced Configurations](#advanced-configurations)
7. [Output Formats](#output-formats)

## Setting up Pythia for Jet Studies {#setting-up-pythia}

### Installation
First, ensure you have Pythia8 properly installed with FastJet support:

```bash
# Download Pythia 8.315
wget https://pythia.org/download/pythia83/pythia8315.tgz
tar xvfz pythia8315.tgz
cd pythia8315

# Configure with FastJet (if available)
./configure --with-fastjet3=/path/to/fastjet
make

# Test installation
cd examples
make main213
./main213
```

### Basic Directory Structure
```
jet_tagging_project/
├── src/
│   ├── jet_generator.cc
│   └── Makefile
├── config/
│   ├── qcd_jets.cmnd
│   └── ttbar_jets.cmnd
└── output/
    └── data/
```

## Basic Event Generation {#basic-event-generation}

### Simple Jet Generator (C++)
Create `src/jet_generator.cc`:

```cpp
#include "Pythia8/Pythia.h"
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include <iostream>
#include <fstream>
#include <vector>

using namespace Pythia8;
using namespace fastjet;

int main() {
    // Initialize Pythia
    Pythia pythia;
    
    // Read configuration
    pythia.readFile("../config/qcd_jets.cmnd");
    pythia.init();
    
    // Setup FastJet
    double R = 0.4;  // Jet radius
    JetDefinition jet_def(antikt_algorithm, R);
    
    // Output file
    ofstream outfile("../output/jet_data.csv");
    outfile << "jet_pt,jet_eta,jet_phi,jet_mass,jet_flavor,constituents\n";
    
    // Event generation loop
    int nEvents = 10000;
    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
        if (!pythia.next()) continue;
        
        // Convert to FastJet particles
        vector<PseudoJet> particles;
        for (int i = 0; i < pythia.event.size(); ++i) {
            if (pythia.event[i].isFinal() && pythia.event[i].isCharged()) {
                particles.push_back(PseudoJet(
                    pythia.event[i].px(),
                    pythia.event[i].py(),
                    pythia.event[i].pz(),
                    pythia.event[i].e()
                ));
                particles.back().set_user_index(i);
            }
        }
        
        // Cluster jets
        ClusterSequence cs(particles, jet_def);
        vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(20.0));
        
        // Process jets
        for (const auto& jet : jets) {
            if (jet.pt() > 20.0 && abs(jet.eta()) < 2.5) {
                // Extract jet features
                double pt = jet.pt();
                double eta = jet.eta();
                double phi = jet.phi();
                double mass = jet.m();
                
                // Determine jet flavor (simplified)
                int flavor = getJetFlavor(jet, pythia.event);
                
                // Count constituents
                vector<PseudoJet> constituents = jet.constituents();
                int n_constituents = constituents.size();
                
                // Write to file
                outfile << pt << "," << eta << "," << phi << "," 
                       << mass << "," << flavor << "," << n_constituents << "\n";
            }
        }
        
        if (iEvent % 1000 == 0) {
            cout << "Processed " << iEvent << " events" << endl;
        }
    }
    
    outfile.close();
    pythia.stat();
    return 0;
}

// Helper function for jet flavor identification
int getJetFlavor(const PseudoJet& jet, const Event& event) {
    // Simplified flavor matching
    vector<PseudoJet> constituents = jet.constituents();
    
    for (const auto& constituent : constituents) {
        int idx = constituent.user_index();
        if (idx >= 0 && idx < event.size()) {
            int pdg = abs(event[idx].id());
            if (pdg == 5) return 5;  // b-quark
            if (pdg == 4) return 4;  // c-quark
        }
    }
    return 1;  // light quark/gluon
}
```

### Configuration Files

Create `config/qcd_jets.cmnd`:
```
! QCD jet production
Beams:eCM = 13000.          ! CM energy of LHC
HardQCD:all = on            ! Switch on all hard QCD processes

! Phase space cuts
PhaseSpace:pTHatMin = 20.   ! Minimum pT in hard process
PhaseSpace:pTHatMax = 500.  ! Maximum pT in hard process

! PDF settings
PDF:pSet = LHAPDF6:NNPDF31_lo_as_0118

! Multiple interactions
MultipartonInteractions:pT0Ref = 2.4024
MultipartonInteractions:ecmPow = 0.25208

! Random seed
Random:setSeed = on
Random:seed = 12345
```

Create `config/ttbar_jets.cmnd` for top quark events:
```
! Top pair production
Beams:eCM = 13000.
Top:gg2ttbar = on
Top:qqbar2ttbar = on

! Force hadronic decays for more jets
24:onMode = off
24:onIfMatch = 1 -2
24:onIfMatch = 3 -4

Random:setSeed = on
Random:seed = 54321
```

## Jet Finding with FastJet {#jet-finding}

### Advanced Jet Finding Setup

```cpp
// Enhanced jet finding with substructure
class JetAnalyzer {
private:
    JetDefinition jet_def;
    JetDefinition subjet_def;
    
public:
    JetAnalyzer(double R = 0.4) : 
        jet_def(antikt_algorithm, R),
        subjet_def(kt_algorithm, R/2) {}
    
    struct JetInfo {
        double pt, eta, phi, mass;
        int flavor;
        int n_constituents;
        double tau1, tau2, tau3;  // N-subjettiness
        double splitting_scale;
        vector<double> constituent_pts;
        vector<double> constituent_etas;
        vector<double> constituent_phis;
    };
    
    vector<JetInfo> analyzeEvent(const vector<PseudoJet>& particles) {
        ClusterSequence cs(particles, jet_def);
        vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(20.0));
        
        vector<JetInfo> jet_infos;
        
        for (const auto& jet : jets) {
            if (jet.pt() > 20.0 && abs(jet.eta()) < 2.5) {
                JetInfo info;
                info.pt = jet.pt();
                info.eta = jet.eta();
                info.phi = jet.phi();
                info.mass = jet.m();
                
                // Get constituents
                vector<PseudoJet> constituents = jet.constituents();
                info.n_constituents = constituents.size();
                
                // Store constituent information
                for (const auto& constituent : constituents) {
                    info.constituent_pts.push_back(constituent.pt());
                    info.constituent_etas.push_back(constituent.eta());
                    info.constituent_phis.push_back(constituent.phi());
                }
                
                // Calculate N-subjettiness (simplified)
                calculateNSubjettiness(jet, info);
                
                jet_infos.push_back(info);
            }
        }
        
        return jet_infos;
    }
    
private:
    void calculateNSubjettiness(const PseudoJet& jet, JetInfo& info) {
        // Simplified N-subjettiness calculation
        vector<PseudoJet> constituents = jet.constituents();
        
        // Find subjets
        ClusterSequence subjet_cs(constituents, subjet_def);
        vector<PseudoJet> subjets = subjet_cs.exclusive_jets(1);
        
        info.tau1 = 0.0;
        for (const auto& constituent : constituents) {
            double min_dR = 999.0;
            for (const auto& subjet : subjets) {
                double dR = constituent.delta_R(subjet);
                if (dR < min_dR) min_dR = dR;
            }
            info.tau1 += constituent.pt() * min_dR;
        }
        info.tau1 /= jet.pt();
        
        // Calculate tau2, tau3 similarly...
        info.tau2 = info.tau1 * 0.7;  // Placeholder
        info.tau3 = info.tau1 * 0.5;  // Placeholder
    }
};
```

## Feature Extraction {#feature-extraction}

### Jet Image Generation

```cpp
class JetImageGenerator {
private:
    int n_pixels;
    double jet_size;
    
public:
    JetImageGenerator(int pixels = 33, double size = 0.8) : 
        n_pixels(pixels), jet_size(size) {}
    
    vector<vector<double>> generateImage(const PseudoJet& jet) {
        vector<vector<double>> image(n_pixels, vector<double>(n_pixels, 0.0));
        
        // Center jet at (0, 0)
        double center_eta = jet.eta();
        double center_phi = jet.phi();
        
        vector<PseudoJet> constituents = jet.constituents();
        
        for (const auto& constituent : constituents) {
            double deta = constituent.eta() - center_eta;
            double dphi = constituent.phi() - center_phi;
            
            // Handle phi wraparound
            if (dphi > M_PI) dphi -= 2 * M_PI;
            if (dphi < -M_PI) dphi += 2 * M_PI;
            
            // Convert to pixel coordinates
            int eta_pixel = static_cast<int>((deta + jet_size/2) * n_pixels / jet_size);
            int phi_pixel = static_cast<int>((dphi + jet_size/2) * n_pixels / jet_size);
            
            // Check bounds
            if (eta_pixel >= 0 && eta_pixel < n_pixels && 
                phi_pixel >= 0 && phi_pixel < n_pixels) {
                image[eta_pixel][phi_pixel] += constituent.pt();
            }
        }
        
        return image;
    }
    
    void saveImage(const vector<vector<double>>& image, 
                   const string& filename) {
        ofstream file(filename);
        for (int i = 0; i < n_pixels; ++i) {
            for (int j = 0; j < n_pixels; ++j) {
                file << image[i][j];
                if (j < n_pixels - 1) file << ",";
            }
            file << "\n";
        }
        file.close();
    }
};
```

## Creating Training Datasets {#creating-datasets}

### Multi-Class Dataset Generator

```cpp
class DatasetGenerator {
private:
    map<string, Pythia*> generators;
    JetAnalyzer analyzer;
    JetImageGenerator imageGen;
    
public:
    DatasetGenerator() : analyzer(0.4), imageGen(33, 0.8) {
        // Initialize different generators for different jet types
        setupGenerators();
    }
    
    void setupGenerators() {
        // QCD jets (background)
        generators["qcd"] = new Pythia();
        generators["qcd"]->readFile("../config/qcd_jets.cmnd");
        generators["qcd"]->init();
        
        // Top jets (signal)
        generators["ttbar"] = new Pythia();
        generators["ttbar"]->readFile("../config/ttbar_jets.cmnd");
        generators["ttbar"]->init();
        
        // W jets
        generators["wjets"] = new Pythia();
        generators["wjets"]->readString("WeakBosonAndParton:qqbar2gmZg = on");
        generators["wjets"]->readString("WeakBosonAndParton:qg2gmZq = on");
        generators["wjets"]->init();
    }
    
    void generateDataset(const string& output_prefix, 
                        int events_per_class = 10000) {
        
        // Output files
        ofstream features_file(output_prefix + "_features.csv");
        ofstream images_dir = output_prefix + "_images/";
        
        // Write header
        features_file << "jet_pt,jet_eta,jet_phi,jet_mass,n_constituents,"
                     << "tau1,tau2,tau3,class_label,image_file\n";
        
        int jet_counter = 0;
        
        for (const auto& [class_name, pythia] : generators) {
            cout << "Generating " << class_name << " jets..." << endl;
            
            int class_label = getClassLabel(class_name);
            int events_generated = 0;
            
            while (events_generated < events_per_class) {
                if (!pythia->next()) continue;
                
                // Convert to FastJet particles
                vector<PseudoJet> particles = getParticles(*pythia);
                
                // Analyze jets
                vector<JetAnalyzer::JetInfo> jets = analyzer.analyzeEvent(particles);
                
                for (const auto& jet_info : jets) {
                    // Apply selection cuts
                    if (jet_info.pt < 20.0 || abs(jet_info.eta) > 2.5) continue;
                    
                    // Create jet for image generation
                    PseudoJet jet = reconstructJet(jet_info);
                    
                    // Generate image
                    auto image = imageGen.generateImage(jet);
                    string image_filename = output_prefix + "_images/jet_" + 
                                          to_string(jet_counter) + ".csv";
                    imageGen.saveImage(image, image_filename);
                    
                    // Write features
                    features_file << jet_info.pt << "," << jet_info.eta << ","
                                 << jet_info.phi << "," << jet_info.mass << ","
                                 << jet_info.n_constituents << ","
                                 << jet_info.tau1 << "," << jet_info.tau2 << ","
                                 << jet_info.tau3 << "," << class_label << ","
                                 << image_filename << "\n";
                    
                    jet_counter++;
                    events_generated++;
                    
                    if (events_generated >= events_per_class) break;
                }
            }
        }
        
        features_file.close();
        cout << "Generated " << jet_counter << " total jets" << endl;
    }
    
private:
    vector<PseudoJet> getParticles(const Pythia& pythia) {
        vector<PseudoJet> particles;
        for (int i = 0; i < pythia.event.size(); ++i) {
            if (pythia.event[i].isFinal() && 
                pythia.event[i].isVisible()) {
                particles.push_back(PseudoJet(
                    pythia.event[i].px(),
                    pythia.event[i].py(),
                    pythia.event[i].pz(),
                    pythia.event[i].e()
                ));
                particles.back().set_user_index(i);
            }
        }
        return particles;
    }
    
    int getClassLabel(const string& class_name) {
        if (class_name == "qcd") return 0;
        if (class_name == "ttbar") return 1;
        if (class_name == "wjets") return 2;
        return -1;
    }
    
    PseudoJet reconstructJet(const JetAnalyzer::JetInfo& info) {
        // Reconstruct jet from constituent information
        vector<PseudoJet> constituents;
        for (size_t i = 0; i < info.constituent_pts.size(); ++i) {
            double pt = info.constituent_pts[i];
            double eta = info.constituent_etas[i];
            double phi = info.constituent_phis[i];
            
            constituents.push_back(PseudoJet());
            constituents.back().reset_PtYPhiM(pt, eta, phi, 0.0);
        }
        
        JetDefinition jet_def(antikt_algorithm, 0.4);
        ClusterSequence cs(constituents, jet_def);
        return cs.inclusive_jets(0.0)[0];
    }
};
```

## Advanced Configurations {#advanced-configurations}

### Systematic Variations

```cpp
// Configuration for systematic studies
struct SystematicConfig {
    double pt_smearing = 0.0;     // Relative pT smearing
    double eta_shift = 0.0;       // Absolute eta shift
    double energy_scale = 1.0;    // Energy scale factor
    bool pileup_enabled = false;  // Add pileup events
    int pileup_mean = 20;         // Mean number of pileup interactions
};

class SystematicGenerator {
public:
    static void applySystematics(vector<PseudoJet>& particles, 
                                const SystematicConfig& config) {
        for (auto& particle : particles) {
            // Apply energy scale
            particle *= config.energy_scale;
            
            // Apply pT smearing
            if (config.pt_smearing > 0) {
                double smear = 1.0 + gRandom->Gaus(0, config.pt_smearing);
                double pt = particle.pt() * smear;
                particle.reset_PtYPhiM(pt, particle.rap(), 
                                     particle.phi(), particle.m());
            }
            
            // Apply eta shift
            if (config.eta_shift != 0) {
                double eta = particle.eta() + config.eta_shift;
                particle.reset_PtYPhiM(particle.pt(), eta, 
                                     particle.phi(), particle.m());
            }
        }
    }
};
```

## Output Formats {#output-formats}

### ROOT Output (Optional)

```cpp
#ifdef USE_ROOT
#include "TFile.h"
#include "TTree.h"

class ROOTWriter {
private:
    TFile* file;
    TTree* tree;
    
    // Branch variables
    Float_t jet_pt, jet_eta, jet_phi, jet_mass;
    Int_t jet_flavor, n_constituents;
    vector<float> constituent_pts;
    vector<float> constituent_etas;
    
public:
    ROOTWriter(const string& filename) {
        file = new TFile(filename.c_str(), "RECREATE");
        tree = new TTree("jets", "Jet data for ML training");
        
        tree->Branch("jet_pt", &jet_pt);
        tree->Branch("jet_eta", &jet_eta);
        tree->Branch("jet_phi", &jet_phi);
        tree->Branch("jet_mass", &jet_mass);
        tree->Branch("jet_flavor", &jet_flavor);
        tree->Branch("n_constituents", &n_constituents);
        tree->Branch("constituent_pts", &constituent_pts);
        tree->Branch("constituent_etas", &constituent_etas);
    }
    
    void writeJet(const JetAnalyzer::JetInfo& info) {
        jet_pt = info.pt;
        jet_eta = info.eta;
        jet_phi = info.phi;
        jet_mass = info.mass;
        jet_flavor = info.flavor;
        n_constituents = info.n_constituents;
        
        constituent_pts = info.constituent_pts;
        constituent_etas = info.constituent_etas;
        
        tree->Fill();
    }
    
    ~ROOTWriter() {
        tree->Write();
        file->Close();
        delete file;
    }
};
#endif
```

### HDF5 Output (Recommended for ML)

```cpp
// For use with Python ML frameworks
class HDF5Writer {
public:
    static void writeDataset(const string& filename,
                            const vector<JetAnalyzer::JetInfo>& jets,
                            const vector<vector<vector<double>>>& images) {
        // Write structured data that can be easily loaded in Python
        // Implementation would use HDF5 C++ library
        
        ofstream metadata(filename + "_metadata.json");
        metadata << "{\n";
        metadata << "  \"n_jets\": " << jets.size() << ",\n";
        metadata << "  \"image_size\": [33, 33],\n";
        metadata << "  \"features\": [\"pt\", \"eta\", \"phi\", \"mass\", \"tau1\", \"tau2\", \"tau3\"]\n";
        metadata << "}\n";
        metadata.close();
    }
};
```

## Compilation and Usage

### Makefile
Create `src/Makefile`:
```makefile
CXX = g++
CXXFLAGS = -O2 -std=c++11 -I$(PYTHIA8)/include
LDFLAGS = -L$(PYTHIA8)/lib -lpythia8 -ldl

# FastJet flags
FASTJET_CONFIG = fastjet-config
CXXFLAGS += `$(FASTJET_CONFIG) --cxxflags`
LDFLAGS += `$(FASTJET_CONFIG) --libs`

# ROOT flags (optional)
# CXXFLAGS += `root-config --cflags` -DUSE_ROOT
# LDFLAGS += `root-config --libs`

SOURCES = jet_generator.cc
TARGET = jet_generator

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: clean
```

### Running the Generator

```bash
# Compile
cd src
make

# Create output directory
mkdir -p ../output/training_images

# Generate datasets
./jet_generator

# The program will create:
# - ../output/training_features.csv (tabular data)
# - ../output/training_images/ (jet images)
```

## Python Interface for Analysis

Create a Python script to load and analyze the generated data:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the generated data
df = pd.read_csv('output/training_features.csv')

# Basic statistics
print(f"Generated {len(df)} jets")
print(f"Class distribution:\n{df['class_label'].value_counts()}")

# Plot some distributions
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
df['jet_pt'].hist(bins=50, ax=axes[0,0])
axes[0,0].set_xlabel('Jet pT [GeV]')

df['jet_eta'].hist(bins=50, ax=axes[0,1])
axes[0,1].set_xlabel('Jet η')

df['n_constituents'].hist(bins=50, ax=axes[1,0])
axes[1,0].set_xlabel('Number of Constituents')

df['tau1'].hist(bins=50, ax=axes[1,1])
axes[1,1].set_xlabel('N-subjettiness τ₁')

plt.tight_layout()
plt.savefig('output/jet_distributions.png')
plt.show()

# Simple ML training example
features = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_mass', 
           'n_constituents', 'tau1', 'tau2', 'tau3']
X = df[features]
y = df['class_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Classification accuracy: {accuracy:.3f}")
```

## Next Steps

1. **Extend jet flavors**: Add more physics processes (Z+jets, Higgs, etc.)
2. **Advanced features**: Implement more sophisticated jet substructure variables
3. **Deep learning**: Use the generated images with CNNs or graph neural networks
4. **Detector simulation**: Add realistic detector effects using Delphes
5. **Systematic studies**: Generate datasets with different systematic variations

## References

- PYTHIA 8.315 Manual
- JetClass: A Large-Scale Dataset for Deep Learning in Jet Physics
- Sample Main Programs in Pythia documentation

This tutorial provides a comprehensive framework for generating jet tagging datasets with Pythia. The modular design allows for easy extension and customization based on specific research needs.
