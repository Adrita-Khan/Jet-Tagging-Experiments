

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/Adrita-Khan/Jet-Tagging)](https://github.com/Adrita-Khan/Jet-Tagging/issues)

----
*Note: This project is ongoing and subject to continuous advancements and modifications.*
----

This repository focuses on **jet tagging**—classifying collimated sprays of particles (jets) from high-energy collisions and identifying their originating partons. This is an **ongoing project** of the [Center for Computational and Data Sciences (CCDS)](https://ccds.ai/).

## Physics-Motivated Approach

We enhance the **Particle Chebyshev Network (PCN)** architecture by integrating physics-informed interaction features derived from the **Lund jet plane** formalism. The Lund plane provides a theoretically motivated representation of the phase space within jets through repeated Cambridge/Aachen declustering.

### Key Physics Features

Our model incorporates four fundamental QCD-inspired interaction features computed from particle 4-momentum vectors:

- **`ln Δ`**: Logarithmic angular separation between particles during jet declustering
- **`ln kT`**: Logarithmic relative transverse momentum at each declustering step  
- **`ln z`**: Logarithmic momentum fraction carried by the softer branch
- **`ln m²`**: Logarithmic invariant mass squared of particle pairs

These coordinates correspond to the natural variables of the Lund jet plane, where QCD emissions are approximately uniformly distributed, providing optimal discrimination between different jet origins and enabling the model to capture fine-grained, QCD-informed inter-particle dependencies.

### Physical Significance

- **Angular Structure (`ln Δ`)**: Captures the geometric distribution of radiation within jets
- **Momentum Hierarchy (`ln kT`)**: Reflects the energy scale of QCD splittings 
- **Energy Sharing (`ln z`)**: Quantifies how energy is distributed between decay products
- **Mass Scale (`ln m²`)**: Encodes the invariant mass information crucial for particle identification

## Dataset

The model is trained and evaluated on the **JetClass dataset** from CERN Open Data, which provides:
- High-quality jet samples from proton-proton collisions
- Multiple jet categories (quark, gluon, W/Z bosons, top quarks, etc.)
- Comprehensive particle-level information for each jet

## Features

- **Physics-informed feature engineering** based on the Lund jet plane formalism
- **Explainable AI architecture** providing interpretable jet classification
- **End-to-end training pipeline** with CERN Open Data integration
- **Comprehensive evaluation** using standard High Energy Physics (HEP) metrics
- **QCD-motivated design** ensuring theoretical consistency with parton shower physics

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- NumPy, SciPy
- Matplotlib, Seaborn (for visualization)

### Installation

```bash
git clone https://github.com/Adrita-Khan/Jet-Tagging.git
cd Jet-Tagging
pip install -r requirements.txt
```

### Quick Start

```bash
# Download and preprocess JetClass data
python scripts/download_data.py

# Train the E-PCN model
python src/train.py --config configs/epcn_config.yaml

# Evaluate model performance
python src/evaluate.py --model_path models/best_model.pth
```

## Repository Structure

```
jet-tagging/
├── data/                    # JetClass dataset and preprocessing
│   ├── raw/                # Raw CERN Open Data files  
│   └── processed/          # Processed jet samples with features
├── src/                    # Core implementation
│   ├── models/            # E-PCN architecture and physics features
│   ├── training/          # Training loops and optimization
│   ├── evaluation/        # Metrics and performance analysis
│   └── utils/             # Helper functions and data loading
├── notebooks/             # Analysis and visualization notebooks
│   ├── feature_analysis.ipynb    # Physics feature exploration
│   ├── model_comparison.ipynb    # Benchmarking against baselines
│   └── interpretability.ipynb   # Explainability analysis
├── configs/               # Model and training configurations
├── models/                # Saved model checkpoints
├── results/               # Evaluation results and plots
└── scripts/               # Data processing and utility scripts
```

## Methodology

### Physics-Informed Features

The Lund jet plane coordinates are computed through iterative declustering:

1. **Declustering Process**: Apply Cambridge/Aachen algorithm to reconstruct jet branching
2. **Coordinate Calculation**: For each splitting, compute (`ln Δ`, `ln kT`, `ln z`, `ln m²`)
3. **Feature Integration**: Incorporate these variables into the PCN architecture
4. **Training**: End-to-end optimization with physics-motivated regularization

### Model Architecture

- **Base Model**: Particle Chebyshev Networks (PCN) for permutation-invariant jet representation
- **Physics Enhancement**: Integration of Lund plane features at multiple network layers
- **Explainability**: Attention mechanisms highlighting important physics relationships
- **Output**: Multi-class jet classification with uncertainty quantification

## Results

Our physics-informed approach demonstrates:
- **Improved Classification**: Enhanced discrimination across jet categories
- **Physics Consistency**: Results aligned with QCD expectations
- **Interpretability**: Clear attribution to physical processes
- **Robustness**: Stable performance across different energy scales

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

- [CERN Open Data Portal](http://opendata.cern.ch/) for providing high-quality collision data
- The particle physics community for developing the theoretical foundations
- Original PCN authors for the base architecture

## References

1. [JetClass: A Large-Scale Dataset for Deep Learning in Jet Physics](https://link.springer.com/article/10.1007/JHEP07(2024)247)
2. [Particle Chebyshev Network (PCN)](https://proceedings.mlr.press/v162/qu22b.html)  
3. [The Lund Jet Plane](https://link.springer.com/article/10.1007/JHEP12(2018)064)
4. [Jet Substructure and Machine Learning](https://iopscience.iop.org/article/10.1088/1674-1137/ad7f3d/meta)
5. [QCD Analysis of Jet Substructure](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.056019)

## Citation

If you use this work in your research, please cite:

```bibtex
@article{khan2025epcn,
  title={Physics-Informed Interaction Feature Guided Explainable Particle Chebyshev Networks (E-PCN) for Jet Tagging},
  author={Khan, Adrita and others},
  journal={arXiv preprint},
  year={2025}
}
```

## Contact

**For inquiries, collaborations, or feedback:**

**Adrita Khan**  
📧 [Email](mailto:adrita.khan.official@gmail.com) | 💼 [LinkedIn](https://www.linkedin.com/in/adrita-khan) | 🐦 [Twitter](https://x.com/Adrita_)

---

*Advancing the intersection of theoretical physics and machine learning for fundamental particle research.*
