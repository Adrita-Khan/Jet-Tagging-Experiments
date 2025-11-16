# Interaction Feature-Guided Explainable Particle Chebyshev Networks (E-PCN) for Jet Tagging

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/Adrita-Khan/Jet-Tagging)](https://github.com/Adrita-Khan/Jet-Tagging/issues)

---

> **Note:** This project is ongoing and subject to continuous advancements and modifications.

---

## Overview

This repository focuses on **jet tagging**—classifying collimated sprays of particles (jets) from high-energy collisions and associating them with their originating particles. It is an ongoing project of the [Center for Computational and Data Sciences (CCDS)](https://ccds.ai/) in collaboration with the [Department of Theoretical Physics, University of Dhaka](https://www.du.ac.bd/body/MissionVision/TPHY) and has strong ties to [CERN](https://home.cern/).


We enhance the **Particle Chebyshev Network (PCN)** architecture by integrating physics-motivated interaction features derived from particle 4-momentum vectors and inspired by the Lund jet plane. This approach improves discrimination capability for jet classification tasks using the **JetClass dataset**.

## Physics-Motivated Interaction Features

| Feature   | Description                                    |
|-----------|------------------------------------------------|
| `ln Δ`    | Logarithm of angular separation: $\ln \Delta R = \ln \sqrt{(\Delta\eta)^2 + (\Delta\phi)^2}$ |
| `ln kT`   | Logarithm of transverse momentum: $\ln k_T = \ln \min(p_{T,j}, p_{T,k}) \cdot \Delta R$ |
| `ln z`    | Logarithm of momentum fraction: $\ln z = \ln \min(p_{T,i}, p_{T,j})/(p_{T,i} + p_{T,j})$ |
| `ln m²`   | Logarithm of squared invariant mass: $\ln m_{ij}^2 = \ln \left( (E_i + E_j)^2 - |\mathbf{p}_i + \mathbf{p}_j|^2 \right)$ |


These features bias the model toward fine-grained, QCD-informed inter-particle dependencies.

## Key Features

- Physics-motivated feature integration into PCN
- End-to-end training with CERN Open Data (JetClass)
- Evaluation with standard HEP metrics

## Getting Started

### Installation

```bash
git clone https://github.com/Adrita-Khan/Jet-Tagging.git
cd Jet-Tagging
pip install -r requirements.txt
```

### Repository Structure

```
jet-tagging/
├── data/          # JetClass dataset
├── notebooks/     # Analysis & training notebooks
├── src/           # Model & feature integration
├── models/        # Saved checkpoints
└── requirements.txt
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

- [CERN Open Data Portal](http://opendata.cern.ch/) for providing high-quality collision data
- Original PCN authors for the base architecture
- [Center for Computational and Data Sciences (CCDS)](https://ccds.ai/)

## References

| # | Title | Link |
|---|-------|------|
| 1 | JetClass: A Large-Scale Dataset for Deep Learning in Jet Physics | [Springer](https://link.springer.com/article/10.1007/JHEP07(2024)247) |
| 2 | Particle Chebyshev Network (PCN) | [PMLR](https://proceedings.mlr.press/v162/qu22b.html) |
| 3 | The Lund Jet Plane | [Springer](https://link.springer.com/article/10.1007/JHEP12(2018)064) |
| 4 | Jet Substructure and Machine Learning | [IOP Science](https://iopscience.iop.org/article/10.1088/1674-1137/ad7f3d/meta) |
| 5 | Jet Tagging via Particle Clouds | [Physical Review D](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.056019) |
| 6 | PCN-Jet-Tagging GitHub Repository | [GitHub](https://github.com/YVSemlani/PCN-Jet-Tagging) |

## Contact

For any inquiries or feedback, please contact:

| Name | Email | LinkedIn | Twitter |
|------|-------|----------|---------|
| **Adrita Khan** | [Email](mailto:adrita.khan.official@gmail.com) | [LinkedIn](https://www.linkedin.com/in/adrita-khan) | [Twitter](https://x.com/Adrita_) |
| **Md Raqibul Islam** | [Email](mailto:raqibul.islam.academic@gmail.com) | [LinkedIn](https://www.linkedin.com/in/raqib03/) | [Twitter](https://x.com/raqib_03) |

---

*Maintained by the CCDS Team*
