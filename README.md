# Jet Tagging with CERN Open Data

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/Adrita-Khan/Jet-Tagging)](https://github.com/Adrita-Khan/Jet-Tagging/issues)

----
*Note: This project is ongoing and subject to continuous advancements and modifications.*
----

This repository focuses on **jet tagging**—classifying collimated sprays of particles (jets) from high-energy collisions and associating them with their originating particles. It is an **ongoing project** of the [Center for Computational and Data Sciences (CCDS)](https://ccds.ai/).

We enhance the **Particle Chebyshev Network (PCN)** architecture by integrating four physics-motivated interaction features:

- `ln Δ`
- `ln kT`
- `ln z`
- `ln m²`

These features, derived from particle 4-momentum vectors and inspired by the Lund jet plane, bias the model toward fine-grained, QCD-informed inter-particle dependencies. The approach improves discrimination capability for jet classification tasks using the **JetClass dataset**.

## Features
- Physics-motivated feature integration into PCN.
- End-to-end training with CERN Open Data (JetClass).
- Evaluation with standard HEP metrics.

## Getting Started
```bash
git clone https://github.com/Adrita-Khan/Jet-Tagging.git
cd Jet-Tagging
pip install -r requirements.txt
````





## Structure

```
jet-tagging/
├── data/          # JetClass dataset
├── notebooks/     # Analysis & training
├── src/           # Model & feature integration
├── models/        # Saved checkpoints
```

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgements

- [CERN Open Data Portal](http://opendata.cern.ch/) for providing high-quality collision data
- Original PCN authors for the base architecture

## References

1. [JetClass: A Large-Scale Dataset for Deep Learning in Jet Physics](https://link.springer.com/article/10.1007/JHEP07(2024)247)
2. [Particle Chebyshev Network (PCN)](https://proceedings.mlr.press/v162/qu22b.html)  
3. [The Lund Jet Plane](https://link.springer.com/article/10.1007/JHEP12(2018)064)
4. [Jet Substructure and Machine Learning](https://iopscience.iop.org/article/10.1088/1674-1137/ad7f3d/meta)
5. [QCD Analysis of Jet Substructure](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.056019)


---


## Contact

**For any inquiries or feedback, please contact:**

**Adrita Khan**  
[Email](mailto:adrita.khan.official@gmail.com) | [LinkedIn](https://www.linkedin.com/in/adrita-khan) | [Twitter](https://x.com/Adrita_)


