# Jet Tagging with CERN Open Data

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/Adrita-Khan/Jet-Tagging)](https://github.com/Adrita-Khan/Jet-Tagging/issues)
[![GitHub stars](https://img.shields.io/github/stars/Adrita-Khan/Jet-Tagging)](https://github.com/Adrita-Khan/Jet-Tagging/stargazers)


This repository focuses on **jet tagging**—classifying collimated sprays of particles (jets) from high-energy collisions and associating them with their originating particles. We enhance the **Particle Chebyshev Network (PCN)** architecture by integrating four physics-motivated interaction features:

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

Download JetClass from [CERN Open Data](http://opendata.cern.ch/) and place it in `data/`.

## Usage

* **Notebook:** `notebooks/jet_tagging_analysis.ipynb`
* **Script:**

```bash
python src/train_model.py --config configs/config.yaml
```

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

* [CERN Open Data Portal](http://opendata.cern.ch/)



---


## Contact

**For any inquiries or feedback, please contact:**

**Adrita Khan**  
[Email](mailto:adrita.khan.official@gmail.com) | [LinkedIn](https://www.linkedin.com/in/adrita-khan) | [Twitter](https://x.com/Adrita_)


