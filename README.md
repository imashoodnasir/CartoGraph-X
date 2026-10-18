
# CartoGraph-X

**Residual GraphSAGE with Gated Fusion for Cross-Country Road Network Selection**

CartoGraph-X is a graph neural network framework for road-segment selection across geographic domains. It investigates road-retention prediction using road geometry, spatial neighborhood representations, residual graph learning, and gated feature fusion.

The framework is evaluated using road-network datasets from Lithuania and Poland under within-country, cross-country, and joint-country training protocols.

## Methodology

The CartoGraph-X methodology consists of:

1. **Data acquisition:** Collection and inspection of geographic road-network datasets.
2. **Road correspondence:** Geometric matching between detailed source roads and generalized reference networks to construct road-retention labels.
3. **Benchmark construction:** Geographic partitioning, feature extraction, dataset harmonization, and leakage-aware evaluation preparation.
4. **Spatial graph construction:** Road segments represented as nodes, with centroid-based proximity relationships.
5. **Graph representation learning:** A two-layer residual GraphSAGE encoder.
6. **Gated feature fusion:** Adaptive combination of projected road-segment features and graph-context representations.
7. **Road-retention prediction:** Class-weighted binary classification with country-conditioned prediction during joint-country training.
8. **Experimental evaluation:** Comparison against classical and graph-based baselines, including ablation, sensitivity, robustness, calibration, and statistical analyses.

## Model Architecture

| Component | Configuration |
|---|---|
| Input features | 23 |
| Graph construction | Centroid-based spatial proximity |
| Maximum neighbors | 6 |
| Neighborhood radius | 250 m |
| Graph encoder | Two-layer residual GraphSAGE |
| Hidden dimension | 64 |
| Aggregation | Mean |
| Dropout | 0.2 |
| Feature fusion | Dimension-wise gated fusion |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Training epochs | 60 |
| Training objective | Class-weighted binary cross-entropy |

These settings describe the reported CartoGraph-X methodology; the repository also contains earlier development implementations.

## Evaluation Protocols

Six evaluation protocols are considered:

| Protocol | Training domain | Evaluation domain |
|---|---|---|
| LT→LT | Lithuania | Lithuania |
| PL→PL | Poland | Poland |
| LT→PL | Lithuania | Poland |
| PL→LT | Poland | Lithuania |
| Joint→LT | Lithuania + Poland | Lithuania |
| Joint→PL | Lithuania + Poland | Poland |

The experiments report Matthews correlation coefficient (MCC), ROC-AUC, and PR-AUC.

Stochastic model experiments use ten matched random seeds (42–51), where applicable.

## Repository Contents

- `scripts/` — Data acquisition, correspondence construction, benchmark preparation, model training, evaluation, ablations, and figure-generation scripts.
- `src/` — Shared Python modules and utilities.
- `configs/` — Development and experiment configuration files.
- `experiments/` — Supporting experimental resources.
- `CartoGraphX_Publication_Figures/` — Publication-related figures.
- `figures_5_9/` — Additional experimental visualizations.
- `requirements.txt` — Python dependencies.

The repository preserves historical development scripts alongside the final CartoGraph-X experiments.

## Installation

Clone the repository:

```bash
git clone https://github.com/imashoodnasir/CartoGraph-X.git
cd CartoGraph-X
```

Create and activate a Python environment.

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Reproducibility

The repository contains scripts for the experimental workflow, including:

- Geographic data acquisition and inspection
- Lithuania and Poland benchmark preparation
- Geometric road correspondence and label construction
- Cross-country dataset harmonization
- Classical machine learning baselines
- Graph neural network baselines
- CartoGraph-X training
- Ten-seed experiments
- Architecture ablations
- Statistical comparisons
- Graph sensitivity analysis
- Training-data sensitivity analysis
- Calibration and reliability evaluation
- Geographic robustness assessment
- Error and failure analysis
- Publication figure generation

The code is provided in its original research-development form. Some scripts may require local data paths, source-specific inputs, or configuration adjustments. An independently verified end-to-end reproduction procedure is not yet available.

## Data Availability

The methodology uses geographic road-network data from Lithuania and Poland.

Large geographic datasets and intermediate artifacts are not necessarily included in this repository. Users must obtain the required source data from the corresponding providers, subject to their licensing conditions.

## Citation

If you use this implementation, please cite:

*CartoGraph-X: Residual GraphSAGE with Gated Fusion for Cross-Country Road Network Selection.*

Full bibliographic information will be added when available.

## Repository

https://github.com/imashoodnasir/CartoGraph-X
