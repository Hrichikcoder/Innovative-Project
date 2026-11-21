# Tier 1 Specialization Retention Metric Experiments

This directory contains the round-level specialization retention evaluation results, empirical logs, and publication-ready plots for **CIFAR-10** and **MNIST**, benchmarked across **FedAvg**, **FedCurv**, and **MI-IAA**.

---

## Metric Definition

The **Specialization Retention Metric** \( R_i^{(t)} \) for client \( i \) at communication round \( t \) measures how much of the client's locally trained latent feature representation is preserved following global aggregation:

\[
R_i^{(t)} = \frac{\widehat{I}(Z_i(W^{(t)}); Y_i)}{\widehat{I}(Z_i(W_i^{(t)}); Y_i)}
\]

Where:
- \( W_i^{(t)} \): Client \( i \)'s local model weights *before* aggregation (post local training).
- \( W^{(t)} \): Global model weights *after* server aggregation.
- \( \widehat{I}(\cdot) \): Mutual information estimated via freshly trained MINE critics on the client's local partition \( D_i \).

---

## Directory Organization

```
specialization_retention_metric/
├── cifar10/
│   ├── logs/       # CSV and JSON retention records + corresponding training logs
│   └── plots/      # Trajectory and entropy-correlation plots
├── mnist/
│   ├── logs/       # CSV and JSON retention records + corresponding training logs
│   └── plots/      # Trajectory and entropy-correlation plots
└── scripts/        # Reproducibility scripts (evaluator, unified runner, plotter, launcher)
```

---

## Contents per Dataset

### 1. CIFAR-10 (`cifar10/`)
* **Logs:**
  * `specialization_retention_fedavg_cifar10_*.csv` & `.json`
  * `specialization_retention_fedcurv_cifar10_*.csv` & `.json`
  * `specialization_retention_mi_iaa_cifar10_*.csv` & `.json`
  * Corresponding training logs for FedAvg, FedCurv, and MI-IAA
* **Plots:**
  * `cifar10_specialization_retention_trajectory.png`: Retention dynamics across rounds.
  * `cifar10_retention_vs_entropy_weight.png`: Retention vs. label entropy and aggregation weight.

### 2. MNIST (`mnist/`)
* **Logs:**
  * `specialization_retention_fedavg_mnist_*.csv` & `.json`
  * `specialization_retention_fedcurv_mnist_*.csv` & `.json`
  * `specialization_retention_mi_iaa_mnist_*.csv` & `.json`
  * Corresponding training logs for FedAvg, FedCurv, and MI-IAA
* **Plots:**
  * `mnist_specialization_retention_trajectory.png`: Retention dynamics across rounds.
  * `mnist_retention_vs_entropy_weight.png`: Retention vs. label entropy and aggregation weight.

---

## Scripts (`scripts/`)
* `retention_evaluator.py`: Modular retention evaluator that computes pre- and post-aggregation MINE estimates.
* `run_specialization_experiments.py`: CLI runner to automate single or multi-method benchmarks.
* `plot_specialization_retention.py`: Multi-panel comparative plotter generating high-resolution figures.
* `run_specialization.ps1`: Quick launch PowerShell automation wrapper.
