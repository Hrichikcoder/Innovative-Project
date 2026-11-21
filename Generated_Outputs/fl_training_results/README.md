# Federated Learning Model Training Results & Reports

This directory stores the complete experimental records, training logs, accuracy plots, and client contribution reports for baseline Federated Learning runs across **CIFAR-10** and **MNIST** datasets under Dirichlet Non-IID partitions (\(\alpha=0.1\), 30 clients).

---

## Directory Organization

```
fl_training_results/
├── cifar10/
│   ├── logs/       # Markdown training logs & advanced FedCurv console logs
│   ├── plots/      # Client, global, and combined accuracy trajectory curves
│   └── reports/    # Client contribution breakdown reports (.md)
└── mnist/
    ├── logs/       # Markdown training logs & advanced FedCurv console logs
    ├── plots/      # Client, global, and combined accuracy trajectory curves
    └── reports/    # Client contribution breakdown reports (.md)
```

---

## Evaluated Methods

1. **FedAvg (Federated Averaging):** Standard baseline federated learning algorithm.
2. **FedCurv (Curvature-Regularized FL):** Uses diagonal Fisher Information Matrix approximation to mitigate catastrophic forgetting.
3. **MI-IAA (Mutual Information–Aware FL):** Our proposed framework estimating latent feature-to-label statistical dependence via MINE critics to guide personalized aggregation.

---

## Contents per Dataset

### 1. CIFAR-10 (`cifar10/`)
* **Reports:**
  * `fedavg_cifar10_client_contributions.md`
  * `fedcurv_cifar10_client_contributions.md`
  * `mine_cifar10_client_contributions.md`
* **Logs:**
  * `cifar10_advanced_fedcurv_log.txt`
  * `training_log_fedavg_cifar10_20261121_135445.md`
  * `training_log_fedcurv_cifar10_20261121_135449.md`
  * `training_log_mine_cifar10_20261121_135453.md`
* **Plots:**
  * Client, global, and combined accuracy curves for FedAvg, FedCurv, and MI-IAA.

### 2. MNIST (`mnist/`)
* **Reports:**
  * `fedavg_mnist_client_contributions.md`
  * `fedcurv_mnist_client_contributions.md`
  * `mine_mnist_client_contributions.md`
* **Logs:**
  * `mnist_advanced_fedcurv_log.txt`
  * `training_log_fedavg_mnist_20261121_131726.md`
  * `training_log_fedcurv_mnist_20261121_131733.md`
  * `training_log_mine_mnist_20261121_131739.md`
* **Plots:**
  * Client, global, and combined accuracy curves for FedAvg, FedCurv, and MI-IAA.
