"""
retention_evaluator.py
Tier 1: Round-Level Specialization Retention Metric Evaluation Module.

Evaluates how much of each client's label-relevant specialization the global
model retains after aggregation, comparing local models (w_i^t) against the
global model (w_g^t) on held-out evaluation data (D_i^val).

Passive observer: Does not modify model weights, losses, or aggregation.
"""

import os
import json
import time
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


# =====================================================================
# 1. MINE Critic Architecture (Per-Feature Decomposed MLP)
# =====================================================================
class PerFeatureMINENetwork(nn.Module):
    """
    Per-feature MINE critic identical to the existing implementation in mine.py.
    Each of the 128 latent dimensions z_k has an independent MLP seeing (z_k, y).
    Vectorized using batched matrix multiplication (bmm).
    """
    def __init__(self, z_dim=128, y_dim=10, hidden=16):
        super(PerFeatureMINENetwork, self).__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.hidden = hidden

        self.W1 = nn.Parameter(torch.randn(z_dim, 1 + y_dim, hidden) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(z_dim, hidden))
        self.W2 = nn.Parameter(torch.randn(z_dim, hidden, 1) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(z_dim, 1))

    def forward(self, z, y):
        batch_size = z.shape[0]
        y_exp = y.unsqueeze(1).expand(batch_size, self.z_dim, self.y_dim)
        z_exp = z.unsqueeze(-1)
        inp = torch.cat([z_exp, y_exp], dim=-1)
        inp = inp.permute(1, 0, 2)
        h = torch.relu(torch.bmm(inp, self.W1) + self.b1.unsqueeze(1))
        out = torch.bmm(h, self.W2) + self.b2.unsqueeze(1)
        return out.squeeze(-1).permute(1, 0)


# =====================================================================
# 2. Latent Feature Extraction
# =====================================================================
@torch.no_grad()
def extract_latents(model, loader, device):
    """
    Pass dataset through model in eval mode to extract final 128-d latents z
    and target labels y.
    """
    model.eval()
    zs, ys = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.view(-1)
        # Model forward returns (logits, z)
        out = model(x)
        if isinstance(out, tuple):
            _, z = out
        else:
            z = out
        zs.append(z.detach())
        ys.append(y.to(device))

    if len(zs) == 0:
        return torch.empty(0, 128, device=device), torch.empty(0, dtype=torch.long, device=device)

    return torch.cat(zs, dim=0), torch.cat(ys, dim=0)


# =====================================================================
# 3. Fresh Critic Training & MI Estimation (Donsker-Varadhan Lower Bound)
# =====================================================================
def train_fresh_mine(z_fit, y_fit_one_hot, num_classes, steps=100, lr=1e-3, device='cuda'):
    """
    Train a brand new MINE critic on held-out critic-fit data.
    Never reuses training critics to prevent optimistic bias.
    """
    z_dim = z_fit.shape[1] if z_fit.numel() > 0 else 128
    critic = PerFeatureMINENetwork(z_dim=z_dim, y_dim=num_classes, hidden=16).to(device)
    critic.train()

    optimizer = optim.Adam(critic.parameters(), lr=lr)
    n_samples = z_fit.shape[0]

    if n_samples < 2:
        return critic, 0.0

    batch_size = min(64, n_samples)
    running_mean_marginal = None
    EMA_DECAY = 0.99
    final_loss = 0.0

    for step in range(steps):
        # Sample minibatch
        if batch_size < n_samples:
            idx = torch.randperm(n_samples, device=device)[:batch_size]
            z_batch = z_fit[idx]
            y_batch = y_fit_one_hot[idx]
        else:
            z_batch = z_fit
            y_batch = y_fit_one_hot

        # Shuffle labels for marginal p(z)p(y)
        shuffle_idx = torch.randperm(z_batch.shape[0], device=device)
        y_shuffled = y_batch[shuffle_idx]

        t_joint = critic(z_batch, y_batch)
        t_marginal = critic(z_batch, y_shuffled)

        mean_exp_marginal = torch.mean(torch.exp(t_marginal), dim=0)

        if running_mean_marginal is None:
            running_mean_marginal = mean_exp_marginal.detach()
        else:
            running_mean_marginal = (
                EMA_DECAY * running_mean_marginal + (1.0 - EMA_DECAY) * mean_exp_marginal.detach()
            )

        debiased_marginal = mean_exp_marginal / (running_mean_marginal + 1e-12)
        loss = -torch.mean(torch.mean(t_joint, dim=0) - debiased_marginal)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    return critic, final_loss


@torch.no_grad()
def estimate_mi(critic, z_rep, y_rep_one_hot, n_shuffles=5, device='cuda'):
    """
    Estimate mutual information on the held-out report half across multiple
    marginal label permutations to control shuffle variance.
    Returns:
        avg_mi (float): mean MI in nats averaged across 128 dimensions
        shuffle_std (float): std of MI across marginal shuffles
        per_feat_mi (list of float): 128 per-feature MI values
    """
    critic.eval()
    n_samples = z_rep.shape[0]
    if n_samples < 2:
        return 0.0, 0.0, [0.0] * 128

    t_joint = critic(z_rep, y_rep_one_hot)  # [N, 128]
    mean_joint = torch.mean(t_joint, dim=0)  # [128]

    estimates = []
    per_feat_runs = []

    for _ in range(n_shuffles):
        shuffle_idx = torch.randperm(n_samples, device=device)
        y_shuffled = y_rep_one_hot[shuffle_idx]
        t_marginal = critic(z_rep, y_shuffled)  # [N, 128]

        # Numerically stable log-mean-exp
        max_t = torch.max(t_marginal, dim=0, keepdim=True)[0]
        log_mean_exp = max_t.squeeze(0) + torch.log(
            torch.mean(torch.exp(t_marginal - max_t), dim=0).clamp(min=1e-12)
        )
        mi_k = mean_joint - log_mean_exp  # [128]
        per_feat_runs.append(mi_k)
        estimates.append(torch.mean(mi_k).item())

    stacked_feats = torch.stack(per_feat_runs, dim=0)  # [n_shuffles, 128]
    avg_per_feat = torch.mean(stacked_feats, dim=0).tolist()
    avg_mi = float(np.mean(estimates))
    shuffle_std = float(np.std(estimates))

    return avg_mi, shuffle_std, avg_per_feat


# =====================================================================
# 4. Metrics & Entropy Computation
# =====================================================================
def compute_label_entropy(labels, num_classes):
    """
    Compute empirical Shannon entropy H(Y) in nats for a client's labels.
    H_i = - sum p(c) * ln(p(c))
    """
    labels = np.asarray(labels).reshape(-1)
    if len(labels) == 0:
        return 0.0
    counts = np.bincount(labels, minlength=num_classes)
    probs = counts / float(len(labels))
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def compute_retention_metrics(i_loc, i_glob, h_i, eps=1e-6):
    """
    Computes the metric tri-factor:
      - Retention Ratio: R = I_glob / (I_loc + eps)
      - Retention Gap: G = I_loc - I_glob
      - Entropy-Normalized Retention Gap: Gn = (I_loc - I_glob) / (h_i + eps)
    """
    r = float(i_glob / (i_loc + eps))
    g = float(i_loc - i_glob)
    gn = float(g / (h_i + eps))
    return {"R": r, "G": g, "Gn": gn}


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    """Compute top-1 classification accuracy on a loader."""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.view(-1).to(device)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return float(100.0 * correct / total) if total > 0 else 0.0


# =====================================================================
# 5. Master RetentionEvaluator Class
# =====================================================================
class SpecializationRetentionEvaluator:
    """
    Per-round evaluator that coordinates latent extraction, fresh critic
    training, MI estimation, metric calculation, and comprehensive logging.
    """
    def __init__(
        self,
        client_val_loaders,
        num_classes,
        device,
        critic_steps=100,
        n_shuffles=5,
        critic_lr=1e-3,
        split_seed=42,
        log_dir=None,
        dataset_name="cifar10",
        method_name="mi_iaa"
    ):
        self.client_val_loaders = client_val_loaders
        self.num_classes = num_classes
        self.device = device
        self.critic_steps = critic_steps
        self.n_shuffles = n_shuffles
        self.critic_lr = critic_lr
        self.dataset_name = dataset_name
        self.method_name = method_name

        self.log_dir = log_dir or "."
        os.makedirs(self.log_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.json_log_path = os.path.join(
            self.log_dir, f"specialization_retention_{method_name}_{dataset_name}_{timestamp}.json"
        )
        self.csv_log_path = os.path.join(
            self.log_dir, f"specialization_retention_{method_name}_{dataset_name}_{timestamp}.csv"
        )

        # Pre-split each client's validation data into 50% critic-fit and 50% report
        # The split is fixed across all rounds and methods.
        self.client_splits = []
        self.client_entropies = []
        self._prepare_data_splits(split_seed)
        self._init_csv_log()

        self.history = []

    def _prepare_data_splits(self, seed):
        """Split held-out D_i^val into fit and report halves deterministically."""
        rng = np.random.RandomState(seed)
        for i, loader in enumerate(self.client_val_loaders):
            dataset = loader.dataset
            indices = list(range(len(dataset)))
            rng.shuffle(indices)

            half = len(indices) // 2
            fit_idx = indices[:half]
            rep_idx = indices[half:]

            # Extract all labels to calculate client entropy H_i
            all_labels = []
            for idx in indices:
                item = dataset[idx]
                label = item[1]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                all_labels.append(label)

            h_i = compute_label_entropy(all_labels, self.num_classes)
            self.client_entropies.append(h_i)

            fit_loader = DataLoader(
                Subset(dataset, fit_idx),
                batch_size=loader.batch_size,
                shuffle=False
            )
            rep_loader = DataLoader(
                Subset(dataset, rep_idx),
                batch_size=loader.batch_size,
                shuffle=False
            )
            self.client_splits.append((fit_loader, rep_loader))

    def _init_csv_log(self):
        """Initialize CSV log file with header."""
        header = [
            "round", "client_id", "method", "dataset", "n_samples", "label_entropy",
            "I_loc", "I_glob", "R", "G", "Gn", "shuffle_std",
            "critic_loss_loc", "critic_loss_glob",
            "local_acc", "global_acc", "agg_weight"
        ]
        with open(self.csv_log_path, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")

    def evaluate_round(
        self,
        round_idx,
        local_models_or_weights,
        global_model,
        local_sample_counts,
        agg_weights,
        local_accuracies,
        model_builder_fn
    ):
        """
        Execute measurement protocol for round t:
          1. Extract latents for local models w_i^t and global model w_g^t.
          2. Fit separate fresh critics for w_i^t and w_g^t.
          3. Estimate MI on report half.
          4. Compute R, G, Gn, summaries.
          5. Log to JSON, CSV, and console.
        """
        start_time = time.time()
        num_clients = len(self.client_val_loaders)
        round_client_records = []

        R_list = []
        Gn_list = []
        weights_norm = np.array(local_sample_counts) / max(1.0, sum(local_sample_counts))

        print(f"\n   [Specialization Retention Evaluation - Round {round_idx}]")

        for i in range(num_clients):
            fit_loader, rep_loader = self.client_splits[i]
            h_i = self.client_entropies[i]
            n_i = local_sample_counts[i]
            agg_w = agg_weights[i] if i < len(agg_weights) else (1.0 / num_clients)
            loc_acc = local_accuracies[i] if i < len(local_accuracies) else 0.0

            # Reconstruct local model if given state_dict
            if isinstance(local_models_or_weights[i], dict):
                loc_model = model_builder_fn().to(self.device)
                loc_model.load_state_dict(local_models_or_weights[i])
            else:
                loc_model = local_models_or_weights[i]

            # 1. Extract latents on fit and report halves
            z_loc_fit, y_loc_fit = extract_latents(loc_model, fit_loader, self.device)
            z_loc_rep, y_loc_rep = extract_latents(loc_model, rep_loader, self.device)

            z_glob_fit, y_glob_fit = extract_latents(global_model, fit_loader, self.device)
            z_glob_rep, y_glob_rep = extract_latents(global_model, rep_loader, self.device)

            # Global model accuracy on client i's validation data
            glob_acc = evaluate_accuracy(global_model, self.client_val_loaders[i], self.device)

            # Convert labels to one-hot
            y_loc_fit_oh = F.one_hot(y_loc_fit, num_classes=self.num_classes).float()
            y_loc_rep_oh = F.one_hot(y_loc_rep, num_classes=self.num_classes).float()

            y_glob_fit_oh = F.one_hot(y_glob_fit, num_classes=self.num_classes).float()
            y_glob_rep_oh = F.one_hot(y_glob_rep, num_classes=self.num_classes).float()

            # 2. Train fresh critic for local model w_i^t
            critic_loc, loss_loc = train_fresh_mine(
                z_loc_fit, y_loc_fit_oh, self.num_classes,
                steps=self.critic_steps, lr=self.critic_lr, device=self.device
            )
            i_loc, shuffle_std_loc, per_feat_loc = estimate_mi(
                critic_loc, z_loc_rep, y_loc_rep_oh,
                n_shuffles=self.n_shuffles, device=self.device
            )

            # 3. Train separate fresh critic for global model w_g^t
            critic_glob, loss_glob = train_fresh_mine(
                z_glob_fit, y_glob_fit_oh, self.num_classes,
                steps=self.critic_steps, lr=self.critic_lr, device=self.device
            )
            i_glob, shuffle_std_glob, per_feat_glob = estimate_mi(
                critic_glob, z_glob_rep, y_glob_rep_oh,
                n_shuffles=self.n_shuffles, device=self.device
            )

            # 4. Compute Retention Metrics
            metrics = compute_retention_metrics(i_loc, i_glob, h_i)
            r_val = metrics["R"]
            g_val = metrics["G"]
            gn_val = metrics["Gn"]

            R_list.append(r_val)
            Gn_list.append(gn_val)

            # Log record per client
            client_record = {
                "round": round_idx,
                "client_id": i + 1,
                "method": self.method_name,
                "dataset": self.dataset_name,
                "n_samples": n_i,
                "label_entropy": round(h_i, 4),
                "I_loc": round(i_loc, 4),
                "I_glob": round(i_glob, 4),
                "R": round(r_val, 4),
                "G": round(g_val, 4),
                "Gn": round(gn_val, 4),
                "shuffle_std": round(shuffle_std_loc, 4),
                "critic_loss_loc": round(loss_loc, 4),
                "critic_loss_glob": round(loss_glob, 4),
                "local_acc": round(loc_acc, 2),
                "global_acc": round(glob_acc, 2),
                "agg_weight": round(float(agg_w), 4),
                "per_feat_loc": [round(v, 4) for v in per_feat_loc],
                "per_feat_glob": [round(v, 4) for v in per_feat_glob]
            }
            round_client_records.append(client_record)

            # Append to CSV
            with open(self.csv_log_path, "a", encoding="utf-8") as f:
                csv_row = [
                    str(round_idx), str(i + 1), self.method_name, self.dataset_name,
                    str(n_i), f"{h_i:.4f}", f"{i_loc:.4f}", f"{i_glob:.4f}",
                    f"{r_val:.4f}", f"{g_val:.4f}", f"{gn_val:.4f}", f"{shuffle_std_loc:.4f}",
                    f"{loss_loc:.4f}", f"{loss_glob:.4f}",
                    f"{loc_acc:.2f}", f"{glob_acc:.2f}", f"{float(agg_w):.4f}"
                ]
                f.write(",".join(csv_row) + "\n")

            # Clean memory
            del critic_loc, critic_glob

        # 5. Compute Round Summaries (Section 5 of specification)
        R_arr = np.array(R_list)
        Gn_arr = np.array(Gn_list)
        weighted_mean_retention = float(np.sum(weights_norm * R_arr))
        unweighted_mean_retention = float(np.mean(R_arr))
        worst_client_retention = float(np.min(R_arr))
        dispersion = float(np.std(R_arr))
        mean_normalized_gap = float(np.mean(Gn_arr))
        elapsed_sec = float(time.time() - start_time)

        round_summary = {
            "round": round_idx,
            "weighted_mean_retention": round(weighted_mean_retention, 4),
            "unweighted_mean_retention": round(unweighted_mean_retention, 4),
            "worst_client_retention": round(worst_client_retention, 4),
            "dispersion": round(dispersion, 4),
            "mean_normalized_gap": round(mean_normalized_gap, 4),
            "measurement_time_sec": round(elapsed_sec, 2),
            "clients": round_client_records
        }

        self.history.append(round_summary)

        # Update JSON history
        with open(self.json_log_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        print(f"   Weighted Retention : {weighted_mean_retention:.4f} | Unweighted: {unweighted_mean_retention:.4f}")
        print(f"   Worst-Client Ret.  : {worst_client_retention:.4f} | Dispersion: {dispersion:.4f}")
        print(f"   Mean Norm. Gap     : {mean_normalized_gap:.4f} | Eval Time: {elapsed_sec:.1f}s")

        return round_summary
