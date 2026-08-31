import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import copy
import os
import math
import sys
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

# Import the centralized data pipeline
import data_pipeline

class Tee:
    """
    Prints everything to both:
      1. Terminal
      2. Log file
    """

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

# ==========================================
# Hyperparameters & Configurations
# ==========================================
NUM_CLIENTS = 30
GLOBAL_ROUNDS = 50
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.5
DIRICHLET_ALPHA = 0.1
MIN_SAMPLES_PER_CLIENT = 250

# MI-IAA Specific Hyperparameters
LAMBDA_MI = 0.01       # Weight of Mutual Information regularizer in local loss
LAMBDA_REG = 0.01      # Weight of FedProx-style proximal penalty
MINE_LR = 0.001        # Learning rate for the MINE network
MI_TEMPERATURE = 5.0
MI_NORM_EPSILON = 1e-8 # Epsilon for Min-Max normalization

# Device Detection Strategy
# Priority: CUDA GPU -> Apple MPS -> CPU fallback.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    device_name = f"CUDA GPU ({torch.cuda.get_device_name(0)})"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    device_name = "Apple MPS"
else:
    DEVICE = torch.device("cpu")
    device_name = "CPU"

# ==========================================
# Model Definitions
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        z = self.relu3(self.fc1(x))  # Z: Latent Features (128-dim)
        out = self.fc2(z)            # Y_hat: Logits
        return out, z

class PerFeatureMINENetwork(nn.Module):
    """
    True per-feature MI critic. Each latent dimension z_k gets its own small
    MLP that sees ONLY (z_k, y) -- not the full latent vector -- so the k-th
    output score is a genuine function of that single feature and the label.
    (The previous version concatenated the *entire* z vector before producing
    128 outputs, so every "per-feature" score actually depended on all 128
    features jointly -- 128 correlated critics of the same input, not a real
    per-dimension decomposition.) Implemented as z_dim independent 2-layer
    MLPs, vectorized with batched matmul (bmm) so it costs about the same as
    a couple of dense layers rather than instantiating z_dim separate modules.
    """
    def __init__(self, z_dim=128, y_dim=10, hidden=16):
        super(PerFeatureMINENetwork, self).__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.hidden = hidden
        # Batched weights: one (1 + y_dim) -> hidden -> 1 MLP per latent feature
        self.W1 = nn.Parameter(torch.randn(z_dim, 1 + y_dim, hidden) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(z_dim, hidden))
        self.W2 = nn.Parameter(torch.randn(z_dim, hidden, 1) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(z_dim, 1))

    def forward(self, z, y):
        # z: [Batch, z_dim], y: [Batch, y_dim]
        batch_size = z.shape[0]
        y_exp = y.unsqueeze(1).expand(batch_size, self.z_dim, self.y_dim)  # [B, z_dim, y_dim]
        z_exp = z.unsqueeze(-1)                                           # [B, z_dim, 1]
        inp = torch.cat([z_exp, y_exp], dim=-1)                           # [B, z_dim, 1+y_dim]
        inp = inp.permute(1, 0, 2)                                        # [z_dim, B, 1+y_dim]
        h = torch.relu(torch.bmm(inp, self.W1) + self.b1.unsqueeze(1))    # [z_dim, B, hidden]
        out = torch.bmm(h, self.W2) + self.b2.unsqueeze(1)                # [z_dim, B, 1]
        return out.squeeze(-1).permute(1, 0)                              # [Batch, z_dim]

# ==========================================
# Local Training (Client)
# ==========================================
class ClientUpdate:
    def __init__(self, dataloader, device, epochs, num_classes):
        self.dataloader = dataloader
        self.device = device
        self.epochs = epochs
        self.num_classes = num_classes
        self.ce_criterion = nn.CrossEntropyLoss()

    def train(self, global_model):
        if len(self.dataloader.dataset) == 0:
            raise ValueError("Client training dataloader contains no samples.")

        model = copy.deepcopy(global_model).to(self.device)
        mine_model = PerFeatureMINENetwork(z_dim=128, y_dim=self.num_classes).to(self.device)
        model.train()
        mine_model.train()

        global_weights = copy.deepcopy(global_model.state_dict())  # Used for FedProx penalty

        optimizer_clf = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        optimizer_mine = optim.Adam(mine_model.parameters(), lr=MINE_LR)

        initial_weights = copy.deepcopy(model.state_dict())
        epoch_loss, epoch_mi, epoch_prox_loss = [], [], []

        # EMA of mean(exp(t_marginal)) per feature -- the standard MINE bias
        # correction. log(mean(exp(.))) estimated from a single minibatch has a
        # known negative bias (Jensen's inequality on a finite sample); using an
        # EMA-based denominator for the *critic's gradient* (while still reporting
        # the plain Donsker-Varadhan bound below for logging) is the fix used in
        # the original MINE paper/reference implementations.
        running_mean_marginal = None
        EMA_DECAY = 0.99

        for epoch in range(self.epochs):
            batch_loss, batch_mi, batch_prox = [], [], []
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.view(-1).long().to(self.device)

                # 1. Single forward pass through the CNN, shared by both objectives below
                logits, z = model(images)
                y_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
                shuffle_idx = torch.randperm(
                    y_one_hot.shape[0], device=y_one_hot.device
                )
                y_shuffled = y_one_hot[shuffle_idx]

                # 2a. --- Update the MINE critic ---
                # z is DETACHED here: the critic's own loss must never backprop into
                # the CNN. Previously loss_mine.backward() (with a live z) and
                # loss_clf.backward() both pushed gradient into `model` through the
                # same graph, so the CNN's features ended up shaped by an effective
                # MI-gradient weight of ~(1 + LAMBDA_MI) instead of the intended
                # LAMBDA_MI -- roughly 100x more than the hyperparameter implies.
                z_det = z.detach()
                t_joint_c = mine_model(z_det, y_one_hot)
                t_marginal_c = mine_model(z_det, y_shuffled)

                mean_exp_marginal = torch.mean(torch.exp(t_marginal_c), dim=0)  # [z_dim]
                if running_mean_marginal is None:
                    running_mean_marginal = mean_exp_marginal.detach()
                else:
                    running_mean_marginal = (
                        EMA_DECAY * running_mean_marginal
                        + (1 - EMA_DECAY) * mean_exp_marginal.detach()
                    )
                # De-biased marginal term used only for the critic's own gradient
                debiased_marginal = mean_exp_marginal / (running_mean_marginal + MI_NORM_EPSILON)
                mine_loss = -torch.mean(torch.mean(t_joint_c, dim=0) - debiased_marginal)

                optimizer_mine.zero_grad()
                mine_loss.backward()
                optimizer_mine.step()

                # 2b. --- MI term used inside the classifier's objective ---
                # mine_model is frozen (requires_grad_(False)) so loss_clf.backward()
                # only ever updates `model`'s parameters via the path through z; it
                # can no longer also update mine_model's parameters as it did before.
                for p in mine_model.parameters():
                    p.requires_grad_(False)

                t_joint = mine_model(z, y_one_hot)
                t_marginal = mine_model(z, y_shuffled)
                # Numerically stable log(mean(exp(.))) via logsumexp. The previous
                # version computed torch.log(torch.mean(torch.exp(.))) directly,
                # which overflows to inf/nan once critic outputs grow, since exp()
                # is applied before any normalization.
                log_mean_exp = torch.logsumexp(t_marginal, dim=0) - math.log(t_marginal.shape[0])
                feature_mi_estimates = torch.mean(t_joint, dim=0) - log_mean_exp  # [z_dim]
                avg_mi_estimate = torch.mean(feature_mi_estimates)  # Scalar average of all 128 features

                for p in mine_model.parameters():
                    p.requires_grad_(True)

                # 3. Proximal / Regularization Penalty (FedProx style)
                proximal_term = 0.0
                for name, param in model.named_parameters():
                    proximal_term += torch.sum((param - global_weights[name].to(self.device)) ** 2)

                # 4. Total Local Objective (classifier side -- critic already updated in 2a)
                loss_task = self.ce_criterion(logits, labels)
                loss_clf = loss_task - (LAMBDA_MI * avg_mi_estimate) + (LAMBDA_REG / 2.0) * proximal_term

                optimizer_clf.zero_grad()
                loss_clf.backward()
                optimizer_clf.step()

                batch_loss.append(loss_clf.item())
                batch_mi.append(avg_mi_estimate.item())
                batch_prox.append(proximal_term.item())

            if not batch_loss:
                raise RuntimeError(
                    f"No training batches were processed in epoch {epoch + 1}."
                )

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_mi.append(sum(batch_mi) / len(batch_mi))
            epoch_prox_loss.append(sum(batch_prox) / len(batch_prox))

        final_weights = model.state_dict()

        # Calculate Layer-Wise Parameter Importance (Feature-Traceable Proxy via Update Magnitude)
        importance_scores = {}
        for key in final_weights.keys():
            update_magnitude = torch.abs(final_weights[key] - initial_weights[key].to(self.device)).mean().item()
            importance_scores[key] = update_magnitude

        num_samples = len(self.dataloader.dataset)

        return (final_weights, sum(epoch_loss)/len(epoch_loss), sum(epoch_mi)/len(epoch_mi),
                sum(epoch_prox_loss)/len(epoch_prox_loss), importance_scores, num_samples)

# ==========================================
# Evaluation
# ==========================================
def evaluate(model_state, dataloader, device, num_classes):
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(model_state)
    model.eval()
    model.to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.view(-1).long().to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total if total > 0 else 0.0

# ==========================================
# Server Aggregation: MI-IAA (Per-Layer Softmax)
# ==========================================
def average_weights_mi_iaa(local_weights, mi_scores, importance_dicts, num_samples):
    if not local_weights:
        raise ValueError("local_weights cannot be empty.")
    if not (
        len(local_weights)
        == len(mi_scores)
        == len(importance_dicts)
        == len(num_samples)
    ):
        raise ValueError(
            "local_weights, mi_scores, importance_dicts, and num_samples "
            "must have the same length."
        )

    total_samples = sum(num_samples)
    if total_samples <= 0:
        raise ValueError("Total client samples must be greater than zero.")

    # Existing MI-IAA normalization: temperature-scaled softmax.
    mi_tensor = torch.tensor(mi_scores, dtype=torch.float32)
    normalized_mi = F.softmax(
        mi_tensor / MI_TEMPERATURE, dim=0
    ).tolist()

    sample_fractions = [n / total_samples for n in num_samples]
    num_clients = len(local_weights)
    w_avg = copy.deepcopy(local_weights[0])
    layer_alpha_logs = {}

    for key in w_avg.keys():
        layer_scores = []
        for i in range(num_clients):
            imp = importance_dicts[i][key]
            layer_scores.append(
                normalized_mi[i] * imp * sample_fractions[i]
            )

        scores_tensor = torch.tensor(
            layer_scores, dtype=torch.float32
        )

        # Existing standardization and final layer-wise softmax.
        if scores_tensor.std() > 1e-8:
            scores_tensor = (
                scores_tensor - scores_tensor.mean()
            ) / scores_tensor.std()

        alphas = F.softmax(scores_tensor, dim=0).tolist()
        layer_alpha_logs[key] = alphas

        # Existing weighted aggregation.
        w_avg[key] = local_weights[0][key] * alphas[0]
        for i in range(1, num_clients):
            w_avg[key] += local_weights[i][key] * alphas[i]

    avg_alphas = [
        sum(layer_alpha_logs[key][i] for key in w_avg.keys())
        / len(w_avg.keys())
        for i in range(num_clients)
    ]

    return (
        w_avg,
        normalized_mi,
        [a * 100 for a in avg_alphas],
        layer_alpha_logs
    )


def calculate_client_contributions(
    local_weights,
    mi_scores,
    importance_dicts,
    num_samples
):
    """
    Calculate per-client MI-IAA aggregation contribution.

    Contribution is the mean of the actual final layer-wise softmax
    aggregation weights across all model parameter tensors.
    """
    _, _, _, layer_alpha_logs = average_weights_mi_iaa(
        local_weights,
        mi_scores,
        importance_dicts,
        num_samples
    )

    if not layer_alpha_logs:
        raise ValueError("No model parameter layers were available.")

    num_clients = len(local_weights)
    num_layers = len(layer_alpha_logs)

    contributions = [
        (
            sum(
                layer_alpha_logs[key][client_idx]
                for key in layer_alpha_logs
            ) / num_layers
        ) * 100.0
        for client_idx in range(num_clients)
    ]

    return contributions, layer_alpha_logs


def save_client_contributions_md(
    filepath,
    dataset_name,
    round_contributions,
    round_mi_scores,
    round_sample_counts
):
    """Save per-client MI-IAA contribution for every round in Markdown."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# MI-IAA Per-Client Contribution Report\n\n")
        f.write(f"- **Dataset:** {dataset_name}\n")
        f.write("- **Contribution definition:** Mean of the actual final "
                "layer-wise MI-IAA aggregation weights across parameter "
                "tensors.\n")
        f.write("- **Interpretation:** Aggregation-weight attribution; "
                "not a causal measure of performance influence.\n\n")

        for round_idx, contributions in enumerate(
            round_contributions, start=1
        ):
            f.write(f"## Round {round_idx}\n\n")
            f.write("| Client | Contribution | MI Score | Samples |\n")
            f.write("|---:|---:|---:|---:|\n")

            mi_scores = round_mi_scores[round_idx - 1]
            samples = round_sample_counts[round_idx - 1]

            for client_idx, contribution in enumerate(contributions):
                f.write(
                    f"| {client_idx + 1} | {contribution:.4f}% | "
                    f"{mi_scores[client_idx]:.6f} | "
                    f"{samples[client_idx]} |\n"
                )
            f.write("\n")


# ==========================================
# Main Simulation
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="MI-IAA Federated Learning")
    parser.add_argument('--dataset', required=True,
                         choices=['cifar10', 'mnist', 'pathmnist', 'bloodmnist'],
                         help="Dataset to train on.")
    args = parser.parse_args()
    dataset_name = args.dataset.lower()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = open(
        f"training_log_mine_{dataset_name}_{timestamp}.md",
        "w",
        encoding="utf-8"
    )

    sys.stdout = Tee(sys.__stdout__, log_file)
    print("==============================================")
    print(f" MI-IAA: Mutual-Information Importance-Aware Aggregation")
    print(f" (Includes Per-Feature MINE & Proximal Penalty)")
    print("==============================================")
    print(f"Dataset         : {dataset_name}")
    print(f"Compute Device   : {device_name}")

    data_pipeline.set_seed()
    num_classes = data_pipeline.get_num_classes(dataset_name)

    # Using centralized data pipeline
    train_dataset, global_test_dataset = data_pipeline.get_dataset(dataset_name)
    global_test_loader = DataLoader(global_test_dataset, batch_size=128, shuffle=False)

    partitions = data_pipeline.ensure_partitions(
        dataset_name=dataset_name,
        num_clients=NUM_CLIENTS,
        alpha=DIRICHLET_ALPHA,
        min_samples=MIN_SAMPLES_PER_CLIENT
    )

    client_train_loaders, client_test_loaders = data_pipeline.build_client_loaders(
        train_dataset, partitions, BATCH_SIZE
    )

    global_model = SimpleCNN(num_classes=num_classes).to(DEVICE)

    history = {'global_acc': [], 'avg_client_acc': []}
    contribution_history = []
    contribution_mi_history = []
    contribution_sample_history = []

    print("\n==============================================")
    print(f" Starting MI-IAA Training ({GLOBAL_ROUNDS} Rounds)")
    print("==============================================")

    for round_idx in range(1, GLOBAL_ROUNDS + 1):
        local_weights, local_losses, local_mis, local_importances = [], [], [], []
        client_personalized_accuracies, client_prox_losses, local_sample_counts = [], [], []

        print(f"\n| Round : {round_idx}/{GLOBAL_ROUNDS} |")

        for idx in range(NUM_CLIENTS):
            client = ClientUpdate(dataloader=client_train_loaders[idx], device=DEVICE,
                                   epochs=LOCAL_EPOCHS, num_classes=num_classes)
            weights, loss, mi_estimate, prox_loss, importances, num_samples = client.train(global_model)

            client_personalized_accuracies.append(evaluate(weights, client_test_loaders[idx], DEVICE, num_classes))
            local_weights.append(weights)
            local_losses.append(loss)
            local_mis.append(mi_estimate)
            client_prox_losses.append(prox_loss)
            local_importances.append(importances)
            local_sample_counts.append(num_samples)

        # Server Aggregation
        (
            global_weights,
            normalized_mi,
            avg_layer_weights,
            layer_alpha_logs
        ) = average_weights_mi_iaa(
            local_weights,
            local_mis,
            local_importances,
            local_sample_counts
        )
        global_model.load_state_dict(global_weights)

        num_layers = len(layer_alpha_logs)
        round_contributions = [
            (
                sum(
                    layer_alpha_logs[key][client_idx]
                    for key in layer_alpha_logs
                ) / num_layers
            ) * 100.0
            for client_idx in range(len(local_weights))
        ]
        contribution_history.append(round_contributions)
        contribution_mi_history.append(list(local_mis))
        contribution_sample_history.append(list(local_sample_counts))

        # Analytics
        global_test_acc = evaluate(global_model.state_dict(), global_test_loader, DEVICE, num_classes)

        print(f"\n   [Server Metrics]")
        print(f"   Global Model Test Accuracy    : {global_test_acc:.2f}%")
        print(f"   Avg Client Personalized Acc   : {np.mean(client_personalized_accuracies):.2f}%")
        print(f"   Avg Proximal Penalty (Drift)  : {np.mean(client_prox_losses):.4f}")
        print(f"   Avg Per-Feature MI Estimate   : {np.mean(local_mis):.4f}")

        print(f"\n   [Detailed Per-Client MI-IAA Metrics]")
        print(f"   | Client | Local Acc % | Raw MI Est. | Norm MI | Avg Layer Weight % | Prox Drift |")
        print(f"   |--------|-------------|-------------|---------|--------------------|------------|")
        for idx in range(NUM_CLIENTS):
            print(f"   | {idx+1:6d} | {client_personalized_accuracies[idx]:11.2f}% | {local_mis[idx]:11.4f} | {normalized_mi[idx]:7.4f} | {avg_layer_weights[idx]:18.2f}% | {client_prox_losses[idx]:10.4f} |")

        history['global_acc'].append(global_test_acc)
        history['avg_client_acc'].append(np.mean(client_personalized_accuracies))

    contribution_file = f"mine_{dataset_name}_client_contributions.md"
    save_client_contributions_md(
        contribution_file,
        dataset_name,
        contribution_history,
        contribution_mi_history,
        contribution_sample_history
    )
    print(f"Per-client contribution report saved to {contribution_file}")

    # --- Plotting Results ---
    print("\n[Simulation Complete] Generating accuracy plots...")
    rounds = list(range(1, GLOBAL_ROUNDS + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, history['avg_client_acc'], marker='o', color='b', label='Avg Client Acc')
    plt.title(f'MI-IAA ({dataset_name}): Communication Rounds vs Avg Client Accuracy')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'mine_{dataset_name}_client_accuracy_plot.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, history['global_acc'], marker='s', color='r', label='Global Model Acc')
    plt.title(f'MI-IAA ({dataset_name}): Communication Rounds vs Global Accuracy')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'mine_{dataset_name}_global_accuracy_plot.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, history['avg_client_acc'], marker='o', color='b', label='Avg Client Acc')
    plt.plot(rounds, history['global_acc'], marker='s', color='r', label='Global Model Acc')
    plt.title(f'MI-IAA ({dataset_name}): Communication Rounds vs Federated Accuracies')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'mine_{dataset_name}_combined_accuracy_plot.png')
    plt.close()

    print("Plots successfully saved as PNG files in the project folder!")

if __name__ == '__main__':
    try:
        main()
    finally:
        sys.stdout = sys.__stdout__
