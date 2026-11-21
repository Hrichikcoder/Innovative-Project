
import os
import sys
import math
import csv
import copy
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import data_pipeline


# ==========================================
# 1. Logging
# ==========================================

class Tee:
    """Print output to both terminal and log file."""

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
# 2. Hyperparameters
# ==========================================

NUM_CLIENTS = 30

GLOBAL_ROUNDS = 50
LOCAL_EPOCHS = 5

BATCH_SIZE = 32

LEARNING_RATE = 0.01
MOMENTUM = 0.5

DIRICHLET_ALPHA = 0.1
MIN_SAMPLES_PER_CLIENT = 250

# MINE parameters
LAMBDA_MI = 0.01
LAMBDA_REG = 0.01

MINE_LR = 0.001
MI_TEMPERATURE = 5.0
MI_NORM_EPSILON = 1e-8

VALIDATION_FRACTION = 0.10
RANDOM_SEED = 42

CHECKPOINT_ROUNDS = [1, 10, 20, 30, 40, 50]


# ==========================================
# 3. Device Detection
# ==========================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")
    DEVICE_NAME = torch.cuda.get_device_name(0)

elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):

    DEVICE = torch.device("mps")
    DEVICE_NAME = "Apple MPS"

else:

    DEVICE = torch.device("cpu")
    DEVICE_NAME = "CPU"


# ==========================================
# 4. CNN Model
# ==========================================

class SimpleCNN(nn.Module):

    def __init__(self, num_classes=10):

        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=3, padding=1
        )

        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, padding=1
        )

        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            64 * 8 * 8, 128
        )

        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(
            128, num_classes
        )

    def forward(self, x):

        x = self.pool1(
            self.relu1(self.conv1(x))
        )

        x = self.pool2(
            self.relu2(self.conv2(x))
        )

        x = self.flatten(x)

        z = self.relu3(self.fc1(x))

        out = self.fc2(z)

        return out, z


# ==========================================
# 5. Per-Feature MINE Network
# ==========================================

class PerFeatureMINENetwork(nn.Module):
    """
    Independent MINE critic for every latent feature.

    Each critic receives only:
        One latent feature z_k
        One-hot class label y

    Output:
        One MI score for each latent feature.
    """

    def __init__(
        self,
        z_dim=128,
        y_dim=10,
        hidden=16
    ):

        super(PerFeatureMINENetwork, self).__init__()

        self.z_dim = z_dim
        self.y_dim = y_dim
        self.hidden = hidden

        self.W1 = nn.Parameter(
            torch.randn(
                z_dim,
                1 + y_dim,
                hidden
            ) * 0.1
        )

        self.b1 = nn.Parameter(
            torch.zeros(z_dim, hidden)
        )

        self.W2 = nn.Parameter(
            torch.randn(
                z_dim,
                hidden,
                1
            ) * 0.1
        )

        self.b2 = nn.Parameter(
            torch.zeros(z_dim, 1)
        )

    def forward(self, z, y):

        batch_size = z.shape[0]

        y_exp = y.unsqueeze(1).expand(
            batch_size,
            self.z_dim,
            self.y_dim
        )

        z_exp = z.unsqueeze(-1)

        inp = torch.cat(
            [z_exp, y_exp],
            dim=-1
        )

        inp = inp.permute(1, 0, 2)

        h = torch.relu(
            torch.bmm(inp, self.W1)
            + self.b1.unsqueeze(1)
        )

        out = (
            torch.bmm(h, self.W2)
            + self.b2.unsqueeze(1)
        )

        return out.squeeze(-1).permute(1, 0)


# ==========================================
# 6. Client Local Training
# ==========================================

class ClientUpdate:

    def __init__(
        self,
        dataloader,
        device,
        epochs,
        num_classes
    ):

        self.dataloader = dataloader
        self.device = device
        self.epochs = epochs
        self.num_classes = num_classes

        self.ce_criterion = nn.CrossEntropyLoss()

    def train(self, global_model):

        if len(self.dataloader.dataset) == 0:

            raise ValueError(
                "Client training dataloader contains no samples."
            )

        model = copy.deepcopy(global_model).to(self.device)

        mine_model = PerFeatureMINENetwork(
            z_dim=128,
            y_dim=self.num_classes
        ).to(self.device)

        model.train()
        mine_model.train()

        global_weights = copy.deepcopy(
            global_model.state_dict()
        )

        initial_weights = copy.deepcopy(
            model.state_dict()
        )

        optimizer_clf = optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=MOMENTUM
        )

        optimizer_mine = optim.Adam(
            mine_model.parameters(),
            lr=MINE_LR
        )

        epoch_loss = []
        epoch_mi = []
        epoch_prox_loss = []

        # Store the 128 individual MI estimates.
        all_feature_mi = []

        running_mean_marginal = None

        EMA_DECAY = 0.99

        # ------------------------------
        # Local training epochs
        # ------------------------------

        for epoch in range(self.epochs):

            batch_loss = []
            batch_mi = []
            batch_prox = []

            for images, labels in self.dataloader:

                images = images.to(self.device)

                labels = (
                    labels.view(-1)
                    .long()
                    .to(self.device)
                )

                # CNN forward pass
                logits, z = model(images)

                y_one_hot = F.one_hot(
                    labels,
                    num_classes=self.num_classes
                ).float()

                shuffle_idx = torch.randperm(
                    y_one_hot.shape[0],
                    device=y_one_hot.device
                )

                y_shuffled = y_one_hot[shuffle_idx]

                # ------------------------------
                # MINE critic update
                # ------------------------------

                z_det = z.detach()

                t_joint_c = mine_model(
                    z_det,
                    y_one_hot
                )

                t_marginal_c = mine_model(
                    z_det,
                    y_shuffled
                )

                mean_exp_marginal = torch.mean(
                    torch.exp(t_marginal_c),
                    dim=0
                )

                if running_mean_marginal is None:

                    running_mean_marginal = (
                        mean_exp_marginal.detach()
                    )

                else:

                    running_mean_marginal = (
                        EMA_DECAY * running_mean_marginal
                        + (1 - EMA_DECAY)
                        * mean_exp_marginal.detach()
                    )

                debiased_marginal = (
                    mean_exp_marginal
                    / (
                        running_mean_marginal
                        + MI_NORM_EPSILON
                    )
                )

                mine_loss = -torch.mean(
                    torch.mean(t_joint_c, dim=0)
                    - debiased_marginal
                )

                optimizer_mine.zero_grad()

                mine_loss.backward()

                optimizer_mine.step()

                # ------------------------------
                # Classifier MI objective
                # ------------------------------

                for p in mine_model.parameters():
                    p.requires_grad_(False)

                t_joint = mine_model(
                    z,
                    y_one_hot
                )

                t_marginal = mine_model(
                    z,
                    y_shuffled
                )

                log_mean_exp = (
                    torch.logsumexp(
                        t_marginal,
                        dim=0
                    )
                    - math.log(t_marginal.shape[0])
                )

                feature_mi_estimates = (
                    torch.mean(t_joint, dim=0)
                    - log_mean_exp
                )

                # Save all 128 feature-wise estimates.
                all_feature_mi.append(
                    feature_mi_estimates.detach().cpu()
                )

                avg_mi_estimate = torch.mean(
                    feature_mi_estimates
                )

                for p in mine_model.parameters():
                    p.requires_grad_(True)

                # ------------------------------
                # FedProx regularization
                # ------------------------------

                proximal_term = 0.0

                for name, param in model.named_parameters():

                    proximal_term += torch.sum(
                        (
                            param
                            - global_weights[name].to(self.device)
                        ) ** 2
                    )

                # ------------------------------
                # Classifier objective
                # ------------------------------

                loss_task = self.ce_criterion(
                    logits,
                    labels
                )

                loss_clf = (
                    loss_task
                    - LAMBDA_MI * avg_mi_estimate
                    + (LAMBDA_REG / 2.0) * proximal_term
                )

                optimizer_clf.zero_grad()

                loss_clf.backward()

                optimizer_clf.step()

                batch_loss.append(loss_clf.item())
                batch_mi.append(avg_mi_estimate.item())
                batch_prox.append(proximal_term.item())

            if not batch_loss:

                raise RuntimeError(
                    f"No training batches in epoch {epoch + 1}."
                )

            epoch_loss.append(
                np.mean(batch_loss)
            )

            epoch_mi.append(
                np.mean(batch_mi)
            )

            epoch_prox_loss.append(
                np.mean(batch_prox)
            )

        # ------------------------------
        # Final client model
        # ------------------------------

        final_weights = copy.deepcopy(
            model.state_dict()
        )

        # Average feature-wise MI estimates.
        mean_feature_mi = torch.stack(
            all_feature_mi,
            dim=0
        ).mean(dim=0)

        # Save trained MINE critic parameters.
        mine_state = {
            key: value.detach().cpu().clone()
            for key, value in mine_model.state_dict().items()
        }

        # Layer-wise parameter update magnitudes.
        importance_scores = {}

        for key in final_weights.keys():

            update_magnitude = torch.abs(
                final_weights[key]
                - initial_weights[key].to(self.device)
            ).mean().item()

            importance_scores[key] = update_magnitude

        num_samples = len(
            self.dataloader.dataset
        )

        return (
            final_weights,
            float(np.mean(epoch_loss)),
            float(np.mean(epoch_mi)),
            float(np.mean(epoch_prox_loss)),
            importance_scores,
            num_samples,
            mean_feature_mi,
            mine_state
        )


# ==========================================
# 7. Model Evaluation
# ==========================================

def evaluate(
    model_state,
    dataloader,
    device,
    num_classes
):

    model = SimpleCNN(
        num_classes=num_classes
    )

    model.load_state_dict(model_state)

    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)

            labels = (
                labels.view(-1)
                .long()
                .to(device)
            )

            outputs, _ = model(images)

            predicted = torch.argmax(
                outputs,
                dim=1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    return (
        100.0 * correct / total
        if total > 0
        else 0.0
    )


# ==========================================
# 8. MI-IAA Aggregation
# ==========================================

def average_weights_mi_iaa(
    local_weights,
    mi_scores,
    importance_dicts,
    num_samples
):

    if not local_weights:

        raise ValueError(
            "local_weights cannot be empty."
        )

    if not (
        len(local_weights)
        == len(mi_scores)
        == len(importance_dicts)
        == len(num_samples)
    ):

        raise ValueError(
            "All client input lists must have equal lengths."
        )

    total_samples = sum(num_samples)

    if total_samples <= 0:

        raise ValueError(
            "Total client samples must be positive."
        )

    # Existing MI normalization.
    mi_tensor = torch.tensor(
        mi_scores,
        dtype=torch.float32
    )

    normalized_mi = F.softmax(
        mi_tensor / MI_TEMPERATURE,
        dim=0
    ).tolist()

    sample_fractions = [
        n / total_samples
        for n in num_samples
    ]

    num_clients = len(local_weights)

    w_avg = copy.deepcopy(
        local_weights[0]
    )

    layer_alpha_logs = {}

    for key in w_avg.keys():

        layer_scores = []

        for i in range(num_clients):

            imp = importance_dicts[i][key]

            layer_scores.append(
                normalized_mi[i]
                * imp
                * sample_fractions[i]
            )

        scores_tensor = torch.tensor(
            layer_scores,
            dtype=torch.float32
        )

        if scores_tensor.std() > 1e-8:

            scores_tensor = (
                scores_tensor - scores_tensor.mean()
            ) / scores_tensor.std()

        alphas = F.softmax(
            scores_tensor,
            dim=0
        ).tolist()

        layer_alpha_logs[key] = alphas

        # Existing weighted aggregation.
        w_avg[key] = (
            local_weights[0][key] * alphas[0]
        )

        for i in range(1, num_clients):

            w_avg[key] += (
                local_weights[i][key] * alphas[i]
            )

    avg_alphas = [
        sum(
            layer_alpha_logs[key][i]
            for key in w_avg.keys()
        ) / len(w_avg.keys())
        for i in range(num_clients)
    ]

    return (
        w_avg,
        normalized_mi,
        [a * 100 for a in avg_alphas],
        layer_alpha_logs
    )


# ==========================================
# 9. Client Contribution Calculation
# ==========================================

def calculate_client_contributions(
    local_weights,
    mi_scores,
    importance_dicts,
    num_samples
):

    _, _, _, layer_alpha_logs = average_weights_mi_iaa(
        local_weights,
        mi_scores,
        importance_dicts,
        num_samples
    )

    if not layer_alpha_logs:

        raise ValueError(
            "No model parameter layers available."
        )

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


# ==========================================
# 10. Save Client Contribution Report
# ==========================================

def save_client_contributions_md(
    filepath,
    dataset_name,
    round_contributions,
    round_mi_scores,
    round_sample_counts
):

    os.makedirs(
        os.path.dirname(filepath) or ".",
        exist_ok=True
    )

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# MI-IAA Per-Client Contribution Report\n\n"
        )

        f.write(
            f"- **Dataset:** {dataset_name}\n"
        )

        f.write(
            "- **Contribution definition:** Mean of final "
            "layer-wise MI-IAA aggregation weights.\n"
        )

        f.write(
            "- **Interpretation:** Aggregation-weight "
            "attribution, not a causal performance measure.\n\n"
        )

        for round_idx, contributions in enumerate(
            round_contributions,
            start=1
        ):

            f.write(
                f"## Round {round_idx}\n\n"
            )

            f.write(
                "| Client | Contribution | MI Score | Samples |\n"
            )

            f.write(
                "|---:|---:|---:|---:|\n"
            )

            mi_scores = round_mi_scores[
                round_idx - 1
            ]

            samples = round_sample_counts[
                round_idx - 1
            ]

            for client_idx, contribution in enumerate(
                contributions
            ):

                f.write(
                    f"| {client_idx + 1} | "
                    f"{contribution:.4f}% | "
                    f"{mi_scores[client_idx]:.6f} | "
                    f"{samples[client_idx]} |\n"
                )

            f.write("\n")


# ==========================================
# 11. Save Feature-wise MI CSV
# ==========================================

def save_feature_mi_csv(
    filepath,
    round_idx,
    client_feature_mis
):

    with open(
        filepath,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        for client_idx, feature_vector in enumerate(
            client_feature_mis
        ):

            for feature_idx, mi_value in enumerate(
                feature_vector
            ):

                writer.writerow([
                    round_idx,
                    client_idx + 1,
                    feature_idx + 1,
                    float(mi_value)
                ])


# ==========================================
# 12. Save Ablation Checkpoints
# ==========================================

def save_ablation_checkpoints(
    checkpoint_dir,
    round_idx,
    global_model,
    local_weights,
    local_mine_states
):

    round_dir = os.path.join(
        checkpoint_dir,
        f"round_{round_idx}"
    )

    os.makedirs(
        round_dir,
        exist_ok=True
    )

    # Global model after aggregation.
    torch.save(
        global_model.state_dict(),
        os.path.join(
            round_dir,
            "global_model.pt"
        )
    )

    # Save local client models and their matching MINE critics.
    for client_idx in range(len(local_weights)):

        torch.save(
            local_weights[client_idx],
            os.path.join(
                round_dir,
                f"client_{client_idx + 1}_model.pt"
            )
        )

        torch.save(
            local_mine_states[client_idx],
            os.path.join(
                round_dir,
                f"client_{client_idx + 1}_mine.pt"
            )
        )

    print(
        f"Saved global model, client models and MINE critics "
        f"for round {round_idx}."
    )


# ==========================================
# 13. Main Federated Simulation
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="MI-IAA Federated Learning with Feature-wise MINE"
    )

    parser.add_argument(
        '--dataset',
        required=True,
        choices=[
            'cifar10',
            'mnist',
            'pathmnist',
            'bloodmnist'
        ]
    )

    parser.add_argument(
        '--rounds',
        type=int,
        default=GLOBAL_ROUNDS
    )

    parser.add_argument(
        '--local_epochs',
        type=int,
        default=LOCAL_EPOCHS
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED
    )

    args = parser.parse_args()

    if args.rounds < 1 or args.local_epochs < 1:

        raise ValueError(
            "Rounds and local epochs must be at least 1."
        )

    dataset_name = args.dataset.lower()

    data_pipeline.set_seed(args.seed)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    # ======================================
    # Output directories
    # ======================================

    experiment_dir = (
        f"mine_ablation_{dataset_name}_{timestamp}"
    )

    os.makedirs(
        experiment_dir,
        exist_ok=True
    )

    checkpoint_dir = os.path.join(
        experiment_dir,
        "checkpoints"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    log_path = os.path.join(
        experiment_dir,
        "training_log.md"
    )

    feature_mi_csv = os.path.join(
        experiment_dir,
        "feature_mi_estimates.csv"
    )

    # ======================================
    # Logging
    # ======================================

    log_file = open(
        log_path,
        "w",
        encoding="utf-8"
    )

    original_stdout = sys.stdout

    sys.stdout = Tee(
        original_stdout,
        log_file
    )

    try:

        print("==============================================")
        print(" MI-IAA: Mutual-Information Importance-Aware Aggregation")
        print(" Per-Feature MINE + FedProx")
        print("==============================================")

        print(f"Dataset       : {dataset_name}")
        print(f"Device        : {DEVICE_NAME}")
        print(f"Clients       : {NUM_CLIENTS}")
        print(f"Global rounds : {args.rounds}")
        print(f"Local epochs  : {args.local_epochs}")
        print(f"Batch size    : {BATCH_SIZE}")
        print(f"Random seed   : {args.seed}")

        # ======================================
        # Load federated data
        # ======================================

        num_classes = data_pipeline.get_num_classes(
            dataset_name
        )

        (
            train_dataset,
            validation_dataset,
            global_test_dataset,
            partitions
        ) = data_pipeline.get_federated_data(
            dataset_name=dataset_name,
            num_clients=NUM_CLIENTS,
            alpha=DIRICHLET_ALPHA,
            min_samples=MIN_SAMPLES_PER_CLIENT,
            val_fraction=VALIDATION_FRACTION,
            seed=args.seed
        )

        global_validation_loader = DataLoader(
            validation_dataset,
            batch_size=128,
            shuffle=False
        )

        global_test_loader = DataLoader(
            global_test_dataset,
            batch_size=128,
            shuffle=False
        )

        client_train_loaders, client_test_loaders = (
            data_pipeline.build_client_loaders(
                train_dataset,
                partitions,
                BATCH_SIZE
            )
        )

        # ======================================
        # Initialize global model
        # ======================================

        global_model = SimpleCNN(
            num_classes=num_classes
        ).to(DEVICE)

        # ======================================
        # Initialize CSV
        # ======================================

        with open(
            feature_mi_csv,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "round",
                "client",
                "feature",
                "mi_estimate"
            ])

        # Include the last round even when using a custom
        # number of rounds, such as --rounds 1.
        selected_checkpoint_rounds = (
            set(CHECKPOINT_ROUNDS)
            | {args.rounds}
        )

        # ======================================
        # Training history
        # ======================================

        history = {
            'global_validation_acc': [],
            'avg_client_acc': []
        }

        contribution_history = []
        contribution_mi_history = []
        contribution_sample_history = []

        # ======================================
        # Federated training
        # ======================================

        print("\n==============================================")
        print(" Starting MI-IAA Federated Training")
        print("==============================================")

        for round_idx in range(
            1,
            args.rounds + 1
        ):

            print(
                f"\n| Round : {round_idx}/{args.rounds} |"
            )

            local_weights = []
            local_losses = []
            local_mis = []
            local_importances = []

            local_feature_mis = []
            local_mine_states = []

            client_personalized_accuracies = []
            client_prox_losses = []
            local_sample_counts = []

            # ==================================
            # Client local training
            # ==================================

            for idx in range(NUM_CLIENTS):

                print(
                    f"\nTraining Client {idx + 1}/{NUM_CLIENTS}"
                )

                client = ClientUpdate(
                    dataloader=client_train_loaders[idx],
                    device=DEVICE,
                    epochs=args.local_epochs,
                    num_classes=num_classes
                )

                (
                    weights,
                    loss,
                    mi_estimate,
                    prox_loss,
                    importances,
                    num_samples,
                    feature_mi_vector,
                    mine_state
                ) = client.train(global_model)

                # Evaluate local client model on its
                # own held-out local test partition.
                local_accuracy = evaluate(
                    weights,
                    client_test_loaders[idx],
                    DEVICE,
                    num_classes
                )

                client_personalized_accuracies.append(
                    local_accuracy
                )

                local_weights.append(weights)
                local_losses.append(loss)
                local_mis.append(mi_estimate)
                client_prox_losses.append(prox_loss)
                local_importances.append(importances)
                local_sample_counts.append(num_samples)

                local_feature_mis.append(
                    feature_mi_vector
                )

                local_mine_states.append(
                    mine_state
                )

                print(
                    f"Client {idx + 1} | "
                    f"Loss: {loss:.4f} | "
                    f"MI: {mi_estimate:.4f} | "
                    f"Local Accuracy: {local_accuracy:.2f}%"
                )

            # ==================================
            # Server aggregation
            # ==================================

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

            global_model.load_state_dict(
                global_weights
            )

            # ==================================
            # Save feature-wise MI estimates
            # ==================================

            save_feature_mi_csv(
                feature_mi_csv,
                round_idx,
                local_feature_mis
            )

            # ==================================
            # Save selected checkpoints
            # ==================================

            if round_idx in selected_checkpoint_rounds:

                save_ablation_checkpoints(
                    checkpoint_dir,
                    round_idx,
                    global_model,
                    local_weights,
                    local_mine_states
                )

            # ==================================
            # Client contribution calculation
            # ==================================

            round_contributions = [
                (
                    sum(
                        layer_alpha_logs[key][client_idx]
                        for key in layer_alpha_logs
                    ) / len(layer_alpha_logs)
                ) * 100.0

                for client_idx in range(NUM_CLIENTS)
            ]

            contribution_history.append(
                round_contributions
            )

            contribution_mi_history.append(
                list(local_mis)
            )

            contribution_sample_history.append(
                list(local_sample_counts)
            )

            # ==================================
            # Global validation evaluation
            # ==================================

            global_validation_acc = evaluate(
                global_model.state_dict(),
                global_validation_loader,
                DEVICE,
                num_classes
            )

            avg_client_acc = float(
                np.mean(client_personalized_accuracies)
            )

            avg_prox_loss = float(
                np.mean(client_prox_losses)
            )

            avg_mi = float(
                np.mean(local_mis)
            )

            print("\n[Server Metrics]")

            print(
                f"Global Validation Accuracy : "
                f"{global_validation_acc:.2f}%"
            )

            print(
                f"Average Client Accuracy    : "
                f"{avg_client_acc:.2f}%"
            )

            print(
                f"Average Proximal Penalty   : "
                f"{avg_prox_loss:.4f}"
            )

            print(
                f"Average MI Estimate        : "
                f"{avg_mi:.4f}"
            )

            print("\n[Detailed Client Metrics]")

            print(
                "| Client | Local Acc | Raw MI | Norm MI | "
                "Layer Weight | Prox Drift |"
            )

            print(
                "|---:|---:|---:|---:|---:|---:|"
            )

            for idx in range(NUM_CLIENTS):

                print(
                    f"| {idx + 1} | "
                    f"{client_personalized_accuracies[idx]:.2f}% | "
                    f"{local_mis[idx]:.4f} | "
                    f"{normalized_mi[idx]:.4f} | "
                    f"{avg_layer_weights[idx]:.2f}% | "
                    f"{client_prox_losses[idx]:.4f} |"
                )

            history['global_validation_acc'].append(
                global_validation_acc
            )

            history['avg_client_acc'].append(
                avg_client_acc
            )

        # ======================================
        # Final test evaluation
        # ======================================

        print("\n==============================================")
        print(" Final Evaluation on Untouched Test Dataset")
        print("==============================================")

        final_test_acc = evaluate(
            global_model.state_dict(),
            global_test_loader,
            DEVICE,
            num_classes
        )

        print(
            f"Final Global Test Accuracy: {final_test_acc:.2f}%"
        )

        # Save final global model.
        torch.save(
            global_model.state_dict(),
            os.path.join(
                experiment_dir,
                "final_global_model.pt"
            )
        )

        # ======================================
        # Save final metrics
        # ======================================

        metrics_path = os.path.join(
            experiment_dir,
            "final_metrics.csv"
        )

        with open(
            metrics_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "dataset",
                "rounds",
                "local_epochs",
                "final_validation_accuracy",
                "final_test_accuracy"
            ])

            writer.writerow([
                dataset_name,
                args.rounds,
                args.local_epochs,
                history['global_validation_acc'][-1],
                final_test_acc
            ])

        # ======================================
        # Save contribution report
        # ======================================

        contribution_file = os.path.join(
            experiment_dir,
            "client_contributions.md"
        )

        save_client_contributions_md(
            contribution_file,
            dataset_name,
            contribution_history,
            contribution_mi_history,
            contribution_sample_history
        )

        print(
            f"\nContribution report saved: {contribution_file}"
        )

        print(
            f"Feature MI estimates saved: {feature_mi_csv}"
        )

        # ======================================
        # Plot validation accuracy
        # ======================================

        rounds = list(
            range(1, args.rounds + 1)
        )

        plt.figure(figsize=(10, 5))

        plt.plot(
            rounds,
            history['global_validation_acc'],
            marker='s',
            label='Global Validation Accuracy'
        )

        plt.title(
            f'MI-IAA ({dataset_name}): Validation Accuracy'
        )

        plt.xlabel('Communication Rounds')
        plt.ylabel('Accuracy (%)')

        plt.grid(True)
        plt.legend()

        plt.savefig(
            os.path.join(
                experiment_dir,
                "validation_accuracy.png"
            )
        )

        plt.close()

        # ======================================
        # Plot client accuracy
        # ======================================

        plt.figure(figsize=(10, 5))

        plt.plot(
            rounds,
            history['avg_client_acc'],
            marker='o',
            label='Average Client Accuracy'
        )

        plt.title(
            f'MI-IAA ({dataset_name}): Client Accuracy'
        )

        plt.xlabel('Communication Rounds')
        plt.ylabel('Accuracy (%)')

        plt.grid(True)
        plt.legend()

        plt.savefig(
            os.path.join(
                experiment_dir,
                "client_accuracy.png"
            )
        )

        plt.close()

        # ======================================
        # Combined plot
        # ======================================

        plt.figure(figsize=(10, 6))

        plt.plot(
            rounds,
            history['avg_client_acc'],
            marker='o',
            label='Average Client Accuracy'
        )

        plt.plot(
            rounds,
            history['global_validation_acc'],
            marker='s',
            label='Global Validation Accuracy'
        )

        plt.title(
            f'MI-IAA ({dataset_name}): Accuracy Comparison'
        )

        plt.xlabel('Communication Rounds')
        plt.ylabel('Accuracy (%)')

        plt.grid(True)
        plt.legend()

        plt.savefig(
            os.path.join(
                experiment_dir,
                "combined_accuracy.png"
            )
        )

        plt.close()

        print("\n==============================================")
        print(" MI-IAA Experiment Completed Successfully")
        print("==============================================")

        print(
            f"All experiment files: {experiment_dir}"
        )

    finally:

        sys.stdout = original_stdout

        log_file.close()


# ==========================================
# 14. Program Entry Point
# ==========================================

if __name__ == '__main__':

    main()
