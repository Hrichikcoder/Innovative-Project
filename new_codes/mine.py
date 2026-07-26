import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import copy
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
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.5
DIRICHLET_ALPHA = 0.1
MIN_SAMPLES_PER_CLIENT = 250

# MI-IAA Specific Hyperparameters
LAMBDA_MI = 0.01       # Weight of Mutual Information regularizer in local loss
LAMBDA_REG = 0.01      # Weight of FedProx-style proximal penalty
MINE_LR = 0.001        # Learning rate for the MINE network
MI_NORM_EPSILON = 1e-8 # Epsilon for Min-Max normalization

# GPU Detection Strategy (Forced CPU for sm_120 incompatibility)
DEVICE = torch.device("cpu")
device_name = "CPU (Forced due to sm_120 incompatibility)"

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
    Calculates MI individually for *each* latent feature dimension.
    Outputs a 128-dimensional vector representing the joint/marginal score for each feature.
    """
    def __init__(self, z_dim=128, y_dim=10):
        super(PerFeatureMINENetwork, self).__init__()
        # Taking concatenated Z and Y, mapping to a hidden layer, then mapping to 128 distinct outputs
        self.fc1 = nn.Linear(z_dim + y_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, z_dim)  # One score per latent feature

    def forward(self, z, y):
        x = torch.cat((z, y), dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)  # Shape: [Batch_Size, z_dim]

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
        model = copy.deepcopy(global_model).to(self.device)
        mine_model = PerFeatureMINENetwork(z_dim=128, y_dim=self.num_classes).to(self.device)
        model.train()
        mine_model.train()

        global_weights = copy.deepcopy(global_model.state_dict())  # Used for FedProx penalty

        optimizer_clf = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        optimizer_mine = optim.Adam(mine_model.parameters(), lr=MINE_LR)

        initial_weights = copy.deepcopy(model.state_dict())
        epoch_loss, epoch_mi, epoch_prox_loss = [], [], []

        for epoch in range(self.epochs):
            batch_loss, batch_mi, batch_prox = [], [], []
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.view(-1).long().to(self.device)

                # 1. Forward Pass CNN
                logits, z = model(images)
                y_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()

                # 2. Per-Feature MINE Forward
                y_shuffled = y_one_hot[torch.randperm(y_one_hot.shape[0])]
                t_joint = mine_model(z, y_one_hot)      # [Batch, 128]
                t_marginal = mine_model(z, y_shuffled)  # [Batch, 128]

                # Donsker-Varadhan Lower Bound (Computed per-feature, then averaged for scalar loss)
                feature_mi_estimates = torch.mean(t_joint, dim=0) - torch.log(torch.mean(torch.exp(t_marginal), dim=0) + 1e-8)
                avg_mi_estimate = torch.mean(feature_mi_estimates)  # Scalar average of all 128 features

                # 3. Proximal / Regularization Penalty (FedProx style)
                proximal_term = 0.0
                for name, param in model.named_parameters():
                    proximal_term += torch.sum((param - global_weights[name].to(self.device)) ** 2)

                # 4. Total Local Objective
                loss_mine = -avg_mi_estimate  # Maximize MI
                loss_task = self.ce_criterion(logits, labels)

                loss_clf = loss_task - (LAMBDA_MI * avg_mi_estimate) + (LAMBDA_REG / 2.0) * proximal_term

                optimizer_mine.zero_grad()
                optimizer_clf.zero_grad()

                loss_mine.backward(retain_graph=True)
                loss_clf.backward()

                optimizer_mine.step()
                optimizer_clf.step()

                batch_loss.append(loss_clf.item())
                batch_mi.append(avg_mi_estimate.item())
                batch_prox.append(proximal_term.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_mi.append(sum(batch_mi) / len(batch_mi))
            epoch_prox_loss.append(sum(batch_prox) / len(batch_prox))

        final_weights = model.state_dict()

        # Calculate Layer-Wise Parameter Importance (Feature-Traceable Proxy via Update Magnitude)
        importance_scores = {}
        for key in final_weights.keys():
            update_magnitude = torch.abs(final_weights[key] - initial_weights[key].to(self.device)).mean().item()
            importance_scores[key] = update_magnitude

        return final_weights, sum(epoch_loss)/len(epoch_loss), sum(epoch_mi)/len(epoch_mi), sum(epoch_prox_loss)/len(epoch_prox_loss), importance_scores

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
def average_weights_mi_iaa(local_weights, mi_scores, importance_dicts):
    mi_min, mi_max = min(mi_scores), max(mi_scores)
    normalized_mi = [(mi - mi_min) / (mi_max - mi_min + MI_NORM_EPSILON) for mi in mi_scores]

    num_clients = len(local_weights)
    w_avg = copy.deepcopy(local_weights[0])
    layer_alpha_logs = {}

    for key in w_avg.keys():
        layer_scores = []
        for i in range(num_clients):
            imp = importance_dicts[i][key]
            layer_scores.append(normalized_mi[i] * imp)

        alphas = F.softmax(torch.tensor(layer_scores, dtype=torch.float32), dim=0).tolist()
        layer_alpha_logs[key] = alphas

        w_avg[key] = local_weights[0][key] * alphas[0]
        for i in range(1, num_clients):
            w_avg[key] += local_weights[i][key] * alphas[i]

    avg_alphas = [sum(layer_alpha_logs[key][i] for key in w_avg.keys()) / len(w_avg.keys()) for i in range(num_clients)]
    return w_avg, normalized_mi, [a * 100 for a in avg_alphas]

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

    print("\n==============================================")
    print(f" Starting MI-IAA Training ({GLOBAL_ROUNDS} Rounds)")
    print("==============================================")

    for round_idx in range(1, GLOBAL_ROUNDS + 1):
        local_weights, local_losses, local_mis, local_importances = [], [], [], []
        client_personalized_accuracies, client_prox_losses = [], []

        print(f"\n| Round : {round_idx}/{GLOBAL_ROUNDS} |")

        for idx in range(NUM_CLIENTS):
            client = ClientUpdate(dataloader=client_train_loaders[idx], device=DEVICE,
                                   epochs=LOCAL_EPOCHS, num_classes=num_classes)
            weights, loss, mi_estimate, prox_loss, importances = client.train(global_model)

            client_personalized_accuracies.append(evaluate(weights, client_test_loaders[idx], DEVICE, num_classes))
            local_weights.append(weights)
            local_losses.append(loss)
            local_mis.append(mi_estimate)
            client_prox_losses.append(prox_loss)
            local_importances.append(importances)

        # Server Aggregation
        global_weights, normalized_mi, avg_layer_weights = average_weights_mi_iaa(
            local_weights, local_mis, local_importances
        )
        global_model.load_state_dict(global_weights)

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
