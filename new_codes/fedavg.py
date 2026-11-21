import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import sys
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

# Import the centralized data pipeline
import data_pipeline


class Tee:  # Saving results to both terminal and log file
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
DIRICHLET_ALPHA = 0.1  # Highly skewed data
MIN_SAMPLES_PER_CLIENT = 250

# GPU Detection Strategy
# Force CPU because standard PyTorch Windows binaries do not yet have sm_120 (RTX 5000) conv kernels
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
# Model Definition
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
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# ==========================================
# Local Training (Client)
# ==========================================
class ClientUpdate:
    def __init__(self, dataloader, device, epochs):
        self.dataloader = dataloader
        self.device = device
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()

    def train(self, global_model):
        model = copy.deepcopy(global_model)
        model.to(self.device)
        model.train()

        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

        if len(self.dataloader.dataset) == 0:
            raise ValueError("Client training dataloader contains no samples.")

        epoch_loss = []
        for epoch in range(self.epochs):
            batch_loss = []
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.view(-1).long().to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            if batch_loss:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        if not epoch_loss:
            raise RuntimeError("No training batches were processed for this client.")

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), len(self.dataloader.dataset)

# ==========================================
# Server Aggregation
# ==========================================
def average_weights(w, num_samples):
    if not w:
        raise ValueError("No client weights were supplied for aggregation.")
    if len(w) != len(num_samples):
        raise ValueError("Client weights and sample counts must have the same length.")

    total_samples = sum(num_samples)
    if total_samples <= 0:
        raise ValueError("Total client samples must be greater than zero.")

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * (num_samples[0] / total_samples)
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * (num_samples[i] / total_samples)
    return w_avg


def calculate_client_contributions(local_weights, num_samples):
    """
    Calculate each client's exact FedAvg server aggregation contribution.

    FedAvg contribution is its sample-based aggregation weight:
        n_i / sum_j(n_j)

    The function reports this weight as a percentage and does not alter
    the FedAvg algorithm.
    """
    if not local_weights:
        raise ValueError("local_weights cannot be empty.")
    if len(local_weights) != len(num_samples):
        raise ValueError("local_weights and num_samples must have the same length.")

    total_samples = sum(num_samples)
    if total_samples <= 0:
        raise ValueError("Total client samples must be greater than zero.")

    return [(samples / total_samples) * 100.0 for samples in num_samples]


def save_client_contributions_md(
    filepath,
    dataset_name,
    round_contributions,
    round_sample_counts
):
    """Save per-client FedAvg contribution for every round in Markdown."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# FedAvg Per-Client Contribution Report\n\n")
        f.write(f"- **Dataset:** {dataset_name}\n")
        f.write("- **Contribution definition:** FedAvg server aggregation weight\n")
        f.write("- **Formula:** `client samples / total samples in round × 100`\n\n")

        for round_idx, contributions in enumerate(round_contributions, start=1):
            f.write(f"## Round {round_idx}\n\n")
            f.write("| Client | Contribution | Samples |\n")
            f.write("|---:|---:|---:|\n")

            samples = round_sample_counts[round_idx - 1]
            for client_idx, contribution in enumerate(contributions):
                f.write(
                    f"| {client_idx + 1} | {contribution:.4f}% | "
                    f"{samples[client_idx]} |\n"
                )
            f.write("\n")


# ==========================================
# Evaluation
# ==========================================
def evaluate(model_state, dataloader, device, num_classes):
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(model_state)
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.view(-1).long().to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0
    accuracy = 100 * correct / total
    return accuracy

# ==========================================
# Main Simulation
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="FedAvg Federated Learning")
    parser.add_argument('--dataset', required=True,
                         choices=['cifar10', 'mnist', 'pathmnist', 'bloodmnist'],
                         help="Dataset to train on.")
    args = parser.parse_args()
    dataset_name = args.dataset.lower()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = open(
        f"training_log_fedavg_{dataset_name}_{timestamp}.md",
        "w",
        encoding="utf-8"
    )

    sys.stdout = Tee(sys.__stdout__, log_file)
    print("==============================================")
    print(f" Advanced Federated Learning Script (FedAvg)")
    print("==============================================")
    print(f"Dataset         : {dataset_name}")
    print(f"Hardware Device : {device_name}")
    print(f"Total Clients   : {NUM_CLIENTS}")
    print(f"Global Rounds   : {GLOBAL_ROUNDS}")
    print(f"Heterogeneity   : Dirichlet (Alpha={DIRICHLET_ALPHA})")

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

    global_model = SimpleCNN(num_classes=num_classes)
    global_model.to(DEVICE)

    history = {'global_acc': [], 'personalized_acc': []}
    contribution_history = []
    contribution_sample_history = []

    print("\n==============================================")
    print(f" Starting FedAvg Training ({GLOBAL_ROUNDS} Rounds)")
    print("==============================================")

    for round_idx in range(1, GLOBAL_ROUNDS + 1):
        local_weights, local_losses, local_sample_counts = [], [], []
        client_personalized_accuracies = []

        print(f"\n| Round : {round_idx}/{GLOBAL_ROUNDS} |")

        for idx in range(NUM_CLIENTS):
            client = ClientUpdate(dataloader=client_train_loaders[idx], device=DEVICE, epochs=LOCAL_EPOCHS)
            personalized_weights, loss, num_samples = client.train(global_model)

            pers_acc = evaluate(personalized_weights, client_test_loaders[idx], DEVICE, num_classes)
            client_personalized_accuracies.append(pers_acc)

            local_weights.append(copy.deepcopy(personalized_weights))
            local_losses.append(loss)
            local_sample_counts.append(num_samples)

        # Aggregate weights on Server
        global_weights = average_weights(local_weights, local_sample_counts)
        global_model.load_state_dict(global_weights)

        # Per-client FedAvg contribution for this round.
        round_contributions = calculate_client_contributions(
            local_weights, local_sample_counts
        )
        contribution_history.append(round_contributions)
        contribution_sample_history.append(list(local_sample_counts))

        # Calculate Analytics
        total_samples_round = sum(local_sample_counts)
        avg_train_loss = sum(local_losses) / len(local_losses)
        avg_personalized_acc = sum(client_personalized_accuracies) / len(client_personalized_accuracies)
        global_test_acc = evaluate(global_model.state_dict(), global_test_loader, DEVICE, num_classes)

        # Print Deep Analytics
        print(f"\n   [Server Metrics]")
        print(f"   Global Model Test Accuracy    : {global_test_acc:.2f}%")
        print(f"   Avg Client Personalized Acc   : {avg_personalized_acc:.2f}%")
        print(f"   Avg Local Train Loss          : {avg_train_loss:.4f}")

        print(f"\n   [Detailed Per-Client Metrics]")
        print(f"   | Client | Contribution % | Retention | Local Loss | Local Acc % |")
        print(f"   |--------|----------------|-----------|------------|-------------|")
        for idx in range(NUM_CLIENTS):
            contribution = round_contributions[idx]
            retention_rate = 100.0  # Assumed 100% since no dropout mechanism is implemented
            print(f"   | {idx+1:6d} | {contribution:13.2f}% | {retention_rate:8.1f}% | {local_losses[idx]:10.4f} | {client_personalized_accuracies[idx]:10.2f}% |")

        history['global_acc'].append(global_test_acc)
        history['personalized_acc'].append(avg_personalized_acc)

    print(f"\nTraining Complete! Final Global Acc: {history['global_acc'][-1]:.2f}%, Final Avg Personalized Acc: {history['personalized_acc'][-1]:.2f}%")

    contribution_file = f'fedavg_{dataset_name}_client_contributions.md'
    save_client_contributions_md(
        contribution_file,
        dataset_name,
        contribution_history,
        contribution_sample_history
    )
    print(f"Per-client contribution report saved to {contribution_file}")

    # --- Plotting Results ---
    print("\n[Simulation Complete] Generating accuracy plots...")
    rounds = list(range(1, GLOBAL_ROUNDS + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, history['personalized_acc'], marker='o', color='b', label='Avg Client Acc')
    plt.title(f'FedAvg ({dataset_name}): Communication Rounds vs Avg Client Accuracy')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'fedavg_{dataset_name}_client_accuracy_plot.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, history['global_acc'], marker='s', color='r', label='Global Model Acc')
    plt.title(f'FedAvg ({dataset_name}): Communication Rounds vs Global Accuracy')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'fedavg_{dataset_name}_global_accuracy_plot.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, history['personalized_acc'], marker='o', color='b', label='Avg Client Acc')
    plt.plot(rounds, history['global_acc'], marker='s', color='r', label='Global Model Acc')
    plt.title(f'FedAvg ({dataset_name}): Communication Rounds vs Federated Accuracies')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'fedavg_{dataset_name}_combined_accuracy_plot.png')
    plt.close()

    print("Plots successfully saved as PNG files in the project folder!")

if __name__ == '__main__':
    try:
        main()
    finally:
        sys.stdout = sys.__stdout__
