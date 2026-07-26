import torch
import torch.nn as nn
import torch.nn.functional as F
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
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.5
DIRICHLET_ALPHA = 0.1
MIN_SAMPLES_PER_CLIENT = 250
FEDCURV_LAMBDA = 1.0

# GPU Detection Strategy (Forced CPU for sm_120 incompatibility)
DEVICE = torch.device("cpu")
device_name = "CPU (Forced due to sm_120 incompatibility)"

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
# FedCurv: Fisher Information Matrix
# ==========================================
def compute_fim(model, dataloader, device):
    fim = {}
    for name, param in model.named_parameters():
        fim[name] = torch.zeros_like(param.data)

    model.eval()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.view(-1).long().to(device)
        model.zero_grad()

        log_probs = F.log_softmax(model(images), dim=1)
        loss = F.nll_loss(log_probs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fim[name] += (param.grad.data ** 2) / len(dataloader)

    return fim

# ==========================================
# Local Training (Client)
# ==========================================
class ClientUpdate:
    def __init__(self, dataloader, device, epochs):
        self.dataloader = dataloader
        self.device = device
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()

    def train(self, global_model, global_fim):
        model = copy.deepcopy(global_model)
        model.to(self.device)
        model.train()

        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

        epoch_loss = []
        for epoch in range(self.epochs):
            batch_loss = []
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.view(-1).long().to(self.device)

                optimizer.zero_grad()
                outputs = model(images)

                # Standard Cross Entropy
                loss = self.criterion(outputs, labels)

                # Add FedCurv Penalty (Fisher Regularization)
                if global_fim is not None:
                    penalty = 0.0
                    for name, param in model.named_parameters():
                        global_param = global_model.state_dict()[name].to(self.device)
                        global_f = global_fim[name].to(self.device)
                        penalty += torch.sum(global_f * (param - global_param) ** 2)
                    loss += (FEDCURV_LAMBDA / 2.0) * penalty

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # Compute local FIM for this client after training
        local_fim = compute_fim(model, self.dataloader, self.device)

        return model.state_dict(), local_fim, sum(epoch_loss) / len(epoch_loss), len(self.dataloader.dataset)

# ==========================================
# Server Aggregation (FedCurv)
# ==========================================
def average_weights(w, num_samples):
    total_samples = sum(num_samples)
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * (num_samples[0] / total_samples)
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * (num_samples[i] / total_samples)
    return w_avg

def average_fim(fims, num_samples):
    total_samples = sum(num_samples)
    fim_avg = copy.deepcopy(fims[0])
    for key in fim_avg.keys():
        fim_avg[key] = fim_avg[key] * (num_samples[0] / total_samples)
    for key in fim_avg.keys():
        for i in range(1, len(fims)):
            fim_avg[key] += fims[i][key] * (num_samples[i] / total_samples)
    return fim_avg

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
    parser = argparse.ArgumentParser(description="FedCurv Federated Learning")
    parser.add_argument('--dataset', required=True,
                         choices=['cifar10', 'mnist', 'pathmnist', 'bloodmnist'],
                         help="Dataset to train on.")
    args = parser.parse_args()
    dataset_name = args.dataset.lower()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = open(
        f"training_log_fedcurv_{dataset_name}_{timestamp}.md",
        "w",
        encoding="utf-8"
        )
    sys.stdout = Tee(sys.__stdout__, log_file)
    print("==============================================")
    print(f" Advanced FedCurv Script")
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

    # FedCurv specific: Initialize Global FIM
    global_fim = None

    history = {'global_acc': [], 'personalized_acc': []}

    print("\n==============================================")
    print(f" Starting FedCurv Training ({GLOBAL_ROUNDS} Rounds)")
    print("==============================================")

    with open(f"{dataset_name}_advanced_fedcurv_log.txt", "w") as f:
        f.write("Advanced FedCurv Log\n")
        f.write("======================================\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Data Distribution: Dirichlet Alpha={DIRICHLET_ALPHA}, {NUM_CLIENTS} Clients\n")
        f.write("======================================\n")

    for round_idx in range(1, GLOBAL_ROUNDS + 1):
        local_weights, local_fims, local_losses, local_sample_counts = [], [], [], []
        client_personalized_accuracies = []

        print(f"\n| Round : {round_idx}/{GLOBAL_ROUNDS} |")

        for idx in range(NUM_CLIENTS):
            client = ClientUpdate(dataloader=client_train_loaders[idx], device=DEVICE, epochs=LOCAL_EPOCHS)
            personalized_weights, local_fim, loss, num_samples = client.train(global_model, global_fim)

            pers_acc = evaluate(personalized_weights, client_test_loaders[idx], DEVICE, num_classes)
            client_personalized_accuracies.append(pers_acc)

            local_weights.append(copy.deepcopy(personalized_weights))
            local_fims.append(copy.deepcopy(local_fim))
            local_losses.append(loss)
            local_sample_counts.append(num_samples)

        # Aggregate weights and FIMs on Server
        global_weights = average_weights(local_weights, local_sample_counts)
        global_model.load_state_dict(global_weights)

        global_fim = average_fim(local_fims, local_sample_counts)

        # Calculate Analytics
        total_samples_round = sum(local_sample_counts)
        avg_train_loss = sum(local_losses) / len(local_losses)
        avg_personalized_acc = sum(client_personalized_accuracies) / len(client_personalized_accuracies)
        global_test_acc = evaluate(global_model.state_dict(), global_test_loader, DEVICE, num_classes)

        # Print Deep Analytics
        print(f"\n   [Server Metrics]")
        print(f"   Global Model Test Accuracy    : {global_test_acc:.2f}%")
        print(f"   Avg Client Personalized Acc   : {avg_personalized_acc:.2f}%  <-- FedCurv Client Performance")
        print(f"   Avg Local Train Loss          : {avg_train_loss:.4f}")

        print(f"\n   [Detailed Per-Client Metrics]")
        print(f"   | Client | Contribution % | Retention | Local Loss | Local Acc % |")
        print(f"   |--------|----------------|-----------|------------|-------------|")
        for idx in range(NUM_CLIENTS):
            contribution = (local_sample_counts[idx] / total_samples_round) * 100
            retention_rate = 100.0  # Assumed 100% since no dropout mechanism is implemented
            print(f"   | {idx+1:6d} | {contribution:13.2f}% | {retention_rate:8.1f}% | {local_losses[idx]:10.4f} | {client_personalized_accuracies[idx]:10.2f}% |")

        history['global_acc'].append(global_test_acc)
        history['personalized_acc'].append(avg_personalized_acc)

        with open(f"{dataset_name}_advanced_fedcurv_log.txt", "a") as f:
            f.write(f"FedCurv - Round {round_idx}: Global Acc={global_test_acc:.2f}%, Personalized Acc={avg_personalized_acc:.2f}%\n")

    print(f"\nTraining Complete! Final Global Acc: {history['global_acc'][-1]:.2f}%, Final Avg Personalized Acc: {history['personalized_acc'][-1]:.2f}%")
    # --- Plotting Results ---
    print("\n[Simulation Complete] Generating accuracy plots...")
    rounds = list(range(1, GLOBAL_ROUNDS + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, history['personalized_acc'], marker='o', color='b', label='Avg Client Acc')
    plt.title(f'FedCurv ({dataset_name}): Communication Rounds vs Avg Client Accuracy')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'fedcurv_{dataset_name}_client_accuracy_plot.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, history['global_acc'], marker='s', color='r', label='Global Model Acc')
    plt.title(f'FedCurv ({dataset_name}): Communication Rounds vs Global Accuracy')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'fedcurv_{dataset_name}_global_accuracy_plot.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, history['personalized_acc'], marker='o', color='b', label='Avg Client Acc')
    plt.plot(rounds, history['global_acc'], marker='s', color='r', label='Global Model Acc')
    plt.title(f'FedCurv ({dataset_name}): Communication Rounds vs Federated Accuracies')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'fedcurv_{dataset_name}_combined_accuracy_plot.png')
    plt.close()

    print("Plots successfully saved as PNG files in the project folder!")

if __name__ == '__main__':
    try:
        main()
    finally:
        sys.stdout = sys.__stdout__