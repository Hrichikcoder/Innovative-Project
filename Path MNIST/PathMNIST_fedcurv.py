import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import matplotlib.pyplot as plt
from medmnist import PathMNIST
from torchvision import transforms

# ==========================================
# Hyperparameters
# ==========================================
NUM_CLIENTS = 30
GLOBAL_ROUNDS = 100
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.5
DIRICHLET_ALPHA = 0.1
MIN_SAMPLES_PER_CLIENT = 250
NUM_CLASSES = 9
FEDCURV_LAMBDA = 0.1  # EWC penalty weight

DEVICE = torch.device("cpu")
device_name = "CPU (Forced due to sm_120 incompatibility)"

# ==========================================
# Data Partitioning
# ==========================================
def get_dirichlet_data_loaders(dataset, num_clients, alpha, batch_size, min_samples):
    targets = np.array([dataset[i][1] for i in range(len(dataset))]).squeeze()
    class_indices = [np.where(targets == i)[0] for i in range(NUM_CLASSES)]
    print(f"\n[Data Partitioning] Splitting PathMNIST for {num_clients} clients (Alpha={alpha}).")
    while True:
        client_indices = {i: [] for i in range(num_clients)}
        for c in range(NUM_CLASSES):
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            n = (proportions * len(class_indices[c])).astype(int)
            diff = len(class_indices[c]) - n.sum()
            for _ in range(diff):
                n[np.random.randint(num_clients)] += 1
            np.random.shuffle(class_indices[c])
            cur = 0
            for i in range(num_clients):
                client_indices[i].extend(class_indices[c][cur: cur + n[i]])
                cur += n[i]
        if min(len(client_indices[i]) for i in range(num_clients)) >= min_samples:
            print(f"[OK] Minimum samples satisfied.")
            break
    train_loaders, test_loaders = [], []
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
        tl = int(0.8 * len(client_indices[i]))
        train_loaders.append(DataLoader(Subset(dataset, client_indices[i][:tl]), batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(Subset(dataset, client_indices[i][tl:]), batch_size=batch_size, shuffle=False))
    return train_loaders, test_loaders

# ==========================================
# Model
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ==========================================
# Fisher Information Matrix
# ==========================================
def compute_fisher(model, loader, device):
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.view(-1).long().to(device)
        model.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.data.clone().pow(2)
    for n in fisher:
        fisher[n] /= len(loader)
    return fisher

def aggregate_fishers(fishers):
    agg = {k: torch.zeros_like(v) for k, v in fishers[0].items()}
    for f in fishers:
        for k in agg:
            agg[k] += f[k]
    for k in agg:
        agg[k] /= len(fishers)
    return agg

# ==========================================
# Local Training with FedCurv Penalty
# ==========================================
def local_train_fedcurv(model, loader, epochs, device, global_params, global_fisher):
    model = copy.deepcopy(model).to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _ in range(epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            penalty = 0.0
            for name, param in model.named_parameters():
                if name in global_fisher:
                    penalty += (global_fisher[name].to(device) * (param - global_params[name].to(device)) ** 2).sum()
            loss = loss + (FEDCURV_LAMBDA / 2.0) * penalty
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return model.state_dict(), sum(losses) / len(losses), len(loader.dataset)

def average_weights(w, sizes):
    total = sum(sizes)
    w_avg = copy.deepcopy(w[0])
    for key in w_avg:
        w_avg[key] = w_avg[key] * (sizes[0] / total)
    for key in w_avg:
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * (sizes[i] / total)
    return w_avg

def evaluate(state_dict, loader, device):
    model = SimpleCNN().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total if total > 0 else 0.0

# ==========================================
# Main
# ==========================================
def main():
    print("==============================================")
    print(" PathMNIST - FedCurv")
    print("==============================================")
    print(f"Device: {device_name} | Clients: {NUM_CLIENTS} | Rounds: {GLOBAL_ROUNDS} | Alpha: {DIRICHLET_ALPHA}")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
    train_dataset = PathMNIST(split='train', transform=tfm, download=True)
    test_dataset  = PathMNIST(split='test',  transform=tfm, download=True)
    global_test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    client_train_loaders, client_test_loaders = get_dirichlet_data_loaders(
        train_dataset, NUM_CLIENTS, DIRICHLET_ALPHA, BATCH_SIZE, MIN_SAMPLES_PER_CLIENT
    )

    global_model = SimpleCNN().to(DEVICE)
    global_fisher = {n: torch.zeros_like(p) for n, p in global_model.named_parameters() if p.requires_grad}
    history = {'global_acc': [], 'avg_client_acc': []}

    for rnd in range(1, GLOBAL_ROUNDS + 1):
        print(f"\n| Round {rnd}/{GLOBAL_ROUNDS} |")
        global_params = copy.deepcopy(global_model.state_dict())
        local_weights, local_sizes, local_losses = [], [], []
        client_accs, client_fishers = [], []

        for idx in range(NUM_CLIENTS):
            w, loss, size = local_train_fedcurv(global_model, client_train_loaders[idx], LOCAL_EPOCHS, DEVICE, global_params, global_fisher)
            acc = evaluate(w, client_test_loaders[idx], DEVICE)
            temp_model = SimpleCNN()
            temp_model.load_state_dict(w)
            fisher = compute_fisher(temp_model, client_train_loaders[idx], DEVICE)
            local_weights.append(w)
            local_sizes.append(size)
            local_losses.append(loss)
            client_accs.append(acc)
            client_fishers.append(fisher)

        global_model.load_state_dict(average_weights(local_weights, local_sizes))
        global_fisher = aggregate_fishers(client_fishers)
        global_acc = evaluate(global_model.state_dict(), global_test_loader, DEVICE)
        avg_client_acc = np.mean(client_accs)
        history['global_acc'].append(global_acc)
        history['avg_client_acc'].append(avg_client_acc)

        print(f"   Global Acc: {global_acc:.2f}% | Avg Client Acc: {avg_client_acc:.2f}% | Avg Loss: {np.mean(local_losses):.4f}")
        print(f"   | Client | Contribution % | Retention | Local Loss | Local Acc % |")
        print(f"   |--------|----------------|-----------|------------|-------------|")
        total_samples = sum(local_sizes)
        for idx in range(NUM_CLIENTS):
            print(f"   | {idx+1:6d} | {(local_sizes[idx]/total_samples)*100:13.2f}% |    100.0% | {local_losses[idx]:10.4f} | {client_accs[idx]:10.2f}% |")

    rounds = list(range(1, GLOBAL_ROUNDS + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, history['avg_client_acc'], marker='o', color='blue', label='Avg Client Acc')
    plt.title('PathMNIST FedCurv: Communication Rounds vs Avg Client Accuracy')
    plt.xlabel('Communication Rounds'); plt.ylabel('Accuracy (%)'); plt.grid(True); plt.legend()
    plt.savefig('PathMNIST_fedcurv_client_accuracy.png'); plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, history['global_acc'], marker='s', color='red', label='Global Model Acc')
    plt.title('PathMNIST FedCurv: Communication Rounds vs Global Accuracy')
    plt.xlabel('Communication Rounds'); plt.ylabel('Accuracy (%)'); plt.grid(True); plt.legend()
    plt.savefig('PathMNIST_fedcurv_global_accuracy.png'); plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, history['avg_client_acc'], marker='o', color='blue', label='Avg Client Acc')
    plt.plot(rounds, history['global_acc'], marker='s', color='red', label='Global Model Acc')
    plt.title('PathMNIST FedCurv: Communication Rounds vs Federated Accuracies')
    plt.xlabel('Communication Rounds'); plt.ylabel('Accuracy (%)'); plt.grid(True); plt.legend()
    plt.savefig('PathMNIST_fedcurv_combined.png'); plt.close()

    print("\n[Done] Plots saved.")
    print(f"Training Complete! Final Global Acc: {history['global_acc'][-1]:.2f}%, Final Avg Personalized Acc: {history['avg_client_acc'][-1]:.2f}%")

if __name__ == '__main__':
    main()
