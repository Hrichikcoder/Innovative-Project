import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import matplotlib.pyplot as plt
from medmnist import BloodMNIST
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
MIN_SAMPLES_PER_CLIENT = 100  # BloodMNIST is smaller (~11k samples), 250 is impossible with alpha=0.1
NUM_CLASSES = 8

LAMBDA_MI = 0.01
LAMBDA_REG = 0.01
MINE_LR = 0.001
MI_NORM_EPSILON = 1e-8

DEVICE = torch.device("cpu")
device_name = "CPU (Forced due to sm_120 incompatibility)"

# ==========================================
# Data Partitioning
# ==========================================
def get_dirichlet_data_loaders(dataset, num_clients, alpha, batch_size, min_samples):
    targets = np.array([dataset[i][1] for i in range(len(dataset))]).squeeze()
    class_indices = [np.where(targets == i)[0] for i in range(NUM_CLASSES)]
    print(f"\n[Data Partitioning] Splitting BloodMNIST for {num_clients} clients (Alpha={alpha}).")
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
# Models
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
        z = F.relu(self.fc1(x))
        return self.fc2(z), z

class PerFeatureMINENetwork(nn.Module):
    def __init__(self, z_dim=128, y_dim=NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(z_dim + y_dim, 256)
        self.fc2 = nn.Linear(256, z_dim)

    def forward(self, z, y):
        x = F.relu(self.fc1(torch.cat((z, y), dim=1)))
        return self.fc2(x)

# ==========================================
# Local Training (MI-IAA)
# ==========================================
class ClientUpdate:
    def __init__(self, dataloader, device, epochs):
        self.dataloader = dataloader
        self.device = device
        self.epochs = epochs

    def train(self, global_model):
        model = copy.deepcopy(global_model).to(self.device)
        mine = PerFeatureMINENetwork().to(self.device)
        model.train(); mine.train()
        global_weights = copy.deepcopy(global_model.state_dict())
        opt_clf = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        opt_mine = optim.Adam(mine.parameters(), lr=MINE_LR)
        init_weights = copy.deepcopy(model.state_dict())
        ce = nn.CrossEntropyLoss()
        epoch_loss, epoch_mi, epoch_prox = [], [], []

        for _ in range(self.epochs):
            bl, bm, bp = [], [], []
            for images, labels in self.dataloader:
                images = images.to(self.device)
                labels = labels.view(-1).long().to(self.device)
                logits, z = model(images)
                y_oh = F.one_hot(labels, num_classes=NUM_CLASSES).float()
                y_sh = y_oh[torch.randperm(y_oh.shape[0])]
                t_j = mine(z, y_oh)
                t_m = mine(z, y_sh)
                feat_mi = torch.mean(t_j, dim=0) - torch.log(torch.mean(torch.exp(t_m), dim=0) + 1e-8)
                avg_mi = torch.mean(feat_mi)
                prox = sum(((p - global_weights[n].to(self.device)) ** 2).sum() for n, p in model.named_parameters())
                loss_mine = -avg_mi
                loss_clf = ce(logits, labels) - (LAMBDA_MI * avg_mi) + (LAMBDA_REG / 2.0) * prox
                opt_mine.zero_grad(); opt_clf.zero_grad()
                loss_mine.backward(retain_graph=True)
                loss_clf.backward()
                opt_mine.step(); opt_clf.step()
                bl.append(loss_clf.item()); bm.append(avg_mi.item()); bp.append(prox.item())
            epoch_loss.append(np.mean(bl)); epoch_mi.append(np.mean(bm)); epoch_prox.append(np.mean(bp))

        final_weights = model.state_dict()
        importance = {k: torch.abs(final_weights[k] - init_weights[k].to(self.device)).mean().item() for k in final_weights}
        return final_weights, np.mean(epoch_loss), np.mean(epoch_mi), np.mean(epoch_prox), importance

# ==========================================
# Evaluation & Aggregation
# ==========================================
def evaluate(state_dict, loader, device):
    model = SimpleCNN().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)
            preds, _ = model(images)
            correct += (preds.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total if total > 0 else 0.0

def average_weights_mi_iaa(local_weights, mi_scores, importance_dicts):
    mi_min, mi_max = min(mi_scores), max(mi_scores)
    norm_mi = [(m - mi_min) / (mi_max - mi_min + MI_NORM_EPSILON) for m in mi_scores]
    w_avg = copy.deepcopy(local_weights[0])
    layer_alphas = {}
    for key in w_avg:
        scores = [norm_mi[i] * importance_dicts[i][key] for i in range(len(local_weights))]
        alphas = F.softmax(torch.tensor(scores, dtype=torch.float32), dim=0).tolist()
        layer_alphas[key] = alphas
        w_avg[key] = sum(local_weights[i][key] * alphas[i] for i in range(len(local_weights)))
    avg_alphas = [sum(layer_alphas[k][i] for k in w_avg) / len(w_avg) * 100 for i in range(len(local_weights))]
    return w_avg, norm_mi, avg_alphas

# ==========================================
# Main
# ==========================================
def main():
    print("==============================================")
    print(" BloodMNIST - Normalized (MI-IAA) - Our Method")
    print("==============================================")
    print(f"Device: {device_name} | Clients: {NUM_CLIENTS} | Rounds: {GLOBAL_ROUNDS} | Alpha: {DIRICHLET_ALPHA}")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
    train_dataset = BloodMNIST(split='train', transform=tfm, download=True)
    test_dataset  = BloodMNIST(split='test',  transform=tfm, download=True)
    global_test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    client_train_loaders, client_test_loaders = get_dirichlet_data_loaders(
        train_dataset, NUM_CLIENTS, DIRICHLET_ALPHA, BATCH_SIZE, MIN_SAMPLES_PER_CLIENT
    )

    global_model = SimpleCNN().to(DEVICE)
    history = {'global_acc': [], 'avg_client_acc': []}

    for rnd in range(1, GLOBAL_ROUNDS + 1):
        print(f"\n| Round {rnd}/{GLOBAL_ROUNDS} |")
        local_weights, local_losses, local_mis, local_importances = [], [], [], []
        client_accs, client_prox = [], []

        for idx in range(NUM_CLIENTS):
            client = ClientUpdate(client_train_loaders[idx], DEVICE, LOCAL_EPOCHS)
            w, loss, mi, prox, imp = client.train(global_model)
            acc = evaluate(w, client_test_loaders[idx], DEVICE)
            local_weights.append(w); local_losses.append(loss); local_mis.append(mi)
            local_importances.append(imp); client_accs.append(acc); client_prox.append(prox)

        global_weights, norm_mi, avg_layer_wts = average_weights_mi_iaa(local_weights, local_mis, local_importances)
        global_model.load_state_dict(global_weights)
        global_acc = evaluate(global_model.state_dict(), global_test_loader, DEVICE)
        avg_client_acc = np.mean(client_accs)
        history['global_acc'].append(global_acc); history['avg_client_acc'].append(avg_client_acc)

        print(f"   Global Acc: {global_acc:.2f}% | Avg Client Acc: {avg_client_acc:.2f}%")
        print(f"   | Client | Local Acc % | Raw MI Est. | Norm MI | Avg Layer Wgt % | Prox Drift |")
        print(f"   |--------|-------------|-------------|---------|--------------------|------------|")
        for idx in range(NUM_CLIENTS):
            print(f"   | {idx+1:6d} | {client_accs[idx]:11.2f}% | {local_mis[idx]:11.4f} | {norm_mi[idx]:7.4f} | {avg_layer_wts[idx]:18.2f}% | {client_prox[idx]:10.4f} |")

    rounds = list(range(1, GLOBAL_ROUNDS + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, history['avg_client_acc'], marker='o', color='blue', label='Avg Client Acc')
    plt.title('BloodMNIST Normalized (MI-IAA): Rounds vs Avg Client Accuracy')
    plt.xlabel('Communication Rounds'); plt.ylabel('Accuracy (%)'); plt.grid(True); plt.legend()
    plt.savefig('BloodMNIST_normalized_client_accuracy.png'); plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, history['global_acc'], marker='s', color='red', label='Global Model Acc')
    plt.title('BloodMNIST Normalized (MI-IAA): Rounds vs Global Accuracy')
    plt.xlabel('Communication Rounds'); plt.ylabel('Accuracy (%)'); plt.grid(True); plt.legend()
    plt.savefig('BloodMNIST_normalized_global_accuracy.png'); plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, history['avg_client_acc'], marker='o', color='blue', label='Avg Client Acc')
    plt.plot(rounds, history['global_acc'], marker='s', color='red', label='Global Model Acc')
    plt.title('BloodMNIST Normalized (MI-IAA): Rounds vs Federated Accuracies')
    plt.xlabel('Communication Rounds'); plt.ylabel('Accuracy (%)'); plt.grid(True); plt.legend()
    plt.savefig('BloodMNIST_normalized_combined.png'); plt.close()

    print("\n[Done] Plots saved.")
    print(f"Training Complete! Final Global Acc: {history['global_acc'][-1]:.2f}%, Final Avg Personalized Acc: {history['avg_client_acc'][-1]:.2f}%")

if __name__ == '__main__':
    main()
