import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import numpy as np
import copy
import matplotlib.pyplot as plt

# ==========================================
# Hyperparameters & Configurations
# ==========================================
NUM_CLIENTS = 30
GLOBAL_ROUNDS = 40
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
# Data Partitioning (Dirichlet Non-IID)
# ==========================================
def get_dirichlet_data_loaders(dataset, num_clients, alpha, batch_size, min_samples):
    num_classes = len(dataset.classes)
    targets = np.array(dataset.targets)
    class_indices = [np.where(targets == i)[0] for i in range(num_classes)]
    
    print(f"\n[Data Partitioning] Splitting data for {num_clients} clients (Alpha={alpha}).")
    while True:
        client_indices = {i: [] for i in range(num_clients)}
        for c in range(num_classes):
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            num_samples_per_client = (proportions * len(class_indices[c])).astype(int)
            diff = len(class_indices[c]) - num_samples_per_client.sum()
            for _ in range(diff):
                num_samples_per_client[np.random.randint(num_clients)] += 1
            np.random.shuffle(class_indices[c])
            current_idx = 0
            for i in range(num_clients):
                num_samples = num_samples_per_client[i]
                client_indices[i].extend(class_indices[c][current_idx : current_idx + num_samples])
                current_idx += num_samples
        if min([len(client_indices[i]) for i in range(num_clients)]) >= min_samples:
            break

    client_train_loaders, client_test_loaders = [], []
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
        train_len = int(0.8 * len(client_indices[i]))
        client_train_loaders.append(DataLoader(Subset(dataset, client_indices[i][:train_len]), batch_size=batch_size, shuffle=True))
        client_test_loaders.append(DataLoader(Subset(dataset, client_indices[i][train_len:]), batch_size=batch_size, shuffle=False))
    return client_train_loaders, client_test_loaders

# ==========================================
# Model Definitions
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self):
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
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        z = self.relu3(self.fc1(x)) # Z: Latent Features (128-dim)
        out = self.fc2(z)           # Y_hat: Logits
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
        self.fc2 = nn.Linear(256, z_dim) # One score per latent feature

    def forward(self, z, y):
        x = torch.cat((z, y), dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x) # Shape: [Batch_Size, 128]

# ==========================================
# Local Training (Client)
# ==========================================
class ClientUpdate:
    def __init__(self, dataloader, device, epochs):
        self.dataloader = dataloader
        self.device = device
        self.epochs = epochs
        self.ce_criterion = nn.CrossEntropyLoss()

    def train(self, global_model):
        model = copy.deepcopy(global_model).to(self.device)
        mine_model = PerFeatureMINENetwork().to(self.device)
        model.train()
        mine_model.train()
        
        global_weights = copy.deepcopy(global_model.state_dict()) # Used for FedProx penalty
        
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
                y_one_hot = F.one_hot(labels, num_classes=10).float()
                
                # 2. Per-Feature MINE Forward
                y_shuffled = y_one_hot[torch.randperm(y_one_hot.shape[0])]
                t_joint = mine_model(z, y_one_hot)      # [Batch, 128]
                t_marginal = mine_model(z, y_shuffled)  # [Batch, 128]
                
                # Donsker-Varadhan Lower Bound (Computed per-feature, then averaged for scalar loss)
                # t_joint.mean(dim=0) -> [128], t_marginal.exp().mean(dim=0).log() -> [128]
                feature_mi_estimates = torch.mean(t_joint, dim=0) - torch.log(torch.mean(torch.exp(t_marginal), dim=0) + 1e-8)
                avg_mi_estimate = torch.mean(feature_mi_estimates) # Scalar average of all 128 features
                
                # 3. Proximal / Regularization Penalty (FedProx style)
                proximal_term = 0.0
                for name, param in model.named_parameters():
                    proximal_term += torch.sum((param - global_weights[name].to(self.device)) ** 2)
                    
                # 4. Total Local Objective
                loss_mine = -avg_mi_estimate # Maximize MI
                loss_task = self.ce_criterion(logits, labels)
                
                # L_local = L_task + \lambda_MI * MI_term + \lambda_reg * R(w_local, w_global)
                loss_clf = loss_task - (LAMBDA_MI * avg_mi_estimate) + (LAMBDA_REG / 2.0) * proximal_term
                
                # Clear gradients for both optimizers
                optimizer_mine.zero_grad()
                optimizer_clf.zero_grad()
                
                # 5. Backpropagate (Compute Gradients)
                loss_mine.backward(retain_graph=True)
                loss_clf.backward()
                
                # 6. Apply Gradients (Update Weights)
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
def evaluate(model_state, dataloader, device):
    model = SimpleCNN()
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
    # 1. Min-Max Normalization of MI Scores across clients
    mi_min, mi_max = min(mi_scores), max(mi_scores)
    normalized_mi = [(mi - mi_min) / (mi_max - mi_min + MI_NORM_EPSILON) for mi in mi_scores]
        
    num_clients = len(local_weights)
    w_avg = copy.deepcopy(local_weights[0])
    layer_alpha_logs = {} 
    
    # 2. Layer-by-Layer Softmax Aggregation
    for key in w_avg.keys():
        layer_scores = []
        for i in range(num_clients):
            # Alpha^l_i = MI_norm_i * Importance^l_i
            imp = importance_dicts[i][key]
            layer_scores.append(normalized_mi[i] * imp)
            
        alphas = F.softmax(torch.tensor(layer_scores, dtype=torch.float32), dim=0).tolist()
        layer_alpha_logs[key] = alphas
        
        w_avg[key] = local_weights[0][key] * alphas[0]
        for i in range(1, num_clients):
            w_avg[key] += local_weights[i][key] * alphas[i]
            
    # Calculate average alpha across all layers for analytics logging
    avg_alphas = [sum(layer_alpha_logs[key][i] for key in w_avg.keys()) / len(w_avg.keys()) for i in range(num_clients)]
    return w_avg, normalized_mi, [a * 100 for a in avg_alphas]

# ==========================================
# Main Simulation
# ==========================================
def main():
    print("==============================================")
    print(f" MI-IAA: Mutual-Information Importance-Aware Aggregation")
    print(f" (Includes Per-Feature MINE & Proximal Penalty)")
    print("==============================================")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    global_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    global_test_loader = DataLoader(global_test_dataset, batch_size=128, shuffle=False)
    
    client_train_loaders, client_test_loaders = get_dirichlet_data_loaders(
        train_dataset, NUM_CLIENTS, DIRICHLET_ALPHA, BATCH_SIZE, MIN_SAMPLES_PER_CLIENT
    )
    
    global_model = SimpleCNN().to(DEVICE)
    
    history = {'global_acc': [], 'avg_client_acc': []}
    
    print("\n==============================================")
    print(f" Starting MI-IAA Training ({GLOBAL_ROUNDS} Rounds)")
    print("==============================================")
    
    for round_idx in range(1, GLOBAL_ROUNDS + 1):
        local_weights, local_losses, local_mis, local_importances = [], [], [], []
        client_personalized_accuracies, client_prox_losses = [], []
        
        print(f"\n| Round : {round_idx}/{GLOBAL_ROUNDS} |")
        
        for idx in range(NUM_CLIENTS):
            client = ClientUpdate(dataloader=client_train_loaders[idx], device=DEVICE, epochs=LOCAL_EPOCHS)
            weights, loss, mi_estimate, prox_loss, importances = client.train(global_model)
            
            client_personalized_accuracies.append(evaluate(weights, client_test_loaders[idx], DEVICE))
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
        global_test_acc = evaluate(global_model.state_dict(), global_test_loader, DEVICE)
        
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

        # Save metrics for plotting
        history['global_acc'].append(global_test_acc)
        history['avg_client_acc'].append(np.mean(client_personalized_accuracies))

    # --- Plotting Results ---
    print("\n[Simulation Complete] Generating accuracy plots...")
    rounds = list(range(1, GLOBAL_ROUNDS + 1))
    
    # 1. Communication vs Avg Client Personalized Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, history['avg_client_acc'], marker='o', color='b', label='Avg Client Acc')
    plt.title('Communication Rounds vs Avg Client Accuracy')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig('client_accuracy_plot.png')
    plt.close()
    
    # 2. Communication vs Global Model Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, history['global_acc'], marker='s', color='r', label='Global Model Acc')
    plt.title('Communication Rounds vs Global Accuracy')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig('global_accuracy_plot.png')
    plt.close()
    
    # 3. Communication vs Both (Combined)
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, history['avg_client_acc'], marker='o', color='b', label='Avg Client Acc')
    plt.plot(rounds, history['global_acc'], marker='s', color='r', label='Global Model Acc')
    plt.title('MI-IAA: Communication Rounds vs Federated Accuracies')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig('combined_accuracy_plot.png')
    plt.close()
    
    print("Plots successfully saved as PNG files in the project folder!")

if __name__ == '__main__':
    main()
