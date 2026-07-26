"""
data_pipeline.py
Single modular file: loads datasets on demand, applies Dirichlet partitioning,
saves/loads client index partitions so all training scripts share the same split.
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SEED = 42


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ==========================================
# 1. Dataset Loading (on demand)
# ==========================================
def get_dataset(name, root='./data'):
    name = name.lower()

    if name == 'cifar10':
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train = datasets.CIFAR10(root, train=True, download=True, transform=tf)
        test = datasets.CIFAR10(root, train=False, download=True, transform=tf)

    elif name == 'mnist':
        tf = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,) * 3, (0.3081,) * 3)
        ])
        train = datasets.MNIST(root, train=True, download=True, transform=tf)
        test = datasets.MNIST(root, train=False, download=True, transform=tf)

    elif name in ('pathmnist', 'bloodmnist'):
        import medmnist
        from medmnist import INFO
        tf = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        info = INFO[name]
        DataClass = getattr(medmnist, info['python_class'])
        train = DataClass(split='train', download=True, transform=tf, root=root)
        test = DataClass(split='test', download=True, transform=tf, root=root)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train, test


def get_targets(dataset, name):
    """Normalize label access across torchvision vs medmnist datasets."""
    name = name.lower()
    if name in ('pathmnist', 'bloodmnist'):
        return np.array(dataset.labels).reshape(-1)
    return np.array(dataset.targets)


def get_num_classes(name):
    name = name.lower()
    if name in ('cifar10', 'mnist'):
        return 10
    if name == 'pathmnist':
        return 9
    if name == 'bloodmnist':
        return 8
    raise ValueError(f"Unknown dataset: {name}")


# ==========================================
# 2. Dirichlet Partitioning (dataset-agnostic)
# ==========================================
def partition_dirichlet(targets, num_clients, alpha, min_samples):
    num_classes = len(np.unique(targets))
    class_indices = [np.where(targets == i)[0] for i in range(num_classes)]

    print(f"\n[Data Partitioning] Splitting data for {num_clients} clients (Alpha={alpha}).")
    print(f"Enforcing minimum of {min_samples} samples per client using rejection sampling...")

    attempts = 0
    while True:
        attempts += 1
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
                n = num_samples_per_client[i]
                client_indices[i].extend(class_indices[c][current_idx: current_idx + n])
                current_idx += n

        min_client_samples = min(len(client_indices[i]) for i in range(num_clients))
        if min_client_samples >= min_samples:
            print(f"Success! Minimum samples found: {min_client_samples}. (Took {attempts} attempt(s))")
            break
        elif attempts % 10 == 0:
            print(f"Attempt {attempts}... Constraint failed (Minimum found: {min_client_samples}). Retrying...")

    # 80/20 train/test split per client, with reporting
    partitions = {}
    print("\n[Data Analytics] Splitting into 80% Train, 20% Local Test...")
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
        total_len = len(client_indices[i])
        train_len = int(0.8 * total_len)

        train_idx = client_indices[i][:train_len]
        test_idx = client_indices[i][train_len:]
        partitions[i] = {"train_idx": train_idx, "test_idx": test_idx}

        labels = targets[client_indices[i]]
        unique, counts = np.unique(labels, return_counts=True)
        class_dist_str = ", ".join(f"{u}:{c}" for u, c in zip(unique, counts))
        print(f"Client {i+1:2d} | Total Data: {total_len:4d} | Classes -> {class_dist_str}")

    return partitions


# ==========================================
# 3. Save / Load partitions
# ==========================================
def partition_path(dataset_name, out_dir='./partitions'):
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"partitions_{dataset_name.lower()}.pt")


def save_partitions(partitions, dataset_name, out_dir='./partitions'):
    path = partition_path(dataset_name, out_dir)
    torch.save(partitions, path)
    print(f"Saved partitions to {path}")


def load_partitions(dataset_name, out_dir='./partitions'):
    path = partition_path(dataset_name, out_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No partition file found at {path}. Run: python data_pipeline.py --dataset {dataset_name}"
        )
    # weights_only=False: this file stores plain Python dicts of numpy index arrays,
    # not model weights, and PyTorch >=2.6 defaults to weights_only=True which
    # blocks numpy globals. Safe here since we generated this file ourselves locally.
    return torch.load(path, weights_only=False)


def ensure_partitions(dataset_name, num_clients=30, alpha=0.1, min_samples=250, out_dir='./partitions'):
    """Load partitions if they exist, else generate them on the fly (used by algo scripts)."""
    path = partition_path(dataset_name, out_dir)
    if os.path.exists(path):
        return load_partitions(dataset_name, out_dir)

    print(f"No partitions found for '{dataset_name}', generating now...")
    set_seed()
    train_dataset, _ = get_dataset(dataset_name)
    targets = get_targets(train_dataset, dataset_name)
    partitions = partition_dirichlet(targets, num_clients, alpha, min_samples)
    save_partitions(partitions, dataset_name, out_dir)
    return partitions


def build_client_loaders(dataset, partitions, batch_size):
    """Turn saved index partitions into DataLoaders for a given dataset object."""
    train_loaders, test_loaders = [], []
    for i in sorted(partitions.keys()):
        train_idx = partitions[i]["train_idx"]
        test_idx = partitions[i]["test_idx"]
        train_loaders.append(DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False))
    return train_loaders, test_loaders


# ==========================================
# 4. CLI: run once per dataset to generate partitions
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Prepare Dirichlet-partitioned federated dataset splits. "
                    "(Note: algo scripts can also call ensure_partitions() to auto-generate on first use.)"
    )
    parser.add_argument('--dataset', required=True,
                         choices=['cifar10', 'mnist', 'pathmnist', 'bloodmnist'])
    parser.add_argument('--num_clients', type=int, default=30)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--min_samples', type=int, default=250)
    parser.add_argument('--out_dir', type=str, default='./partitions')
    args = parser.parse_args()

    set_seed()

    train_dataset, _ = get_dataset(args.dataset)
    targets = get_targets(train_dataset, args.dataset)

    partitions = partition_dirichlet(targets, args.num_clients, args.alpha, args.min_samples)
    save_partitions(partitions, args.dataset, args.out_dir)


if __name__ == '__main__':
    main()
