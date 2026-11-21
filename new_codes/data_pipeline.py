
"""
data_pipeline.py

Modular federated learning data pipeline.

Features:
1. Loads CIFAR10, MNIST, PathMNIST and BloodMNIST.
2. Creates a stratified global train/validation split.
3. Applies Dirichlet partitioning across clients.
4. Creates local 80/20 train/test splits.
5. Saves and loads reproducible partitions.
6. Keeps the original test dataset untouched.
"""

import os
import argparse
import random
import numpy as np
import torch

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


SEED = 42


# ==========================================
# 1. Reproducibility
# ==========================================

def set_seed(seed=SEED):
    """Set all relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# 2. Dataset Loading
# ==========================================

def get_dataset(name, root='./data'):
    name = name.lower()

    if name == 'cifar10':

        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)
            )
        ])

        train = datasets.CIFAR10(
            root, train=True, download=True, transform=tf
        )

        test = datasets.CIFAR10(
            root, train=False, download=True, transform=tf
        )

    elif name == 'mnist':

        tf = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,) * 3,
                (0.3081,) * 3
            )
        ])

        train = datasets.MNIST(
            root, train=True, download=True, transform=tf
        )

        test = datasets.MNIST(
            root, train=False, download=True, transform=tf
        )

    elif name in ('pathmnist', 'bloodmnist'):

        import medmnist
        from medmnist import INFO

        tf = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])

        info = INFO[name]

        DataClass = getattr(
            medmnist,
            info['python_class']
        )

        train = DataClass(
            split='train',
            download=True,
            transform=tf,
            root=root
        )

        test = DataClass(
            split='test',
            download=True,
            transform=tf,
            root=root
        )

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train, test


def get_targets(dataset, name):
    """Normalize label access across datasets."""

    name = name.lower()

    if name in ('pathmnist', 'bloodmnist'):
        return np.asarray(dataset.labels).reshape(-1)

    return np.asarray(dataset.targets).reshape(-1)


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
# 3. Stratified Train / Validation Split
# ==========================================

def stratified_train_val_split(
    targets,
    val_fraction=0.10,
    seed=SEED
):
    """
    Split the original training dataset into:

        90% federated training data
        10% global validation data

    Class proportions are approximately preserved.
    """

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")

    targets = np.asarray(targets).reshape(-1)

    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []

    for class_id in np.unique(targets):

        class_indices = np.where(
            targets == class_id
        )[0].copy()

        rng.shuffle(class_indices)

        if len(class_indices) < 2:
            raise ValueError(
                f"Class {class_id} has fewer than 2 samples."
            )

        val_count = max(
            1,
            int(round(len(class_indices) * val_fraction))
        )

        val_count = min(
            val_count,
            len(class_indices) - 1
        )

        val_indices.extend(
            class_indices[:val_count]
        )

        train_indices.extend(
            class_indices[val_count:]
        )

    train_indices = np.asarray(
        train_indices,
        dtype=np.int64
    )

    val_indices = np.asarray(
        val_indices,
        dtype=np.int64
    )

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


# ==========================================
# 4. Dirichlet Partitioning
# ==========================================

def partition_dirichlet(
    targets,
    num_clients,
    alpha,
    min_samples
):
    """Partition training targets using Dirichlet distribution."""

    targets = np.asarray(targets).reshape(-1)

    if targets.size == 0:
        raise ValueError("targets must contain samples.")

    if num_clients < 1:
        raise ValueError("num_clients must be at least 1.")

    if alpha <= 0:
        raise ValueError("alpha must be greater than 0.")

    if min_samples < 1:
        raise ValueError("min_samples must be at least 1.")

    if num_clients * min_samples > len(targets):
        raise ValueError(
            f"Impossible partition constraint: "
            f"{num_clients} clients x {min_samples} samples "
            f"exceeds {len(targets)} available samples."
        )

    classes = np.unique(targets)

    class_indices = [
        np.where(targets == c)[0].copy()
        for c in classes
    ]

    print(
        f"\n[Data Partitioning] "
        f"Clients={num_clients}, Alpha={alpha}"
    )

    attempts = 0

    while True:

        attempts += 1

        client_indices = {
            i: [] for i in range(num_clients)
        }

        for indices in class_indices:

            indices = indices.copy()

            proportions = np.random.dirichlet(
                np.repeat(alpha, num_clients)
            )

            samples_per_client = (
                proportions * len(indices)
            ).astype(int)

            difference = (
                len(indices) - samples_per_client.sum()
            )

            for _ in range(difference):
                samples_per_client[
                    np.random.randint(num_clients)
                ] += 1

            np.random.shuffle(indices)

            current_idx = 0

            for client_id in range(num_clients):

                count = samples_per_client[client_id]

                client_indices[client_id].extend(
                    indices[
                        current_idx:current_idx + count
                    ]
                )

                current_idx += count

        min_found = min(
            len(client_indices[i])
            for i in range(num_clients)
        )

        if min_found >= min_samples:

            print(
                f"Partition successful! "
                f"Minimum samples={min_found}, "
                f"Attempts={attempts}"
            )

            break

        if attempts % 10 == 0:

            print(
                f"Attempt {attempts}: "
                f"minimum samples={min_found}. Retrying..."
            )

    # Local 80/20 train/test split.

    partitions = {}

    print("\n[Client Data Distribution]")

    for i in range(num_clients):

        np.random.shuffle(client_indices[i])

        total_len = len(client_indices[i])

        train_len = int(0.8 * total_len)

        train_idx = np.asarray(
            client_indices[i][:train_len],
            dtype=np.int64
        )

        test_idx = np.asarray(
            client_indices[i][train_len:],
            dtype=np.int64
        )

        partitions[i] = {
            "train_idx": train_idx,
            "test_idx": test_idx
        }

        labels = targets[client_indices[i]]

        unique, counts = np.unique(
            labels,
            return_counts=True
        )

        distribution = ", ".join(
            f"{u}:{c}"
            for u, c in zip(unique, counts)
        )

        print(
            f"Client {i + 1:2d} | "
            f"Total={total_len:5d} | "
            f"Classes={distribution}"
        )

    return partitions


# ==========================================
# 5. Existing Partition Save / Load
# ==========================================

def partition_path(
    dataset_name,
    out_dir='./partitions',
    num_clients=None,
    alpha=None,
    min_samples=None
):

    os.makedirs(out_dir, exist_ok=True)

    dataset_name = dataset_name.lower()

    if (
        num_clients is None
        or alpha is None
        or min_samples is None
    ):

        return os.path.join(
            out_dir,
            f"partitions_{dataset_name}.pt"
        )

    return os.path.join(
        out_dir,
        f"partitions_{dataset_name}_clients{num_clients}"
        f"_alpha{alpha:g}_min{min_samples}.pt"
    )


def save_partitions(
    partitions,
    dataset_name,
    out_dir='./partitions',
    num_clients=None,
    alpha=None,
    min_samples=None
):

    path = partition_path(
        dataset_name,
        out_dir,
        num_clients,
        alpha,
        min_samples
    )

    torch.save(partitions, path)

    print(f"Saved partitions to {path}")


def load_partitions(
    dataset_name,
    out_dir='./partitions',
    num_clients=None,
    alpha=None,
    min_samples=None
):

    path = partition_path(
        dataset_name,
        out_dir,
        num_clients,
        alpha,
        min_samples
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"No partition file found: {path}"
        )

    return torch.load(
        path,
        weights_only=False
    )


def ensure_partitions(
    dataset_name,
    num_clients=30,
    alpha=0.1,
    min_samples=250,
    out_dir='./partitions'
):
    """Backward-compatible old partition function."""

    path = partition_path(
        dataset_name,
        out_dir,
        num_clients,
        alpha,
        min_samples
    )

    if os.path.exists(path):

        return load_partitions(
            dataset_name,
            out_dir,
            num_clients,
            alpha,
            min_samples
        )

    print(
        f"No partitions found for {dataset_name}. "
        "Generating..."
    )

    set_seed()

    train_dataset, _ = get_dataset(dataset_name)

    targets = get_targets(
        train_dataset,
        dataset_name
    )

    partitions = partition_dirichlet(
        targets,
        num_clients,
        alpha,
        min_samples
    )

    save_partitions(
        partitions,
        dataset_name,
        out_dir,
        num_clients,
        alpha,
        min_samples
    )

    return partitions


# ==========================================
# 6. NEW: Federated Data Preparation
# ==========================================

def get_federated_data(
    dataset_name,
    num_clients=30,
    alpha=0.1,
    min_samples=250,
    val_fraction=0.10,
    seed=SEED,
    out_dir='./partitions'
):
    """
    Prepare federated training, global validation,
    and untouched final test datasets.

    Returns:
        federated_train_dataset
        validation_dataset
        test_dataset
        partitions

    Client partition indices are relative to
    federated_train_dataset.
    """

    os.makedirs(out_dir, exist_ok=True)

    dataset_name = dataset_name.lower()

    split_path = os.path.join(
        out_dir,
        f"federated_{dataset_name}"
        f"_clients{num_clients}"
        f"_alpha{alpha:g}"
        f"_min{min_samples}"
        f"_val{val_fraction:g}"
        f"_seed{seed}.pt"
    )

    # Load original datasets.

    full_train_dataset, test_dataset = get_dataset(
        dataset_name
    )

    all_targets = get_targets(
        full_train_dataset,
        dataset_name
    )

    # Load saved split if available.

    if os.path.exists(split_path):

        saved = torch.load(
            split_path,
            map_location='cpu',
            weights_only=False
        )

        train_indices = saved["train_indices"]
        val_indices = saved["val_indices"]
        partitions = saved["partitions"]

        print(
            f"\nLoaded saved federated split: {split_path}"
        )

    else:

        print(
            "\nCreating stratified global train/validation split..."
        )

        train_indices, val_indices = (
            stratified_train_val_split(
                all_targets,
                val_fraction=val_fraction,
                seed=seed
            )
        )

        # Get labels only from the federated training subset.

        train_targets = all_targets[train_indices]

        # Generate client partitions.

        set_seed(seed)

        partitions = partition_dirichlet(
            train_targets,
            num_clients=num_clients,
            alpha=alpha,
            min_samples=min_samples
        )

        # Save all indices and partitions together.

        torch.save(
            {
                "train_indices": train_indices,
                "val_indices": val_indices,
                "partitions": partitions
            },
            split_path
        )

        print(
            f"Saved federated split: {split_path}"
        )

    # Create dataset subsets.

    federated_train_dataset = Subset(
        full_train_dataset,
        train_indices
    )

    validation_dataset = Subset(
        full_train_dataset,
        val_indices
    )

    print("\n========== FEDERATED DATA SUMMARY ==========")

    print(
        f"Original training samples : "
        f"{len(full_train_dataset)}"
    )

    print(
        f"Federated training samples: "
        f"{len(federated_train_dataset)}"
    )

    print(
        f"Global validation samples : "
        f"{len(validation_dataset)}"
    )

    print(
        f"Final test samples         : "
        f"{len(test_dataset)}"
    )

    print(
        f"Number of clients          : "
        f"{len(partitions)}"
    )

    print("============================================")

    return (
        federated_train_dataset,
        validation_dataset,
        test_dataset,
        partitions
    )


# ==========================================
# 7. Build Client DataLoaders
# ==========================================

def build_client_loaders(
    dataset,
    partitions,
    batch_size
):

    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1."
        )

    if not partitions:
        raise ValueError(
            "Partitions must not be empty."
        )

    train_loaders = []
    test_loaders = []

    for i in sorted(partitions.keys()):

        train_idx = partitions[i]["train_idx"]
        test_idx = partitions[i]["test_idx"]

        train_loaders.append(
            DataLoader(
                Subset(dataset, train_idx),
                batch_size=batch_size,
                shuffle=True
            )
        )

        test_loaders.append(
            DataLoader(
                Subset(dataset, test_idx),
                batch_size=batch_size,
                shuffle=False
            )
        )

    return train_loaders, test_loaders


# ==========================================
# 8. CLI
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="Prepare federated learning datasets."
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
        '--num_clients',
        type=int,
        default=30
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.1
    )

    parser.add_argument(
        '--min_samples',
        type=int,
        default=250
    )

    parser.add_argument(
        '--val_fraction',
        type=float,
        default=0.10
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=SEED
    )

    parser.add_argument(
        '--out_dir',
        type=str,
        default='./partitions'
    )

    args = parser.parse_args()

    set_seed(args.seed)

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        partitions
    ) = get_federated_data(
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        alpha=args.alpha,
        min_samples=args.min_samples,
        val_fraction=args.val_fraction,
        seed=args.seed,
        out_dir=args.out_dir
    )

    print("\nData preparation completed successfully.")


if __name__ == '__main__':
    main()
