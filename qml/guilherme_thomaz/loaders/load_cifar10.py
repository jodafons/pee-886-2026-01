from typing import Tuple
import torch
import torchvision.transforms as transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader

def load_cifar10_iid(
    partition_id: int, num_partitions: int, batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """Load and partition the dataset for federated learning."""

    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% validation
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # Define preprocessing: convert CIFAR-10 images to tensor and normalize pixel values to mean=0.5, std=0.5
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = torch.stack([pytorch_transforms(img) for img in batch["img"]])
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    valloader = DataLoader(
        partition_train_test["test"], batch_size=batch_size, num_workers=0
    )  # validation split
    return trainloader, valloader

