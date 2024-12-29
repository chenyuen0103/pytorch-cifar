import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from datasets import load_from_disk
from PIL import Image
import os
import psutil


def get_num_workers():
    total_cores = psutil.cpu_count(logical=False)
    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    workers_per_gpu = max(1, total_cores // num_gpus)
    return workers_per_gpu


class CustomImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        data = self.dataset[idx]
        image = data["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        label = data["label"]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)
    


def get_dataloader(dataset_name, batch_size, num_workers=None, is_train=True, return_dataset=False, sampler=None):
    """
    Load dataset and create DataLoader.

    Args:
        dataset_name (str): Name of the dataset ('cifar10', 'cifar100', 'imagenet').
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of worker threads.
        is_train (bool): Whether to load training or testing data.

    Returns:
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        num_classes (int): Number of classes in the dataset.
    """

    if num_workers is None:
        allocated_cores = len(psutil.Process().cpu_affinity())  # Cores assigned to this process
        num_workers = max(1, allocated_cores - 1)  # Reserve 1 core for the main process
        print(f"Using {num_workers} DataLoader workers (allocated cores: {allocated_cores}).")

    if dataset_name in ['cifar10', 'cifar100']:
        # Define different transformations for train and test
        if is_train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        # Load CIFAR dataset
        if dataset_name == 'cifar10':
            dataset = datasets.CIFAR10(root='../data', train=is_train, download=True, transform=transform)
            num_classes = 10
        elif dataset_name == 'cifar100':
            dataset = datasets.CIFAR100(root='../data', train=is_train, download=True, transform=transform)
            num_classes = 100

    elif dataset_name == 'imagenet':
        # Define ImageNet transformations
        if is_train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        # Define the path for ImageNet
        imagenet_path = '/projects/bdeb/chenyuen0103/imagenet1k'

        # Load ImageNet dataset from Hugging Face's `load_from_disk`
        hf_dataset = load_from_disk(imagenet_path)

        # Select appropriate split
        split = "train" if is_train else "validation"
        dataset_split = hf_dataset[split]

        # Wrap the Hugging Face dataset with PyTorch Dataset
        dataset = CustomImageNetDataset(dataset_split, transform)
        num_classes = 1000

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if return_dataset:
        return dataset, num_classes
    else:
        if sampler:
            dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)

        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)
        # print(f"Batch size: {batch_size}")
        # print(f"Sampler: {sampler}")
        # print(f"Shuffle: {is_train}")

        return dataloader, num_classes
