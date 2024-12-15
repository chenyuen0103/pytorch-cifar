'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import csv
import time
import psutil

from dataset_loader import get_dataloader
from model_builder import get_model
from train_utils import train, test, log_metrics, SGDTrainer, DiveBatchTrainer
from adaptive_batch_size import update_batch_size, compute_gradient_diversity

import socket

def find_free_port():
    """Find an available port dynamically."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# Set MASTER_PORT dynamically if not already set
if "MASTER_PORT" not in os.environ:
    os.environ["MASTER_PORT"] = str(find_free_port())
num_gpus = torch.cuda.device_count()

os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", str(num_gpus))
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
if "MASTER_PORT" not in os.environ:
    os.environ["MASTER_PORT"] = str(find_free_port())
if dist.is_available() and dist.is_initialized():
    num_workers = max(os.cpu_count() // dist.get_world_size(), 1)
else:
    num_workers = os.cpu_count() // 2

print(f"RANK: {os.environ['RANK']}")
print(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}")
print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
print(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
print(f"NUM_WORKERS: {num_workers}")

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data_dir', default='../data', type=str)
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10','cifar100', 'imagenet'],
                    help='Dataset to train on (cifar10 or imagenet)')
parser.add_argument('--model', default='resnet18', type=str, help='Model to train')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train')
parser.add_argument('--algorithm', default='sgd', type=str, choices=['sgd', 'adam'],)
parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
parser.add_argument('--batch_size', '-bs', default=128, type=int, help='Batch size')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--log_dir', default='../logs', type=str, help='Directory to save logs')
parser.add_argument('--checkpoint_dir', default='/projects/bdeb/chenyuen0103/checkpoint', type=str, help='Directory to save checkpoints')
parser.add_argument('--local_rank', type=int, default=0,  help='Local rank for distributed training')
args = parser.parse_args()




# Initialize distributed process group
if not dist.is_initialized():
    dist.init_process_group(backend='nccl', init_method='env://')



best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

log_dir = os.path.join(args.log_dir, args.model, args.dataset)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


log_file = os.path.join(log_dir, f'{args.algorithm}_lr{args.lr}_bs{args.batch_size}.csv')


fieldnames = [
    'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
    'learning_rate', 'batch_size', 'epoch_time', 'eval_time', 'memory_allocated', 'memory_reserved'
]

# Check if the log file already exists
log_exists = os.path.exists(log_file)
file_mode = 'w' if not args.resume else 'a'


# breakpoint()
# Open the log file
if dist.get_rank() == 0:
    with open(log_file, file_mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # Write the header only if the file is new
        if not log_exists or not args.resume:
            writer.writeheader()




# Data
print(f'==> Preparing data for {args.dataset}..')

train_sampler = DistributedSampler(get_dataloader(args.dataset, args.batch_size, is_train=True, return_dataset=True, num_workers=num_workers)[0])
trainloader, num_classes = get_dataloader(
    args.dataset, batch_size=args.batch_size, is_train=True, sampler=train_sampler
)
testloader, _ = get_dataloader(args.dataset, batch_size=args.batch_size, is_train=False)

# Model
print('==> Building model..')
net = get_model(model_name=args.model, num_classes=num_classes, pretrained=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    # Move model to the current device
    torch.cuda.set_device(args.local_rank)
    net = net.to(torch.device(f'cuda:{args.local_rank}'))

    # Wrap the model with DistributedDataParallel
    net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank)
else:
    # Use CPU
    net = DDP(net)


# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()





checkpoint_dir = os.path.join(args.checkpoint_dir, args.model, args.dataset)
print(f"Checkpoint directory: {checkpoint_dir}")

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')

    assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
    checkpoint_file = f"{args.algorithm}_lr{args.lr}_bs{args.batch_size}_ckpt.pth"
    checkpoint = torch.load(f'{checkpoint_dir}/{checkpoint_file}')
    net.load_state_dict(checkpoint['net'])
    
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

if args.algorithm.lower() == 'sgd':
    trainer = SGDTrainer(net, optimizer, criterion, device)
elif args.algorithm.lower() == 'divebatch':
    trainer = DiveBatchTrainer(net, optimizer, criterion, device, resize_freq=5, max_batch_size=4096)
elif args.algorithm.lower() == 'adabatch':
    trainer = AdaBatchTrainer(net, optimizer, criterion, device, resize_freq=5, max_batch_size=4096)
else:
    raise ValueError(f"Unknown algorithm: {args.algorithm}")



min_batch_size = 32
max_batch_size = 4096
delta = 0.1

remaining_epochs = args.epochs - start_epoch
if remaining_epochs <= 0:
    print("No epochs remaining. Exiting...")
    exit()

for epoch in range(start_epoch, start_epoch + remaining_epochs):
    print(f"Epoch {epoch + 1}/{args.epochs}")
    train_sampler.set_epoch(epoch)  # Ensure randomness across epochs

    start_time = time.time()
    train_metrics = trainer.train_epoch(net, trainloader, optimizer, criterion, device, epoch, progress_bar)
    end_time = time.time()
    train_time = end_time - start_time

    # Adjust batch size dynamically if applicable
    if "grad_diversity" in train_metrics:  # Check if gradient diversity is available
        gradient_diversity = train_metrics["grad_diversity"]
        new_batch_size = update_batch_size(
            batch_size,
            gradient_diversity,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            delta=delta
        )
        if new_batch_size != batch_size:
            print(f"Adjusting batch size: {batch_size} -> {new_batch_size}")
            batch_size = new_batch_size
            trainloader = get_dataloader(
                args.dataset,
                batch_size=batch_size,
                is_train=True,
                sampler=train_sampler
            )[0]

    dist.barrier() if dist.is_initialized() else None
    val_loss, val_acc, best_acc = test(net, testloader, criterion, device, epoch, progress_bar, best_acc, args.checkpoint_dir)
    eval_time = time.time() - end_time
    # Log metrics
    # Log metrics
    log_metrics(
        log_file,
        epoch,
        train_loss=train_metrics["train_loss"],
        train_acc=train_metrics["train_accuracy"],
        val_loss=val_loss,
        val_acc=val_acc,
        lr=scheduler.get_last_lr()[0],
        batch_size=batch_size,
        epoch_time=train_time,
        eval_time=eval_time,
        memory_allocated=torch.cuda.memory_allocated() if device == 'cuda' else 0,
        memory_reserved=torch.cuda.memory_reserved() if device == 'cuda' else 0,
        grad_diversity=gradient_diversity if "grad_diversity" in train_metrics else None
    )
    scheduler.step()

if dist.get_rank() == 0:
    print("Training complete. Model saved to checkpoint.")