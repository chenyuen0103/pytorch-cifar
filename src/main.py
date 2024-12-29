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
from efficiency.function import set_seed
import csv
import time
import psutil

from dataset_loader import get_dataloader
from model_builder import get_model
from train_utils import test, log_metrics, SGDTrainer, DiveBatchTrainer
from adaptive_batch_size import update_batch_size




parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data_dir', default='../data', type=str)
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10','cifar100', 'imagenet'],
                    help='Dataset to train on (cifar10 or imagenet)')
parser.add_argument('--model', default='resnet18', type=str, help='Model to train')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train')
parser.add_argument('--algorithm', default='sgd', type=str, choices=['sgd', 'divebatch','adabatch'],)
parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
parser.add_argument('--batch_size', '-bs', default=128, type=int, help='Batch size')
parser.add_argument('--milestones', default=[60, 120, 160], type=int, nargs='+', help='Learning rate milestones')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--log_dir', default='../logs_multistep', type=str, help='Directory to save logs')
parser.add_argument('--checkpoint_dir', default='/projects/bdeb/chenyuen0103/checkpoint_multistep', type=str, help='Directory to save checkpoints')
parser.add_argument('--seed', default=1, type=int, help='Random seed')
args = parser.parse_args()

# num_workers = max(1, psutil.cpu_count(logical=False) - 1)
# num_workers = max(1, psutil.cpu_count(logical=False)//4 - 1)
num_workers = 1


# Initialize distributed process group
# if not dist.is_initialized():
#     dist.init_process_group(backend='nccl', init_method='env://')



best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

log_dir = os.path.join(args.log_dir, args.model, args.dataset)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)


log_file = os.path.join(log_dir, f'{args.algorithm}_lr{args.lr}_bs{args.batch_size}_s{args.seed}.csv')


fieldnames = [
    'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
    'learning_rate', 'batch_size', 'epoch_time', 'eval_time', 'memory_allocated', 'memory_reserved'
]






def get_last_log_row(log_file):
    """
    Efficiently retrieves the last row of a CSV log file.
    """
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)[-1] if os.path.getsize(log_file) > 0 else None



# Data
print(f'==> Preparing data for {args.dataset}..')

# train_sampler = DistributedSampler(get_dataloader(args.dataset, args.batch_size, is_train=True, return_dataset=True, num_workers=num_workers)[0])
trainloader, num_classes = get_dataloader(
    args.dataset, batch_size=args.batch_size, num_workers=num_workers, is_train=True, sampler=None
)
testloader, _ = get_dataloader(args.dataset, batch_size=args.batch_size*4, is_train=False)

# Model
print('==> Building model..')
net = get_model(model_name=args.model, num_classes=num_classes, pretrained=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    # net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank)
# else:
    # Use CPU
    # net = DDP(net)
# 

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


set_seed(args.seed)


checkpoint_dir = os.path.join(args.checkpoint_dir, args.model, args.dataset)
print(f"Checkpoint directory: {checkpoint_dir}")
checkpoint_file = f"{args.algorithm}_lr{args.lr}_bs{args.batch_size}_s{args.seed}_ckpt.pth"
# Check if the log file already exists
log_exists = os.path.exists(log_file)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.2)



file_mode = 'w'
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    latest_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')

    if os.path.exists(latest_checkpoint_path):
        # breakpoint()
        try:
            checkpoint = torch.load(latest_checkpoint_path)
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch'] + 1
            file_mode = 'a'
            print(f"Checkpoint loaded successfully: Resuming from epoch {start_epoch}, best accuracy: {best_acc}")

            # Read the log file to get the last epoch and best accuracy
            if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
                try:
                    last_row = get_last_log_row(log_file)
                    if last_row:
                        # Ensure start_epoch and best_acc are consistent
                        start_epoch = max(start_epoch, int(last_row['epoch']) + 1)
                        best_acc = max(best_acc, float(last_row['val_acc']))
                        print(f"Adjusted resume point: Start epoch {start_epoch}, Best accuracy {best_acc}")
                    else:
                        print("Warning: Log file is empty. Resuming from checkpoint defaults.")
                except (ValueError, KeyError) as e:
                    print(f"Error reading log file: {e}. Using checkpoint defaults.")
            else:
                print("Warning: Log file not found or empty. Resuming from checkpoint defaults.")
        except RuntimeError as e:
            print(f"Error loading checkpoint: {e}")
            args.resume = False


    elif log_exists and args.resume:
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            # if last epoch > args.epochs, then training is complete
            if int(list(reader)[-1]['epoch']) >= args.epochs-1:
                print("Training is already complete. Exiting...")
                exit()

    else:
        # No checkpoint or log file found, training from scratch
        print("No checkpoint found. Training from scratch.")
        args.resume = False




# breakpoint()
# Open the log file

with open(log_file, file_mode, newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    # Write the header only if the file is new
    if not log_exists or not args.resume:
        writer.writeheader()







if args.algorithm.lower() == 'sgd':
    trainer = SGDTrainer(net, optimizer, criterion, device)
elif args.algorithm.lower() == 'divebatch':
    trainer = DiveBatchTrainer(net, optimizer, criterion, device, resize_freq=5, max_batch_size=4096)
elif args.algorithm.lower() == 'adabatch':
    trainer = AdaBatchTrainer(net, optimizer, criterion, device, resize_freq=5, max_batch_size=4096)
else:
    raise ValueError(f"Unknown algorithm: {args.algorithm}")


batch_size = args.batch_size
min_batch_size = 32
max_batch_size = 4096
delta = 0.1

remaining_epochs = args.epochs - start_epoch
if remaining_epochs <= 0:
    print("No epochs remaining. Exiting...")
    exit()

for epoch in range(start_epoch, start_epoch + remaining_epochs):
    print(f"Epoch {epoch + 1}/{args.epochs}")
    # train_sampler.set_epoch(epoch)  # Ensure randomness across epochs

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
                sampler=None
            )[0]

    # dist.barrier() if dist.is_initialized() else None
    val_loss, val_acc, best_acc = test(net, optimizer, scheduler, testloader, criterion, device, epoch, progress_bar, best_acc, checkpoint_dir, checkpoint_file)
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


    if args.algorithm == 'divebatch':
        if epoch % args.resize_freq == 0:
            new_batch_size = trainer.resize_batch()
            if new_batch_size != args.batch_size:
                # Update the batch size argument
                args.batch_size = new_batch_size
                print(f"Updating DataLoader with new batch size: {args.batch_size}")

                # Recreate DataLoader with the new batch size
                trainloader = torch.utils.data.DataLoader(
                    trainset, 
                    batch_size=args.batch_size, 
                    shuffle=True, 
                    num_workers=2, 
                    pin_memory=True
                )
                print(f"Recreated trainloader with batch size: {args.batch_size}")
    scheduler.step()

# if dist.get_rank() == 0:
#     print("Training complete. Model saved to checkpoint.")