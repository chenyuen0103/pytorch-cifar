# main.py
#python main_imgnet.py -a resnet50 --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
# '../../../../scratch/bcxt/yian3/imagenet-1k'
#python main_imgnet.py -a resnet50 --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
# '../../../../scratch/bcxt/yian3/tiny_imagenet'
"""
python main_imgnet.py -a resnet18 -b 256 --accumulation-steps 4 --lr 0.0001 --dist-backend 'nccl' --multiprocessing-distributed --world-size 1
"""

import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import csv
import socket

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

from dataset_loader import CustomImageNetDataset, get_dataloader
from train_utils import test, DiveBatchTrainer, SGDTrainer
from backpack import extend
from utils import progress_bar

from datasets import load_from_disk

# Define model names
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Argument parser setup
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training (Single Node)')
parser.add_argument('--dataset', metavar='DIR', nargs='?', default='/projects/bdeb/chenyuen0103/tinyimagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--algorithm', default='sgd', type=str,
                    choices=['sgd', 'divebatch', 'adabatch'],
                    help='Training algorithm to use')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--adaptive_lr',  action='store_true', help='Rescale learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--accumulation-steps', default=1, type=int,
                    help='Number of steps to accumulate gradients over before performing an optimization step')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--log_dir', default='../logs', type=str,
                    help='Directory to save logs')
parser.add_argument('--checkpoint_dir', default='/projects/bdeb/chenyuen0103/checkpoint', type=str,
                    help='Directory to save checkpoints')

# New arguments for DiveBatchTrainer
parser.add_argument('--resize_freq', default=20, type=int,
                    help='Frequency (in epochs) to resize batch size')
parser.add_argument('--max_batch_size', default=256, type=int,
                    help='Maximum allowable batch size')
parser.add_argument('--delta', default=0.02, type=float,
                    help='Threshold delta for gradient diversity')
parser.add_argument('--seed', default=1, type=int,
                    help='Seed for random number generator')


best_acc1 = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        num_gpus = torch.backends.mps.device_count()
    else:
        device = torch.device("cpu")
        num_gpus = 0

    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs with DataParallel")
    elif num_gpus == 1:
        print("Using 1 GPU")
    else:
        print("Using CPU")

    # Create model
    if args.pretrained:
        print(f"=> using pre-trained model '{args.arch}'")
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print(f"=> creating model '{args.arch}'")
        model = models.__dict__[args.arch]()

    

    if args.algorithm == 'divebatch':
        def extend_all_modules(model):
            """
            Recursively extend all submodules of the model with BackPACK.
            """
            for module in model.modules():
                extend(module)
        extend_all_modules(model)

    model = model.to(device)


    if num_gpus > 1:
        model = nn.DataParallel(model)

    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Scale learning rate based on batch size and accumulation steps
    effective_bs = args.batch_size * args.accumulation_steps
    scale_factor = effective_bs / args.batch_size
    scaled_lr = args.lr * scale_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = scaled_lr

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Checkpoint setup
    

    # Create log directory and file
    log_dir = os.path.join(args.log_dir, args.arch, 'imagenet')
    os.makedirs(log_dir, exist_ok=True)

    if args.algorithm == 'sgd':
        log_file = os.path.join(log_dir, f'{args.algorithm}_lr{scaled_lr}_bs{effective_bs}_seed{args.seed}.csv')
    elif args.algorithm == 'divebatch':
        log_file = os.path.join(log_dir, f'{args.algorithm}_lr{args.lr}_bs{args.batch_size}_rf{args.resize_freq}_mbs{args.max_batch_size}_delta{args.delta}_seed{args.seed}.csv')
    elif args.algorithm == 'adabatch':
        log_file = os.path.join(log_dir, f'{args.algorithm}_lr{args.lr}_bs{args.batch_size}_seed{args.seed}.csv')
    else:
        raise ValueError(f"Unknown algorithm {args.algorithm}")

    if args.algorithm != 'sgd' and args.adaptive_lr:
        log_file = log_file.replace('.csv', '_rescale.csv')

    checkpoint_dir = os.path.join(args.checkpoint_dir, args.arch, 'imagenet')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = f"{args.algorithm}_lr{args.lr}_bs{args.batch_size}_ckpt.pth"

    log_exists = os.path.exists(log_file)
    batch_size = args.batch_size
    # Optionally resume from a checkpoint
    best_acc1 = 0
    if args.algorithm == 'sgd':
        checkpoint_file = f"{args.algorithm}_lr{args.lr}_bs{args.batch_size}_s{args.seed}_ckpt.pth"
    elif args.algorithm == 'divebatch':
        checkpoint_file = f"{args.algorithm}_lr{args.lr}_bs{args.batch_size}_rf{args.resize_freq}_mbs{args.max_batch_size}_delta{args.delta}_s{args.seed}_ckpt.pth"
    elif args.algorithm == 'adabatch':
        checkpoint_file = f"{args.algorithm}_lr{args.lr}_bs{args.batch_size}_rf{args.resize_freq}_mbs{args.max_batch_size}_s{args.seed}_ckpt.pth"
    latest_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    best_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file.replace('.pth', '_best.pth'))

    if args.adaptive_lr and args.algorithm != 'sgd':
        latest_checkpoint_path = latest_checkpoint_path.replace('.pth', '_rescale.pth')
        best_checkpoint_path = best_checkpoint_path.replace('.pth', '_rescale.pth')
        
    if args.resume:
        # breakpoint()
        if os.path.exists(latest_checkpoint_path):
            # breakpoint()
            print(f"=> loading checkpoint {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            batch_size = checkpoint['batch_size']
            print(f"=> loaded checkpoint (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{latest_checkpoint_path}'")

        row = None
        # load the batch size from the csv file
        epochs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        epochs.append(int(row['epoch'])) if row else None
                # if log file is not empty
                if row:
                    max_epoch_log = max(epochs)
                    
                    if max_epoch_log >= args.epochs-1:
                        # exit the program
                        print(f"Epoch {args.epochs} already exists in the log file. Exiting...")
                        exit()
            except ValueError:
                print("Error reading log file, starting from epoch 0")
                args.start_epoch = 0


    
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'epoch', 'train_loss', 'train_acc1',
            'val_loss', 'val_acc1', 'val_acc5',
            'best_acc1', 'learning_rate', 'batch_size',
            'epoch_time', 'eval_time', 'abs_time',
            'memory_allocated_mb', 'memory_reserved_mb', 'grad_diversity'
        ])
        if not log_exists or not args.resume:
            writer.writeheader()

    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        # Define dataset name based on the path or add an argument to specify
        dataset_name = 'imagenet'  # Modify if necessary

        if dataset_name == 'imagenet':
            train_dataset = get_imagenet_dataset(args, is_train=True)
            val_dataset = get_imagenet_dataset(args, is_train=False)
            num_classes = 1000
            # breakpoint()
        else:
            traindir = os.path.join(args.dataset, 'train')
            valdir = os.path.join(args.dataset, 'val')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

            val_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
            num_classes = 1000  # Ensure this matches your dataset

    # Data loaders
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    # Training and Testing
    if args.algorithm == 'divebatch':
        trainer_cls = DiveBatchTrainer
    # elif args.algorithm == 'adabatch':
    #     trainer_cls = DiveBatchTrainer
    else:
        trainer_cls = SGDTrainer
    trainer_args = {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "device": device,
        "num_classes": num_classes,
    }
    if args.algorithm == 'divebatch':
        trainer_args.update({
            "delta": args.delta,
            "resize_freq": args.resize_freq,
            "max_batch_size": args.max_batch_size,
        })
    # breakpoint()
    trainer = trainer_cls(**trainer_args)
    old_grad_diversity = 1.0 if args.algorithm == 'divebatch' else None
    # Training loop
    abs_start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        train_metrics = trainer.train_epoch(trainloader, epoch)
        # val_loss, val_acc, eval_time = test(epoch)
        # Evaluate on validation set
        val_start_time = time.time()
        val_loss, val_acc1, val_acc5 = validate(val_loader, model, criterion, args)
        eval_time = time.time() - val_start_time
        scheduler.step()

        if (epoch + 1) % args.resize_freq == 0 and args.algorithm in ['divebatch', 'adabatch'] and batch_size < args.max_batch_size:
            old_batch_size = batch_size
            if args.algorithm == 'divebatch':
                grad_diversity = train_metrics.get("grad_diversity")
                rescale_ratio = max((grad_diversity / old_grad_diversity),1)
            elif args.algorithm == 'adabatch':
                rescale_ratio = 2

            batch_size = int(min(old_batch_size * rescale_ratio, args.max_batch_size))
            
            if batch_size != old_batch_size:
                # Update the batch size argument
                # batch_size = new_batch_size
                # print(f"Updating DataLoader with new batch size: {batch_size}")
                # trainer.accum_steps = new_batch_size // args.batch_size
                # Recreate DataLoader with the new batch size
                print(f"Recreating trainloader with batch size: {batch_size}...")
                trainloader = torch.utils.data.DataLoader(
                    train_dataset, 
                    batch_size=batch_size,
                    shuffle=True, 
                    num_workers=1, 
                    pin_memory=True
                )

            if args.algorithm != 'sgd' and args.adaptive_lr:
                # rescale the learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= rescale_ratio
                

        # Update best accuracy
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        state = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
            'batch_size': batch_size,  # Save the current batch size
            'is_best': is_best
        }
        torch.save(state, latest_checkpoint_path)
        if is_best:
            torch.save(state, best_checkpoint_path)
        # breakpoint()
        # epoch_time = time.time() - epoch_start_time
        log_metrics(
            log_file=log_file,
            epoch=epoch + 1,
            train_loss=train_metrics["train_loss"],
            train_acc=train_metrics["train_accuracy"],
            val_loss=val_loss,
            val_acc1=val_acc1,
            val_acc5=val_acc5,
            lr=optimizer.param_groups[0]['lr'],
            batch_size=batch_size,
            epoch_time= val_start_time - epoch_start_time,
            eval_time=eval_time,
            abs_time = time.time() - abs_start_time,
            memory_allocated=torch.cuda.memory_allocated() if device == 'cuda' else 0,
            memory_reserved=torch.cuda.memory_reserved() if device == 'cuda' else 0,
            grad_diversity=train_metrics.get("grad_diversity"),
        )


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.to(device)
                target = target.to(device)

            elif torch.backends.mps.is_available():
                images = images.to('mps')
                target = target.to('mps')
            else:
                images = images.to("cpu")
            
            target = target.to(device)

            # Compute output
            output = model(images)
            loss = criterion(output, target)

            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i + 1)

    progress.display_summary()

    return losses.avg, top1.avg, top5.avg

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = f'{{name}} {{val{self.fmt}}} ({{avg{self.fmt}}})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        if self.summary_type == Summary.NONE:
            return ""
        elif self.summary_type == Summary.AVERAGE:
            return f'{self.name} {self.avg:.3f}'
        elif self.summary_type == Summary.SUM:
            return f'{self.name} {self.sum:.3f}'
        elif self.summary_type == Summary.COUNT:
            return f'{self.name} {self.count:.3f}'
        else:
            raise ValueError('Invalid summary type')

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters if meter.summary()]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# Add this function above or below your existing functions
def get_imagenet_dataset(args, is_train=True):
    """
    Loads the ImageNet dataset using Hugging Face's load_from_disk and wraps it with a custom PyTorch Dataset.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        is_train (bool): Whether to load the training set or validation set.
    
    Returns:
        dataset (CustomImageNetDataset): Wrapped PyTorch Dataset.
    """
    # Define ImageNet transformations
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225)),
        ])

    # Define the path for ImageNet
    imagenet_path = args.dataset  # Use the dataset path from arguments

    # Load ImageNet dataset from Hugging Face's `load_from_disk`
    hf_dataset = load_from_disk(imagenet_path)

    # Select appropriate split
    split = "train" if is_train else "valid"
    dataset_split = hf_dataset[split]

    # Wrap the Hugging Face dataset with PyTorch Dataset
    dataset = CustomImageNetDataset(dataset_split, transform)
    
    return dataset


def log_metrics(log_file, epoch, train_loss, train_acc, val_loss, val_acc1, val_acc5, lr, batch_size, epoch_time, eval_time, abs_time, memory_allocated, memory_reserved, grad_diversity=None, abs_start_time=None):
    memory_allocated_mb = memory_allocated / (1024 * 1024)
    memory_reserved_mb = memory_reserved / (1024 * 1024)
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc1','val_acc5',
            'learning_rate', 'batch_size', 'epoch_time', 'eval_time', 'abs_time', 'memory_allocated_mb', 'memory_reserved_mb', 'grad_diversity'
        ])
        writer.writerow({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc1': val_acc1,
            'val_acc5': val_acc5,
            'learning_rate': lr,
            'batch_size': batch_size,
            'epoch_time': epoch_time,  # Time taken for one epoch
            'eval_time': eval_time,  # Time taken for evaluation
            'abs_time': abs_time,  # Absolute time taken
            'memory_allocated_mb': round(memory_allocated_mb, 2),
            'memory_reserved_mb': round(memory_reserved_mb, 2),
            'grad_diversity': round(grad_diversity, 4) if grad_diversity is not None else None,
        })


if __name__ == '__main__':
    main()
