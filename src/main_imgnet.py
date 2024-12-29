#python main_imgnet.py -a resnet50 --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
# '../../../scratch/bcxt/yian3/imagenet-1k'
import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import csv

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
import socket
from dataset_loader import CustomImageNetDataset, get_dataloader
from train_utils import log_metrics, test, DiveBatchTrainer, SGDTrainer
import time
import csv
from backpack import extend
from utils import progress_bar



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', metavar='DIR', nargs='?', default='/projects/bdeb/chenyuen0103/tinyimagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--algorithm', default='sgd', type=str,
                    choices=['sgd', 'divebatch', 'adam'],
                    help='Training algorithm to use')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
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
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                     help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--log_dir', default='../logs', type=str,
                    help='Directory to save logs')
parser.add_argument('--checkpoint_dir', default='/projects/bdeb/chenyuen0103/checkpoint', type=str,
                    help='Directory to save checkpoints')


best_acc1 = 0

def find_free_port():
    """Find an available port dynamically."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to an available port
        return s.getsockname()[1]  # Return the port number


def main():
    args = parser.parse_args()
    args.dist_url = f'tcp://{os.environ.get("MASTER_ADDR", "127.0.0.1")}:{os.environ.get("MASTER_PORT", "23456")}'

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

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
        if ngpus_per_node == 1 and args.dist_backend == "nccl":
            warnings.warn("nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"],0)
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                # args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # Scale learning rate
    effective_bs = args.batch_size * args.accumulation_steps * args.world_size
    scale_factor = effective_bs / args.batch_size
    scaled_lr = args.lr * scale_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = scaled_lr

    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.arch, 'imagenet')
    checkpoint_file = f"{args.algorithm}_lr{args.lr}_bs{args.batch_size}_ckpt.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(checkpoint_path)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(checkpoint_path, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            batch_size = checkpoint['batch_size']
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))


    # Create checkpoint directory
    if args.rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Create log directory and file

    log_dir = os.path.join(args.log_dir, args.arch, 'imagenet')

    if args.rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        if args.algorithm == 'sgd':
            log_file = os.path.join(log_dir, f'{args.algorithm}_lr{scaled_lr}_bs{effective_bs}.csv')
        elif args.algorithm == 'divebatch':
            log_file = os.path.join(log_dir, f'{args.algorithm}_lr{args.lr}_bs{args.batch_size}_rf{args.resize_freq}_mbs{args.max_batch_size}_delta{args.delta}_s{args.seed}.csv')
        elif args.algorithm == 'adabatch':
            log_file = os.path.join(log_dir, f'{args.algorithm}_lr{args.lr}_bs{args.batch_size}_rf{args.resize_freq}_mbs{args.max_batch_size}_s{args.seed}.csv')

        if args.adaptive_lr:
            log_file = log_file.replace('.csv', '_rescale.csv')
        log_exists = os.path.exists(log_file)
        with open(log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'epoch', 'train_loss', 'train_acc1', 'train_acc5',
                'val_loss', 'val_acc1', 'val_acc5',
                'best_acc1', 'learning_rate', 'batch_size',
                'epoch_time', 'eval_time',
                'memory_allocated_mb', 'memory_reserved_mb', 'grad_diversity'
            ])
            if not log_exists or args.start_epoch == 0:
                writer.writeheader()


    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
    # Check if you're using imagenet
    # You can add an argument to specify the dataset if needed
        dataset_name = 'imagenet'  # Set this based on your requirement

    if dataset_name == 'imagenet':
        train_dataset = get_imagenet_dataset(args, is_train=True)
        val_dataset = get_imagenet_dataset(args, is_train=False)
        num_classes = 1000
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


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    # Instantiate the appropriate Trainer
    if args.algorithm == 'sgd':
        trainer = SGDTrainer(model, optimizer, criterion, device)
    elif args.algorithm == 'divebatch':
        trainer = DiveBatchTrainer(
            model, optimizer, criterion, device,
            resize_freq=args.resize_freq,
            max_batch_size=args.max_batch_size,
            delta=args.delta,
            dataset_size=len(train_loader.dataset)
        )

    old_grad_diversity = 1.0 if args.algorithm == 'divebatch' else None 

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_metrics = trainer.train_epoch(train_loader, epoch)

        if args.algorithm == 'divebatch':
            new_batch_size = trainer.suggest_new_batch_size(
                current_batch_size=args.batch_size,
                grad_diversity=train_metrics.get("grad_diversity", 1.0),
                args=args
            )

            # Broadcast the new batch size from rank 0 to all other ranks
            new_batch_size_tensor = torch.tensor(new_batch_size, dtype=torch.int)
            if args.distributed:
                # Only rank 0 computes the new batch size
                if args.rank != 0:
                    new_batch_size_tensor = torch.tensor(0, dtype=torch.int)
                dist.broadcast(new_batch_size_tensor, src=0)
                if args.rank != 0:
                    new_batch_size = new_batch_size_tensor.item()
            
            # Ensure all ranks have the updated batch size
            if new_batch_size != args.batch_size:
                print(f"Rank {args.rank}: Updating batch size from {args.batch_size} to {new_batch_size}")
                args.batch_size = new_batch_size

                # Recreate the DataLoader with the new batch size
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                    num_workers=args.workers, pin_memory=True, sampler=train_sampler
                )
        val_start_time = time.time()
        # evaluate on validation set
        val_loss, val_acc1, val_acc5 = validate(val_loader, model, criterion, args)
        eval_time = time.time() - val_start_time
        
        
        
        # remember best acc@1 and save checkpoint
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):

            checkpoint_filename = f'checkpoint_epoch_{epoch+1}.pth.tar'
            checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_filename)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, filename=checkpoint_path)
            # save_checkpoint(checkpoint, is_best, filename=checkpoint_path)

        epoch_time = time.time() - epoch_start_time
                # **Insert Your Logging Code Here**
        if args.rank == 0:
            memory_allocated = torch.cuda.memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0
            memory_reserved = torch.cuda.memory_reserved(device) / (1024 * 1024) if torch.cuda.is_available() else 0
            with open(log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'epoch', 'train_loss', 'train_acc1', 'train_acc5',
                    'val_loss', 'val_acc1', 'val_acc5',
                    'best_acc1', 'learning_rate', 'batch_size',
                    'epoch_time', 'eval_time',
                    'memory_allocated_mb', 'memory_reserved_mb'
                ])
                writer.writerow({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics["train_loss"],
                    'train_acc1': train_metrics["train_accuracy"],
                    'train_acc5': None
                    'val_loss': val_loss,
                    'val_acc1': val_acc1,
                    'val_acc5': val_acc5,
                    'best_acc1': best_acc1,
                    'learning_rate': scheduler.get_last_lr()[0],
                    'batch_size': effective_bs,
                    'epoch_time': round(epoch_time, 2),
                    'eval_time': round(eval_time, 2),
                    'memory_allocated_mb': round(memory_allocated, 2),
                    'memory_reserved_mb': round(memory_reserved, 2),
                    'grad_diversity': train_metrics.get("grad_diversity")
                })
            scheduler.step()
        if (epoch + 1) % args.resize_freq == 0:
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
                    trainset, 
                    batch_size=batch_size,
                    shuffle=True, 
                    num_workers=1, 
                    pin_memory=True
                )

            if args.adaptive_lr:
                # rescale the learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= rescale_ratio

        




def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))


        loss.backward()
        if (i + 1) % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i + 1)

    # Handle remaining gradients
    if len(train_loader) % args.accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()



    
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

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

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


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
        entries += [meter.summary() for meter in self.meters]
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


from datasets import load_from_disk

# Add this function above or below your existing functions
def get_imagenet_dataset(args, is_train=True):
    """
    Loads the imagenet dataset using Hugging Face's load_from_disk and wraps it with a custom PyTorch Dataset.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        is_train (bool): Whether to load the training set or validation set.
    
    Returns:
        dataset (CustomImageNetDataset): Wrapped PyTorch Dataset.
    """
    # Define imagenet transformations
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

    # Define the path for imagenet
    imagenet_path = '/projects/bdeb/chenyuen0103/tinyimagenet'  # Update if different

    # Load imagenet dataset from Hugging Face's `load_from_disk`
    hf_dataset = load_from_disk(imagenet_path)

    # Select appropriate split
    split = "train" if is_train else "valid"
    dataset_split = hf_dataset[split]

    # Wrap the Hugging Face dataset with PyTorch Dataset
    dataset = CustomImageNetDataset(dataset_split, transform)
    
    return dataset




if __name__ == '__main__':
    main()