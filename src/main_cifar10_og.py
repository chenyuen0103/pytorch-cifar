'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from efficiency.function import set_seed
import os
import argparse

from models import *
from utils import progress_bar
from train_utils import log_metrics, test, DiveBatchTrainer, SGDTrainer
import time
import csv
from backpack import extend



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--data_dir', default='../data', type=str)
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10','cifar100', 'imagenet'],
                    help='Dataset to train on (cifar10 or imagenet)')
parser.add_argument('--model', default='resnet18', type=str, help='Model to train')
parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to train')
parser.add_argument('--algorithm', default='sgd', type=str, choices=['sgd', 'divebatch','adabatch'],)
parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
parser.add_argument('--adaptive_lr',  action='store_true', help='Rescale learning rate')
parser.add_argument('--batch_size', '-bs', default=128, type=int, help='Batch size')
parser.add_argument('--milestones', default=[60, 120, 160], type=int, nargs='+', help='Learning rate milestones')
parser.add_argument('--resize_freq', default=10, type=int, help='Resize frequency for DiveBatch')
parser.add_argument('--max_batch_size', default=2048, type=int, help='Maximum batch size for DiveBatch')
parser.add_argument('--delta', default=0.1, type=float, help='Delta for GradDiversity')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--log_dir', default='../logs_multistep', type=str, help='Directory to save logs')
parser.add_argument('--checkpoint_dir', default='/projects/bdeb/chenyuen0103/checkpoint_multistep', type=str, help='Directory to save checkpoints')
parser.add_argument('--seed', default=1, type=int, help='Random seed')
args = parser.parse_args()




device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(args.seed)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


log_dir = os.path.join(args.log_dir, args.model, args.dataset)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)


if args.algorithm == 'sgd':
    log_file = os.path.join(log_dir, f'{args.algorithm}_lr{args.lr}_bs{args.batch_size}_s{args.seed}.csv')
elif args.algorithm == 'divebatch':
    log_file = os.path.join(log_dir, f'{args.algorithm}_lr{args.lr}_bs{args.batch_size}_rf{args.resize_freq}_mbs{args.max_batch_size}_delta{args.delta}_s{args.seed}.csv')
elif args.algorithm == 'adabatch':
    log_file = os.path.join(log_dir, f'{args.algorithm}_lr{args.lr}_bs{args.batch_size}_rf{args.resize_freq}_mbs{args.max_batch_size}_s{args.seed}.csv')

fieldnames = [
    'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
    'learning_rate', 'batch_size', 'epoch_time', 'eval_time', 'memory_allocated', 'memory_reserved', 'gradient_diversity'
]



# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# if args.algorithm == 'divebatch':
#     net = extend(net)
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



if args.algorithm == 'divebatch':
    def extend_all_modules(model):
        """
        Recursively extend all submodules of the model with BackPACK.
        """
        for module in model.modules():
            extend(module)
    extend_all_modules(net)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True





checkpoint_dir = os.path.join(args.checkpoint_dir, args.model, args.dataset)
print(f"Checkpoint directory: {checkpoint_dir}")
if args.algorithm == 'sgd':
    checkpoint_file = f"{args.algorithm}_lr{args.lr}_bs{args.batch_size}_s{args.seed}_ckpt.pth"
elif args.algorithm == 'divebatch':
    checkpoint_file = f"{args.algorithm}_lr{args.lr}_bs{args.batch_size}_rf{args.resize_freq}_mbs{args.max_batch_size}_delta{args.delta}_s{args.seed}_ckpt.pth"
elif args.algorithm == 'adabatch':
    checkpoint_file = f"{args.algorithm}_lr{args.lr}_bs{args.batch_size}_rf{args.resize_freq}_mbs{args.max_batch_size}_s{args.seed}_ckpt.pth"
# Check if the log file already exists
log_exists = os.path.exists(log_file)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

batch_size = args.batch_size
# Resume from checkpoint
file_mode = 'w'
if args.resume:
    latest_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    if os.path.exists(latest_checkpoint_path):
        checkpoint = torch.load(latest_checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        file_mode = 'a'
        # load the batch size from the csv file
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pass
            batch_size = int(row['batch_size'])
            
# Data
# breakpoint()
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size = batch_size, shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
elif args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(
        root='../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)


with open(log_file, file_mode, newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    # Write the header only if the file is new
    if not log_exists or not args.resume:
        writer.writeheader()


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    epoch_start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        epoch_time = time.time() - epoch_start_time
        train_acc = 100. * correct / total       
    return train_loss / len(trainloader), train_acc, epoch_time



def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    eval_start_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)

    state = {
        'epoch': epoch + 1,  # Save the next epoch number
        'net': net.state_dict(),
        'best_acc': best_acc,
        'current_acc': acc,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'is_best': is_best
    }

    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    torch.save(state, latest_checkpoint_path)
    if is_best:
        best_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file.replace('.pth', '_best.pth'))
        torch.save(state, best_checkpoint_path)
        print(f"Updated best checkpoint: {best_checkpoint_path}")

    eval_time = time.time() - eval_start_time

    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(testloader),
        acc,
        eval_time
    ))

    return test_loss / len(testloader), acc, eval_time


if not os.path.exists(log_file) or file_mode == 'w':
    with open(log_file, file_mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
# Training and Testing
if args.algorithm == 'divebatch':
    trainer_cls = DiveBatchTrainer
# elif args.algorithm == 'adabatch':
#     trainer_cls = DiveBatchTrainer
else:
    trainer_cls = SGDTrainer
trainer_args = {
    "model": net,
    "optimizer": optimizer,
    "criterion": criterion,
    "device": device,
}
if args.algorithm == 'divebatch':
    trainer_args.update({
        "delta": args.delta,
        "resize_freq": args.resize_freq,
        "max_batch_size": args.max_batch_size,
    })
trainer = trainer_cls(**trainer_args)
old_grad_diversity = 1.0 if args.algorithm == 'divebatch' else None
for epoch in range(start_epoch, args.epochs):
    train_metrics = trainer.train_epoch(trainloader, epoch)
    val_loss, val_acc, eval_time = test(epoch)

    log_metrics(
        log_file=log_file,
        epoch=epoch + 1,
        train_loss=train_metrics["train_loss"],
        train_acc=train_metrics["train_accuracy"],
        val_loss=val_loss,
        val_acc=val_acc,
        lr=scheduler.get_last_lr()[0],
        batch_size=batch_size,
        epoch_time=train_metrics.get("epoch_time", 0),
        eval_time=eval_time,
        memory_allocated=torch.cuda.memory_allocated() if device == 'cuda' else 0,
        memory_reserved=torch.cuda.memory_reserved() if device == 'cuda' else 0,
        grad_diversity=train_metrics.get("grad_diversity")
    )
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
                
    # torch.cuda.empty_cache()

