import os
import torch
import torch.nn as nn
from models import ResNet18  # Adjust this to match the model you're using
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from models import *
from utils import progress_bar

# Import your test dataset loader
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Define the loss criterion
criterion = nn.CrossEntropyLoss()


# Load the saved checkpoint
print("==> Loading checkpoint..")
checkpoint_path = './checkpoint/ckpt.pth'  # Update path if needed
assert os.path.isfile(checkpoint_path), "Checkpoint not found!"


# check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    checkpoint = torch.load(checkpoint_path)
else:
    # Load GPU model on CPU
    checkpoint = torch.load(checkpoint_path,
                            map_location=torch.device('cpu'))

net = ResNet18()
net = net.to(device)

# Adjust for DataParallel checkpoints
state_dict = checkpoint['net']
if 'module.' in list(state_dict.keys())[0]:  # Check if 'module.' exists in the keys
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v  # Strip the 'module.' prefix
    state_dict = new_state_dict

net.load_state_dict(state_dict)


# Restore model state
# net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
print(f"Checkpoint loaded: Epoch {start_epoch}, Best Accuracy {best_acc:.2f}%")

# Evaluate the model on the test set
def evaluate_model():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
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

    acc = 100.*correct/total
    print(f"Final Test Accuracy: {acc:.2f}%")
    return acc

# Call evaluation function
evaluate_model()
