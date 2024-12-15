# src/train_utils.py

import csv
import os
import torch
from backpack import backpack, extend
from backpack.extensions import (
    BatchGrad,
    BatchL2Grad,
)
from adaptive_batch_size import compute_gradient_diversity




class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def compute_grad_sum_norm(self, accumulated_grads):
        """Compute the norm of the sum of accumulated gradients."""
        flattened_grads = [grad.flatten() for grad in accumulated_grads]
        concatenated_grads = torch.cat(flattened_grads)
        return torch.norm(concatenated_grads).item() ** 2

    def train_epoch(self, dataloader, epoch):
        """Abstract method for training one epoch."""
        raise NotImplementedError("Subclasses must implement train_epoch!")





class SGDTrainer(Trainer):
    def train_epoch(self, model, dataloader, optimizer, criterion, device, epoch, progress_bar):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        return {
            "train_loss": train_loss / len(dataloader),
            "train_accuracy": 100. * correct / total,
        }

class DiveBatchTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, device, resize_freq, max_batch_size):
        super().__init__(model, optimizer, criterion, device)
        self.resize_freq = resize_freq
        self.max_batch_size = max_batch_size

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        train_loss, correct, total = 0, 0, 0
        accumulated_grads = [torch.zeros_like(param) for param in self.model.parameters()]
        individual_grad_norm_sum = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            if self.resize_freq != 0 and (epoch + 1) % self.resize_freq == 0:
                with backpack(BatchGrad(), BatchL2Grad()):
                    loss.backward()
                for j, param in enumerate(self.model.parameters()):
                    accumulated_grads[j] += param.grad.detach().cpu().clone()
                    individual_grad_norm_sum += (param.batch_l2).sum().item()
            else:
                loss.backward()

            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        grad_sum_norm = self.compute_grad_sum_norm(accumulated_grads)
        grad_diversity = self.compute_gradient_diversity(grad_sum_norm, individual_grad_norm_sum)
        return {
            "train_loss": train_loss / len(dataloader),
            "train_accuracy": 100. * correct / total,
            "grad_diversity": grad_diversity
        }
    def compute_gradient_diversity(self, grad_sum_norm, individual_grad_norms):
        return individual_grad_norms /(grad_sum_norm + 1e-10)


def train(model, trainloader, optimizer, criterion, device, epoch, progress_bar, resize_freq = 0, max_batch_size = 4096):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    accumulated_grads = [torch.zeros_like(param) for param in model.parameters()]
    individual_grad_norm_sum = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)


        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if resize_freq != 0 and ((epoch + 1) % resize_freq == 0) and len(targets) < max_batch_size:
            with backpack(BatchGrad(), BatchL2Grad()):
                loss.backward()
            for j, param in enumerate(model.parameters()):
                accumulated_grads[j] += param.grad.detach().cpu().clone()  # Accumulate gradients
                individual_grad_norm_sum += (param.batch_l2).sum().item()  # Accumulate gradient norms
        else:
            # Standard backward pass
            loss.backward()

        optimizer.step()  # Update model parameters
        
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    if resize_freq != 0 and ((epoch + 1) % resize_freq == 0) and len(targets) < max_batch_size:
        flattened_grads = [grad.flatten() for grad in accumulated_grads]
        concatenated_grads = torch.cat(flattened_grads)
        grad_sum_norm = torch.norm(concatenated_grads).item() ** 2
        grad_diversity = compute_gradient_diversity(grad_sum_norm, individual_grad_norm_sum)
    else:
        grad_diversity = 1

    return train_loss / len(trainloader), 100. * correct / total, grad_diversity




def test(model, testloader, criterion, device, epoch, progress_bar, best_acc, checkpoint_dir):
    print('\nValidation...')
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint if best accuracy is achieved
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving checkpoint...')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(state, os.path.join(checkpoint_dir, 'ckpt.pth'))
        best_acc = acc

    return test_loss / len(testloader), acc, best_acc




def log_metrics(log_file, epoch, train_loss, train_acc, val_loss, val_acc, lr, batch_size, epoch_time, eval_time, memory_allocated, memory_reserved, grad_diversity=None):
    memory_allocated_mb = memory_allocated / (1024 * 1024)
    memory_reserved_mb = memory_reserved / (1024 * 1024)
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
            'learning_rate', 'batch_size', 'epoch_time', 'eval_time','memory_allocated_mb', 'memory_reserved_mb', 'grad_diversity'
        ])
        writer.writerow({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': lr,
            'batch_size': batch_size,
            'epoch_time': epoch_time,  # Time taken for one epoch
            'eval_time': eval_time,  # Time taken for evaluation
            'memory_allocated_mb': round(memory_allocated_mb, 2),
            'memory_reserved_mb': round(memory_reserved_mb, 2),
            'grad_diversity': round(grad_diversity, 4) if grad_diversity is not None else None,
        })
