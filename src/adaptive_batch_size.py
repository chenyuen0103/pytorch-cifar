import math
import torch

def update_batch_size(old_batch_size, new_gd, dataset_size, min_batch_size = 128, max_batch_size = 2048, delta = 0.0001,):
    if not isinstance(new_gd, torch.Tensor):
        new_gd = torch.tensor(new_gd, device='cuda' if torch.cuda.is_available() else 'cpu')
    if torch.isnan(new_gd).any() or torch.isinf(new_gd).any():
        return max_batch_size
    new_batch_size = int(min(max(delta * new_gd * dataset_size, old_batch_size), max_batch_size))
    return new_batch_size

def compute_gradient_diversity(grad_sum_norm, individual_grad_norms):
    """
    Calculate gradient diversity based on accumulated gradients and individual gradient norms.
    Arguments:
    - accumulated_grads: Accumulated gradients from all samples in the batch.
    - individual_grad_norms: Sum of squared norms of individual gradients.
    Returns:
    - Gradient diversity metric.
    """
    return grad_sum_norm/ (individual_grad_norms + 1e-10)
