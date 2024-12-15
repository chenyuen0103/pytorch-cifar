import math

def update_batch_size(current_batch_size, gradient_diversity, min_batch_size, max_batch_size, delta):
    """
    Adjust batch size based on gradient diversity.
    Arguments:
    - current_batch_size: Current batch size.
    - gradient_diversity: Calculated gradient diversity metric.
    - min_batch_size: Minimum allowable batch size.
    - max_batch_size: Maximum allowable batch size.
    - delta: Adjustment factor for batch size changes.
    Returns:
    - Updated batch size.
    """
    if gradient_diversity > delta:
        new_batch_size = min(max_batch_size, current_batch_size * 2)  # Increase batch size
    elif gradient_diversity < delta / 2:
        new_batch_size = max(min_batch_size, current_batch_size // 2)  # Decrease batch size
    else:
        new_batch_size = current_batch_size
    return new_batch_size


def compute_gradient_diversity(accumulated_grads, individual_grad_norms):
    """
    Calculate gradient diversity based on accumulated gradients and individual gradient norms.
    Arguments:
    - accumulated_grads: Accumulated gradients from all samples in the batch.
    - individual_grad_norms: Sum of squared norms of individual gradients.
    Returns:
    - Gradient diversity metric.
    """
    accumulated_norm = sum(torch.norm(g).item() ** 2 for g in accumulated_grads)
    return accumulated_norm / individual_grad_norms
