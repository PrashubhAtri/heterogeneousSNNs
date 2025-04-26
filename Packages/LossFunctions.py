import torch

def binaryCustomLoss(mempotentials, labels):
    """
        inputs : (mempotentials -> (batch_size, no_of_classes, no_of_time_steps), labels -> (batch_size))
    """
    # Set debugger
    # import pdb; pdb.set_trace()
    labels = labels.long()
    non_labels = 1-labels

    batch_idx = torch.arange(mempotentials.shape[0])

    correct = mempotentials[batch_idx, labels]
    non_correct = mempotentials[batch_idx, non_labels]

    diff = non_correct - correct
    diff_activated = torch.where(diff > 0, diff, torch.zeros_like(diff))
    return (diff_activated).mean()


def multiclassMembraneMarginLoss(mempotentials, labels):
    """
    mem: Tensor of shape [batch_size, num_classes, time_steps]
    labels: Tensor of shape [batch_size]
    """
    batch_size, num_classes, time_steps = mempotentials.shape
    labels = labels.long()

    # Get the membrane potentials for the correct classes: [batch_size, time_steps]
    correct_class_potentials = mempotentials[torch.arange(batch_size), labels, :]  # [batch_size, time_steps]

    # Expand for broadcasting: [batch_size, 1, time_steps]
    correct_class_potentials = correct_class_potentials.unsqueeze(1)

    # Subtract correct potentials from all class potentials
    diff = mempotentials - correct_class_potentials  # [batch_size, num_classes, time_steps]

    # Mask out the correct class
    mask = torch.ones_like(diff, dtype=torch.bool)
    mask[torch.arange(batch_size), labels, :] = False

    # Apply the mask
    diff_masked = diff[mask].view(batch_size, num_classes - 1, time_steps)

    # Hinge activation: penalize only when wrong class > correct
    diff_activated = torch.where(diff_masked > 0, diff_masked, torch.zeros_like(diff_masked))

    # Average over all samples, wrong classes, and time steps
    return diff_activated.mean()
