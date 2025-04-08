import torch

def spike_to_label(spike_train, scheme = 'highest_voltage'):
    """Convert spike train to the label in one-hot encoded class

    Args:
        spike_train (tensor): spike train with shape [time_steps, batch_size, classes]
        scheme(string): options: 'most_spikes' and 'first_spike'
        
    Return:
        one label for each sample: [batch_size,]
    """
    if scheme == 'most_spikes':
        # count number of spikes along the time_steps dimension. Result is [batch_size, classes]
        spike_counts = spike_train.count_nonzero(dim=0)
        
        # pick the index of along the clsses dimension
        result = spike_counts.argmax(dim=-1)
    elif scheme == 'highest_voltage':
        mem_out_aux = torch.max(spike_train, dim = 0)[0]
        result = torch.argmax(mem_out_aux, dim=1)
    else:
        raise Exception('Undefined Scheme')
    
    return result

def spike_count(spike_train):
    """count number of spikes for each spike train

    Args:
        spike_train (tensor): shape v[time_steps, batch_size, classes]

    Returns:
        tensor: [batch_size, classes]
    """
    return spike_train.count_nonzero(dim=0)