"""
author: Caner Mercan
"""

import torch

def squash(s, dim=-1):
    """
    s is input tensor with size: 
        [batch_size, num_capsule, ..., dim_capsule]
    dim is the dimension for which the function will squash s, default is dim_capsule
    
    if s is small:    returns vector with length closer to 0 
    elif s is large:  returns vector with almost unit length (1)
    else:             returns vector with length somewhere between 0 and 1.
    """
    s_normsq = torch.sum(s**2, dim = dim, keepdim = True)
    coeff    = s_normsq / (s_normsq + 1)
    v        = coeff * (s / torch.sqrt(s_normsq))
    return v


def max_caps(v, dim=-1):
    """
    the probability of capsule output vectors v of shape: [batch_size, num_capsules, dim_capsule]
    computes the norm (length) of vector
    """
    v_norm = torch.sqrt(torch.sum(v**2, dim=dim))
    return torch.max(v_norm, dim=1) # get max predictions of each batch.


def mask_caps(v, phase='train', targ=None):
    """
    masks the capsule output w.r.t. to phase:
    if phase=='train': get the capsule output that correspond to the correct class label
    else: get the capsule with the highest probability (longest)
    """
    batch_size = v.size(0)
    if phase == 'train':
        assert targ is not None # target class label info cannot be empty if phase==train
        return v[range(batch_size), targ, :]
    else:
        _, mx = max_caps(v)
        return v[range(batch_size), mx, :]