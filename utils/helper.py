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