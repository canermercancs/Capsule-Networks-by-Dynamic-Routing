"""
author: Caner Mercan
"""

import numpy as np
import torch.nn as nn
from .layers.convolutional_layer import ConvLayer
from .layers.capsule_layer import PrimaryCapsLayer
from .layers.capsule_layer import DigitCapsLayer

# keywords the argument 'args' must have
__args_keywords__ = 'device', 'batch_size'

class CapsNet(nn.Module):
    # The whole Capsule Network.

    def __init__(self, args, convlayer_props, capslayer_props, digitlayer_props):
        super().__init__()
        self.__assert_args__(args)
        
        self.convlayer = ConvLayer(**convlayer_props)
        self.capslayer = PrimaryCapsLayer(**capslayer_props)
        self.digilayer = DigitCapsLayer(args, **digitlayer_props)
                
    def forward(self, inp):
        inp = self.convlayer(inp)
        inp = self.capslayer(inp)
        inp = self.digilayer(inp)
        return inp

    def __assert_args__(self, args):
        try:
            assert np.all(np.array([k in args.keys() for k in __args_keywords__]))
        except AssertionError:
            print(f'args dict must include keys {__args_keywords__}')