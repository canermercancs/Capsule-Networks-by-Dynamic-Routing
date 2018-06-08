"""
author: Caner Mercan
"""

import numpy as np
import torch.nn as nn
import utils.helper as helper
from .layers.convolutional_layer import ConvLayer
from .layers.capsule_layer import PrimaryCapsLayer, DigitCapsLayer
from .layers.decoder_block import Decoder


# keywords the argument 'args' must have
__args_keywords__ = 'device', 'batch_size'

class CapsNet(nn.Module):
    # The whole Capsule Network.

    def __init__(self, args, convlayer_props, capslayer_props, digitlayer_props, decoder_props):
        super().__init__()  
        self.args = args
        self.__assert_args__()
        
        self.convlayer  = ConvLayer(**convlayer_props)
        self.capslayer  = PrimaryCapsLayer(**capslayer_props)
        self.digilayer  = DigitCapsLayer(args, **digitlayer_props)
        self.decolayer  = Decoder(**decoder_props) 
                
    def forward(self, inp, phase='val', targ=None):
        outconv = self.convlayer(inp)
        outcaps = self.capslayer(outconv)
        outdigi = self.digilayer(outcaps)
        if self.args['use_decoder']:
            out_masked  = helper.mask_caps(outdigi, phase=phase, targ=targ)
            outdeco     = self.decolayer(out_masked)
            return outdigi, outdeco
        else:
            return outdigi, None

    def __assert_args__(self):
        try:
            assert np.all(np.array([k in self.args.keys() for k in __args_keywords__]))
        except AssertionError:
            print(f'args dict must include keys {__args_keywords__}')