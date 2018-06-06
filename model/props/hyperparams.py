"""
author: Caner Mercan
"""

import math
import torch
# image size after convolutional layer;
# l: image_height / image_width
# p: padding
# k: kernel size
# s: stride
convim = lambda l,p,k,s: math.ceil((l+2*p - (k-1))/s)
# system level arguments
args = { 
    'device'       : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), # DO NOT CHANGE THIS
    'batch_size'   : 4,
    'image_size'   : [28, 28, 1] # modify this based on original input image size
}
# 1st convolutional layer parameters (2dconvolutional unit + relu function)
convlayer_props = {
    'in_channels'  : args['image_size'][-1],
    'out_channels' : 256,
    'kernel_size'  : 9,
    'stride'       : 1,
    'padding'      : 0
}
# 1st capsule layer (primary capsule) parameters (num_unit independent 2dconvolutional units + squash function)
capslayer_props = {
    'in_channels'  : convlayer_props['out_channels'], 
    'out_channels' : 32, 
    'num_units'    : 8, 
    'kernel_size'  : 9, 
    'stride'       : 2, 
    'padding'      : 0 
}
# 2dn capsule layer (digit capsule) parameters (routing by agreement)
im_size_conv = [convim(args['image_size'][i], convlayer_props['padding'], convlayer_props['kernel_size'], convlayer_props['stride']) for i in range(2)]
im_size_caps = [convim(im_size_conv[i],       capslayer_props['padding'], capslayer_props['kernel_size'], capslayer_props['stride']) for i in range(2)]
digilayer_props = {
    'in_channels'  : capslayer_props['out_channels']*im_size_caps[0]*im_size_caps[1], 
    'in_units'     : capslayer_props['num_units'], 
    'out_channels' : 10, 
    'out_units'    : 16, 
    'routing_epoch': 3
}