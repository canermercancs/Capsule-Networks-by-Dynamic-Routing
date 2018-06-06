"""
author: Caner Mercan
"""

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.helper import squash

class PrimaryCapsLayer(nn.Module):
    """
    Primary Capsule Layer: no routing here.
    Only conv2d operations applied for num_unit times separately. (not sequentially!)
    Finally, output vectors are squashed to have values between 0 and 1.
    """
    def __init__(self, in_channels=256, out_channels=32, num_units=8, kernel_size=9, stride=2, padding=False, args=None):
        super().__init__()
        
        # the number of convolutional units; dimension of a capsule.
        self.num_units = num_units
        
        # there are num_units(8 in paper) number of conv2d units
        # whose outputs when concatenated, will give us the u vector (unsquashed)
        # num_units * conv2d layer list
        self.conv2d_list = nn.ModuleList([
                                nn.Conv2d(
                                    in_channels  = in_channels,
                                    out_channels = out_channels, 
                                    kernel_size  = kernel_size,
                                    stride       = stride,
                                    padding      = padding )
                                for _ in range(num_units)])
        # squashing unit at the end
        self.squash = squash
        
        
    def forward(self, inp):
        """
        # input going from convolutional layer to capsule layer
        # input shape [64, 256, 20, 20]

        # batch_size = inp.size(0) # 64
        # in_channels = inp.size(1) # 256
        # in_patch_size = inp.size(2), inp.size(3) # 20, 20

        # num_units  = 8 # len(self.conv2d_list) # dimensinality of capsule vector
        # out_channels = 32 
        # out_patch_size = 6, 6
        """
        batch_size = inp.size(0)

        # running #num_units conv2d layers on input; unit_list is a list of size 8, each containing [64, 32x6x6] sized tensor. 
        unit_list = [conv2d(inp).view((batch_size, -1, 1)) for conv2d in self.conv2d_list]
        # convert unit_list to torch array of size: [64, 32x6x6, 8] (batch_size, out_channels x patch_height x patch_width, num_units)
        s = torch.cat(unit_list, dim=-1)
        # squash each 32x6x6 capsule unit on the last dimension (num_units:8) 
        v = self.squash(s, dim=-1)
        # v is of shape [64, 1152, 8]
        return v
    

class DigitCapsLayer(nn.Module):
    """
    Digits Capsule Layer: routing is done here!
    
    """
    def __init__(self, args, in_channels=1152, in_units=8, out_channels=10, out_units=16, routing_epoch=3):
        super().__init__()
        self.args           = args 
        self.in_channels    = in_channels
        self.in_units       = in_units
        self.out_channels   = out_channels
        self.out_units      = out_units      
        self.routing_epoch  = routing_epoch
        self.__init_params__()
                     
    def forward(self, inp):   
        # sanity check
        assert inp.size(0) == self.args['batch_size']

        # the prediction vector u_hat; is of shape [batch_size, in_channels(1152), out_channels(10), out_units(16), 1]
        u_hat = self.__f_u_hat__(inp)        
        # b is of shape [1, in_channels(1152), out_channels(10), 1, 1]
        # b is reset to zeros at the start of each epoch!
        b = torch.zeros(1, self.in_channels, self.out_channels, 1, 1)
        b = b.to(self.args['device'])
        
        # run dynamic routing algo loop for routing_epoch times 
        for _ in range(self.routing_epoch):            
            # compute c value as softmax(b): coefficients to the prediction vector u_hat
            # c is shared across all batches; hence the multiplication of list with batch_size.
            # c is of shape: [batch_size, 1152, 10, 1, 1]
            c = F.softmax(b, dim=2)        
            c = torch.cat([c]*self.args['batch_size'], dim=0)        
            # s is of shape: [64, 1, 10, 16, 1] (summed over in_channels(i))
            s = torch.sum(u_hat*c, dim=1, keepdim=True)
            v = squash(s, dim=3)            
            # update b
            b += self.__f_delta_b__(u_hat, v)
            print(v.shape)
            print(u_hat.shape)
            
        # v is of shape: [64, 1, 10, 16, 1] to [64, 10, 16]
        return v.squeeze()

    def __init_params__(self):        
        # W is of shape [1, in_channels(1152), out_channels(10), out_units(16), in_units(8)]
        self.W = nn.Parameter(torch.randn(1, self.in_channels, self.out_channels, self.out_units, self.in_units))
    
    def __f_u_hat__(self, inp):
        """
        helper function for the computation of prediction_vector u_hat(j given i):
        u_hat_j|i = W_ij * u_i
        """
        # inp is of shape: [batch_size, in_channels, in_units]
        # u is of shape [batch_size, in_channels(1152), out_channels(10), in_units(8), 1]
        u = torch.stack(self.out_channels*[inp], dim=2).unsqueeze(-1)
        # W is shared across all batches; hence the multiplication of list with batch_size
        W = torch.cat(self.args['batch_size']*[self.W], dim=0)#.to(self.args['device']) 
        # the prediction vector u_hat; is of shape [batch_size, in_channels(1152), out_channels(10), out_units(16), 1]
        return torch.matmul(W, u)

    def __f_delta_b__(self, u_hat, v):
        """
        helper function for the computation of update value of b:
        deltab_ij = u_hat_j|i * v_j
        """
        # reshape u_hat and v for matrix multiplication. [1x16][16x1]=[1]
        # v is of shape: [64, 1152, 10, 16, 1]
        v = torch.cat([v]*self.in_channels, dim=1)        
        # transpose the last two dims; u_hat is of shape now: [batch_size, in_channels(1152), out_channels(10), 1, out_units(16)]
        u_hat = u_hat.transpose(-1,-2)
        # compute delta b from the mean of (u_hat.T*v) over all batches.
        return torch.matmul(u_hat, v).mean(dim=0, keepdim=True)