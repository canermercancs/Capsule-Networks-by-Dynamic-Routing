"""
author: Caner Mercan
"""
import os
import pdb
import copy
import torch
import numpy as np
import torch.nn as nn
import utils.helper as helper
from model.layers.convolutional_layer import ConvLayer
from model.layers.capsule_layer import PrimaryCapsLayer, DigitCapsLayer
from model.layers.decoder_block import Decoder
from utils.loss import margin_loss, margin_loss_v2, recons_loss

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
            assert targ is not None
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


    def fit(self, data_loader, optimizer, scheduler, num_epochs=100):
        # self(capsule network), data_loader, optimizer, scheduler must be built-in Pytorch Class objects
        # fit model to data_loader['train']

        best_params = copy.deepcopy(self.state_dict())
        data_size = {p: len(data_loader[p].dataset) for p in data_loader.keys()}
        best_acc = 0.0

        # pdb.set_trace()
        for epoch in range(num_epochs):
            for p in data_loader.keys(): # 'train', 'val'
                if p == 'train': 
                    scheduler.step()
                    self.train()
                else: 
                    self.eval()
                #correct_pred = 0.0
                running_loss = 0.0
                running_corrects = 0.0                   
                for idx, (input_batch, label_batch) in enumerate(data_loader[p]):
                    input_batch = input_batch.to(self.args['device'])
                    label_batch = label_batch.to(self.args['device'])
                    
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(p == 'train'):
                        caps_out, deco_out = self(input_batch, p, label_batch)
                        
                        # loss = margin_loss(caps_out, label_batch)
                        loss = margin_loss_v2(caps_out, label_batch)    
                        if self.args['use_decoder']:
                            loss_deco = recons_loss(deco_out, input_batch)
                            loss = loss + self.args['scale_decoder']*loss_deco
                        if p == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item()
                    if idx % 50 == 49:    # print every 50 minibatches
                        print('[%d, %d] loss: %.3f' % (epoch+1, idx+1, running_loss/50))
                        running_loss = 0.0
                    
                    # statistics
                    probs, preds = helper.max_caps(caps_out)
                    #correct_pred += torch.sum(preds == label_batch).item()
                    running_corrects += torch.sum(preds == label_batch).item()

                epoch_loss = running_loss / data_size[p]
                epoch_acc = running_corrects / data_size[p]
                print(f'{p} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if p != 'train' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_params = copy.deepcopy(self.state_dict())
                    torch.save(self.state_dict(), os.path.join(os.path.expanduser('~'), 'capsuleNet_bestmodel'))

        return best_params
