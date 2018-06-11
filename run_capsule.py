"""
Provides main method for running Capsule Network on Mnist data set.

author: Caner Mercan
"""

import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import utils.helper as helper
import data.online_data as ondata
import model.props.config as cf
from model.capsule_network import CapsNet

capsule_net = CapsNet(cf.args, cf.convlayer_props, cf.capslayer_props, cf.digilayer_props, cf.decoder_props)
capsule_net = capsule_net.to(cf.args['device'])
# use Adam optimizer.
optimizer = optim.Adam(capsule_net.parameters(), lr=0.01)
# set exponential decay for the learning rate.
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
#scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
# load data
mnist_loader = ondata.load_MNIST(cf.args['batch_size'])
# run capsule model on data
best_params = capsule_net.fit(mnist_loader, optimizer, scheduler, num_epochs=100)
#capsule_net.load_state_dict(best_params)
