"""
author: Caner Mercan
"""

import torch
import torch.nn as nn

class Decoder(nn.Module):
	def __init__(self, in_units, out_pixels, fc1=512, fc2=1024):
		super().__init__()

		# self.in_units 	= in_units
		# self.out_pixels = out_pixels
		self.classifier = nn.Sequential(
									nn.Linear(in_units, fc1),
									nn.ReLU(inplace = False),
									nn.Linear(fc1, fc2),
									nn.ReLU(inplace = False),
									nn.Linear(fc2, out_pixels),
									nn.Sigmoid())
	def forward(self, inp):
		return self.classifier(inp)
			