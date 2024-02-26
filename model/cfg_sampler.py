import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model, scaler):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.scaler = scaler

    def forward(self, x, timesteps, tpose):
        # assert cond_mode in ['text', 'action']
        out = self.model(x, timesteps, tpose)
        out_uncond = self.model(x, timesteps, tpose, True)
        return out_uncond + (self.scaler * (out - out_uncond))