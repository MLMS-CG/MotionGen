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
        self.alpha_shape = 0.8
        self.alpha_act = 0.7
        self.alpha_cond = 0.8

    def forward(self, x, timesteps, y):
        # out = self.model(x, timesteps, y)
        yuncond = deepcopy(y)
        # fullcond = self.model(x, timesteps, yuncond)
        yuncond['actioncond']-=1
        shapecond = self.model(x, timesteps, yuncond)
        yuncond['actioncond']+=1
        yuncond['shapecond']-=1
        actioncond = self.model(x, timesteps, yuncond)
        yuncond['actioncond']-=1
        out_uncond = self.model(x, timesteps, yuncond)
        return out_uncond + \
            self.alpha_shape * (shapecond - out_uncond) + \
            self.alpha_act * (actioncond - out_uncond)
        # return out_uncond + \
        #     self.alpha_shape * (shapecond - out_uncond)
        # return out_uncond
        # return out_uncond + \
        #     self.alpha_cond * (fullcond - out_uncond)