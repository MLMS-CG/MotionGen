import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encdec import LearnedPooling


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.latent_dim = 256

        self.module_static = LearnedPooling(sizes_downsample=[1024,64,32])

    def forward(self, x):

        resize = False

        if x.dim() == 4:
            bs, f, _, _ = x.shape
            resize = True
            x = torch.flatten(x, 0, 1)

        output_static = self.module_static.enc(x)

        if resize:
            output_static = output_static.view(bs, f, -1)
        
        output_cond = output_static

        resize = False
        if output_cond.dim() == 3:
            bs, f, _ = output_cond.shape
            resize = True
            output_cond_f = torch.flatten(output_cond, 0, 1)

        output = self.module_static.dec(output_cond_f)

        return output, output_cond