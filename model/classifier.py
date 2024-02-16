import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encdec import Decoder, LearnedPooling


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.latent_dim = 256
        self.positional_encoding = PositionalEncoding(self.latent_dim)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.positional_encoding)

        self.module_static = LearnedPooling(sizes_downsample=[1024,64,32])
        self.rep_decoder = Decoder(
            encoder_features = [1,16,32],
            sizes_downsample=[256,64,16],
            latent_space=256,
            activation=nn.ELU
        )
        self.rep_decoder_ln = nn.Linear(self.latent_dim*3, self.latent_dim)

    def forward(self, x, timesteps):
        
        emb = self.embed_timestep(timesteps).unsqueeze(1)

        resize = False

        if x.dim() == 4:
            bs, f, _, _ = x.shape
            resize = True
            x = torch.flatten(x, 0, 1)

        output_static = self.module_static.enc(x)

        if resize:
            output_static = output_static.view(bs, f, -1)
        
        output_cond = emb + output_static

        resize = False
        if output_cond.dim() == 3:
            bs, f, _ = output_cond.shape
            resize = True
            output_cond_f = torch.flatten(output_cond, 0, 1)

        output = self.module_static.dec(output_cond_f)
        output_pre = self.rep_decoder(output_cond_f)
        output_pre = torch.flatten(output_pre,-2,-1)
        output_pre = self.rep_decoder_ln(output_pre)
        if resize:
            output = output.view(bs, f, -1, 3)
            output_pre = output_pre.view(bs, f, -1)

        return output, output_pre
    



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])
