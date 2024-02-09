import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encdec import Encoder, Decoder, LearnedPooling


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.positional_encoding = PositionalEncoding(self.latent_dim)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.positional_encoding)

        self.module_static = LearnedPooling()


    def forward(self, x, timesteps):
        
        emb = self.embed_timestep(timesteps) 
        output_static =  x = self.module_static.enc(x)
        
        output_cond = emb + output_static

        output = self.module_static.dec(output_cond)

        return output
    



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

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
