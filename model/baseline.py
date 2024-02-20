import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encdec import Decoder, LearnedPooling
from model.classifier import Classifier


class Baseline(nn.Module):
    def __init__(self, *args,**kargs):
        super().__init__()

        self.size_window = kargs["size_window"]
        self.latent_dim = kargs["latent_dim"]

        self.ff_size = kargs["ff_size"]
        self.num_layers = kargs["num_layers"]
        self.num_heads = kargs["num_heads"]

        self.dropout = kargs["dropout"]
        self.activation = kargs["activation"]
        self.batch_size = kargs["batch_size"]
        self.n_frames = kargs["n_frames"]

        self.gender = kargs["gender_in"]
        self.shape = kargs["shape"]

        if kargs["t_emb"] == "concat":
            self.t_emb = lambda a, b: torch.concat([a, b], dim=1)
        if kargs["t_emb"] =="add":
            self.t_emb = lambda a, b: a+b

        self.tpose_ae = Classifier()

        self.positional_encoding = PositionalEncoding(self.latent_dim)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.positional_encoding)
        if self.gender:
            self.embed_gender = TimestepEmbedder(self.latent_dim, self.positional_encoding)
        
        self.encoder_features = [3,32,64]
        self.sizes_downsample = [1026,64,32]
        if self.shape:
            self.shape_enc = nn.Linear(self.latent_dim, self.latent_dim)

        self.module_static = LearnedPooling()

        # Transformer to extract temporal information
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=self.num_layers
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=self.num_layers
        )
        self.encoder_first_linear = nn.Linear(self.latent_dim, self.latent_dim)
        self.encoder_end_linear = nn.Linear(self.latent_dim, self.latent_dim)

        # self.decoder_first_linear = nn.Linear(self.latent_dim, self.latent_dim)
        # self.decoder_end_linear = nn.Linear(self.latent_dim, self.latent_dim)


    def forward(self, x, timesteps, tpose=None):
        """
        x: [batch_size, sequence_len, nbfreq, 3], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        # embpre = self.embed_timestep(timesteps-1)
        output_static = self.enc_static(x)
        
        _ , beta_emb = self.tpose_ae(tpose)

        # timesteps embedding
        output_cond = self.t_emb(emb, output_static) 

        # output_cond_shape = beta_emb + output_cond

        output_transformer = self.enc_transformer(beta_emb + output_cond)[:,-self.size_window:,:]

        # res
        output = self.dec_static(output_transformer+output_cond[:,-self.size_window:,:])

        # nores
        # output = self.dec_static(output_transformer)

        return output
    
    # def parameters(self):
    #     return [p for name, p in self.named_parameters()]
    
    def enc_static(self, x):
        resize = False

        if x.dim() == 4:
            resize = True
            x = torch.flatten(x, 0, 1)

        x = self.module_static.enc(x)

        if resize:
            x = x.view(self.batch_size, self.n_frames, -1)

        return x

    def dec_static(self, x):
        resize = False

        if x.dim() == 3:
            resize = True
            x = torch.flatten(x, 0, 1)

        x = self.module_static.dec(x)

        if resize:
            x = x.view(self.batch_size, self.n_frames, -1, 3)

        return x
    
    def dec_transformer(self, y, fenc):
        # y = self.decoder_first_linear(y)
        y = y.permute(1,0,2)
        fenc = fenc.permute(1,0,2)
        y = self.positional_encoding(y)
        y = self.transformer_decoder(y, fenc)
        y = y.permute(1, 0, 2)
        # y = self.encoder_end_linear(y)
        return y


    def enc_transformer(self, x):

        x = self.encoder_first_linear(x)

        x = x.permute(1, 0, 2)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.encoder_end_linear(x)

        return x



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

class PredictSigma(Baseline):
    def __init__(self, *args,**kargs):
        super().__init__(*args, **kargs)
        self.decoder = Decoder(self.encoder_features, self.sizes_downsample, kargs["latent_dim"], nn.ELU)
    
    def dec_var(self, x):
        resize = False

        if x.dim() == 3:
            batch_size = x.size(0)
            n_frames = x.size(1)
            resize = True
            x = torch.flatten(x, 0, 1)

        x = self.decoder(x)

        if resize:
            x = x.view(batch_size, n_frames, -1, 3)

        return x
    
    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, sequence_len, nbfreq, 3], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        output_static = self.enc_static(x)
        
        # timesteps embedding
        output_static = self.t_emb(emb, output_static) 

        output_transformer = self.enc_transformer(output_static)[:,-self.size_window:,:]

        output = self.dec_static(output_transformer)

        output_var = self.dec_var(output_transformer)
        return torch.concat([output, output_var], dim=1)
