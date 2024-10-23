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

        self.cond_mask_prob = 0

        self.positional_encoding = PositionalEncoding(self.latent_dim)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.positional_encoding)
        if self.gender:
            self.embed_gender = TimestepEmbedder(self.latent_dim, self.positional_encoding)
        
        self.encoder_features = [3,32,64]
        self.sizes_downsample = [1026,64,32]
        if self.shape:
            self.shape_enc = nn.Linear(self.latent_dim, self.latent_dim)

        self.module_static = LearnedPooling(self.latent_dim)
        self.mod = "mesh"
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

        self.decoder_first_linear = nn.Linear(self.latent_dim, self.latent_dim)
        self.decoder_end_linear = nn.Linear(self.latent_dim, self.latent_dim)

        self.mlp_gamma = nn.Linear(self.latent_dim, self.latent_dim)
        self.mlp_beta = nn.Linear(self.latent_dim, self.latent_dim)
        # self.embed_beta = nn.Linear(256, self.latent_dim)

        self.embed_action = EmbedAction(100, self.latent_dim)

    def mask_cond(self, cond, uncond=False):
        bs = cond.shape[0]
        uncond = uncond[(...,)+(None,)*(cond.dim()-1)]
        if self.training and self.cond_mask_prob>0:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * (1 - self.cond_mask_prob))
            mask = mask[(...,)+(None,)*(cond.dim()-1)]
            cond = cond*mask
        return cond*uncond

    def stylization(self, x, emb):
        x = x*self.mlp_gamma(emb) + self.mlp_beta(emb)
        return x

    def mode(self, mod):
        self.mod = mod
        self.tpose_ae = nn.Linear(16,self.latent_dim)
        self.beta_enc_static = nn.Sequential(
            nn.Linear(77, 64),
            nn.SiLU(),
            nn.Linear(64, self.latent_dim),
        )
        self.beta_dec_static = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 77)
        )

    def forward(self, x, timesteps, y):
        """
        x: [batch_size, sequence_len, nbfreq, 3], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        # embpre = self.embed_timestep(timesteps-1)
        if self.mod=="beta":
            output_static = self.beta_enc_static(x)
        else:
            output_static = self.enc_static(x)
        
        action_emb = self.embed_action(y['action'])
        emb += self.mask_cond(action_emb, y["actioncond"]).unsqueeze(1)
        # action_emb = self.mask_cond(action_emb, y["actioncond"]).unsqueeze(1)

        if self.mod=="beta":
            beta_emb = self.tpose_ae(y["tpose"])
        else:
            _ , beta_emb = self.tpose_ae(y["tpose"])
        beta_emb = self.mask_cond(beta_emb, y["shapecond"])

        # timesteps embedding
        output_cond = self.t_emb(emb, output_static) 

        # 28 févr sty
        output_transformer = self.enc_transformer(self.stylization(output_cond, beta_emb))[:,-self.size_window:,:]

        # nores
        if self.mod=="beta":
            output = self.beta_dec_static(output_transformer)
        else:
            output = self.dec_static(output_transformer)

        return output
    
    # def parameters(self):
    #     return [p for name, p in self.named_parameters()]
    
    def enc_static(self, x):
        resize = False

        if 4 == x.dim():
            resize = True
            x = torch.flatten(x, 0, 1)

        x = self.module_static.enc(x)

        if resize:
            x = x.view(self.batch_size, self.n_frames, -1)

        return x

    def dec_static(self, x):
        resize = False

        if 3 == x.dim():
            resize = True
            x = torch.flatten(x, 0, 1)

        x = self.module_static.dec(x)

        if resize:
            x = x.view(self.batch_size, self.n_frames, -1, 3)

        return x
    
    def dec_transformer(self, y, fenc):
        y = self.decoder_first_linear(y)
        y = y.permute(1,0,2)
        fenc = fenc.permute(1,0,2)
        y = self.positional_encoding(y)
        y = self.transformer_decoder(y, fenc)
        y = y.permute(1, 0, 2)
        y = self.encoder_end_linear(y)
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


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output