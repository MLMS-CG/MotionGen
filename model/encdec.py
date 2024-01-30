import torch
import torch.nn as nn


class MatMulLayer(torch.nn.Module):
    def __init__(self, dim_0, dim_1) -> None:
        super().__init__()
        self.mat = torch.nn.Parameter(
            torch.zeros(dim_0, dim_1), requires_grad=True)

    def forward(self, x):
        return torch.matmul(x, self.mat)

class Encoder(torch.nn.Module):
    def __init__(
            self,
            encoder_features,
            sizes_downsample,
            latent_space,
            activation
        ):
        super().__init__()

        sizes_convs_encode = [3 for _ in range(len(encoder_features))]

        encoder_linear = [
            encoder_features[-1] * sizes_downsample[-1],
            latent_space,
        ]

        self.encoder_features = torch.nn.Sequential()

        for i in range(len(encoder_features) - 1):
            self.encoder_features.append(
                torch.nn.Conv1d(
                    encoder_features[i],
                    encoder_features[i + 1],
                    sizes_convs_encode[i],
                    padding=sizes_convs_encode[i] // 2,
                )
            )
            self.encoder_features.append(
                MatMulLayer(sizes_downsample[i], sizes_downsample[i + 1])
                # torch.nn.Linear(sizes_downsample[i], sizes_downsample[i + 1])
            )
            self.encoder_features.append(activation())

        self.encoder_linear = torch.nn.Sequential()

        for i in range(len(encoder_linear) - 1):
            self.encoder_linear.append(
                torch.nn.Linear(encoder_linear[i], encoder_linear[i + 1])
            )


    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.encoder_features(x)

        x = torch.flatten(x, start_dim=1, end_dim=2)

        x = self.encoder_linear(x)

        return x

class Decoder(torch.nn.Module):
    def __init__(
            self,
            encoder_features,
            sizes_downsample,
            latent_space,
            activation,
        ):
        super().__init__()
        sizes_convs_encode = [3
                              for _ in range(len(encoder_features))]
        sizes_convs_decode = sizes_convs_encode[::-1]
        
        self.size_view = sizes_downsample[-1]
        
        decoder_features = encoder_features[::-1]
        decoder_features[-1] = decoder_features[-2]
        
        decoder_linear = [
            latent_space,
            encoder_features[-1] * sizes_downsample[-1],
        ]
        
        sizes_upsample = sizes_downsample[::-1]

        self.decoder_linear = torch.nn.Sequential()

        for i in range(len(decoder_linear) - 1):
            self.decoder_linear.append(
                torch.nn.Linear(decoder_linear[i], decoder_linear[i + 1])
            )

        self.decoder_features = torch.nn.Sequential()

        for i in range(len(decoder_features) - 1):
            self.decoder_features.append(
                MatMulLayer(sizes_upsample[i], sizes_upsample[i + 1])
                # torch.nn.Linear(sizes_upsample[i], sizes_upsample[i + 1])
            )

            self.decoder_features.append(
                torch.nn.Conv1d(
                    decoder_features[i],
                    decoder_features[i + 1],
                    sizes_convs_decode[i],
                    padding=sizes_convs_decode[i] // 2,
                )
            )
            self.decoder_features.append(activation())

        self.last_conv = torch.nn.Conv1d(
            decoder_features[-1],
            3,
            sizes_convs_decode[-1],
            padding=sizes_convs_decode[-1] // 2,
        )

    def forward(self, x):
        x = self.decoder_linear(x)

        x = x.view(x.shape[0], -1, self.size_view)

        x = self.decoder_features(x)

        x = self.last_conv(x)

        x = x.permute(0, 2, 1)

        return x

class LearnedPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()

        encoder_features = [3,32,64]

        sizes_downsample = [1024,64,32]

        self.latent_space = 256

        self.activation = nn.ELU

        # if opt["activation_autoencoder"] == "ReLU":
        #     self.activation = nn.ReLU
        # elif opt["activation_autoencoder"] == "Tanh":
        #     self.activation = nn.Tanh
        # elif opt["activation_autoencoder"] == "Sigmoid":
        #     self.activation = nn.Sigmoid
        # elif opt["activation_autoencoder"] == "LeakyReLU":
        #     self.activation = nn.LeakyReLU
        # elif opt["activation_autoencoder"] == "ELU":
        #     self.activation = nn.ELU
        # else:
        #     print("Wrong activation")
        #     exit()

        # Encoder

        self.encoder = Encoder(encoder_features, sizes_downsample, self.latent_space, self.activation)

        # Decoder

        self.decoder = Decoder(encoder_features, sizes_downsample, self.latent_space, self.activation)

    def enc(self, x):
        return self.encoder(x)

    def dec(self, x):
        return self.decoder(x)

    def forward(self, x):
        unsqueeze = False

        if x.dim() == 2:
            x = x.unsqueeze(0)
            unsqueeze = True

        latent = self.enc(x)
        output = self.dec(latent)

        if unsqueeze:
            output = output.squeeze(0)

        return output
