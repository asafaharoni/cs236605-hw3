import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO: Implement a CNN. Save the layers in the modules list.
        # The input shape is an image batch: (N, in_channels, H_in, W_in).
        # The output shape should be (N, out_channels, H_out, W_out).
        # You can assume H_in, W_in >= 64.
        # Architecture is up to you, but you should use at least 3 Conv layers.
        # You can use any Conv layer parameters, use pooling or only strides,
        # use any activation functions, use BN or Dropout, etc.
        # ====== YOUR CODE: ======
        modules = (
            nn.Conv2d(in_channels=in_channels, out_channels=180, kernel_size=6, stride=2, padding=1), # b, [1, 60, 31, 31]
            nn.ReLU(),
            nn.BatchNorm2d(180),
            nn.Conv2d(in_channels=180, out_channels=360, kernel_size=3, stride=2, padding=0), # b, [1, 180, 15, 15]
            nn.BatchNorm2d(360),
            nn.ReLU(),
            nn.Conv2d(in_channels=360, out_channels=540, kernel_size=3, stride=3, padding=0), # b, [1, 540, 5, 5]
            nn.BatchNorm2d(540),
            nn.ReLU(),
            nn.Conv2d(in_channels=540, out_channels=out_channels, kernel_size=2, stride=1, padding=0), # b, [1, 1024, 1, 1]
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # x0 = torch.rand((1,3,64,64))
        # print(f'The shape of x0 is {x0.shape}')
        # cnn1 =  nn.Sequential(*modules[0:3])
        # print(f'The shape of x1 is {cnn1(x0).shape}')
        # cnn2 =  nn.Sequential(*modules[0:6])
        # print(f'The shape of x2 is {cnn2(x0).shape}')
        # cnn3 =  nn.Sequential(*modules[0:9])
        # print(f'The shape of x3 is {cnn3(x0).shape}')
        # cnn4 =  nn.Sequential(*modules)
        # print(f'The shape of x4 is {cnn4(x0).shape}')
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO: Implement the "mirror" CNN of the encoder.
        # For example, instead of Conv layers use transposed convolutions,
        # instead of pooling do unpooling (if relevant) and so on.
        # You should have the same number of layers as in the Encoder,
        # and they should produce the same volumes, just in reverse order.
        # Output should be a batch of images, with same dimensions as the
        # inputs to the Encoder were.
        # ====== YOUR CODE: ======
        modules = (
            nn.ReLU(),
            nn.BatchNorm2d(in_channels),
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=540, kernel_size=2, stride=1, padding=0), # b, 540, 4, 4
            nn.ReLU(),
            nn.BatchNorm2d(540),
            nn.ConvTranspose2d(in_channels=540, out_channels=360, kernel_size=3, stride=3, padding=0), # b, 180, 8, 8
            nn.ReLU(),
            nn.BatchNorm2d(360),
            nn.ConvTranspose2d(in_channels=360, out_channels=180, kernel_size=3, stride=2, padding=0), # b, 60, 16, 16
            nn.ReLU(),
            nn.BatchNorm2d(180),
            nn.ConvTranspose2d(in_channels=180, out_channels=out_channels, kernel_size=6, stride=2, padding=1), # b, 3, 64, 64
            nn.BatchNorm2d(out_channels)
        )




        # x0 = torch.rand((1, 1024, 1, 1))
        # print(f'The shape of x0 is {x0.shape}')
        # cnn1 = nn.Sequential(*modules[0:2])
        # print(f'The shape of x1 is {cnn1(x0).shape}')
        # cnn2 = nn.Sequential(*modules[0:4])
        # print(f'The shape of x2 is {cnn2(x0).shape}')
        # cnn3 = nn.Sequential(*modules[0:7])
        # print(f'The shape of x3 is {cnn3(x0).shape}')
        # cnn4 = nn.Sequential(*modules)
        # print(f'The shape of x4 is {cnn4(x0).shape}')
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        # TODO: Add parameters needed for encode() and decode().
        # ====== YOUR CODE: ======
        self.W_hmu = nn.Linear(in_features=n_features, out_features=z_dim, bias=True)
        self.W_hlsig2 = nn.Linear(in_features=n_features, out_features=z_dim, bias=True)
        self.W_zh = nn.Linear(in_features=z_dim, out_features=n_features, bias=True)
        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h)//h.shape[0]

    def encode(self, x):
        # TODO: Sample a latent vector z given an input x.
        # 1. Use the features extracted from the input to obtain mu and
        # log_sigma2 (mean and log variance) of the posterior p(z|x).
        # 2. Apply the reparametrization trick.
        # ====== YOUR CODE: ======
        h = self.features_encoder(x)
        h = h.view(-1, torch.numel(h)//h.shape[0])
        mu = self.W_hmu(h)
        log_sigma2 = self.W_hlsig2(h)
        sigma2 = torch.exp(log_sigma2)
        random_tensor = torch.randn_like(sigma2)
        z = mu + sigma2 * random_tensor
        # ========================

        return z, mu, log_sigma2

    def decode(self, z):
        # TODO: Convert a latent vector back into a reconstructed input.
        # 1. Convert latent to features.
        # 2. Apply features decoder.
        # ====== YOUR CODE: ======
        h = self.W_zh(z)
        h = h.view(-1, *self.features_shape)
        x_rec = self.features_decoder(h)
        # ========================

        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # TODO: Sample from the model.
            # Generate n latent space samples and return their reconstructions.
            # Remember that for the model, this is like inference.
            # ====== YOUR CODE: ======
            for _ in range(n):
                z = torch.randn(self.z_dim).to(device)
                h = self.decode(z)
                h = h.squeeze(dim=0).cpu()
                samples.append(h)            # for _ in range(n):
            # ========================
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    # TODO: Implement the VAE pointwise loss calculation.
    # Remember:
    # 1. The covariance matrix of the posterior is diagonal.
    # 2. You need to average over the batch dimension.
    # ====== YOUR CODE: ======
    flat_it = lambda a: a.view(a.shape[0],-1)
    x = flat_it(x)
    xr = flat_it(xr)

    data_loss = (x-xr) ** 2
    data_loss = data_loss.sum() / data_loss.numel() / x_sigma2

    kldiv_loss = z_log_sigma2.exp().sum(dim=1)
    kldiv_loss += (z_mu ** 2).sum(dim=1)
    kldiv_loss -= z_mu.size(1)
    kldiv_loss -= z_log_sigma2.sum(dim=1)
    kldiv_loss = kldiv_loss.sum() / kldiv_loss.numel()


    loss = data_loss + kldiv_loss
    # ========================

    return loss, data_loss, kldiv_loss
