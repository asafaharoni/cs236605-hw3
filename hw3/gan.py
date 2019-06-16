from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from .autoencoder import EncoderCNN, DecoderCNN


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        # To extract image features you can use the EncoderCNN from the VAE
        # section or implement something new.
        # You can then use either an affine layer or another conv layer to
        # flatten the features.
        # ====== YOUR CODE: ======
        ndf = 64 # num of feature maps
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.in_size[0], ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (aka logits, not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        # No need to apply sigmoid to obtain probability - we'll combine it
        # with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        y = self.main(x)
        y = y.view(x.shape[0], -1)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        # To combine image features you can use the DecoderCNN from the VAE
        # section or implement something new.
        # You can assume a fixed image size.
        # ====== YOUR CODE: ======
        ngf = 64  # Size of feature maps in generator
        nc = 3 # Number of channels in the training images.
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should track
        gradients or not. I.e., whether they should be part of the generator's
        computation graph or standalone tensors.
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        # Generate n latent space samples and return their reconstructions.
        # Don't use a loop.
        # ====== YOUR CODE: ======
        with torch.set_grad_enabled(with_grad):
            z = torch.randn(n,self.z_dim).to(device)
            samples = self.forward(z)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        # Don't forget to make sure the output instances have the same scale
        # as the original (real) images.
        # ====== YOUR CODE: ======
        z = z.unsqueeze(dim=2).unsqueeze(dim=3)
        x = self.main(z)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.

    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).

    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).

    :param data_label: 0 or 1, label of instances coming from the real dataset.

    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].

    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO: Implement the discriminator loss.
    # See torch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    criterion = nn.BCEWithLogitsLoss()

    noise = (torch.rand_like(y_data)-0.5)*label_noise
    data_labels = data_label + noise
    loss_data = criterion(y_data, data_labels)

    noise = (torch.rand_like(y_data)-0.5)*label_noise
    generated_labels = noise + 0
    loss_generated = criterion(y_generated, generated_labels)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    # TODO: Implement the Generator loss.
    # Think about what you need to compare the input to, in order to
    # formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    criterion = nn.BCEWithLogitsLoss()
    labels = torch.ones_like(y_generated) * data_label
    loss = criterion(y_generated, labels)
    # ========================
    return loss


def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    # 1. Show the discriminator real and generated data
    # 2. Calculate discriminator loss
    # 3. Update discriminator parameters
    # ====== YOUR CODE: ======
    n = x_data.size()[0]
    dsc_optimizer.zero_grad()
    dsc_loss = dsc_loss_fn(dsc_model(x_data), dsc_model(gen_model.sample(n)))
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    # 1. Show the discriminator generated data
    # 2. Calculate generator loss
    # 3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()
    gen_loss = gen_loss_fn(dsc_model(gen_model.sample(n, with_grad=True)))
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()
