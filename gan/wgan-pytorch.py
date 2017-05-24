import torch
import torch.nn
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data

# Load data
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

# Model hyperparameters
img_height = img_width = 28
mb_size = 32
z_dim = 100
X_dim = img_height * img_width
h_dim = 128
generator_lr = 1e-4
discriminator_lr = 1e-3
w_lim = 0.01

# Generator model
generator_model = torch.nn.Sequential(
    torch.nn.Linear(z_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, X_dim),
    torch.nn.Sigmoid()
)

# Dicriminator model
discriminator_model = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, 1),
)


# reset gradient to 0 for both models
def reset_gradients():
    generator_model.zero_grad()
    discriminator_model.zero_grad()

generator_opt = optim.Adam(generator_model.parameters(), lr=generator_lr)
discriminator_opt = optim.Adam(discriminator_model.parameters(), lr=discriminator_lr)

plot_frequency = 100
num_epochs = 10000
cnt = 0
for it in range(num_epochs):
    for _ in range(5):
        # Sample data
        z = Variable(torch.randn(mb_size, z_dim))
        X_train, _ = mnist.train.next_batch(mb_size)
        X = Variable(torch.from_numpy(X_train))

        # Dicriminator forward-loss-backward-update
        G_sample = generator_model(z)
        D_real = discriminator_model(X)
        D_fake = discriminator_model(G_sample)

        discriminator_loss = -(torch.mean(D_real) - torch.mean(D_fake))

        discriminator_loss.backward()
        discriminator_opt.step()

        # Weight clipping
        for w in discriminator_model.parameters():
            w.data.clamp_(-w_lim, w_lim)

        # reset gradient
        reset_gradients()

    # Generator forward-loss-backward-update
    X, _ = mnist.train.next_batch(mb_size)
    X = Variable(torch.from_numpy(X))
    z = Variable(torch.randn(mb_size, z_dim))

    G_sample = generator_model(z)
    D_fake = discriminator_model(G_sample)

    generator_loss = -torch.mean(D_fake)

    generator_loss.backward()
    generator_opt.step()

    # reset gradient
    reset_gradients()

    # Print and plot every now and then
    if it % plot_frequency == 0:
        print('Iteration:{}; D_loss: {}; G_loss: {}'
              .format(it, discriminator_loss.data.numpy(), generator_loss.data.numpy()))

        num_plot_samples = 9
        samples = generator_model(z).data.numpy()[:num_plot_samples]

        fig = plt.figure(figsize=(3, 3))
        gs = gridspec.GridSpec(3, 3)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(img_width, img_height), cmap='Greys_r')

        plt.savefig('generated/wgan-{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)
