from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers.core import Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop, Adam
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image
# import math
import random
from tqdm import tqdm

img_rows, img_cols = 28, 28


# load mnist data - shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype(np.float32) / 255.0
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

print 'X_train shape:', X_train.shape
print 'No. of train samples:', X_train.shape[0]

# Model hyperparameters
generator_input_dim = 100
dropout = 0.4
discriminator_opt = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
# discriminator_opt = Adam(lr=1e-3)
combined_opt = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
# combined_opt = Adam(lr=1e-4)

# Construct generator model
generator_model = Sequential()
generator_model.add(Dense(input_dim=generator_input_dim, output_dim=1024))
generator_model.add(BatchNormalization(momentum=0.9))
generator_model.add(Activation('relu'))
generator_model.add(Dropout(dropout))
generator_model.add(Dense(128 * 7 * 7))
generator_model.add(BatchNormalization(momentum=0.9))
generator_model.add(Activation('relu'))
generator_model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
generator_model.add(UpSampling2D(size=(2, 2)))
generator_model.add(Convolution2D(64, 5, 5, border_mode='same', init='glorot_uniform'))
generator_model.add(Activation('relu'))
generator_model.add(UpSampling2D(size=(2, 2)))
generator_model.add(Convolution2D(1, 5, 5, border_mode='same', init='glorot_uniform'))
generator_model.add(Activation('sigmoid'))
print "Generator:"
generator_model.summary()

# Discriminator model
discriminator_model = Sequential()
discriminator_model.add(Convolution2D(64, 5, 5, border_mode='same', input_shape=(28, 28, 1)))
discriminator_model.add(LeakyReLU(alpha=0.2))
# discriminator_model.add(Dropout(dropout))
discriminator_model.add(MaxPooling2D(pool_size=(2, 2)))
discriminator_model.add(Convolution2D(128, 5, 5))
discriminator_model.add(LeakyReLU(alpha=0.2))
discriminator_model.add(MaxPooling2D(pool_size=(2, 2)))
discriminator_model.add(Flatten())
discriminator_model.add(Dense(1024))
discriminator_model.add(Activation('tanh'))
discriminator_model.add(Dense(2))
discriminator_model.add(Activation('softmax'))
discriminator_model.compile(loss='categorical_crossentropy', optimizer=discriminator_opt, metrics=['accuracy'])
print "Discriminator:"
discriminator_model.summary()

# Stacked model
stacked_model = Sequential()
stacked_model.add(generator_model)
discriminator_model.trainable = False
stacked_model.add(discriminator_model)
stacked_model.compile(loss='binary_crossentropy', optimizer=combined_opt, metrics=['accuracy'])
# stacked_model.summary()

# Training

# Pre-train discriminator network
ntrain = 10000
trainidx = random.sample(range(0, X_train.shape[0]), ntrain)
XT = X_train[trainidx, :, :, :]

noise_gen = np.random.uniform(0, 1, size=[XT.shape[0], 100])
generated_images = generator_model.predict(noise_gen)
X = np.concatenate((XT, generated_images))
n = XT.shape[0]
y = np.zeros([2 * n, 2])
y[:n, 1] = 1
y[n:, 0] = 1

discriminator_model.trainable = True
discriminator_model.fit(X, y, nb_epoch=2, batch_size=128)
print "Done initial training of generator"


def plot_loss(d_losses, g_losses):
    """ Plot losses of generator and discriminator """
    plt.figure(figsize=(10, 8))
    plt.plot(d_losses, label='discriminitive loss')
    plt.plot(g_losses, label='generative loss')
    plt.legend()
    plt.show()


def show_generated_images(n_ex=16, dim=(4, 4), figsize=(10, 10)):
    noise = np.random.uniform(0, 1, size=[n_ex, 100])
    generated_images = generator_model.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        img = generated_images[i, 0, :, :]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Start training
num_epochs = 5000
plot_frequency = 25
BATCH_SIZE = 32

d_losses = []
g_losses = []

for e in tqdm(range(num_epochs)):
    # Make generative images
    image_batch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE), :, :, :]
    noise_gen = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
    generated_images = generator_model.predict(noise_gen)

    # Train discriminator on generated images
    X = np.concatenate((image_batch, generated_images))
    y = np.zeros([2 * BATCH_SIZE, 2])
    y[0:BATCH_SIZE, 1] = 1
    y[BATCH_SIZE:, 0] = 1

    discriminator_model.trainable = True
    # discriminator trained 5 times for each training cycle
    for _ in range(5):
        d_loss = discriminator_model.train_on_batch(X, y)
    d_losses.append(d_loss)

    # train Generator-Discriminator stack on input noise to non-generated output class
    noise_tr = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
    y2 = np.zeros([BATCH_SIZE, 2])
    y2[:, 1] = 1

    # make_trainable(discriminator,False)
    discriminator_model.trainable = False
    g_loss = stacked_model.train_on_batch(noise_tr, y2)
    g_losses.append(g_loss)

    # Updates plots
    if e % plot_frequency == plot_frequency - 1:
        plot_loss(d_losses, g_losses)
        # show_generated_images()
