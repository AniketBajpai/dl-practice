import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import xavier_initializer
from scipy.misc import toimage

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

# dimensions of variables
mb_size = 64
X_dim = mnist.train.images.shape[1]
z_dim = 64
y_dim = mnist.train.labels.shape[1]
h_dim = 128

# encoder - Q(z|X)
X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

Q_W1 = tf.get_variable("Q_W1", shape=[X_dim, h_dim], initializer=xavier_initializer())
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2_mu = tf.get_variable("Q_W2_mu", shape=[h_dim, z_dim], initializer=xavier_initializer())
Q_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]))

Q_W2_logvar = tf.get_variable("Q_W2_logvar", shape=[h_dim, z_dim], initializer=xavier_initializer())
Q_b2_logvar = tf.Variable(tf.zeros(shape=[z_dim]))


def Q(X):
    h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
    z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
    z_logvar = tf.matmul(h, Q_W2_logvar) + Q_b2_logvar
    return z_mu, z_logvar

# decoder - P(y|z)
P_W1 = tf.get_variable("P_W1", shape=[z_dim, h_dim], initializer=xavier_initializer())
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W2 = tf.get_variable("P_W2", shape=[h_dim, X_dim], initializer=xavier_initializer())
P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))


def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits

# TRAINING

z_mu, z_logvar = Q(X)
z_sample = z_mu + tf.exp(z_logvar / 2) * tf.random_normal(shape=tf.shape(z_mu))     # sample from normal distribution
_, logits = P(z_sample)

# Sampling from random z
X_samples, _ = P(z)

# E[log P(X|z)]
reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)
# D_KL(Q(z|X) || P(z|X))
kl_divergence_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
# VAE loss
vae_loss = tf.reduce_mean(reconstruction_loss + kl_divergence_loss)

solver = tf.train.AdamOptimizer().minimize(vae_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('outputs/'):
    os.makedirs('outputs/')

if not os.path.exists('models/'):
    os.makedirs('models/')

counter = 0
max_iter = 1000000
num_samples = 20
check_interval = 1000
save_interval = 10000

for i in range(max_iter):
    X_mb, _ = mnist.train.next_batch(mb_size)
    _, loss = sess.run([solver, vae_loss], feed_dict={X: X_mb})

    if i % check_interval == 0:
        print('Iteration: {}'.format(i))
        print('Loss: {:.4}'. format(loss))
        print('----------------------------------------')

        samples = sess.run(X_samples, feed_dict={z: np.random.randn(num_samples, z_dim)})

        for j, sample in enumerate(samples):
            dirname = 'outputs/{}'.format(str(counter).zfill(3))
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            im = toimage(sample.reshape(28, 28))
            im.save('{}/{}.png'.format(dirname, j))

        counter += 1

        if i % save_interval == 0:
            save_path = tf.train.saver.save(sess, "./models/model_{}.ckpt".format(i))
            print("Model saved in file: %s" % save_path)
