from __future__ import print_function
import tensorflow as tf


class Generator:
    def __init__(self, size=4):
        self.filters = {'fc': 1024, 'deconv1': 512, 'deconv2': 256, 'deconv3': 128, 'deconv4': 3}
        self.size = size
        self.dropout = 0.4
        self.reuse = False

    def __call__(self, inputs, training=False):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope('g', reuse=self.reuse):
            # first fully connected layer
            with tf.variable_scope('fc'):
                outputs = tf.layers.dense(inputs, self.filters['fc'] * self.size * self.size)
                outputs = tf.reshape(outputs, [-1, self.size, self.size, self.filters['fc']])
                outputs = tf.layers.batch_normalization(outputs, training=training, momentum=0.9)
                outputs = tf.nn.relu(outputs)
                outputs = tf.layers.dropout(outputs, rate=self.dropout, training=training, name='outputs')
            # deconvolution layers (4)
            with tf.variable_scope('deconv1'):
                outputs = tf.layers.conv2d_transpose(outputs, self.filters['deconv1'], [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
                outputs = tf.layers.batch_normalization(outputs, training=training, momentum=0.9)
                outputs = tf.nn.relu(outputs)
                outputs = tf.layers.dropout(outputs, rate=self.dropout, training=training, name='outputs')
            with tf.variable_scope('deconv2'):
                outputs = tf.layers.conv2d_transpose(outputs, self.filters['deconv2'], [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
                outputs = tf.layers.batch_normalization(outputs, training=training, momentum=0.9)
                outputs = tf.nn.relu(outputs, name='outputs')
            with tf.variable_scope('deconv3'):
                outputs = tf.layers.conv2d_transpose(outputs, self.filters['deconv3'], [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
                outputs = tf.layers.batch_normalization(outputs, training=training, momentum=0.9)
                outputs = tf.nn.relu(outputs, name='outputs')
            with tf.variable_scope('deconv4'):
                outputs = tf.layers.conv2d_transpose(outputs, self.filters['deconv4'], [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
            # output images in [0,1]
            with tf.variable_scope('sigmoid'):
                outputs = tf.sigmoid(outputs, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return outputs


class Discriminator:
    def __init__(self, depths=[64, 128, 256, 512]):
        self.filters = {'init': 3, 'conv1': 64, 'conv2': 128, 'conv3': 256, 'conv4': 512}
        self.dropout = 0.4
        self.reuse = False

    def __call__(self, inputs, training=False, name=''):
        def leaky_relu(x, leak=0.2, name='leaky_output'):
            return tf.maximum(x, x * leak, name=name)
        outputs = tf.convert_to_tensor(inputs)

        with tf.name_scope('d' + name), tf.variable_scope('d', reuse=self.reuse):
            # convolution layers (4)
            with tf.variable_scope('conv1'):
                outputs = tf.layers.conv2d(outputs, self.filters['conv1'], [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
                outputs = tf.layers.batch_normalization(outputs, training=training, momentum=0.9)
                outputs = leaky_relu(outputs)
                outputs = tf.layers.dropout(outputs, rate=self.dropout, training=training, name='outputs')
            with tf.variable_scope('conv2'):
                outputs = tf.layers.conv2d(outputs, self.filters['conv2'], [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
                outputs = tf.layers.batch_normalization(outputs, training=training, momentum=0.9)
                outputs = leaky_relu(outputs)
                outputs = tf.layers.dropout(outputs, rate=self.dropout, training=training, name='outputs')
            with tf.variable_scope('conv3'):
                outputs = tf.layers.conv2d(outputs, self.filters['conv3'], [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
                outputs = tf.layers.batch_normalization(outputs, training=training, momentum=0.9)
                outputs = leaky_relu(outputs, name='outputs')
            with tf.variable_scope('conv4'):
                outputs = tf.layers.conv2d(outputs, self.filters['conv4'], [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
                outputs = tf.layers.batch_normalization(outputs, training=training, momentum=0.9)
                outputs = leaky_relu(outputs, name='outputs')
            with tf.variable_scope('classify'):
                batch_size = outputs.get_shape()[0].value
                reshape = tf.reshape(outputs, [batch_size, -1])
                outputs = tf.layers.dense(reshape, 2, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
        return outputs


class DCGAN:
    def __init__(self, batch_size=128, s_size=4, z_dim=100):
        self.batch_size = batch_size
        self.s_size = s_size
        self.z_dim = z_dim
        self.g = Generator(size=self.s_size)
        self.d = Discriminator()
        self.z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.0, maxval=1.0)

    def loss(self, train_data):
        """ build models, calculate losses (given training data) """
        generated = self.g(self.z, training=True)
        g_outputs = self.d(generated, training=True, name='g')
        t_outputs = self.d(train_data, training=True, name='t')
        # add each losses to collection
        tf.add_to_collection(
            'g_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.batch_size], dtype=tf.int64),
                    logits=g_outputs)))
        tf.add_to_collection(
            'd_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.batch_size], dtype=tf.int64),
                    logits=t_outputs)))
        tf.add_to_collection(
            'd_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.zeros([self.batch_size], dtype=tf.int64),
                    logits=g_outputs)))
        return {
            self.g: tf.add_n(tf.get_collection('g_losses'), name='total_g_loss'),
            self.d: tf.add_n(tf.get_collection('d_losses'), name='total_d_loss'),
        }

    def train(self, losses, learning_rate=0.0002, beta1=0.5):
        """ return training op given losses dict """
        g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        g_opt_op = g_opt.minimize(losses[self.g], var_list=self.g.variables)
        d_opt_op = d_opt.minimize(losses[self.d], var_list=self.d.variables)
        with tf.control_dependencies([g_opt_op, d_opt_op]):
            return tf.no_op(name='train')

    def sample_images(self, row=8, col=8, inputs=None):
        if inputs is None:
            inputs = self.z
        images = self.g(inputs, training=True)
        images = tf.image.convert_image_dtype(tf.div(tf.add(images, 1.0), 2.0), tf.uint8)
        images = [image for image in tf.split(images, self.batch_size, axis=0)]
        rows = []
        for i in range(row):
            rows.append(tf.concat(images[col * i + 0:col * i + col], 2))
        image = tf.concat(rows, 1)
        return tf.image.encode_jpeg(tf.squeeze(image, [0]))

# Load input data
BATCH_SIZE = 128
s_size = 6
CROP_IMAGE_SIZE = s_size * 16


# mnist = input_data.read_data_sets('~/Documents/dl-code/DCGAN-tensorflow/data/mnist/')
def inputs(batch_size, s_size):
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once('/home/quantumcoder/Documents/dl-code/DCGAN-tensorflow/data/celebA/*.jpg'))
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    image = tf.cast(tf.image.decode_jpeg(value, channels=3), tf.float32)
    image = tf.image.resize_image_with_crop_or_pad(image, CROP_IMAGE_SIZE, CROP_IMAGE_SIZE)
    image = tf.image.random_flip_left_right(image)

    min_queue_examples = 1000     # no. of train examples to use
    images = tf.train.shuffle_batch(
        [image],
        batch_size=batch_size,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
    tf.summary.image('images', images)
    return tf.div(tf.image.resize_images(images, [s_size * 16, s_size * 16]), 255)      # return images in [0,1]

# Train DCGAN
dcgan = DCGAN(batch_size=BATCH_SIZE, s_size=s_size, z_dim=100)
train_data = inputs(dcgan.batch_size, dcgan.s_size)
losses = dcgan.loss(train_data)
train_op = dcgan.train(losses)

# Add L2 loss between features to GAN
feature_loss_weight = 0.05
graph = tf.get_default_graph()
features_g = tf.reduce_mean(graph.get_tensor_by_name('dg/d/conv4/outputs:0'), 0)
features_t = tf.reduce_mean(graph.get_tensor_by_name('dt/d/conv4/outputs:0'), 0)
losses[dcgan.g] += tf.multiply(tf.nn.l2_loss(features_g - features_t), feature_loss_weight)

logdir = './logs/'
tf.summary.scalar('g loss', losses[dcgan.g])
tf.summary.scalar('d loss', losses[dcgan.d])
train_op = dcgan.train(losses, learning_rate=0.0001)
summary_op = tf.summary.merge_all()

g_saver = tf.train.Saver(dcgan.g.variables, max_to_keep=15)
d_saver = tf.train.Saver(dcgan.d.variables, max_to_keep=15)
g_checkpoint_path = logdir + 'g.ckpt'
d_checkpoint_path = logdir + 'd.ckpt'

num_epochs = 10000
log_frequency = 100
checkpt_frequency = 500
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)
    sess.run(tf.global_variables_initializer())

    sample_z = sess.run(tf.random_uniform([dcgan.batch_size, dcgan.z_dim], minval=-1.0, maxval=1.0))
    images = dcgan.sample_images(5, 5, inputs=sample_z)

    tf.train.start_queue_runners(sess=sess)

    for step in range(num_epochs):
        _, g_loss_value, d_loss_value = sess.run([train_op, losses[dcgan.g], losses[dcgan.d]])
        print ("Losses: (G: {}, D: {})".format(g_loss_value, d_loss_value))

        # save generated images
        if step % log_frequency == 0:
            # summary
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
            # sample images
            filename = './generated/' + ('%05d.jpg' % step)
            with open(filename, 'wb') as f:
                f.write(sess.run(images))
        # save variables
        if step % checkpt_frequency == 0:
            g_saver.save(sess, g_checkpoint_path, global_step=step)
            d_saver.save(sess, d_checkpoint_path, global_step=step)


if __name__ == '__main__':
    tf.app.run()
