# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914, R0902

import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import utils


def mnist_data_iteratior():
    mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
    def iterator(hparams, num_batches):
        for _ in range(num_batches):
            yield mnist.train.next_batch(hparams.batch_size)
    return iterator


def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0 / (fan_in + fan_out))
    high = constant*np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


def encoder(hparams, x_ph, scope_name, reuse):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        w1 = tf.get_variable('w1', initializer=xavier_init(hparams.n_input, hparams.n_hidden_recog_1))
        b1 = tf.get_variable('b1', initializer=tf.zeros([hparams.n_hidden_recog_1], dtype=tf.float32))
        hidden1 = hparams.transfer_fct(tf.matmul(x_ph, w1) + b1)

        w2 = tf.get_variable('w2', initializer=xavier_init(hparams.n_hidden_recog_1, hparams.n_hidden_recog_2))
        b2 = tf.get_variable('b2', initializer=tf.zeros([hparams.n_hidden_recog_2], dtype=tf.float32))
        hidden2 = hparams.transfer_fct(tf.matmul(hidden1, w2) + b2)

        w3 = tf.get_variable('w3', initializer=xavier_init(hparams.n_hidden_recog_2, hparams.n_z))
        b3 = tf.get_variable('b3', initializer=tf.zeros([hparams.n_z], dtype=tf.float32))
        z_mean = tf.matmul(hidden2, w3) + b3

        w4 = tf.get_variable('w4', initializer=xavier_init(hparams.n_hidden_recog_2, hparams.n_z))
        b4 = tf.get_variable('b4', initializer=tf.zeros([hparams.n_z], dtype=tf.float32))
        z_log_sigma_sq = tf.matmul(hidden2, w4) + b4

    return z_mean, z_log_sigma_sq


def sampler(hparams, z_mean, z_log_sigma_sq):
    n_z = hparams.n_z
    eps = tf.random_normal((hparams.batch_size, n_z), 0, 1, dtype=tf.float32)

    # z = mu + sigma*epsilon
    z_sigma = tf.sqrt(tf.exp(z_log_sigma_sq))
    z = z_mean + z_sigma*eps
    return z


def generator(hparams, z, scope_name, reuse):

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        w1 = tf.get_variable('w1', initializer=xavier_init(hparams.n_z, hparams.n_hidden_gener_1))
        b1 = tf.get_variable('b1', initializer=tf.zeros([hparams.n_hidden_gener_1], dtype=tf.float32))
        hidden1 = hparams.transfer_fct(tf.matmul(z, w1) + b1)

        w2 = tf.get_variable('w2', initializer=xavier_init(hparams.n_hidden_gener_1, hparams.n_hidden_gener_2))
        b2 = tf.get_variable('b2', initializer=tf.zeros([hparams.n_hidden_gener_2], dtype=tf.float32))
        hidden2 = hparams.transfer_fct(tf.matmul(hidden1, w2) + b2)

        w3 = tf.get_variable('w3', initializer=xavier_init(hparams.n_hidden_gener_2, hparams.n_input))
        b3 = tf.get_variable('b3', initializer=tf.zeros([hparams.n_input], dtype=tf.float32))
        logits = tf.matmul(hidden2, w3) + b3
        x_reconstr_mean = tf.nn.sigmoid(logits)

    return logits, x_reconstr_mean


def get_loss(x, logits, z_mean, z_log_sigma_sq):
    reconstr_losses = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits), 1)
    latent_losses = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
    total_loss = tf.reduce_mean(reconstr_losses + latent_losses, name='total_loss')
    return total_loss


class Hparams(object):
    def __init__(self):
        self.n_hidden_recog_1 = 500  # 1st layer encoder neurons
        self.n_hidden_recog_2 = 500  # 2nd layer encoder neurons
        self.n_hidden_gener_1 = 500  # 1st layer decoder neurons
        self.n_hidden_gener_2 = 500  # 2nd layer decoder neurons
        self.n_input = 784  # MNIST data input (img shape: 28*28)
        self.n_z = 20   # dimensionality of latent space
        self.transfer_fct = tf.nn.softplus
        self.learning_rate = 0.001
        self.batch_size = 100
        self.training_epochs = 100
        self.summary_epoch = 1
        self.ckpt_epoch = 5
        self.num_samples = 60000
        self.ckpt_dir = './models/mnist-vae/'
        self.sample_dir = './samples/mnist-vae/'


def main():
    hparams = Hparams()

    # Set up some stuff according to hparams
    utils.set_up_dir(hparams.ckpt_dir)
    utils.set_up_dir(hparams.sample_dir)
    utils.print_hparams(hparams)

    x_ph = tf.placeholder(tf.float32, [None, hparams.n_input], name='x_ph')
    z_mean, z_log_sigma_sq = encoder(hparams, x_ph, 'enc', reuse=False)
    z = sampler(hparams, z_mean, z_log_sigma_sq)
    z_ph = tf.placeholder(tf.float32, [None, hparams.n_z], name='x_ph')
    logits, x_reconstr_mean = generator(hparams, z, 'gen', reuse=False)
    _, x_sample = generator(hparams, z_ph, 'gen', reuse=True)

    # define loss and update op
    total_loss = get_loss(x_ph, logits, z_mean, z_log_sigma_sq)
    opt = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)
    update_op = opt.minimize(total_loss)

    # Sanity checks
    for var in tf.global_variables():
        print var.op.name
    print ''

    # Get a new session
    sess = tf.Session()

    # Model checkpointing setup
    model_saver = tf.train.Saver()

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Attempt to restore variables from checkpoint
    start_epoch = utils.try_restore(hparams, sess, model_saver)

    # Get data iterator
    iterator = mnist_data_iteratior()

    # Training
    for epoch in range(start_epoch+1, hparams.training_epochs):
        avg_loss = 0.0
        num_batches = hparams.num_samples // hparams.batch_size
        batch_num = 0
        for (x_batch_val, _) in iterator(hparams, num_batches):
            batch_num += 1
            feed_dict = {x_ph: x_batch_val}
            _, loss_val = sess.run([update_op, total_loss], feed_dict=feed_dict)
            avg_loss += loss_val / hparams.num_samples * hparams.batch_size

            if batch_num % 100 == 0:
                x_reconstr_mean_val = sess.run(x_reconstr_mean, feed_dict={x_ph: x_batch_val})

                z_val = np.random.randn(hparams.batch_size, hparams.n_z)
                x_sample_val = sess.run(x_sample, feed_dict={z_ph: z_val})

                utils.save_images(np.reshape(x_reconstr_mean_val, [-1, 28, 28]),
                                  [10, 10],
                                  '{}/reconstr_{:02d}_{:04d}.png'.format(hparams.sample_dir, epoch, batch_num))
                utils.save_images(np.reshape(x_batch_val, [-1, 28, 28]),
                                  [10, 10],
                                  '{}/orig_{:02d}_{:04d}.png'.format(hparams.sample_dir, epoch, batch_num))
                utils.save_images(np.reshape(x_sample_val, [-1, 28, 28]),
                                  [10, 10],
                                  '{}/sampled_{:02d}_{:04d}.png'.format(hparams.sample_dir, epoch, batch_num))


        if epoch % hparams.summary_epoch == 0:
            print "Epoch:", '%04d' % (epoch), 'Avg loss = {:.9f}'.format(avg_loss)

        if epoch % hparams.ckpt_epoch == 0:
            save_path = os.path.join(hparams.ckpt_dir, 'mnist_vae_model')
            model_saver.save(sess, save_path, global_step=epoch)



    save_path = os.path.join(hparams.ckpt_dir, 'mnist_vae_model')
    model_saver.save(sess, save_path, global_step=hparams.training_epochs-1)


if __name__ == '__main__':
    main()
