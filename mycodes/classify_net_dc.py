from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
from datetime import datetime
import os.path
import time
import random
import tensorflow as tf
import numpy as np
import argparse
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, choices=['train', 'validate', 'predict'], default='train',
                    help='Net work mode: train 、 validate and predict.')
parser.add_argument('--label_txt', type=str, default='labels.txt',
                    help="when mode is 'validate' or 'predict' , provide a  Directory where to acquire all labels ")
parser.add_argument("--supervisor", default=None,
                    help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--checkpoint", default=None,
                    help="directory with summary to resume training from or use for testing")
parser.add_argument('--output_dir', type=str, default='classifynet',
                    help='Directory where to write event logs 、rained models and checkpoints.')
parser.add_argument('--data_dir', type=str, default='classify_train',
                    help='Path to the data directory containing classification images patches. Multiple directories are separated with colon.')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='Number of epochs to run.')
parser.add_argument('--batch_size', type=int, default=30,
                    help='Number of images to process in a batch.')
parser.add_argument('--image_size', type=int, default=256,
                    help='Image size (height, width) in pixels.')
parser.add_argument('--epoch_size', type=int,
                    help='Number of batches per epoch. Auto caculate')
parser.add_argument('--embedding_size', type=int, default=128,
                    help='Dimensionality of the embedding.')
parser.add_argument('--random_flip',default=True,
                    help='Performs random horizontal flipping of training images.', action='store_true')
parser.add_argument('--keep_probability', type=float,
                    help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
parser.add_argument('--weight_decay', type=float,
                    help='L2 weight regularization.', default=5e-4)
parser.add_argument('--center_loss_factor', type=float,
                    help='Center loss factor.', default=2e-2)
parser.add_argument('--center_loss_alfa', type=float,
                    help='Center update rate for center loss.', default=0.9)
parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                    help='The optimization algorithm to use', default='ADAM')
parser.add_argument('--learning_rate', type=float, default=-1,
                    help='Initial learning rate. If set to a negative value a learning rate ' +
                         'schedule can be specified in the file "learning_rate_schedule.txt"')
parser.add_argument('--learning_rate_decay_epochs', type=int,
                    help='Number of epochs between learning rate decay.', default=100)
parser.add_argument('--learning_rate_decay_factor', type=float,
                    help='Learning rate decay factor.', default=1.0)
parser.add_argument('--moving_average_decay', type=float,
                    help='Exponential decay for tracking of training parameters.', default=0.9999)
parser.add_argument('--seed', type=int,
                    help='Random seed.')
parser.add_argument('--nrof_preprocess_threads', type=int,
                    help='Number of preprocessing (data loading and augumentation) threads.', default=4)
parser.add_argument('--log_histograms',
                    help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
parser.add_argument('--learning_rate_schedule_file', type=str, default='learning_rate.txt',
                    help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', )
args = parser.parse_args()

alllabels = []

subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
output_dir = os.path.join(args.output_dir, subdir)
if not os.path.isdir(output_dir):  # Create the output directory if it doesn't exist
    os.makedirs(output_dir)


def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate


def save_variables_and_metagraph(sess, saver, summary_writer, output_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(output_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    # metagraph_filename = os.path.join(output_dir, 'model-%s.meta' % model_name)
    # save_time_metagraph = 0
    # if not os.path.exists(metagraph_filename):
    #     print('Saving metagraph')
    #     start_time = time.time()
    #     saver.export_meta_graph(metagraph_filename)
    #     save_time_metagraph = time.time() - start_time
    #     print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    # summary = tf.Summary()
    # pylint: disable=maybe-no-member
    # summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    # summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    # summary_writer.add_summary(summary, step)


def get_dataset(base_path):
    image_paths_flat = []
    labels_flat = []
    global alllabels
    classes_num = len(alllabels)
    for i in range(classes_num):
        class_name = alllabels[i]
        imgdir = os.path.join(base_path, class_name)
        if os.path.isdir(imgdir):
            images = os.listdir(imgdir)
            image_paths = [os.path.join(imgdir, img) for img in images]
            image_paths_flat = image_paths_flat+image_paths
            labels_flat = labels_flat + [i]*len(image_paths)
    return image_paths_flat, labels_flat

def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    # tf.gather(params,indices,axis=0 )
    # 从params的axis维根据indices的参数值获取切片
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers


def _add_loss_summaries(total_loss):
    """Add summaries for losses.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def trainop(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars,
            log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')

        grads = opt.compute_gradients(total_loss, update_gradient_vars)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


# Inception-Renset-A
def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 32, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Inception-Renset-B
def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 128, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 128, [1, 7],
                                        scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 128, [7, 1],
                                        scope='Conv2d_0c_7x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Inception-Resnet-C
def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 192, [1, 3],
                                        scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [3, 1],
                                        scope='Conv2d_0c_3x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


def reduction_a(net, k, l, m, n):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, n, 3, stride=2, padding='VALID',
                                 scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1_0 = slim.conv2d(net, k, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1_0, l, 3,
                                    scope='Conv2d_0b_3x3')
        tower_conv1_2 = slim.conv2d(tower_conv1_1, m, 3,
                                    stride=2, padding='VALID',
                                    scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_conv2 = slim.conv2d(net, m, 3, stride=2, padding='VALID',
                                     scope='Conv2d_1a_3x3')
    net = tf.concat([tower_conv, tower_conv1_2, tower_conv2], 3)
    return net


def reduction_b(net):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                   padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv2_1 = slim.conv2d(tower_conv2, 256, 3,
                                    scope='Conv2d_0b_3x3')
        tower_conv2_2 = slim.conv2d(tower_conv2_1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_3'):
        tower_conv3 = slim.conv2d(net,896, 3, stride=2, padding='VALID',
                                     scope='Conv2d_1a_3x3')
    net = tf.concat([tower_conv_1, tower_conv1_1,
                     tower_conv2_2, tower_conv3], 3)
    return net


# inception resnet v1
def inference(images, keep_probability, phase_train=True, bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return inception_resnet_v1(images, is_training=phase_train,
                                   dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size,
                                   reuse=reuse)


def inception_resnet_v1(inputs, is_training=True, dropout_keep_prob=0.8, bottleneck_layer_size=128, reuse=None,
                        scope='InceptionResnetV1'):
    """Creates the Inception Resnet V1 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
            able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
    end_points = {}

    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
                # 149 x 149 x 32
                net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
                                  scope='Conv2d_1a_3x3')
                end_points['Conv2d_1a_3x3'] = net
                # 147 x 147 x 32
                net = slim.conv2d(net, 32, 3, padding='VALID',
                                  scope='Conv2d_2a_3x3')
                end_points['Conv2d_2a_3x3'] = net
                # 147 x 147 x 64
                net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                end_points['Conv2d_2b_3x3'] = net
                # 73 x 73 x 64
                net = slim.conv2d(net, 64, 3, stride=2, padding='VALID',
                                      scope='Conv2d_3a_3x3')
                end_points['Conv2d_3a_3x3'] = net
                # 73 x 73 x 80
                net = slim.conv2d(net, 80, 1, padding='VALID',
                                  scope='Conv2d_3b_1x1')
                end_points['Conv2d_3b_1x1'] = net
                # 71 x 71 x 192
                net = slim.conv2d(net, 192, 3, padding='VALID',
                                  scope='Conv2d_4a_3x3')
                end_points['Conv2d_4a_3x3'] = net
                # 35 x 35 x 256
                net = slim.conv2d(net, 256, 3, stride=2, padding='VALID',
                                  scope='Conv2d_4b_3x3')
                end_points['Conv2d_4b_3x3'] = net

                # 5 x Inception-resnet-A
                net = slim.repeat(net, 5, block35, scale=0.17)

                # Reduction-A
                with tf.variable_scope('Mixed_6a'):
                    net = reduction_a(net, 192, 192, 256, 384)
                end_points['Mixed_6a'] = net

                # 10 x Inception-Resnet-B
                net = slim.repeat(net, 10, block17, scale=0.10)

                # Reduction-B
                with tf.variable_scope('Mixed_7a'):
                    net = reduction_b(net)
                end_points['Mixed_7a'] = net

                # 5 x Inception-Resnet-C
                net = slim.repeat(net, 5, block8, scale=0.20)
                net = block8(net, activation_fn=None)

                with tf.variable_scope('Logits'):
                    end_points['PrePool'] = net
                    # pylint: disable=no-member
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                          scope='AvgPool_1a_7x7')
                    net = slim.flatten(net)

                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='Dropout')

                    end_points['PreLogitsFlatten'] = net

                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None,
                                           scope='Bottleneck', reuse=False)

    return net, end_points


def train(sess, epoch, max_epochs, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder,
          labels_placeholder,
          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
          loss, train_op, summary_op, summary_writer, regularization_losses,cross_entropy_mean):
    batch_number = 0

    # 获取学习率
    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = get_learning_rate_from_file(args.learning_rate_schedule_file, epoch)

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch), 1)
    image_paths_array = np.expand_dims(np.array(image_epoch), 1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: True,
                     batch_size_placeholder: args.batch_size}
        if ((batch_number % 100 == 0) | batch_number == args.epoch_size - 1):
            # 每100个batch进行一次summary
            total_loss, _, reg_loss,cross_loss, summary_str = sess.run(
                [loss, train_op, regularization_losses,cross_entropy_mean,summary_op], feed_dict=feed_dict)
            step = sess.run(global_step)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            total_loss, _, reg_loss ,cross_loss = sess.run([loss, train_op, regularization_losses,cross_entropy_mean],
                                               feed_dict=feed_dict)
            step = sess.run(global_step)
        duration = time.time() - start_time
        rate = args.batch_size / (time.time() - start_time)
        remaining = (args.epoch_size * max_epochs - step) * args.batch_size / rate
        print('Epoch: [%d][%d/%d]\tTime %.3f\tTotal_Loss %2.3f\tCrossLoss %2.3f\tRegLoss %2.3f\timage/sec %0.1f\tremaining %dm' %
              (epoch, batch_number + 1, args.epoch_size, duration, total_loss,cross_loss, np.sum(reg_loss), rate, remaining / 60))
        f = open(os.path.join(output_dir, 'printRecord.txt'), 'a')
        f.write('Epoch: [%d][%d/%d]\tTime %.3f\tTotal_Loss %2.3f\tCrossLoss %2.3f\tRegLoss %2.3f\timage/sec %0.1f\tremaining %dm\n' %
                (
                    epoch, batch_number + 1, args.epoch_size, duration, total_loss, cross_loss,np.sum(reg_loss), rate,
                    remaining / 60))
        f.close()
        batch_number += 1
        train_time += duration

    return step


def train1(sess, epoch, max_epochs, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder,
          labels_placeholder,
          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
          loss, train_op, summary_op, summary_writer, regularization_losses,cross_entropy_mean):
    batch_number = 0

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch), 1)
    image_paths_array = np.expand_dims(np.array(image_epoch), 1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {phase_train_placeholder: True,
                     batch_size_placeholder: args.batch_size}
        if ((batch_number % 100 == 0) | batch_number == args.epoch_size - 1):
            # 每100个batch进行一次summary
            total_loss,  reg_loss,cross_loss = sess.run(
                [loss, regularization_losses,cross_entropy_mean], feed_dict=feed_dict)
            step = sess.run(global_step)
        else:
            total_loss, reg_loss ,cross_loss = sess.run([loss, regularization_losses,cross_entropy_mean],
                                               feed_dict=feed_dict)
            step = sess.run(global_step)
        duration = time.time() - start_time
        rate = args.batch_size / (time.time() - start_time)
        remaining = (args.epoch_size * max_epochs - step) * args.batch_size / rate
        print('Epoch: [%d][%d/%d]\tTime %.3f\tTotal_Loss %2.3f\tCrossLoss %2.3f\tRegLoss %2.3f\timage/sec %0.1f\tremaining %dm' %
              (epoch, batch_number + 1, args.epoch_size, duration, total_loss,cross_loss, np.sum(reg_loss), rate, remaining / 60))
        f = open(os.path.join(output_dir, 'printRecord.txt'), 'a')
        f.write('Epoch: [%d][%d/%d]\tTime %.3f\tTotal_Loss %2.3f\tCrossLoss %2.3f\tRegLoss %2.3f\timage/sec %0.1f\tremaining %dm\n' %
                (
                    epoch, batch_number + 1, args.epoch_size, duration, total_loss, cross_loss,np.sum(reg_loss), rate,
                    remaining / 60))
        f.close()
        batch_number += 1
        train_time += duration

    return step


def indexOfTopk(list, top_k):
    tmplist = sorted(list, reverse=True)
    maxk_index = [list.index(one) for one in tmplist[0:top_k]]
    return maxk_index


def validate(sess, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, phase_train_placeholder,
             labels_placeholder,
             batch_size_placeholder, path_batch, label_batch, logits, predicts, accuracy_top1_batch,
             accuracy_top5_batch,
             cross_entropy_mean,
             regularization_losses, total_loss):
    
    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]
    
    # Enqueue
    labels_array = np.expand_dims(np.array(label_epoch), 1)
    image_paths_array = np.expand_dims(np.array(image_epoch), 1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

    batch_number = 0
    all_top1_acu = []
    all_top5_acu = []
    while batch_number < args.epoch_size:
        feed_dict = {phase_train_placeholder: True, batch_size_placeholder: args.batch_size}
        path_batch_in, label_batch_in, logits_in, predicts_in, accuracy_top1_batch_in, accuracy_top5_batch_in, total_loss_in, cross_loss_in, reg_loss_in \
            = sess.run(
            [path_batch, label_batch, logits, predicts, accuracy_top1_batch, accuracy_top5_batch,total_loss, 
             cross_entropy_mean,
             regularization_losses], feed_dict=feed_dict)
        batch_number += 1
        all_top1_acu = all_top1_acu + accuracy_top1_batch_in.tolist()
        all_top5_acu = all_top5_acu + accuracy_top5_batch_in.tolist()
        print("cross loss:%f\n"%(cross_loss_in))

        predicts_list = predicts_in.tolist()
        for i in range(len(predicts_list)):
            predict = predicts_list[i]
            img_path_in = path_batch_in.tolist()[i]
            img_name_in, _ = os.path.splitext(os.path.basename(str(img_path_in, encoding="utf-8")))
            img_label_in = label_batch_in.tolist()[i]

            colors = ['b', 'b', 'b', 'b', 'b']
            top5_indexlist = indexOfTopk(predict, 5)
            if img_label_in in top5_indexlist:
                colors[top5_indexlist.index(img_label_in)] = 'r'

            top5_ticks = [alllabels[index] for index in top5_indexlist]
            top5_probability = [predict[index] for index in top5_indexlist]

            # 坐标范围
            plt.ylim([0, 1.1])
            plt.xlim([0, 6])
            # 标签
            plt.xticks(range(1, 6), top5_ticks, rotation=10)
            plt.bar(range(1, 6), top5_probability, color=colors)
            # 带数据
            for x, y in zip(range(1, 6), top5_probability):
                plt.text(x - 0.3, y + 0.05, '%.2f' % (y * 100) + "%")
            # 图片大小：384*256
            plt.rcParams['figure.figsize'] = (6.0, 4.0)
            plt.rcParams['savefig.dpi'] = 64

            plt.savefig(os.path.join(output_dir, "bar_imgs", img_name_in + ".png"))
            plt.close()
    print("top1 accuracy:", np.mean(all_top1_acu))
    print("top5 accuracy:", np.mean(all_top5_acu))
    f = open(os.path.join(output_dir, 'accuracy.txt'), 'w')
    f.write("top1 accuracy:%.4f\ntop5 accuracy:%.4f"%(np.mean(all_top1_acu),np.mean(all_top5_acu)))
    f.close()


#        需要保存在验证集上的整体准确率（top1、top5），并为每张图片生成一张概率直方图


def main():
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 31 - 1)

    global alllabels
    f = open(args.label_txt, 'r')
    for line in f:
        line = line.strip()
        alllabels.append(line)

    if args.mode != "predict":
        # train、validte
        # Get a list of image paths and their labels
        image_path_list, label_list = get_dataset(args.data_dir)
        assert len(image_path_list) > 0, 'The dataset should not be empty'
    else:
        images = os.listdir(args.data_dir)
        image_path_list = [os.path.join(args.data_dir, img) for img in images]

    if args.epoch_size is None:
        args.epoch_size = int(math.ceil(len(image_path_list)/ args.batch_size))

    for k, v in args._get_kwargs():
        print(k, "=", v)
    print("all labels = ", alllabels)

    with open(os.path.join(output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create a queue that produces indices into the image_list and label_list
    labels = ops.convert_to_tensor(label_list, dtype=tf.int32)  # list转tensor
    range_size = array_ops.shape(labels)[0]  # 样本个数

    # 准备placeholder以及图片shuffle
    index_queue = tf.train.range_input_producer(range_size, num_epochs=None, shuffle=True, seed=None, capacity=32)
    index_dequeue_op = index_queue.dequeue_many(args.batch_size * args.epoch_size, 'index_dequeue')

    learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
    batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')
    labels_placeholder = tf.placeholder(tf.int64, shape=(None, 1), name='labels')

    input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                          dtypes=[tf.string, tf.int64],
                                          shapes=[(1,), (1,)],
                                          shared_name=None, name=None)
    enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder], name='enqueue_op')

    # 图像预处理：4线程
    nrof_preprocess_threads = 4
    images_labels_paths = []
    for _ in range(nrof_preprocess_threads):
        filenames, label = input_queue.dequeue()
        images = []
        for filename in tf.unstack(filenames):
            file_contents = tf.read_file(filename)
            image = tf.image.decode_jpeg(file_contents)
            if (args.mode == "train") & args.random_flip:
                image = tf.image.random_flip_left_right(image)
            image.set_shape((args.image_size, args.image_size, 3))
            images.append(tf.image.per_image_standardization(image))
        #     整幅图片标准化（不是归一化），加速神经网络的训练
        images_labels_paths.append([images, label, filenames])

    # 取训练batch
    image_batch, label_batch, path_batch = tf.train.batch_join(
        images_labels_paths, batch_size=batch_size_placeholder,
        shapes=[(args.image_size, args.image_size, 3), (), ()], enqueue_many=True,
        capacity=4 * nrof_preprocess_threads * args.batch_size,
        allow_smaller_final_batch=True)
    image_batch = tf.identity(image_batch, 'image_batch')
    image_batch = tf.identity(image_batch, 'input')
    label_batch = tf.identity(label_batch, 'label_batch')
    path_batch = tf.identity(path_batch, 'path_batch')

    if args.mode == "train":
        print('Total number of classes: %d' % len(alllabels))
        print('Total number of examples: %d' % len(image_path_list))

    # global shep 更新、初始化
    global_step = tf.train.get_or_create_global_step()
    # 损失优化器中global_step可以自动加1

    # 搭建网络
    prelogits, _ = inference(image_batch, args.keep_probability,
                             phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
                             weight_decay=args.weight_decay)
    logits = slim.fully_connected(prelogits, len(alllabels), activation_fn=None,
                                  weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                  weights_regularizer=slim.l2_regularizer(args.weight_decay),
                                  scope='Logits', reuse=False)
    # 大小：batchsize*classnum 每类的概率
    predicts = tf.nn.softmax(logits, name="predicts")
    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

    # top1准确率
    accuracy_top1_batch = tf.cast(tf.equal(tf.argmax(logits, 1), label_batch), tf.float32)
    # top5准确率
    accuracy_top5_batch = tf.cast(tf.nn.in_top_k(predicts, label_batch, 5), tf.float32)

    # center loss
    if args.center_loss_factor > 0.0:
        prelogits_center_loss, center_points = center_loss(prelogits, label_batch, args.center_loss_alfa,
                                                           len(alllabels))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)
    # tf.add_to_collection（）将tensor对象放入同一个集合 ,前一个参数是集合名

    # 交叉熵损失
    #  tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name=None)
    # 计算logits 和 labels 之间的稀疏softmax 交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch, logits=logits,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # total losses
    # 下式计算在4.10 5中使用
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')
    # 下式计算在4.10 1、2、3、4中使用
    # regularization_losses = prelogits_center_loss * args.center_loss_factor
    # total_loss = tf.add_n([cross_entropy_mean] + [regularization_losses], name='total_loss')

    # 学习率
    learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                               args.learning_rate_decay_epochs * args.epoch_size,
                                               args.learning_rate_decay_factor, staircase=True)

    # Build a Graph that trains the model with one batch of examples and updates the model parameters
    train_op = trainop(total_loss, global_step, args.optimizer,
                       learning_rate, args.moving_average_decay, tf.global_variables(), args.log_histograms)

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", image_batch)

    with tf.name_scope("predicts_summary"):
        tf.summary.histogram(predicts.op.name + "/values", predicts)

    tf.summary.scalar("center_loss", prelogits_center_loss)
    tf.summary.scalar("cross_loss", cross_entropy_mean)
    tf.summary.scalar("total_loss", total_loss)

    tf.summary.scalar('learning_rate', learning_rate)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    # Create a saver
    # 4.10训练
    # saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
    # 4.11
    saver = tf.train.Saver(max_to_keep=1)
    # Build the summary operation based on the TF collection of Summaries.
    log_dir = output_dir
    if args.supervisor is not None:
        log_dir = args.supervisor
    sv = tf.train.Supervisor(logdir=log_dir, save_summaries_secs=0, saver=None)
    # Start running operations on the Graph.

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())

        if args.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(args.checkpoint)
            saver.restore(sess, checkpoint)

        options = None
        run_metadata = None

        # 2021.4.11前的summary
        # summary_op = sv.summary.merge_all()
        # summary_writer = tf.summary.FileWriter(output_dir, sess.graph)
        # 2021.04.11调整
        summary_op = sv.summary_op
        summary_writer = sv.summary_writer

        coord = tf.train.Coordinator()
        # 线程管理器
        tf.train.start_queue_runners(coord=coord, sess=sess)

        if args.mode == "train":
            # Training loop
            print('Running training')
            step = sess.run(sv.global_step)
            max_epochs = math.ceil(step/args.epoch_size) + args.max_epochs
            epoch = math.ceil(step / args.epoch_size) + 1  # 取整
            while epoch <= max_epochs:
                # Train for one epoch
                step = train(sess, epoch, max_epochs, image_path_list, label_list, index_dequeue_op, enqueue_op,
                             image_paths_placeholder,
                             labels_placeholder,
                             learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder,
                             global_step,
                             total_loss, train_op, summary_op, summary_writer, regularization_losses,cross_entropy_mean)
                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, output_dir, subdir, step)
                epoch = epoch + 1
        elif args.mode == "validate":
            print('Running validation')
            sess.run(sv.global_step)
            if not os.path.isdir(os.path.join(output_dir,"bar_imgs")): 
                # Create the bar imagin directory if it doesn't exist
                os.makedirs(os.path.join(output_dir, "bar_imgs"))
            validate(sess,image_path_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, phase_train_placeholder,
                     labels_placeholder,
                     batch_size_placeholder, path_batch, label_batch, logits, predicts, accuracy_top1_batch,
                     accuracy_top5_batch, cross_entropy_mean, regularization_losses,
                     total_loss)
        sess.close()
    return output_dir


if __name__ == '__main__':
    main()
