import netvlad_tf.layers as layers
import numpy as np
import tensorflow as tf


# Modified from  https://github.com/uzh-rpg/netvlad_tf_open (not ours, MIT License)
def vgg16Netvlad(image_batch):
    ''' Assumes rank 4 input, first 3 axiss fixed or dynamic, last axis 1 or 3.
    '''
    assert len(image_batch.shape) == 4

    with tf.variable_scope('vgg16_netvlad_pca'):
        # Gray to color if necessary.
        if image_batch.shape[3] == 1:
            x = tf.nn.conv2d(image_batch, np.ones((1, 1, 1, 3)),
                             np.ones(4).tolist(), 'VALID')
        else:
            assert image_batch.shape[3] == 3
            x = image_batch

        # Subtract trained average image.
        average_rgb = tf.get_variable(
            'average_rgb', 3, dtype=image_batch.dtype)
        x = x - average_rgb

        # VGG16
        def vggConv(inputs, numbers, out_axis, with_relu):
            if with_relu:
                activation = tf.nn.relu
            else:
                activation = None
            return tf.layers.conv2d(inputs, out_axis, [3, 3], 1, padding='same',
                                    activation=activation,
                                    name='conv%s' % numbers)

        def vggPool(inputs):
            return tf.layers.max_pooling2d(inputs, 2, 2)

        x = vggConv(x, '1_1', 64, True)
        x = vggConv(x, '1_2', 64, False)
        x = vggPool(x)
        x = tf.nn.relu(x)

        x = vggConv(x, '2_1', 128, True)
        x = vggConv(x, '2_2', 128, False)
        x = vggPool(x)
        x = tf.nn.relu(x)

        x = vggConv(x, '3_1', 256, True)
        x = vggConv(x, '3_2', 256, True)
        x = vggConv(x, '3_3', 256, False)
        x = vggPool(x)
        x = tf.nn.relu(x)

        x = vggConv(x, '4_1', 512, True)
        x = vggConv(x, '4_2', 512, True)
        x = vggConv(x, '4_3', 512, False)
        x = vggPool(x)
        x = tf.nn.relu(x)

        x = vggConv(x, '5_1', 512, True)
        x = vggConv(x, '5_2', 512, True)
        grad_in = vggConv(x, '5_3', 512, False)

        # NetVLAD
        x = tf.nn.l2_normalize(grad_in, axis=-1)
        x = layers.netVLAD(x, 64)

    return x, grad_in


# Modified from  https://github.com/uzh-rpg/netvlad_tf_open (not ours, MIT License)
def vgg16(image_batch):
    ''' Assumes rank 4 input, first 3 axiss fixed or dynamic, last axis 1 or 3.
    '''
    assert len(image_batch.shape) == 4

    with tf.variable_scope('vgg16_netvlad_pca'):
        # Gray to color if necessary.
        if image_batch.shape[3] == 1:
            x = tf.nn.conv2d(image_batch, np.ones((1, 1, 1, 3)),
                             np.ones(4).tolist(), 'VALID')
        else:
            assert image_batch.shape[3] == 3
            x = image_batch

        # Subtract trained average image.
        average_rgb = tf.get_variable(
            'average_rgb', 3, dtype=image_batch.dtype)
        x = x - average_rgb

        # VGG16
        def vggConv(inputs, numbers, out_axis, with_relu):
            if with_relu:
                activation = tf.nn.relu
            else:
                activation = None
            return tf.layers.conv2d(inputs, out_axis, [3, 3], 1, padding='same',
                                    activation=activation,
                                    name='conv%s' % numbers)

        def vggPool(inputs):
            return tf.layers.max_pooling2d(inputs, 2, 2)

        x = vggConv(x, '1_1', 64, True)
        x = vggConv(x, '1_2', 64, False)
        x = vggPool(x)
        x = tf.nn.relu(x)

        x = vggConv(x, '2_1', 128, True)
        x = vggConv(x, '2_2', 128, False)
        x = vggPool(x)
        x = tf.nn.relu(x)

        x = vggConv(x, '3_1', 256, True)
        x = vggConv(x, '3_2', 256, True)
        x = vggConv(x, '3_3', 256, False)
        x = vggPool(x)
        x = tf.nn.relu(x)

        x = vggConv(x, '4_1', 512, True)
        x = vggConv(x, '4_2', 512, True)
        x = vggConv(x, '4_3', 512, False)
        x = vggPool(x)
        x = tf.nn.relu(x)

        x = vggConv(x, '5_1', 512, True)
        x = vggConv(x, '5_2', 512, True)
        grad_in = vggConv(x, '5_3', 512, False)

        # NetVLAD
        x = tf.nn.l2_normalize(grad_in, axis=-1)

    return x, grad_in