"""
First time setup:

pip2 install tensorflow==1.4.1 --user

Running:

python2 network.py

"""

from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from keras.layers import Embedding, LSTM
from keras import backend as K
from utils import _conv2d_wrapper
from layer import capsules_init, capsule_flatten, capsule_conv_layer, capsule_fc_layer
# import tensorflow.contrib.slim as slim


def capsule_net(X,
                feature_extractor="lstm",
                input_dim=300,
                feature_dim=32,
                caps1_dim=16,
                caps2_dim=16,
                out_dim=16):
    """Capsule network with either an LSTM or ConvNet feature extractor.

    Args:
        feature_extractor: Either "lstm" or "conv".
        input_dim: Length of input embeddings.
        feature_dim: Length of feature representation.
        caps1_dim: Dimension of capsules in first conv-caps layer.
        caps2_dim: Dimension of capsules in second conv-caps layer.
        out_dim: Dimensionality of capsules in output.

    Returns:
        Tensor of poses.
        Tensor of activation values.
    """

    with tf.variable_scope("capsule_net"):

        # Extract features with either an LSTM or 
        if feature_extractor == "lstm":
            nets = LSTM(feature_dim, return_sequences=True, unroll=True)(X)
            nets = tf.expand_dims(nets, 2)
        elif feature_extractor == "conv":
            X = X[..., tf.newaxis] 
            nets = _conv2d_wrapper(
                    X, shape=[3, input_dim, 1, feature_dim], strides=[1, 2, 1, 1], padding='VALID', 
                    add_bias=True, activation_fn=tf.nn.relu, name='conv'
                )

        tf.logging.info("feature tensor:", nets.shape)
        print("feature tensor:", nets.shape)
        # nets.shape: (10, ?, 1, 32)

        nets = capsules_init(nets, shape=[1, 1, feature_dim, caps1_dim], strides=[1, 1, 1, 1], 
                             padding='VALID', pose_shape=16, add_bias=True, name='primary')
        print("primary tensors:", nets[0].shape, nets[1].shape)                     
        nets = capsule_conv_layer(nets, shape=[3, 1, caps1_dim, caps2_dim], strides=[1, 1, 1, 1], iterations=3, name='caps1')
        print("conv1 tensors:", nets[0].shape, nets[1].shape)
        # nets = capsule_conv_layer(nets, shape=[3, 1, caps2_dim, out_dim], strides=[1, 1, 1, 1], iterations=3, name='caps1')
        # print("conv2 tensors:", nets[0].shape, nets[1].shape)
        return nets


def capsule_model_A(X):
    """Model taking from paper."""
    with tf.variable_scope('capsule_'+str(3)):
        print("pre conv", X.shape)
        
        print("after conv", nets.shape)
        # nets.shape: (10, 49, 1, 32)
        tf.logging.info('output shape: {}'.format(nets.get_shape()))
        nets = capsules_init(nets, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1], 
                             padding='VALID', pose_shape=16, add_bias=True, name='primary')                        
        nets = capsule_conv_layer(nets, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1], iterations=3, name='conv2')
        return nets


def test():
    seq_len = 100
    batch_size = 10
    embed_size = 300

    X = tf.zeros([batch_size, seq_len, embed_size])
    nets = capsule_net(X, "conv")
    print("output tensors:", nets[0].shape, nets[1].shape)


if __name__ == "__main__":
    test()