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
                num_prim_caps=16,
                num_conv_caps1=16,
                num_conv_caps2=16,
                pose_shape=16,
                routing_iterations=3):
    """Capsule network with either an LSTM or ConvNet feature extractor.

    Args:
        feature_extractor: Either "lstm" or "conv".
        input_dim: Length of input embeddings.
        feature_dim: Length of feature representation.
        num_prim_caps: The number of capsules in the primary capsule layer.
        num_conv_caps1: The number of capsules in the first convolutional capsule layer.
        num_conv_caps2: The number of capsules in the second convolutional capsule layer.
        pose_shape: The length of capsules.
        routing_iterations: The number of iterations for dynamic routing.

    Returns:
        Tensor of poses.
        Tensor of activation values.
    """
    with tf.variable_scope("capsule_net"):

        # Extract features with either an LSTM or ConvNet.
        if feature_extractor == "lstm":
            nets = LSTM(feature_dim, return_sequences=True, unroll=True)(X)
            nets = tf.expand_dims(nets, 2)
        elif feature_extractor == "conv":
            X = X[..., tf.newaxis] 
            nets = _conv2d_wrapper(
                    X, shape=[3, input_dim, 1, feature_dim], strides=[1, 2, 1, 1], padding='VALID', 
                    add_bias=True, activation_fn=tf.nn.relu, name='conv'
                )
        print("feature tensor:", nets.shape)

        # Convert the features at each time step to capsules.
        nets = capsules_init(nets, shape=[1, 1, feature_dim, num_prim_caps], strides=[1, 1, 1, 1], 
                             padding='VALID', pose_shape=pose_shape, add_bias=True, name='prim-caps')
        print("primary-caps tensors:", nets[0].shape, nets[1].shape)

        # Apply the first convolutional layer.
        nets = capsule_conv_layer(nets,
                                  shape=[3, 1, num_prim_caps, num_conv_caps1],
                                  strides=[1, 1, 1, 1],
                                  iterations=routing_iterations,
                                  name='conv-caps1')
        print("conv-caps1 tensors:", nets[0].shape, nets[1].shape)

        # Apply the second convolutional layer if it is not disabled.
        if num_conv_caps2 != -1:
            nets = capsule_conv_layer(nets,
                                      shape=[1, 1, num_conv_caps1, num_conv_caps2],
                                      strides=[1, 1, 1, 1],
                                      iterations=routing_iterations,
                                      name='conv-caps2')
            print("conv-conv2 tensors:", nets[0].shape, nets[1].shape)

        # Remove the extra dimension in the tensors.
        poses, activations = nets
        return tf.squeeze(poses), tf.squeeze(activations)


def test_graph():
    """Build a graph for dummy data."""
    seq_len = 100
    batch_size = 10
    embed_size = 300
    feature_extractor = "conv"

    X = tf.zeros([batch_size, seq_len, embed_size])
    nets = capsule_net(X, feature_extractor)
    print("output tensors:", nets[0].shape, nets[1].shape)


if __name__ == "__main__":
    test_graph()
