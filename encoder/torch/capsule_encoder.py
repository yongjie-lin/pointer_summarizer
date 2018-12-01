import logging

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import numpy as np

from capsule_layer_1d import CapsuleLayer1D


class CapsuleEncoder(nn.Module):

    # TODO: Look at the hyperparameters in the TF version:
    # https://github.com/andyweizhao/capsule_text_classification/blob/master/network.py

    # TODO: Can we also use some analog of reconstruction loss?

    # TODO: Verify dimensions of outputs, insert dim?

    def __init__(self,
                 sequence_length,

                 num_embeddings=1000,
                 embedding_dim=300,
                 lstm_dim=300,

                 primary_num_capsules=256,
                 primary_capsule_dim=8,
                 primary_kernel_size=9,
                 primary_stride=1,

                 secondary_num_capsules=256,
                 secondary_capsule_dim=8,
                 secondary_kernel_size=9,
                 secondary_stride=1):

        """Construct a capsule encoder.

        Args:
            num_embeddings: Number of words in embedding vocabulary.
            embedding_dim: Dimension of embedding vectors.
            lstm_dim: Dimension of LSTM hidden/cell state.

            primary_num_capsules: Number of capsules in primary layer.
            primary_capsule_dim: 
        """

        super(CapsuleEncoder, self).__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=lstm_dim,
                            # Batch size first to match Conv1d interface.
                            batch_first=True)

        self.primary_capsules = CapsuleLayer1D(num_capsules=primary_num_capsules,
                                             num_route_nodes=-1,
                                             in_channels=sequence_length,
                                             out_channels=primary_capsule_dim,
                                             kernel_size=primary_kernel_size,
                                             stride=primary_stride)

        # TODO: Need CapsuleLayer with 1D convolution.

        # Here, num_route_nodes gives the length of the vote vectors.
        self.secondary_capsules = CapsuleLayer1D(num_capsules=secondary_num_capsules,
                                               # TODO: Where does magic 6 * 6 come from?
                                               num_route_nodes=primary_capsule_dim * 6 * 6,
                                               in_channels=primary_num_capsules,
                                               out_channels=primary_capsule_dim,
                                               kernel_size=secondary_kernel_size,
                                               stride=secondary_stride)

    def forward(self, x):
        """Compute a sequence of encoder states from the input sequence x."""
        print("input:", x.size())
        x = self.embedding(x)
        print("after embed:", x.size())
        x, _ = self.lstm(x)
        print("after LSTM:", x.size())
        x = self.primary_capsules(x)
        # TODO: Need to expand dims here somehow.
        print("after prim-caps:", x.size())
        x = self.secondary_capsules(x)
        print("output:", x.size())
        return x


def test():
    seq_len = 100
    batch_size = 10

    logging.info("Creating CapsuleEncoder..")
    encoder = CapsuleEncoder(seq_len)

    logging.info("Passing dummy input through encoder..")
    x = Variable(torch.zeros([batch_size, seq_len]).long())
    y = encoder(x)


if __name__ == "__main__":
    test()

