import torch
import torch.nn.functional as F
from torch import nn

from capsule_network import CapsuleLayer


class CapsuleEncoder(nn.Module):

    # TODO: Look at the hyperparameters in the TF version:
    # https://github.com/andyweizhao/capsule_text_classification/blob/master/network.py

    def __init__(self,
                 num_embeddings=1000,
                 embedding_dim=300,
                 lstm_dim=300,

                 primary_num_capsules=256,
                 primary_capsule_dim=8,
                 primary_kernel_size=9,
                 primary_stride=2,

                 secondary_num_capsules=256,
                 secondary_capsule_dim=8,
                 secondary_kernel_size=9,
                 secondary_stride=2):

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
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_dim)

        self.primary_capsules = CapsuleLayer(num_capsules=primary_num_capsules,
                                             num_route_nodes=-1,
                                             in_channels=lstm_dim,
                                             out_channels=primary_capsule_dim,
                                             kernel_size=primary_kernel_size,
                                             stride=primary_stride)

        # Here, num_route_nodes gives the length of the vote vectors.
        self.secondary_capsules = CapsuleLayer(num_capsules=secondary_num_capsules,
                                               # TODO: Where does magic 6 * 6 come from?
                                               num_route_nodes=primary_capsule_dim * 6 * 6,
                                               in_channels=primary_num_capsules,
                                               out_channels=primary_capsule_dim,
                                               kernel_size=secondary_kernel_size,
                                               stride=secondary_stride)

    def forward(self, x):
        """Compute a sequence of encoder states from the input sequence x."""
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.primary_capsules(x)
        x = self.secondary_capsules(x)
        return x


def test():
    seq_len = 100
    batch_size = 10
    encoder = CapsuleEncoder()
    x = torch.zeros([seq_len, batch_size])
    y = encoder(x)


if __name__ == "__main__":
    test()

