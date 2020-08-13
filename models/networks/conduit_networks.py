import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.np_networks import VanillaNN


class ConduitVanillaNN(nn.Module):
    """
    A vanilla neural network with weights set to 0 for the property that is being predicted. As described in
    J. Chem. Inf. Model. 2019, 59, 3, 1197–1204.
    """

    def __init__(self, in_dim, out_dim, hidden_dim, idx, non_linearity=F.tanh):
        """
        :param in_dim: (int) Dimensionality of the input.
        :param out_dim: (int) Dimensionality of the input.
        :param hidden_dim: (int) Dimensionality of the NN hidden layer.
        :param idx: (int) Index of the property being predicted.
        :param non_linearity: Non-linear activation function to apply after each linear transformation,
                                e.g. relu or tanh.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = [hidden_dim, ]
        self.non_linearity = non_linearity
        self.idx = idx

        self.network = VanillaNN(self.in_dim, self.out_dim, self.hidden_dims,
                                 self.non_linearity)

        # Fix the weighting of the property being predicted to 0.
        self.network.layers[0].weight[:, idx].data.fill_(0)

    def forward(self, x):
        return self.network.forward(x)


class ConduitSetOfVanillaNNs(nn.Module):
    """
    A set of vanilla neural networks with weights set to 0 for each property that is being predicted. As described in
    J. Chem. Inf. Model. 2019, 59, 3, 1197–1204.
    """

    def __init__(self, in_dim, out_dim, hidden_dim, non_linearity=F.tanh):
        """
        :param in_dim: (int) Dimensionality of the input.
        :param out_dim: (int) Dimensionality of the input.
        :param hidden_dim: (int) Dimensionality of the NN hidden layer.
        :param non_linearity: Non-linear activation function to apply after each linear transformation,
                                e.g. relu or tanh.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.networks = nn.ModuleList()

        for i in range(out_dim):
            self.networks.append(ConduitVanillaNN(in_dim=self.in_dim, out_dim=1, hidden_dim=hidden_dim, idx=i,
                                                  non_linearity=non_linearity))

    def forward(self, x):
        """
        :param x: (torch.tensor, (batch_size, in_dim)) Input to the network
        :return: y (torch.tensor, (batch_size, out_dim)) Output of the network
        """
        batch_size = x.shape[0]
        y = torch.zeros((batch_size, self.out_dim))
        for i in range(self.out_dim):
            y_i = self.networks[i](x)
            y[:, i] = y_i[:, 0]
        return y


