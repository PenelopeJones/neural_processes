import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class VanillaNN(nn.Module):
    """
    A `vanilla` neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dims, non_linearity=F.relu):
        """
        :param in_dim: (int) Dimensionality of the input.
        :param out_dim: (int) Dimensionality of the output.
        :param hidden_dims: (list of ints) Architecture of the network.
        :param non_linearity: Non-linear activation function to apply after each linear transformation,
                                e.g. relu or tanh.
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.non_linearity = non_linearity

        self.layers = nn.ModuleList()

        for dim in range(len(hidden_dims) + 1):
            if dim == 0:
                self.layers.append(nn.Linear(self.in_dim, hidden_dims[dim]))
            elif dim == len(hidden_dims):
                self.layers.append(nn.Linear(hidden_dims[-1], self.out_dim))
            else:
                self.layers.append(nn.Linear(hidden_dims[dim - 1],
                                             hidden_dims[dim]))

    def forward(self, x):
        """
        :param self:
        :param x: (torch tensor, (batch_size, in_dim)) Input to the network.
        :return: (torch tensor, (batch_size, out_dim)) Output of the network.
        """
        assert len(x.shape) == 2, 'Input must be of shape [batch_size, in_dim].'

        for i in range(len(self.layers) - 1):
            x = self.non_linearity(self.layers[i](x))

        return self.layers[-1](x)

class BayesianVanillaNN(nn.Module):
    """
    A `vanilla' NN whose output is the natural parameters of a normal distribution over y (as opposed to a point
    estimate of y).
    """
    def __init__(self, in_dim, out_dim, hidden_dims, non_linearity=F.relu, initial_sigma=None):
        """
        :param in_dim: (int) Dimensionality of the input.
        :param out_dim: (int) Dimensionality of the target for which a distribution is being obtained.
        :param hidden_dims: (list of ints) Architecture of the network.
        :param non_linearity: Non-linear activation function to apply after each linear transformation,
                                e.g. relu or tanh.
        :param initial_sigma:
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.network = VanillaNN(in_dim, 2*out_dim, hidden_dims, non_linearity)

        if initial_sigma is not None:
            self.network.layers[-1].bias.data = torch.cat([
                1e-6 * torch.randn(out_dim),
                np.log(np.exp(initial_sigma ** 0.5) - 1)
                + 1e-6 * torch.randn(out_dim)])

    def forward(self, x):
        """

        :param x: x: (torch tensor, (batch_size, in_dim)) Input to the network.
        :return:
        """
        assert len(x.shape) == 2, 'Input must be of shape [batch_size, in_dim].'

        out = self.network(x)
        mu = out[:, :self.out_dim]
        sigma = 0.01 + 0.99*F.sigmoid(out[:, self.out_dim:])

        return mu, sigma