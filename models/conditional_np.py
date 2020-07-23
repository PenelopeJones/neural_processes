import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from models.networks import VanillaNN, BayesianVanillaNN
from utils.data_utils import metrics_calculator

import pdb


class CNP():
    """
    The Conditional Neural Process model.
    """

    def __init__(self, x_dim, y_dim, r_dim, encoder_dims, decoder_dims,
                 encoder_non_linearity=F.relu, decoder_non_linearity=F.relu):
        """

        :param x_dim: (int) Dimensionality of x, the input to the CNP
        :param y_dim: (int) Dimensionality of y, the target.
        :param r_dim: (int) Dimensionality of the deterministic embedding, r.
        :param encoder_dims: (list of ints) Architecture of the encoder network.
        :param decoder_dims: (list of ints) Architecture of the decoder network.
        :param encoder_non_linearity: Non-linear activation function to apply after each linear transformation,
                                in the encoder network e.g. relu or tanh.
        :param decoder_non_linearity: Non-linear activation function to apply after each linear transformation,
                                in the decoder network e.g. relu or tanh.
        :param lr: (float) Optimiser learning rate.
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim

        self.encoder = VanillaNN((x_dim + y_dim), r_dim, encoder_dims, encoder_non_linearity)
        self.decoder = BayesianVanillaNN((x_dim + r_dim), y_dim, decoder_dims, decoder_non_linearity)

    def forward(self, x_context, y_context, x_target, batch_size):
        """

        :param x_context: (torch tensor of dimensions [batch_size*n_context, x_dim])
        :param y_context: (torch tensor of dimensions [batch_size*n_context, y_dim])
        :param x_target: (torch tensor of dimensions [batch_size*n_target, x_dim])
        :return: mu_y, sigma_y: (both torch tensors of dimensions [batch_size*n_target, y_dim])
        """
        assert x_target.shape[0] % batch_size == 0
        assert len(x_context.shape) == 2, 'Input must be of shape [batch_size*n_context, x_dim].'
        assert len(y_context.shape) == 2, 'Input must be of shape [batch_size*n_context, y_dim].'
        assert len(x_target.shape) == 2, 'Input must be of shape [batch_size*n_target, x_dim].'

        n_target = int(x_target.shape[0] / batch_size)

        r = self.encoder.forward(torch.cat((x_context, y_context), dim=-1).float())  # [batch_size*n_context, r_dim]

        r = r.view(batch_size, -1, self.r_dim)  # [batch_size, n_context, r_dim]
        r = torch.mean(r, dim=1).reshape(-1, self.r_dim)  # [batch_size, r_dim]

        r = torch.repeat_interleave(r, n_target, dim=0)  # [batch_size*n_target, r_dim]

        mu_y, var_y = self.decoder.forward(torch.cat((x_target.float(), r), dim=-1))  # [batch_size*n_target, y_dim] x2

        return mu_y, var_y

    def train(self, x, y, x_test=None, y_test=None, x_scaler=None, y_scaler=None,
              batch_size=10, lr=0.001, epochs=3000, print_freq=100, VERBOSE=True,
              dataname=None):
        """

        :param x: [n_functions, [n_train, x_dim]]
        :param y: [n_functions, [n_train, y_dim]]
        :param lr:
        :param iterations:
        :return:
        """
        self.optimiser = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr)

        n_functions = len(x)

        for epoch in range(epochs):
            self.optimiser.zero_grad()

            # Sample the function from the set of functions
            idx_function = np.random.randint(n_functions)
            x_train = x[idx_function]
            y_train = y[idx_function]

            max_target = x_train.shape[0]

            # Sample n_target points from the function, and randomly select n_context points to condition on (these
            # will be a subset of the target set).
            num_target = torch.randint(low=4, high=int(max_target), size=(1,))
            num_context = torch.randint(low=3, high=int(num_target), size=(1,))

            idx = [np.random.permutation(x_train.shape[0])[:num_target] for i in
                   range(batch_size)]
            idx_context = [idx[i][:num_context] for i in range(batch_size)]

            x_target = [x_train[idx[i], :] for i in range(batch_size)]
            y_target = [y_train[idx[i], :] for i in range(batch_size)]
            x_context = [x_train[idx_context[i], :] for i in range(batch_size)]
            y_context = [y_train[idx_context[i], :] for i in range(batch_size)]

            x_target = torch.stack(x_target).view(-1, self.x_dim)  #[batch_size*n_target, x_dim]
            y_target = torch.stack(y_target).view(-1, self.y_dim)  #[batch_size*n_target, y_dim]
            x_context = torch.stack(x_context).view(-1, self.x_dim) #[batch_size*n_context, x_dim]
            y_context = torch.stack(y_context).view(-1, self.y_dim) #[batch_size, n_context, y_dim]

            # Make a forward pass through the CNP to obtain a distribution over the target set.
            mu_y, var_y = self.forward(x_context, y_context, x_target, batch_size) #[batch_size*n_target, y_dim] x2

            log_ps = MultivariateNormal(mu_y, torch.diag_embed(var_y)).log_prob(y_target.float())

            # Calculate the loss function.
            loss = -torch.mean(log_ps)
            self.losslogger = loss

            if epoch % print_freq == 0:
                print('Epoch {:.0f}: Loss = {:.5f}'.format(epoch, loss))

                if VERBOSE:
                    metrics_calculator(self, 'cnp', x, y, x_test, y_test, dataname, epoch, x_scaler, y_scaler)

            loss.backward()
            self.optimiser.step()

