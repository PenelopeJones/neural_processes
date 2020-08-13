import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data_utils import to_natural_params, from_natural_params
from models.networks.np_networks import VanillaNN, ProbabilisticVanillaNN

import pdb

class ProbabilisticNN(nn.Module):
    """
    Probabilistic NN which outputs the natural parameters that each of the context properties contributes to the
    total distribution over the output.
    """
    def __init__(self, in_dim, out_dim, hidden_dims, non_linearity=F.relu,
                 min_var=0.01, initial_sigma=None):
        """
        :param in_dim: (int) Dimensionality of the input.
        :param out_dim: (int) Dimensionality of the target for which a distribution is being obtained.
        :param hidden_dims: (list of ints) Architecture of the network.
        :param non_linearity: Non-linear activation function to apply after each linear transformation,
                                e.g. relu or tanh.
        :param min_var: (float) Minimum variance of the output.
        :param initial_sigma: (float) If not None, adds a small bias to the output of the network.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.network = ProbabilisticVanillaNN(in_dim, out_dim, hidden_dims, non_linearity,
                                              min_var, initial_sigma)

    def forward(self, x, context_mask=None):
        # x is shape [batch_size, in_dim]
        # mask is shape [batch_size, 1] if defined
        if context_mask is not None:
            nu_1 = torch.zeros(x.shape[0], self.out_dim)
            nu_2 = torch.zeros(x.shape[0], self.out_dim)
            idx_in = torch.where(~context_mask)[0]
            x = x[idx_in]

        # if context_mask=None, [batch_size, out_dim]
        # if context_mask is not None, [len(idx_in), out_dim]
        mu, var = self.network(x)

        if context_mask is not None:
            nu_1[idx_in], nu_2[idx_in] = to_natural_params(mu, var)
        else:
            nu_1, nu_2 = to_natural_params(mu, var)
        return nu_1, nu_2  # [batch_size, out_dim] x2


class ProbabilisticNN_FiLM(nn.Module):
    """
    Probabilistic NN with a FiLM layer at the final layer. The FiLM layer is property dependent, and
    linearly transforms the output with the linear parameters being dependent on the property being
    predicted.
    """
    def __init__(self, in_dim, out_dim, hidden_dims, n_films,
                 initial_sigma=None, min_var=0.01, non_linearity=F.relu):
        """
        :param in_dim: (int) Dimensionality of the input.
        :param out_dim: (int) Dimensionality of the target for which a distribution is being obtained.
        :param hidden_dims: (list of ints) Architecture of the network.
        :param non_linearity: Non-linear activation function to apply after each linear transformation,
                                e.g. relu or tanh.
        :param n_films: (int) The number of FiLM sub-networks in the final FiLM layer
                        (i.e. total number of properties being predicted)
        :param min_var: (float) Minimum variance of the output.
        :param initial_sigma: (float) If not None, adds a small bias to the output of
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_films = n_films
        self.min_var = min_var
        self.non_linearity = non_linearity

        self.network = VanillaNN(in_dim, hidden_dims[-1], hidden_dims[:-1],
                                self.non_linearity)
        self.films = nn.ModuleList()

        for i in range(self.n_films):
            self.films.append(nn.Linear(hidden_dims[-1], 2*out_dim))

        if initial_sigma is not None:
            for film in self.films:
                film.bias.data = torch.cat([
                    1e-6 * torch.randn(out_dim),
                    np.log(np.exp(initial_sigma ** 0.5) - 1)
                    + 1e-6 * torch.randn(out_dim)])

    def forward(self, x, mask=None):
        # x is shape [batch_size, in_dim]
        # mask is shape [batch_size, 1] if defined

        # Set the initial natural parameters to 0.
        nu_1 = torch.zeros(x.shape[0], self.n_films, self.out_dim)
        nu_2 = torch.zeros(x.shape[0], self.n_films, self.out_dim)

        if mask is not None:
            idx = torch.where(~mask)[0]
            x = x[idx]

        x = self.non_linearity(self.network(x))

        for p, film in enumerate(self.films):
            x_p = self.films[p](x)
            mu = x_p[:, :self.out_dim]
            var = self.min_var + (1.0-self.min_var)*F.softplus(x_p[:, self.out_dim:])

            # Calculate the natural parameters of the output distribution.
            if mask is not None:
                nu_1[idx, p, :], nu_2[idx, p, :] = to_natural_params(mu, var)
            else:
                nu_1[:, p, :], nu_2[:, p, :] = to_natural_params(mu, var)

        return nu_1, nu_2     # [batch_size, n_films, out_dim] x2


class ProbabilisticNN_MultiFiLM(nn.Module):
    """
    Probabilistic NN with multiple FiLM layers. The FiLM layers are property dependent, and
    linearly transforms the output with the linear parameters being dependent on the property being
    predicted.
    """
    def __init__(self, in_dim, out_dim, hidden_dims, n_films,
                 initial_sigma=None, non_linearity=F.relu):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_films = n_films
        self.non_linearity = non_linearity

        self.layers = nn.ModuleList()

        for i in range(len(hidden_dims)):
            if i == 0:
                self.layers.append(nn.Linear(self.in_dim, hidden_dims[i]))
            else:
                self.layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))

        self.film_layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            film_layer = nn.ModuleList()
            for j in range(self.n_films):
                if i == (len(hidden_dims) - 1):
                    film_layer.append(nn.Linear(hidden_dims[i], 2 * out_dim))
                else:
                    film_layer.append(nn.Linear(hidden_dims[i],
                                                hidden_dims[i]))
            self.film_layers.append(film_layer)

        if initial_sigma is not None:
            for film in self.film_layers[-1]:
                film.bias.data = torch.cat([
                    1e-6 * torch.randn(out_dim),
                    np.log(np.exp(initial_sigma ** 0.5) - 1)
                    + 1e-6 * torch.randn(out_dim)])

    def forward(self, x, mask=None):
        nu_1 = torch.zeros(x.shape[0], self.n_films, self.out_dim)
        nu_2 = torch.zeros(x.shape[0], self.n_films, self.out_dim)
        pdb.set_trace()
        if mask is not None:
            idx = torch.where(~mask)[0]
            x = x[idx]
        pdb.set_trace()
        x = x.unsqueeze(2).repeat(1, 1, self.n_films)
        pdb.set_trace()
        for p, films in enumerate(self.film_layers):
            x_p = x[:, :, p]
            for layer, film in zip(self.layers, films):
                x_p = self.nonlinearity(layer(x_p))
                x_p = film(x_p)

            mu = x_p[:, :self.out_dim]
            var = F.softplus(x_p[:, self.out_dim:])

            if mask is not None:
                nu_1[idx, p, :], nu_2[idx, p, :] = to_natural_params(mu, var)
            else:
                nu_1[:, p, :], nu_2[:, p, :] = to_natural_params(mu, var)
        pdb.set_trace()
        return nu_1, nu_2   # [batch_size, n_films, out_dim]x2


class NPFiLM_Encoder(nn.Module):
    """

    """
    def __init__(self, d_dim, n_properties, z_dim, d_hidden_dims,
                 p_hidden_dims, non_linearity='tanh'):
        super().__init__()
        self.d_dim = d_dim
        self.n_properties = n_properties
        self.z_dim = z_dim

        if non_linearity == 'tanh':
            self.non_linearity = F.tanh
        elif non_linearity == 'relu':
            self.non_linearity = F.relu
        elif non_linearity == 'leaky_relu':
            self.non_linearity = F.leaky_relu
        else:
            raise ValueError('Choose from relu, leaky_relu or tanh.')

        # There are separate encoders for the descriptor and for each property.
        self.d_encoder = ProbabilisticNN_FiLM(d_dim, z_dim,
                                            d_hidden_dims, n_properties,
                                            initial_sigma=None,
                                            non_linearity=self.non_linearity)

        self.p_encoders = nn.ModuleList()
        for i in range(n_properties):
            self.p_encoders.append(ProbabilisticNN_FiLM(1, z_dim,
                                                      p_hidden_dims,
                                                      n_properties,
                                                      initial_sigma=None,
                                                      non_linearity=self.non_linearity))

    def forward(self, x_d, x_ps, mask):
        # nu_1 is [batch_size, n_properties, z_dim]
        nu_d1, nu_d2 = self.d_encoder(x_d)

        # add the prior over the descriptor latent variables

        nu_prior_1 = torch.zeros(nu_d1.shape)
        nu_prior_2 = -0.5*torch.ones(nu_d1.shape)

        nu_d1 += nu_prior_1
        nu_d2 += nu_prior_2

        mu_d, var_d = from_natural_params(nu_d1, nu_d2)

        nu_p1 = nu_prior_1
        nu_p2 = nu_prior_2
        for p, p_encoder in enumerate(self.p_encoders):
            x_p = x_ps[:, p].unsqueeze(1)
            nu_1_p, nu_2_p = p_encoder(x_p, mask[:, p])
            nu_p1 += nu_1_p
            nu_p2 += nu_2_p

        mu_p, var_p = from_natural_params(nu_p1, nu_p2)

        mu = torch.cat((mu_d, mu_p), dim=-1)
        var = torch.cat((var_d, var_p), dim=-1)
        mu[torch.where(torch.isnan(mu))] = 0.0
        var[torch.where(torch.isnan(var))] = 0.0
        return mu, var


class NPFiLM_Decoder(nn.Module):
    """

    """

    def __init__(self, in_dim, out_dim, hidden_dims, n_films,
                 non_linearity='relu', min_var=0.01):
        """
        :param input_size: An integer describing the dimensionality of the input, in this case
                           r_size, (the dimensionality of the embedding r)
        :param output_size: An integer describing the dimensionality of the output, in this case
                            output_size = x_size
        :param decoder_n_hidden: An integer describing the number of hidden layers in the neural
                                 network
        :param decoder_hidden_size: An integer describing the number of nodes in each layer of the
                                    neural network
        """

        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.n_films = n_films
        self.min_var = min_var

        if non_linearity == 'tanh':
            self.non_linearity = F.tanh
        elif non_linearity == 'relu':
            self.non_linearity = F.relu
        elif non_linearity == 'leaky_relu':
            self.non_linearity = F.leaky_relu
        else:
            raise ValueError('Choose from relu, leaky_relu or tanh.')

        self.layers = nn.ModuleList()
        self.in_films = nn.ModuleList()
        self.out_films = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        for i in range(self.n_films):
            self.in_films.append(nn.Linear(self.in_dim, hidden_dims[0]))
            self.out_films.append(nn.Linear(hidden_dims[-1], 2 * self.out_dim))

    def forward(self, x, mask=None):
        # x is [batch_size, n_properties, in_dim]
        mu = torch.zeros(x.shape[0], self.n_films, self.out_dim).fill_(
            np.nan)
        var = copy.deepcopy(mu)

        if mask is not None:
            for p, (in_film, out_film) in enumerate(zip(self.in_films,
                                                        self.out_films)):
                p_idx = torch.where(~mask[:, p])[0]
                x_p = x[p_idx, p, :]
                x_p = in_film(x_p)
                for layer in self.layers:
                    x_p = self.non_linearity(layer(x_p))
                x_p = out_film(x_p)

                mu[p_idx, p, :] = x_p[:, :self.out_dim]
                var[p_idx, p, :] = self.min_var + (1.0-self.min_var) * F.softplus(x_p[:, self.out_dim:])
        else:
            for p, (in_film, out_film) in enumerate(zip(self.in_films,
                                                        self.out_films)):
                x_p = x[:, p, :]
                x_p = in_film(x_p)
                for layer in self.layers:
                    x_p = self.non_linearity(layer(x_p))
                x_p = out_film(x_p)

                mu[:, p, :] = x_p[:, :self.out_dim]
                var[:, p, :] = self.min_var + (1.0-self.min_var) * F.softplus(x_p[:, self.out_dim:])

        return mu, var
