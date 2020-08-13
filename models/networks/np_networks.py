import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.attention import MultiHeadAttention, dot_product_attention, uniform_attention, laplace_attention

import pdb
# BASIC NETWORKS

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


class ProbabilisticVanillaNN(nn.Module):
    """
    A `vanilla' NN whose output is the natural parameters of a normal distribution over y (as opposed to a point
    estimate of y).
    """

    def __init__(self, in_dim, out_dim, hidden_dims, non_linearity=F.relu, min_var=0.01, initial_sigma=None):
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
        self.min_var = min_var
        self.network = VanillaNN(in_dim, 2 * out_dim, hidden_dims, non_linearity)

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
        var = self.min_var + (1.0 - self.min_var) * F.softplus(out[:, self.out_dim:])

        return mu, var

class MultiProbabilisticVanillaNN(nn.Module):
    """

    """
    def __init__(self, in_dim, out_dim, hidden_dims, n_properties,
                 non_linearity=F.relu, min_var=0.01):
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
        self.n_properties = n_properties
        self.min_var = min_var
        self.non_linearity = non_linearity

        self.network = nn.ModuleList()

        for i in range(self.n_properties):
            self.network.append(ProbabilisticVanillaNN(self.in_dim, self.out_dim,
                                                       self.hidden_dims))

    def forward(self, x, mask):
        """

        :param mask: Torch matrix showing which of the
        :return:
        """
        mus_y = []
        vars_y = []
        for p in range(self.n_properties):
            mu_y, var_y = self.network[p].forward(x[~mask[:, p]])
            mus_y.append(mu_y)
            vars_y.append(var_y)

        return mus_y, vars_y



# ATTENTION MODULES (AS IMPLEMENTED IN ATTENTIVE NEURAL PROCESSES PAPER).

class AttentiveProbabilisticEncoder(nn.Module):
    """
    Attentive probabilistic encoder as implemented in the ANP paper,
    where it is described as the Latent Encoder. Includes option of self attention only.
    """

    def __init__(self, in_dim, r_dim, attention_dims, probabilistic_dims=None, self_att=True, min_var=0.001):
        """
        :param in_dim: An integer describing the dimensionality of the input to the encoder;
                           in this case the sum of x_dim and y_dim
        :param r_dim: An integer describing the dimensionality of the embedding, r_i
        :param encoder_n_hidden: An integer describing the number of hidden layers in the neural
                                 network
        :param encoder_hidden_dim: An integer describing the number of nodes in each layer of
                                    the neural network
        """

        super().__init__()
        self.in_dim = in_dim
        self.r_dim = r_dim
        self.attention_dims = attention_dims
        if probabilistic_dims is None:
            self.probabilistic_dims = [self.r_dim, self.r_dim]
        else:
            self.probabilistic_dims = probabilistic_dims
        self.self_att = self_att

        self.self_attentive_network = SelfAttentiveVanillaNN(in_dim=self.in_dim, out_dim=self.r_dim, hidden_dims=self.attention_dims,
                                    non_linearity=F.relu, self_att=self.self_att)

        self.probabilistic_network = ProbabilisticVanillaNN(in_dim=r_dim, out_dim=r_dim,
                                                       hidden_dims=self.probabilistic_dims,
                                                       non_linearity=F.relu, min_var=min_var,
                                                       initial_sigma=None)

    def forward(self, x, batch_size):
        """
        :param x: A tensor of dimensions [batch_dim*number of context points
                  N_context, x_dim+y_dim]. In this case each value of x
                  is the concatenation of the input x with the output y
        :return: The embeddings, a tensor of dimensionality [batch_dim*N_context,
                 r_dim]
        """
        assert len(x.shape) == 2, 'Input must be of shape [batch_size, in_dim].'

        # Encode the inputs (x, y)_i to obtain the embedding r_i
        x = self.self_attentive_network.forward(x, batch_size)   # [batch_size*n_context, r_dim]

        # Aggregate the embeddings r = mean(r_i)
        x = torch.mean(x.view(batch_size, -1, self.r_dim), dim=1).reshape(batch_size, -1) # [batch_size, r_dim]

        # Obtain a distribution over the latent variable z, which is a function of r
        mu_z, var_z = self.probabilistic_network(x)  # [batch_size, r_dim]

        return mu_z, var_z


class AttentiveDeterministicEncoder(nn.Module):
    """
    Self and cross attentive deterministic encoder, as implemented in the ANP paper, where
    it is described as the Deterministic Encoder.
    """
    def __init__(self, x_dim, y_dim, r_dim,
                 hidden_dims, self_att=True, attention_type="uniform"):
        """
        :param input_size: An integer describing the dimensionality of the input to the encoder;
                           in this case the sum of x_dim and y_dim
        :param r_dim: An integer describing the dimensionality of the embedding, r_i
        :param n_hidden: An integer describing the number of hidden layers in the neural
                                 network
        :param hidden_dim: An integer describing the number of nodes in each layer of
                                    the neural network
        """
        super().__init__()
        self.in_dim = x_dim + y_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.hidden_dims = hidden_dims
        self.self_att = self_att
        self.cross_attention_type = attention_type

        self.self_attentive_network = SelfAttentiveVanillaNN(in_dim=self.in_dim, out_dim=self.r_dim, hidden_dims=self.hidden_dims,
                                                             non_linearity=F.relu, self_att=self.self_att)

        self.cross_attentive_network = CrossAttentionNN(x_dim=self.x_dim, key_dim=self.hidden_dims[-1],
                                                        value_dim=self.r_dim, attention_type=self.cross_attention_type)

    def forward(self, x_context, y_context, x_target, batch_size):
        """
        :param x: A tensor of dimensions [batch_size, number of context points
                  N_context, x_dim]. In this case each value of x is the concatenation
                  of the input x with the output y
        :param y:
        :param x_target:
        :return: The embeddings, a tensor of dimensionality [batch_size, N_context,
                 r_dim]
        """
        assert len(x_context.shape) == 2, 'Input must be of shape [batch_size*n_context, x_dim].'
        assert len(x_target.shape) == 2, 'Input must be of shape [batch_size*n_target, x_dim].'
        assert len(y_context.shape) == 2, 'Input must be of shape [batch_size*n_context, y_dim].'

        r = self.self_attentive_network.forward(torch.cat((x_context, y_context), dim=-1).float(), batch_size=batch_size) #[batch_size*n_context, r_dim]
        r = self.cross_attentive_network.forward(keys=x_context, values=r, queries=x_target, batch_size=batch_size)

        return r    # [batch_size*n_target, r_dim]



# NETWORKS THAT INCLUDE SELF / CROSS ATTENTION.

class CrossAttentionNN(nn.Module):
    def __init__(self, x_dim, key_dim, value_dim, attention_type, key_hidden_dims=None):
        """
               :param in_dim: (int) Dimensionality of the input.
               :param out_dim: (int) Dimensionality of the output.
               :param hidden_dims: (list of ints) Architecture of the network.
               :param non_linearity: Non-linear activation function to apply after each linear transformation,
                                       e.g. relu or tanh.
               """
        super().__init__()

        self.x_dim = x_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.attention_type = attention_type

        if key_hidden_dims is None:
            self.key_hidden_dims = [self.key_dim,]
        else:
            self.key_hidden_dims = key_hidden_dims

        self.key_transform = VanillaNN(self.x_dim, self.key_dim, hidden_dims=self.key_hidden_dims, non_linearity=F.relu)

        if self.attention_type == "multihead":
            self.cross_attention = MultiHeadAttention(key_dim=self.key_dim,
                                                      value_dim=self.value_dim,
                                                      num_heads=4,
                                                      key_hidden_dim=self.key_dim,
                                                      normalise=True)

    def forward(self, keys, values, queries, batch_size):
        """
        :param self:
        :param x: (torch tensor, (batch_size*n_context, in_dim)) Input to the network.
        :return: (torch tensor, (batch_size*n_context, out_dim)) Output of the network.
        """
        assert len(keys.shape) == 2, 'Input must be of shape [batch_size, in_dim].'

        queries = self.key_transform(queries.float()).view(batch_size, -1, self.key_dim)
        keys = self.key_transform(keys.float()).view(batch_size, -1, self.key_dim)

        if self.attention_type == "multihead":
            output = self.cross_attention.forward(queries=queries.float(), keys=keys.float(),
                                                  values=values.float())
        elif self.attention_type == "uniform":
            output = uniform_attention(queries=queries.float(), values=values)

        elif self.attention_type == "laplace":
            output = laplace_attention(queries=queries.float(), keys=keys.float(), values=values,
                                       scale=1.0, normalise=True)

        elif self.attention_type == "dot_product":
            output = dot_product_attention(queries=queries.float(), keys=keys.float(),
                                           values=values, normalise=True)

        else:
            raise Exception('Select a type of cross attention from multihead, uniform, laplace or dot'
                            'product.')

        # output is [batch_size, n_target, self.r_dim].

        return output.reshape(-1, self.value_dim)  # [batch_size*n_target, value_size]


class SelfAttentiveVanillaNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, non_linearity=F.relu, self_att=True):
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
        self.hidden_dims = hidden_dims
        self.self_att = self_att
        self.network = VanillaNN(in_dim, out_dim, hidden_dims, non_linearity=self.non_linearity)

        if self.self_att:
            print("Using multihead self attention.")
            self.self_attention = MultiHeadAttention(key_dim=self.hidden_dims[-1],
                                                     value_dim=self.hidden_dims[-1],
                                                     num_heads=4,
                                                     key_hidden_dim=self.hidden_dims[-1])
        else:
            print("Not using multihead self attention.")

    def forward(self, x, batch_size):
        """
        :param self:
        :param x: (torch tensor, (batch_size*n_context, in_dim)) Input to the network.
        :return: (torch tensor, (batch_size*n_context, out_dim)) Output of the network.
        """
        assert len(x.shape) == 2, 'Input must be of shape [batch_size, in_dim].'

        for i in range(len(self.network.layers) - 1):
            x = self.non_linearity(self.network.layers[i](x))  # [batch_size*n_context, hidden_dims[-1]]

        if self.self_att:
            x = x.view(batch_size, -1, self.hidden_dims[-1])
            x = self.self_attention.forward(x)
            x = x.view(-1, self.self_attention._value_dim)

        return self.network.layers[-1](x)  # [batch_size*n_context, value_size]


