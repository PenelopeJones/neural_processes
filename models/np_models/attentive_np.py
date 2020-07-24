import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence

from models.networks.networks import AttentiveProbabilisticEncoder, AttentiveDeterministicEncoder, ProbabilisticVanillaNN
from utils.data_utils import metrics_calculator, batch_sampler


class AttentiveNP():
    """
    The Attentive Neural Process model.
    """

    def __init__(self, x_dim, y_dim, r_dim, det_encoder_dims, prob_encoder_dims, decoder_dims,
                 decoder_non_linearity=F.relu):
        """

        :param x_dim: (int) Dimensionality of x, the input to the NP
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

        self.det_encoder = AttentiveDeterministicEncoder(x_dim=self.x_dim, y_dim=self.y_dim, r_dim=self.r_dim,
                                                         hidden_dims=det_encoder_dims, self_att=True,
                                                         attention_type="multihead")
        self.prob_encoder = AttentiveProbabilisticEncoder(in_dim=self.x_dim+self.y_dim, r_dim=self.r_dim,
                                                          attention_dims=prob_encoder_dims, min_var=0.001)


        self.decoder = ProbabilisticVanillaNN(in_dim=self.x_dim + 2*self.r_dim, out_dim=self.y_dim,
                                         hidden_dims=decoder_dims, non_linearity=decoder_non_linearity,
                                              min_var=0.001)

    def forward(self, x_context, y_context, x_target, y_target=None, nz_samples=10, ny_samples=10, batch_size=1):
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

            # Deterministic encoding (uses self and cross attention)
            r = self.det_encoder.forward(x_context, y_context, x_target, batch_size) # [batch_size*n_target, r_dim]

            # Concatenate r and x_target, reshape and repeat nz_samples times.
            r = torch.cat((x_target.float(), r), dim=-1).reshape(batch_size, 1, -1, self.x_dim+self.r_dim) #[batch_size, 1, n_target, x_dim+r_dim]
            r = r.repeat(1, nz_samples, 1, 1) #[batch_size, nz_samples, n_target, x_dim+r_dim]

            # Probabilistic encoding (uses only self attention)
            mu_z, var_z = self.prob_encoder.forward(torch.cat((x_context, y_context), dim=-1).float(), batch_size) # [batch_size, r_dim] x 2

            # During training, also need to encode the target distribution so that the KL divergence between prior and posterior over z
            # can be computed
            if y_target is not None:
                mu_z_posterior, var_z_posterior = self.prob_encoder.forward(torch.cat((x_target, y_target), dim=-1).float(), batch_size)

            # Sample from resulting distribution nz_samples times.
            samples_z = [MultivariateNormal(mu_z, torch.diag_embed(var_z)).rsample() for i in range(nz_samples)]
            samples_z = torch.stack(samples_z).transpose(1, 0).view(batch_size, -1, 1, self.r_dim) # [batch_size, nz_samples, 1, r_dim]
            samples_z = samples_z.repeat(1, 1, n_target, 1)   #[batch_size, nz_samples, n_target, r_dim]

            mu_y, var_y = self.decoder.forward(torch.cat((r.float(), samples_z), dim=-1).reshape(-1, self.x_dim +
                                                                                                 2*self.r_dim))  #[batch_size*nz_samples*n_target, y_dim]

            # Sample from the distribution over y.
            samples_y = [MultivariateNormal(mu_y, torch.diag_embed(var_y)).rsample() for i in range(ny_samples)] #[ny_samples, batch_size*nz_samples*n_target, y_dim]
            samples_y = torch.stack(samples_y).reshape(ny_samples, batch_size, nz_samples, n_target, -1)
            samples_y = samples_y.transpose(1, 0).reshape(batch_size, -1, n_target, self.y_dim)

            mu_y = torch.mean(samples_y, dim=1).reshape(-1, self.y_dim) # [batch_size*n_target, y_dim]
            var_y = torch.var(samples_y, dim=1).reshape(-1, self.y_dim) # [batch_size*n_target, y_dim]

            if y_target is not None:
                return mu_y, var_y, mu_z, var_z, mu_z_posterior, var_z_posterior # [batch_size*n_target, y_dim] x2
            else:
                return mu_y, var_y

    def train(self, x, y, x_test=None, y_test=None, x_scaler=None, y_scaler=None,
              nz_samples=1, ny_samples=1, batch_size=10, lr=0.001, epochs=3000, print_freq=100, VERBOSE=False,
              dataname=None):
        """

        :param x: [n_functions, [n_train, x_dim]]
        :param y: [n_functions, [n_train, y_dim]]
        :param lr:
        :param iterations:
        :return:
        """
        self.optimiser = optim.Adam(list(self.det_encoder.parameters()) +
                                    list(self.prob_encoder.parameters()) +
                                    list(self.decoder.parameters()), lr=lr)

        for epoch in range(epochs):
            self.optimiser.zero_grad()

            # Sample the function from the set of functions
            x_context, y_context, x_target, y_target = batch_sampler(x, y, batch_size)

            # Make a forward pass through the ANP to obtain a distribution over the target set.
            mu_y, var_y, mus_z, vars_z, mus_z_posterior, vars_z_posterior = self.forward(x_context, y_context, x_target,
                                                                                         y_target, nz_samples, ny_samples,
                                                                                         batch_size)  #[batch_size*n_target, y_dim] x2

            # Measure the log probability of observing y_target given mu_y, var_y.
            log_ps = MultivariateNormal(mu_y, torch.diag_embed(var_y)).log_prob(y_target.float())
            log_ps = log_ps.reshape(batch_size, -1).sum(dim=-1)
            log_ps = torch.mean(log_ps)

            # Compute the KL divergence between prior and posterior over z
            z_posteriors = [MultivariateNormal(mu, torch.diag_embed(var)) for mu, var in zip(mus_z_posterior, vars_z_posterior)]
            z_priors = [MultivariateNormal(mu, torch.diag_embed(var)) for mu, var in zip(mus_z, vars_z)]


            kl_div = [kl_divergence(z_posterior, z_prior).float() for z_posterior, z_prior
                      in zip(z_posteriors, z_priors)]
            kl_div = torch.mean(torch.stack(kl_div))

            # Calculate the loss function from this.
            loss = -(log_ps - kl_div)
            self.losslogger = loss

            if epoch % print_freq == 0:
                print('Epoch {:.0f}: Loss = {:.5f} \t LL = {:.5f} \t KL = {:.5f}'.format(epoch, loss, log_ps, kl_div))
                if epoch % int(10*print_freq) == 0:
                    if VERBOSE:
                        metrics_calculator(self, 'anp', x, y, x_test, y_test, dataname, epoch, x_scaler, y_scaler)

            loss.backward()
            self.optimiser.step()

