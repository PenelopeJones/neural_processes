"""
In this model: the descriptor vector and each property has their own latent encoder.
Each latent encoder is a linear neural network with a FiLM layer as the final layer, allowing
for conditioning the output on the property we are trying to predict.
The output of each latent encoder is a distribution over either z_d or z_pi (i.e. means and covariances);
the distribution over property latent variable z_p is formed by summing the natural parameters of the
distributions over each z_pi. The overall function latent variable z is the concatenation of z_d and z_p.
z is then the input to the decoder, which is also a linear NN with FiLM layers that allow for conditioning
on the property that we are trying to predict. The output is again a distribution over y_i
(mean and variance).

NB Here, no skip connections.

"""

import pdb
import copy

import numpy as np
import torch
import torch.optim as optim
from torch.distributions.kl import kl_divergence
from torch.distributions import MultivariateNormal

from models.networks.npfilm_networks import NPFiLM_Encoder, NPFiLM_Decoder
from models.networks.npfilm_networks import ProbabilisticNN_MultiFiLM
from utils.imputation_utils import npfilm_metrics_calculator


class NPFiLM:
    """
    The Neural Process + FiLM: a model for chemical data imputation.
    """

    def __init__(self, in_dim, out_dim, z_dim, n_properties,
                 d_encoder_dims, p_encoder_dims,
                 decoder_dims):
        """

        :param in_dim: (int) dimensionality of the input x
        :param out_dim: (int) dimensionality of the target variable y
        :param z_dim: (int) dimensionality of the embedding / context vector r
        :param n_properties: (int) the number of unknown properties. Adrenergic = 5; Kinase = 159.
        :param d_encoder_dims: (list of ints) architecture of the descriptor encoder NN.
        :param p_encoder_dims: (list of ints) architecture of the property encoder NN.
        :param decoder_hidden_dims: (list of ints) architecture of the decoder NN.
        """

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.z_dim = z_dim
        self.d_dim = in_dim - n_properties
        self.n_properties = n_properties

        self.encoder = NPFiLM_Encoder(self.d_dim, self.n_properties, self.z_dim,
                               d_encoder_dims, p_encoder_dims, non_linearity='relu')

        self.decoder = NPFiLM_Decoder(in_dim=2*self.z_dim, out_dim=self.out_dim,
                                      hidden_dims=decoder_dims,
                                      n_films=self.n_properties)

    def train(self, x, epochs, batch_size, file, print_freq=50,
              x_test=None, means=None, stds=None, lr=0.001):

        optimiser = optim.Adam(list(self.encoder.parameters()) +
                               list(self.decoder.parameters()), lr)

        for epoch in range(epochs):
            optimiser.zero_grad()

            batch_idx = torch.randperm(x.shape[0])[:batch_size]
            x_batch = x[batch_idx, ...] # [batch_size, x.shape[1]]
            target_batch = x_batch[:, -self.n_properties:]

            mask_batch = torch.isnan(x_batch[:, -self.n_properties:])
            mask_context = copy.deepcopy(mask_batch)

            batch_properties = [torch.where(~mask_batch[i, ...])[0] for i in
                                range(mask_batch.shape[0])]

            for i, properties in enumerate(batch_properties):
                ps = np.random.choice(properties.numpy(),
                    size=np.random.randint(low=1, high=properties.shape[0] + 1),
                                      replace=False)

                # add property to those being masked
                mask_context[i, ps] = True

            mu_priors, sigma_priors = self.encoder(x_batch[:, :-self.n_properties],
                                                   x_batch[:, -self.n_properties:],
                                                   mask_context)
            mu_posts, sigma_posts = self.encoder(x_batch[:, :-self.n_properties],
                                                 x_batch[:, -self.n_properties:],
                                                 mask_batch)

            z = mu_priors + torch.randn_like(mu_priors) * sigma_priors
            recon_mu, recon_sigma = self.decoder(z, mask_batch)

            likelihood_term = 0
            for target, mu, sigma in zip(target_batch, recon_mu, recon_sigma):
                target = torch.stack([t for t in target if not torch.isnan(t)])
                mu = torch.stack([m for m in mu if not torch.isnan(m)])
                sigma = torch.stack([s for s in sigma if not torch.isnan(s)])

                ll = (- 0.5 * np.log(2 * np.pi) - torch.log(sigma)
                      - 0.5 * ((target - mu) ** 2 / sigma ** 2))

                likelihood_term += torch.sum(ll)

            kl_term = 0
            for p in range(self.n_properties):
                idx_p = torch.where(~mask_batch[:, p])[0]
                mu_prior = mu_priors[idx_p, p, :]
                mu_post = mu_posts[idx_p, p, :]
                sigma_prior = sigma_priors[idx_p, p, :]
                sigma_post = sigma_posts[idx_p, p, :]
                prior = MultivariateNormal(mu_prior, torch.diag_embed(
                    sigma_prior))
                post = MultivariateNormal(mu_post, torch.diag_embed(
                    sigma_post))
                kl_term += torch.sum(kl_divergence(post, prior))

            likelihood_term /= torch.sum(~mask_batch)
            kl_term /= torch.sum(~mask_batch)

            loss = (- likelihood_term + kl_term)

            if epoch % print_freq == 0:

                file.write('\n Epoch {} Loss: {:4.4f} LL: {:4.4f} KL: {:4.4f}'.format(
                    epoch, loss.item(), likelihood_term.item(),
                    kl_term.item()))

                r2_scores, nlpds = npfilm_metrics_calculator(self, x,
                                                      self.n_properties,
                                                      num_samples=25,
                                                      means=means,
                                                      stds=stds)
                r2_scores = np.array(r2_scores)
                nlpds = np.array(nlpds)
                file.write('\n R^2 score (train): {:.3f}+- {:.3f}'.format(
                    np.mean(r2_scores), np.std(r2_scores)))
                file.write('\n NLPD (train): {:.3f}+- {:.3f} \n'.format(
                    np.mean(nlpds), np.std(nlpds)))

                file.write(str(r2_scores))
                file.flush()

                if x_test is not None:
                    r2_scores, nlpds = npfilm_metrics_calculator(self, x_test,
                                                          self.n_properties,
                                                          num_samples=50,
                                                          means=means,
                                                          stds=stds)
                    r2_scores = np.array(r2_scores)
                    nlpds = np.array(nlpds)
                    file.write('\n R^2 score (test): {:.3f}+- {:.3f}'.format(
                        np.mean(r2_scores), np.std(r2_scores)))
                    file.write('\n NLPD (test): {:.3f}+- {:.3f} \n'.format(
                        np.mean(nlpds), np.std(nlpds)))
                    file.write(str(r2_scores) + '\n')
                    file.flush()

            loss.backward()
            optimiser.step()