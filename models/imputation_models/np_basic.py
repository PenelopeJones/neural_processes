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
from models.networks.np_networks import ProbabilisticVanillaNN, MultiProbabilisticVanillaNN
from utils.imputation_utils import npbasic_metrics_calculator


class NPBasic:
    """
    The Neural Process + FiLM: a model for chemical data imputation.
    """

    def __init__(self, in_dim, out_dim, z_dim, n_properties,
                 encoder_dims, decoder_dims):
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

        self.encoder = ProbabilisticVanillaNN(in_dim=self.d_dim+self.n_properties, out_dim=self.z_dim,
                                              hidden_dims=encoder_dims)
        self.decoder = MultiProbabilisticVanillaNN(in_dim=self.z_dim, out_dim=1, n_properties=self.n_properties,
                                                   hidden_dims=decoder_dims)

    def train(self, x, epochs, batch_size, file, print_freq=50,
              x_test=None, means=None, stds=None, lr=0.001):

        optimiser = optim.Adam(list(self.encoder.parameters()) +
                               list(self.decoder.parameters()), lr)

        for epoch in range(epochs):
            optimiser.zero_grad()

            # Select a batch
            batch_idx = torch.randperm(x.shape[0])[:batch_size]
            x_batch = x[batch_idx, ...] # [batch_size, x.shape[1]]
            target_batch = x_batch[:, -self.n_properties:]

            # Mask of the properties that are missing
            mask_batch = torch.isnan(x_batch[:, -self.n_properties:])


            # To form the context mask we will add properties to the missing values
            mask_context = copy.deepcopy(mask_batch)
            batch_properties = [torch.where(~mask_batch[i, ...])[0] for i in
                                range(mask_batch.shape[0])]

            for i, properties in enumerate(batch_properties):
                ps = np.random.choice(properties.numpy(),
                    size=np.random.randint(low=0, high=properties.shape[0] + 1),
                                      replace=False)

                # add property to those being masked
                mask_context[i, ps] = True
            input_batch = copy.deepcopy(x_batch)
            input_batch[:, -self.n_properties:][mask_batch] = 0.0

            # First use this to compute the posterior distribution over z
            mus_post, vars_post = self.encoder(input_batch)

            context_batch = copy.deepcopy(input_batch)
            # Now set the property values of the context mask to 0 too.
            context_batch[:, -self.n_properties:][mask_context] = 0.0

            mus_prior, vars_prior = self.encoder(input_batch)

            # Sample from the distribution over z.
            z = mus_prior + torch.randn_like(mus_prior) * vars_prior**0.5

            mus_y, vars_y = self.decoder.forward(z, mask_batch)

            likelihood_term = 0

            for p in range(self.n_properties):
                target = target_batch[:, p][~mask_batch[:, p]]
                mu_y = mus_y[p]
                var_y = vars_y[p]

                ll = (- 0.5 * np.log(2 * np.pi) - 0.5*torch.log(var_y)
                      - 0.5 * ((target - mu_y) ** 2 / var_y))

                likelihood_term += torch.sum(ll)

            likelihood_term /= torch.sum(~mask_batch)

            # Compute the KL divergence between prior and posterior over z
            z_posteriors = [MultivariateNormal(mu, torch.diag_embed(var**0.5)) for mu, var in
                                zip(mus_post, vars_post)]
            z_priors = [MultivariateNormal(mu, torch.diag_embed(var**0.5)) for mu, var in zip(mus_prior, vars_prior)]
            kl_div = [kl_divergence(z_posterior, z_prior).float() for z_posterior, z_prior
                          in zip(z_posteriors, z_priors)]
            kl_div = torch.sum(torch.stack(kl_div))
            kl_div /= torch.sum(~(mask_context==mask_batch).all(dim=1))

            loss = kl_div - likelihood_term

            if epoch % print_freq == 0:

                file.write('\n Epoch {} Loss: {:4.4f} LL: {:4.4f} KL: {:4.4f}'.format(
                    epoch, loss.item(), likelihood_term.item(),
                    kl_div.item()))

                r2_scores, nlpds = npbasic_metrics_calculator(self, x,
                                                      self.n_properties,
                                                      num_samples=50,
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
                    r2_scores, nlpds = npbasic_metrics_calculator(self, x_test,
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