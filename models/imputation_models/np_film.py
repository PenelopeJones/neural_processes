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
import os
import copy

import numpy as np
import torch
import torch.optim as optim
from torch.distributions.kl import kl_divergence
from torch.distributions import MultivariateNormal
from sklearn.metrics import r2_score, mean_squared_error

from models.networks.npfilm_networks import NPFiLM_Encoder, NPFiLM_Decoder
from utils.metric_utils import mll



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

        self.means = means
        self.stds = stds
        self.dir_name = os.path.dirname(file.name)
        self.file_start = file.name[len(self.dir_name) + 1:-4]

        for epoch in range(epochs):
            self.epoch = epoch
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

                if epoch > 0:
                    r2_scores, mlls, rmses = self.metrics_calculator(x, n_samples=100, plot=False)
                    r2_scores = np.array(r2_scores)
                    mlls = np.array(mlls)
                    rmses = np.array(rmses)
                    file.write('\n R^2 score (train): {:.3f}+- {:.3f}'.format(
                        np.mean(r2_scores), np.std(r2_scores)))
                    file.write('\n MLL (train): {:.3f}+- {:.3f} \n'.format(
                        np.mean(mlls), np.std(mlls)))
                    file.write('\n RMSE (train): {:.3f}+- {:.3f} \n'.format(
                        np.mean(rmses), np.std(rmses)))
                    file.flush()

                    if x_test is not None:
                        r2_scores, mlls, rmses = self.metrics_calculator(x_test, n_samples=100, plot=True)
                        r2_scores = np.array(r2_scores)
                        mlls = np.array(mlls)
                        rmses = np.array(rmses)
                        file.write('\n R^2 score (test): {:.3f}+- {:.3f}'.format(
                            np.mean(r2_scores), np.std(r2_scores)))
                        file.write('\n MLL (test): {:.3f}+- {:.3f} \n'.format(
                            np.mean(mlls), np.std(mlls)))
                        file.write('\n RMSE (test): {:.3f}+- {:.3f} \n'.format(
                            np.mean(rmses), np.std(rmses)))
                        file.write(str(r2_scores) + '\n')
                        file.flush()

                        if (self.epoch % 1000) == 0 and (self.epoch > 0):
                            path_to_save = self.dir_name + '/' + self.file_start + '_' + str(self.epoch)
                            np.save(path_to_save + 'r2_scores.npy', r2_scores)
                            np.save(path_to_save + 'mll_scores.npy', mlls)
                            np.save(path_to_save + 'rmse_scores.npy', mlls)

            loss.backward()
            optimiser.step()

    def metrics_calculator(self, x, n_samples=1, plot=True):
        mask = torch.isnan(x[:, -self.n_properties:])
        r2_scores = []
        mlls = []
        rmses = []
        for p in range(0, self.n_properties, 1):
            p_idx = torch.where(~mask[:, p])[0]
            if p_idx.shape[0] > 40:
                x_p = x[p_idx]
                target = x_p[:, (-self.n_properties + p)]

                mask_context = copy.deepcopy(mask[p_idx, :])
                mask_context[:, p] = True
                mask_p = torch.zeros_like(mask_context).fill_(True)
                mask_p[:, p] = False

                # [test_size, n_properties, z_dim]
                mu_priors, sigma_priors = self.encoder(x_p[:, :-self.n_properties],
                                                        x_p[:, -self.n_properties:],
                                                        mask_context)
                samples = []
                for i in range(n_samples):
                    z = mu_priors + sigma_priors * torch.randn_like(mu_priors)
                    recon_mu, recon_sigma = self.decoder(z, mask_p)
                    recon_mu = recon_mu.detach()
                    recon_sigma = recon_sigma.detach()
                    recon_mu = recon_mu[:, p]
                    recon_sigma = recon_sigma[:, p]
                    sample = recon_mu + recon_sigma * torch.randn_like(recon_mu)
                    samples.append(sample.transpose(0, 1))

                samples = torch.cat(samples)
                predict_mean = torch.mean(samples, dim=0)
                predict_std = torch.std(samples, dim=0)

                if (self.means is not None) and (self.stds is not None):
                    predict_mean = (predict_mean.numpy() * self.stds[-self.n_properties + p] +
                                    self.means[-self.n_properties + p])
                    predict_std = predict_std.numpy() * self.stds[-self.n_properties + p]
                    target = (target.numpy() * self.stds[-self.n_properties + p] +
                              self.means[-self.n_properties + p])
                    r2_scores.append(r2_score(target, predict_mean))
                    mlls.append(mll(predict_mean, predict_std ** 2, target))
                    rmses.append(np.sqrt(mean_squared_error(target, predict_mean)))

                    path_to_save = self.dir_name + '/' + self.file_start + str(p)

                    #np.save(path_to_save + '_mean.npy', predict_mean)
                    #np.save(path_to_save + '_std.npy', predict_std)
                    #np.save(path_to_save + '_target.npy', target)

                    #if plot:
                    #    confidence_curve(predict_mean, predict_std**2, target,
                    #                     filename=path_to_save + '_rmse_conf_curve.png',
                    #                     metric='rmse')
                    #    confidence_curve(predict_mean, predict_std**2, target,
                    #                     filename=path_to_save + '_r2_conf_curve.png',
                    #                     metric='r2')

                else:
                    r2_scores.append(r2_score(target.numpy(), predict_mean.numpy()))
                    mlls.append(mll(predict_mean.numpy(), predict_std.numpy() ** 2, target.numpy()))
                    rmses.append(np.sqrt(mean_squared_error(target.numpy(), predict_mean.numpy())))

        return r2_scores, mlls, rmses
