"""
Conditional Neural Process inspired imputation model.
"""
import os
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error

from models.networks.np_networks import VanillaNN, MultiProbabilisticVanillaNN
from utils.metric_utils import mll

import pdb


class CNPBasic(nn.Module):
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

        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.z_dim = z_dim
        self.d_dim = in_dim - n_properties
        self.n_properties = n_properties
        self.encoder = VanillaNN(in_dim=self.d_dim+self.n_properties, out_dim=self.z_dim,
                                 hidden_dims=encoder_dims)
        self.decoder = MultiProbabilisticVanillaNN(in_dim=self.z_dim, out_dim=1, n_properties=self.n_properties,
                                                   hidden_dims=decoder_dims, restrict_var=False)

    def train_model(self, x, epochs, batch_size, file, print_freq=50,
              x_test=None, means=None, stds=None, lr=0.001):
        """
        :param x:
        :param epochs:
        :param batch_size:
        :param file:
        :param print_freq:
        :param x_test:
        :param means:
        :param stds:
        :param lr:
        :return:
        """
        self.means = means
        self.stds = stds
        self.dir_name = os.path.dirname(file.name)
        self.file_start = file.name[len(self.dir_name) + 1:-4]

        optimiser = optim.Adam(list(self.encoder.parameters()) +
                               list(self.decoder.parameters()), lr)

        self.epoch = 0
        for epoch in range(epochs):
            self.epoch = epoch
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
                    size = np.random.randint(low=0, high=properties.shape[0] + 1),
                                      replace=False)

                # add property to those being masked
                mask_context[i, ps] = True
            input_batch = copy.deepcopy(x_batch)
            input_batch[:, -self.n_properties:][mask_context] = 0.0

            z = self.encoder(input_batch)

            mus_y, vars_y = self.decoder.forward(z, mask_batch)

            likelihood_term = 0

            for p in range(self.n_properties):
                target = target_batch[:, p][~mask_batch[:, p]]
                mu_y = mus_y[p].squeeze(1)
                var_y = vars_y[p].squeeze(1)

                ll = (- 0.5 * np.log(2 * np.pi) - 0.5*torch.log(var_y)
                      - 0.5 * ((target - mu_y) ** 2 / var_y))

                likelihood_term += torch.sum(ll)

            likelihood_term /= torch.sum(~mask_batch)

            loss = - likelihood_term

            if (epoch % print_freq == 0) and (epoch > 0):
                file.write('\n Epoch {} Loss: {:4.4f} LL: {:4.4f}'.format(
                    epoch, loss.item(), likelihood_term.item()))

                r2_scores, mlls, rmses = self.metrics_calculator(x, test=False)
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
                    r2_scores, mlls = self.metrics_calculator(x_test, test=True)
                    r2_scores = np.array(r2_scores)
                    mlls = np.array(mlls)
                    rmses = np.array(rmses)

                    file.write('\n R^2 score (test): {:.3f}+- {:.3f}'.format(
                        np.mean(r2_scores), np.std(r2_scores)))
                    file.write('\n MLL (test): {:.3f}+- {:.3f} \n'.format(
                        np.mean(mlls), np.std(mlls)))
                    file.write('\n RMSE (test): {:.3f}+- {:.3f} \n'.format(
                        np.mean(rmses), np.std(rmses)))
                    file.flush()

                    if (self.epoch % 2000) == 0 and (self.epoch > 0):
                        path_to_save = self.dir_name + '/' + self.file_start + '_' + str(self.epoch)
                        np.save(path_to_save + 'r2_scores.npy', r2_scores)
                        np.save(path_to_save + 'mll_scores.npy', mlls)
                        np.save(path_to_save + 'rmse_scores.npy', rmses)

            loss.backward()

            optimiser.step()

    def metrics_calculator(self, x, test=True):
        mask = torch.isnan(x[:, -self.n_properties:])
        r2_scores = []
        mlls = []
        rmses = []

        for p in range(0, self.n_properties, 1):
            p_idx = torch.where(~mask[:, p])[0]
            x_p = x[p_idx]

            input_p = copy.deepcopy(x_p)
            input_p[:, -self.n_properties:][mask[p_idx]] = 0.0
            input_p[:, (-self.n_properties + p)] = 0.0

            mask_p = torch.zeros_like(mask[p_idx, :]).fill_(True)
            mask_p[:, p] = False
            z = self.encoder(input_p)
            predict_mean, predict_var = self.decoder.forward(z, mask_p)

            predict_mean = predict_mean[p].reshape(-1).detach()
            predict_std = (predict_var[p]**0.5).reshape(-1).detach()

            target = x_p[:, (-self.n_properties + p)]

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

                if (self.epoch % 2000) == 0 and (self.epoch > 0):
                    if test:
                        np.save(path_to_save + '_mean.npy', predict_mean)
                        np.save(path_to_save + '_std.npy', predict_std)
                        np.save(path_to_save + '_target.npy', target)

            else:
                r2_scores.append(r2_score(target.numpy(), predict_mean.numpy()))
                mlls.append(mll(predict_mean, predict_std ** 2, target))
                rmses.append(np.sqrt(mean_squared_error(target.numpy(), predict_mean.numpy())))
        return r2_scores, mlls, rmses
