import copy
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.networks.conduit_networks import ConduitSetOfVanillaNNs
from utils.imputation_utils import conduit_r2_calculator

class SetofConduitModels:
    def __init__(self, in_dim, hidden_dim, n_properties, n_networks):
        """
        :param in_dim: (int) Dimensionality of the input.
        :param out_dim: (int) Dimensionality of the output.
        :param hidden_dim: (int) Dimensionality of the hidden layer of the NN.
        :param n_properties: (int) Number of properties being predicted.
        """
        self.in_dim = in_dim
        self.out_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_properties = n_properties
        self.n_networks = n_networks

        self.models = nn.ModuleList()
        for i in range(self.n_networks):
            self.models.append(ConduitModel(self.in_dim, self.hidden_dim, self.n_properties))

    def train_models(self, x, total_epochs, n_cycles, batch_size, file, print_freq=50,
              x_test=None, means=None, stds=None, lr=0.001):

        # Initially set the values of missing data (i.e. NaN values) to 0.
        input = copy.deepcopy(x)
        input[torch.where(torch.isnan(input))] = 0.0

        rounds = total_epochs // print_freq

        for round in rounds:
            for model in models:
                model.train(x, n_cycles, batch_size, print_freq)

            file.write('\n Epoch {} Loss: {:2.2f}'.format(
                int(round * print_freq), self.losslogger.item()))
            r2_scores, nlpds = conduit_metrics_calculator(self, x, self.n_properties,
                                                          means, stds, n_cycles)
            r2_scores = np.array(r2_scores)
            nlpds = np.array(nlpds)
            file.write('\n R^2 score (train): {:.3f}+- {:.3f}'.format(
                np.mean(r2_scores), np.std(r2_scores)))
            file.write('\n NLPD (train): {:.3f}+- {:.3f} \n'.format(
                np.mean(nlpds), np.std(nlpds)))
            file.write(str(r2_scores))
            file.flush()

            if x_test is not None:
                r2_scores, nlpds = conduit_metrics_calculator(self, x_test, self.n_properties,
                                                              means, stds, n_cycles)
                r2_scores = np.array(r2_scores)
                nlpds = np.array(nlpds)

                file.write('\n R^2 score (test): {:.3f}+- {:.3f}'.format(
                    np.mean(r2_scores), np.std(r2_scores)))
                file.write('\n NLPD (test): {:.3f}+- {:.3f} \n'.format(
                    np.mean(nlpds), np.std(nlpds)))

                file.write('\n R2 scores: \n')
                file.write(str(r2_scores))
                file.write('\n NLPDs: \n')
                file.write(str(nlpds))

                file.flush()


class ConduitModel(nn.Module):
    """
    Set of vanilla NNs with outputs recycled through the networks some number of times. As described in
    J. Chem. Inf. Model. 2019, 59, 3, 1197–1204.
    """
    def __init__(self, in_dim, hidden_dim, n_properties, lr):
        """
        :param in_dim: (int) Dimensionality of the input.
        :param out_dim: (int) Dimensionality of the output.
        :param hidden_dim: (int) Dimensionality of the hidden layer of the NN.
        :param n_properties: (int) Number of properties being predicted.
        """
        self.in_dim = in_dim
        self.out_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_properties = n_properties

        self.network = ConduitSetOfVanillaNNs(in_dim=self.in_dim, out_dim=self.out_dim, hidden_dim=self.hidden_dim,
                                       non_linearity=F.tanh)

        # Use the Adam optimiser.
        self.optimiser = optim.Adam(self.network.parameters(), lr)

    def forward(self, input, n_cycles):
        """
        :param input: (torch.tensor (batch_size, in_dim)
        :param n_cycles: (int) Number of times to re-cycle the output through the network.
        :return:
        """
        for cycle in range(n_cycles):
            output = self.network(input)
            input = 0.5 * (input + output)

        return output

    def train(self, x, n_cycles, batch_size, epochs):
        for epoch in range(epochs):
            self.optimiser.zero_grad()

            # Randomly select a batch of functions
            batch_idx = torch.randperm(x.shape[0])[:batch_size]
            x_batch = x[batch_idx, ...]  # [batch_size, n_descriptors + n_properties]

            # The input batch includes descriptors and properties; the target batch only includes the properties we want
            # to predict.
            input_batch = input[batch_idx, ...]
            target_batch = x_batch[:, -self.n_properties:]

            # Mask is the values which are not missing in the target batch. Identify the indices of these values as they
            # are the ones that will be used to compute the loss function.
            mask_batch = ~torch.isnan(target_batch)
            idx_target = torch.where(mask_batch)

            # Predict the target properties.
            output_batch = self.forward(input_batch, n_cycles=n_cycles)
            output_batch = output_batch[:, -self.n_properties:]

            # Compute the loss function.
            loss = torch.mean((target_batch[idx_target] - output_batch[idx_target]) ** 2)
            self.losslogger = loss

            # Back-propagate.
            loss.backward()

            for i in range(self.in_dim):
                # Fill the first layer with zeroes.
                self.network.networks[i].network.layers[0].weight.grad[:, i].fill_(0)

            self.optimiser.step()

    def train_model(self, x, total_epochs, n_cycles, batch_size, file, print_freq=50,
              x_test=None, means=None, stds=None):
        """

        :param x: (torch.tensor (batch_size, in_dim))
        :param epochs: (int) Number of training epochs
        :param n_cycles: (int) Number of times to re-cycle the output through the network.
        :param batch_size: (int) Batch size
        :param file:
        :param print_freq:
        :param x_test:
        :param means:
        :param stds:
        :param lr:
        :return:
        """
        self.losslogger = 0.0

        # Initially set the values of missing data (i.e. NaN values) to 0.
        input = copy.deepcopy(x)
        input[torch.where(torch.isnan(input))] = 0.0

        rounds = total_epochs // print_freq

        for round in range(rounds):
            self.train(x, n_cycles, batch_size, print_freq)

            file.write('\n Epoch {} Loss: {:2.2f}'.format(
                int(round*print_freq), self.losslogger.item()))
            r2_scores, nlpds = conduit_metrics_calculator(self, x, self.n_properties,
                                                          means, stds, n_cycles)
            r2_scores = np.array(r2_scores)
            nlpds = np.array(nlpds)
            file.write('\n R^2 score (train): {:.3f}+- {:.3f}'.format(
                np.mean(r2_scores), np.std(r2_scores)))
            file.write('\n NLPD (train): {:.3f}+- {:.3f} \n'.format(
                np.mean(nlpds), np.std(nlpds)))
            file.write(str(r2_scores))
            file.flush()

            if x_test is not None:
                r2_scores, nlpds = conduit_metrics_calculator(self, x_test, self.n_properties,
                                                              means, stds, n_cycles)
                r2_scores = np.array(r2_scores)
                nlpds = np.array(nlpds)

                file.write('\n R^2 score (test): {:.3f}+- {:.3f}'.format(
                    np.mean(r2_scores), np.std(r2_scores)))
                file.write('\n NLPD (test): {:.3f}+- {:.3f} \n'.format(
                    np.mean(nlpds), np.std(nlpds)))

                file.write('\n R2 scores: \n')
                file.write(str(r2_scores))
                file.write('\n NLPDs: \n')
                file.write(str(nlpds))

                file.flush()



        for epoch in range(epochs):
            optimiser.zero_grad()

            # Randomly select a batch of functions
            batch_idx = torch.randperm(x.shape[0])[:batch_size]
            x_batch = x[batch_idx, ...]  # [batch_size, n_descriptors + n_properties]

            # The input batch includes descriptors and properties; the target batch only includes the properties we want
            # to predict.
            input_batch = input[batch_idx, ...]
            target_batch = x_batch[:, -self.n_properties:]

            # Mask is the values which are not missing in the target batch. Identify the indices of these values as they
            # are the ones that will be used to compute the loss function.
            mask_batch = ~torch.isnan(target_batch)
            idx_target = torch.where(mask_batch)

            # Predict the target properties.
            output_batch = self.forward(input_batch, n_cycles=n_cycles)
            output_batch = output_batch[:, -self.n_properties:]

            # Compute the loss function.
            loss = torch.mean((target_batch[idx_target] - output_batch[idx_target]) ** 2)

            if epoch % print_freq == 0:
                # Write to file the value of the loss, the R2 score and RMSE for both the training and test data.
                file.write('\n Epoch {} Loss: {:2.2f}'.format(
                    epoch, loss.item()))
                r2_scores, nlpds = conduit_metrics_calculator(self, x, self.n_properties,
                                                  means, stds, n_cycles)
                r2_scores = np.array(r2_scores)
                nlpds = np.array(nlpds)
                file.write('\n R^2 score (train): {:.3f}+- {:.3f}'.format(
                    np.mean(r2_scores), np.std(r2_scores)))
                file.write('\n NLPD (train): {:.3f}+- {:.3f} \n'.format(
                    np.mean(nlpds), np.std(nlpds)))
                file.write(str(r2_scores))
                file.flush()

                if x_test is not None:
                    r2_scores, nlpds = conduit_metrics_calculator(self, x_test, self.n_properties,
                                                      means, stds, n_cycles)
                    r2_scores = np.array(r2_scores)
                    nlpds = np.array(nlpds)

                    file.write('\n R^2 score (test): {:.3f}+- {:.3f}'.format(
                        np.mean(r2_scores), np.std(r2_scores)))
                    file.write('\n NLPD (test): {:.3f}+- {:.3f} \n'.format(
                        np.mean(nlpds), np.std(nlpds)))

                    file.write('\n R2 scores: \n')
                    file.write(str(r2_scores))
                    file.write('\n NLPDs: \n')
                    file.write(str(nlpds))


                    file.flush()

            # Back-propagate.
            loss.backward()

            for i in range(self.in_dim):
                # Fill the first layer with zeroes.
                self.network.networks[i].network.layers[0].weight.grad[:, i].fill_(0)

            optimiser.step()


