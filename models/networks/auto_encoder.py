import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.networks.networks import VanillaNN

import pdb

class AutoEncoder(nn.Module):
    """
    ss
    """
    def __init__(self, in_dim, out_dim, encoder_hidden_dims, decoder_hidden_dims):
        """

        :param in_dim:
        :param out_dim:
        :param encoder_n_hidden:
        :param encoder_hidden_size:
        :param decoder_n_hidden:
        :param decoder_hidden_size:
        :param lr:
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims

        
        self.encoder = VanillaNN(in_dim, out_dim, encoder_hidden_dims)
        self.decoder
        
        

        # Encoder function taking as input a function x and outputting the lower dimensional representation r
        for i in range(self.encoder_n_hidden + 1):
            if i == 0:
                self.encoder_fcs.append(nn.Linear(in_dim, encoder_hidden_size))

            elif i == encoder_n_hidden:
                self.encoder_fcs.append(nn.Linear(encoder_hidden_size, out_dim))

            else:
                self.encoder_fcs.append(nn.Linear(encoder_hidden_size, encoder_hidden_size))

        for i in range(self.decoder_n_hidden + 1):
            if i == 0:
                self.decoder_fcs.append(nn.Linear(out_dim, decoder_hidden_size))

            elif i == decoder_n_hidden:
                self.decoder_fcs.append(nn.Linear(decoder_hidden_size, in_dim))

            else:
                self.decoder_fcs.append(nn.Linear(decoder_hidden_size, decoder_hidden_size))

    def train_model(self, x, iterations, print_freq, lr):
        self.optimiser = optim.Adam(self.parameters(), lr=lr)
        loss_function = nn.MSELoss()

        for iteration in range(iterations):
            self.optimiser.zero_grad()

            z = self.forward(x)

            loss = loss_function(x, z)
            self.losslogger = loss


            if iteration % print_freq == 0:
                print("Iteration " + str(iteration) + ":, Loss = {:.3f}".format(loss.item()))

            loss.backward()
            self.optimiser.step()

    def forward(self, x):
        r = self.encode(x)
        r = self.decode(r)
        return r

    def encode(self, x):
        for fc in self.encoder_fcs[:-1]:
            x = F.relu(fc(x))
        x = self.encoder_fcs[-1](x)
        return x

    def decode(self, x):
        for fc in self.decoder_fcs[:-1]:
            x = F.relu(fc(x))
        x = self.decoder_fcs[-1](x)
        return x








