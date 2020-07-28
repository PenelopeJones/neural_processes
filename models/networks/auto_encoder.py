import torch.nn as nn
import torch.optim as optim

from models.networks.networks import VanillaNN

class AutoEncoder(nn.Module):
    """
    Model that can be used to reduce the dimensionality of data from in_dim to out_dim.
    """
    def __init__(self, in_dim, out_dim, encoder_dims, decoder_dims):
        """

        :param in_dim: (int) Dimensionality of the input data
        :param out_dim: (int) Dimensionality of the output data
        :param encoder_dims: (list of ints) NN architecture of the encoder
        :param decoder_dims: (list of ints) NN architecture of the decoder
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        self.encoder = VanillaNN(self.in_dim, self.out_dim, self.encoder_dims)
        self.decoder = VanillaNN(self.out_dim, self.in_dim, self.decoder_dims)

    def forward(self, x):
        r = self.encode(x)
        r = self.decode(r)
        return r

    def encode(self, x):
        return self.encoder.forward(x)

    def decode(self, x):
        return self.decoder.forward(x)

    def train_model(self, x, epochs, print_freq, lr, loss_function=nn.MSELoss()):
        self.optimiser = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            self.optimiser.zero_grad()

            z = self.forward(x)

            loss = loss_function(x, z)
            self.losslogger = loss

            if epoch % print_freq == 0:
                print("Epoch {:.0f}:, Loss = {:.3f}".format(epoch, loss.item()))

            loss.backward()
            self.optimiser.step()









