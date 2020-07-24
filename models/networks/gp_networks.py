import numpy as np
import torch
import gpytorch

# We Exact GP Model.
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultiFunctionExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultiFunctionExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def train_model(self, x_trains, y_trains, lr=0.001, epochs=100, print_freq=5, batch_size=50):

        n_functions = len(x_trains)

        optimizer = torch.optim.Adam([{'params': self.parameters()}, ], lr=lr)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for j in range(epochs):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            loss = 0

            idx = np.random.permutation(n_functions)[0:batch_size]
            for i in idx:
                x = x_trains[i]
                y = y_trains[i].reshape(-1)
                self.set_train_data(inputs=x, targets=y, strict=False)
                # Output from model
                output = self.forward(x)
                # Calc loss and backprop gradients
                loss += -mll(output, y)
            loss = torch.mean(loss)
            loss.backward()
            if j % print_freq == 0:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    j + 1, epochs, loss.item(),
                    self.covar_module.base_kernel.lengthscale.item(),
                    self.likelihood.noise.item()
                ))
            optimizer.step()
