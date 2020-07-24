import pdb
import numpy as np
import tensorflow as tf
import gpflow

from gpflow.utilities import print_summary, ops
from gpflow.config import default_jitter
from gpflow.likelihoods import Gaussian
from gpflow.kernels import SquaredExponential
from gpflow.kernels import Matern12

import pdb

"""
GPSampler is suitable for when you have access to only one function. In this case, the 
NP can be trained using samples from a GP prior which is fit to that single function. 

This will result in the NP learning to model a GP prior. 
"""
class GPSampler():
    def __init__(self, data, Z=None, kernel=Matern12(),
                 likelihood=Gaussian(), mean_function=None, maxiter=1000):
        # Use full Gaussian processes regression model for now. Could
        # implement SVGP in the future is dataset gets too big.
        if Z is None:
            m = gpflow.models.GPR(data, kernel=kernel,
                                  mean_function=mean_function)
            # Implements the L-BFGS-B algorithm for optimising hyperparameters
            opt = gpflow.optimizers.Scipy()

            def objective_closure():
                return - m.log_marginal_likelihood()

            opt_logs = opt.minimize(objective_closure, m.trainable_variables,
                                    options=dict(maxiter=maxiter))
        else:
            # Sparse variational Gaussian process for big data (see Hensman)
            m = gpflow.models.SVGP(kernel, likelihood, Z, num_data=data[0].shape[0])

            @tf.function
            def optimization_step(optimizer, m, batch):
                with tf.GradientTape() as t:
                    t.watch(m.tranable_variables)
                    objective = - model.elbo(batch)
                    grads = tape.gradient(objective, m.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                return objective

            adam = tf.optimizers.Adam()
            for i in range(maxiter):
                elbo = - optimization_step(adam, m, data)
                if step % 100 == 0:
                    print('Iteration: {} ELBO: {.3f}'.format(i, elbo))

        print_summary(m)
        # Cannot simply set self.gp_model = m as need to sample from prior,
        # not the posterior.
        self.kernel = m.kernel
        self.likelihood = m.likelihood

    def sample(self, batch_size, train_size, num_context, x_min, x_max):
        # [batch_size, num_context]
        x = np.random.uniform(x_min, x_max, size=(batch_size, train_size))
        x = np.expand_dims(x, 2)  # [batch_size, train_size=num_context + num_target, 1]
        x_context = np.array([np.random.choice(x[i, :, 0],
                                               size=num_context,
                                               replace=False)
                              for i in range(batch_size)])
        x_context = np.expand_dims(x_context, 2)

        knn = self.kernel(x)  # [batch_size, train_size, train_size]
        knn = ops.add_to_diagonal(knn, default_jitter())
        Lnn = np.linalg.cholesky(knn)
        Vnn = np.random.normal(size=(batch_size, train_size, 1))

        y = Lnn @ Vnn  # [batch_size, train_size]

        idx = [np.random.permutation(train_size)[:num_context] for i in
               range(batch_size)]
        x_context = [x[i, idx[i], :] for i in range(batch_size)]
        x_context = np.array(x_context)
        y_context = [y[i, idx[i], :] for i in range(batch_size)]
        y_context = np.array(y_context)

        return x_context, y_context, x, y

"""
The GPDataGenerator can be used when we wish to generate functions all sampled from the same, fixed GP prior.
"""
class GPDataGenerator():
    def __init__(self, kernel,
                 likelihood=Gaussian(), noise_scale=0.1):
        self.kernel = kernel
        self.likelihood = likelihood
        self.noise_scale = noise_scale

    def sample(self, train_size, test_size, x_min, x_max):
        # [num_context]
        x_train = np.random.uniform(x_min, x_max, size=(train_size))
        x_test = np.random.uniform(x_min - 1, x_max + 1, size=(test_size))
        x = np.concatenate((x_train, x_test))

        x = np.expand_dims(x, 1)  # [train_size + test_size, 1]

        knn = self.kernel(x)  # [batch_size, train_size, train_size]

        y = np.random.multivariate_normal(np.zeros(train_size + test_size), knn, 1).transpose(1, 0)
        y += np.random.normal(loc=0.0, scale=self.noise_scale, size=(train_size + test_size, 1))

        #Vnn = np.random.normal(size=(train_size + test_size, 1))

        #y = Lnn @ Vnn  # [train_size + test_size]

        x_train = x[:train_size, :]
        y_train = y[:train_size, :]
        x_test = x[train_size:, :]
        y_test = y[train_size:, :]
        return x_train, y_train, x_test, y_test