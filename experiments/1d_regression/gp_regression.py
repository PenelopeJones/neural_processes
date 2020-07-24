import sys

sys.path.append('../../')

import warnings
import argparse

import numpy as np
import torch
import gpytorch

from models.networks.gp_networks import MultiFunctionExactGPModel
from utils.data_utils import torch_from_numpy_list, plotter1d
import pdb


def main(args):
    """
    """
    warnings.filterwarnings('ignore')

    X_trains = np.load(args.directory + args.dataname + '_X_trains.npy', allow_pickle=True)
    y_trains = np.load(args.directory + args.dataname + '_y_trains.npy', allow_pickle=True)
    X_tests = np.load(args.directory + args.dataname + '_X_tests.npy', allow_pickle=True)
    y_tests = np.load(args.directory + args.dataname + '_y_tests.npy', allow_pickle=True)

    n_functions = len(X_trains)

    # Convert the data for use in PyTorch.
    X_trains = torch_from_numpy_list(X_trains)
    y_trains = torch_from_numpy_list(y_trains)
    X_tests = torch_from_numpy_list(X_tests)
    y_tests = torch_from_numpy_list(y_tests)

    print('... building model')

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = MultiFunctionExactGPModel(train_x=X_trains[0], train_y=y_trains[0].reshape(-1), likelihood=likelihood)

    model.train()
    likelihood.train()

    model.train_model(x_trains=X_trains, y_trains=y_trains, lr=args.lr, epochs=args.epochs,
                      print_freq=args.print_freq, batch_size=args.batch_size)


    model.eval()
    likelihood.eval()

    save_directory = 'results/' + args.dataname + '/gp/'

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, 1000, 100):
            fig_name = args.dataname + '_f' + str(i)
            path_to_save_test = save_directory + fig_name + '_gp_test.png'
            path_to_save_no_test = save_directory + fig_name + '_gp_no_test.png'
            x_train = X_trains[i]
            y_train = y_trains[i].reshape(-1)
            x_test = X_tests[i]
            y_test = y_tests[i].reshape(-1)

            model.set_train_data(inputs=x_train, targets=y_train, strict=False)

            x_uniform = torch.linspace(-4, 4, 200).reshape(-1, 1)

            observed_pred = likelihood(model(x_uniform.double()))
            pdb.set_trace()
            lower, upper = observed_pred.confidence_region()
            mu_y = observed_pred.mean.numpy()
            var_y = observed_pred.variance.numpy()

            plotter1d(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, x_uniform=x_uniform, mu_y=mu_y, var_y=var_y,
                      path_to_save=path_to_save_test, plot_test=True)
            plotter1d(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, x_uniform=x_uniform, mu_y=mu_y,
                      var_y=var_y, path_to_save=path_to_save_no_test, plot_test=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='data/toy_data/1DGP_MaternCombo/')
    parser.add_argument('--dataname', type=str, default='1DGP_MaternCombo')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training iterations.')
    parser.add_argument('--print_freq', type=int, default=5,
                        help='Number of training iterations.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='The training learning rate.')

    args = parser.parse_args()

    main(args)
