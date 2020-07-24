import sys
sys.path.append('../../')

import time
import warnings
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from models.conditional_np import CNP
from utils.data_utils import torch_from_numpy_list
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

    # Build the Conditional Neural Process model, with the following architecture:
    # (x, y)_i --> encoder --> r_i
    # r = average(r_i)
    # (x*, r) --> decoder --> y_mean*, y_var*
    # The encoder and decoder functions are neural networks, with size and number of layers being
    # hyperparameters to be selected.
    cnp = CNP(x_dim=X_trains[0].shape[-1], y_dim=y_trains[0].shape[-1], r_dim=args.r_dim,
              encoder_dims=args.encoder_dims, decoder_dims=args.decoder_dims,
              encoder_non_linearity=F.relu, decoder_non_linearity=F.relu)

    print('... training.')

    # Train the model(NB can replace x_test, y_test with x_valid and y_valid if planning to use
    # a cross validation set)
    cnp.train(x=X_trains, y=y_trains, x_test=X_tests, y_test=y_tests, x_scaler=None, y_scaler=None, batch_size=args.batch_size, lr=args.lr,
              epochs=args.epochs, print_freq=args.print_freq, dataname=args.dataname)

    pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='../data/toy_data/1DGP_2SE/')
    parser.add_argument('--dataname', type=str, default='1DGP_2SE')
    parser.add_argument('--r_dim', type=int, default=8,
                        help='Dimensionality of context encoding, r.')
    parser.add_argument('--encoder_dims', type=int, nargs='+',
                        default=[16, 16],
                        help='Dimensions of encoder network.')
    parser.add_argument('--decoder_dims', type=int, nargs='+',
                        default=[16, 16],
                        help='Dimensions of decoder network.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--epochs', type=int, default=80000,
                        help='Number of training iterations.')
    parser.add_argument('--print_freq', type=int, default=10000,
                        help='Number of training iterations.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='The training learning rate.')

    args = parser.parse_args()

    main(args)