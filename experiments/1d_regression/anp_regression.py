import sys
sys.path.append('../../')

import warnings
import argparse

import numpy as np
import torch.nn.functional as F

from models.np_models.attentive_np import AttentiveNP
from utils.data_utils import torch_from_numpy_list


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

    anp = AttentiveNP(x_dim=X_trains[0].shape[-1], y_dim=y_trains[0].shape[-1], r_dim=args.r_dim,
                      det_encoder_dims=args.det_encoder_dims, prob_encoder_dims=args.prob_encoder_dims,
                      decoder_dims=args.decoder_dims, decoder_non_linearity=F.relu)

    print('... training.')

    anp.train(x=X_trains, y=y_trains, x_test=X_tests, y_test=y_tests, x_scaler=None, y_scaler=None,
              nz_samples=args.nz_samples, ny_samples=args.ny_samples, batch_size=args.batch_size, lr=args.lr,
              epochs=args.epochs, print_freq=args.print_freq, VERBOSE=args.VERBOSE, dataname=args.dataname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='data/toy_data/1DGP_MaternCombo/')
    parser.add_argument('--dataname', type=str, default='1DGP_MaternCombo')
    parser.add_argument('--r_dim', type=int, default=8,
                        help='Dimensionality of context encoding, r.')
    parser.add_argument('--det_encoder_dims', type=int, nargs='+',
                        default=[16, 16, 16],
                        help='Dimensions of encoder network.')
    parser.add_argument('--prob_encoder_dims', type=int, nargs='+',
                        default=[16, 16, 16],
                        help='Dimensions of encoder network.')
    parser.add_argument('--decoder_dims', type=int, nargs='+',
                        default=[32, 32],
                        help='Dimensions of decoder network.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--nz_samples', type=int, default=5,
                        help='Number of z samples.')
    parser.add_argument('--ny_samples', type=int, default=50,
                        help='Number of y samples for each z sample.')
    parser.add_argument('--epochs', type=int, default=80000,
                        help='Number of training iterations.')
    parser.add_argument('--print_freq', type=int, default=500,
                        help='Number of training iterations.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='The training learning rate.')
    parser.add_argument('--VERBOSE', type=bool, default=True,
                        help='Whether to calculate metrics and plot.')

    args = parser.parse_args()

    main(args)