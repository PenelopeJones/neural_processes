"""
Script for training various models for the goal of data imputation.
In particular, two variants on the neural process are implemented, both including FiLM layers
but one allowing for the possibility of also including skip connections. A competing model
(previously applied by Conduit et al.) is also implemented.
"""
import os
import sys
sys.path.append('../../')
import warnings
import argparse

import numpy as np
import torch

from models.imputation_models.conduit_models import SetofConduitModels
from models.imputation_models.np_basic import NPBasic
from models.imputation_models.cnp_basic import CNPBasic
from models.imputation_models.np_film import NPFiLM
from utils.data_utils import nan_transform_data, select_descriptors, parse_boolean
from utils.metric_utils import baseline_metrics_calculator
from sklearn.decomposition import PCA


import pdb

def main(args):
    """
    :return:
    """

    warnings.filterwarnings('ignore')
    torch.set_default_dtype(torch.float64)

    filename = args.dataname + str(args.num) + '_' + args.model_name + '_' + str(args.run_number)

    with open('results/{}/{}/{}.txt'.format(args.dataname, args.model_name, filename), 'w+') as f:
        f.write('\n Input data:')
        f.write('\n Dataname = ' + args.dataname)
        f.write('\n Split = ' + str(args.num))
        f.write('\n Number of properties = ' + str(args.n_properties))
        if args.n_descriptors > 0:
            f.write('\n Number of descriptors = ' + str(args.n_descriptors))
        else:
            f.write('\n Using all descriptors.')
        f.write('\n Number of PCA components (0 if no PCA used) = ' + str(args.pca_components) + '\n')

        f.write('\n Model architecture:')
        f.write('\n Model name = ' + args.model_name)

        if args.model_name == 'baseline':
            f.write('\n Imputing mean and standard deviation of each property. \n')

        elif args.model_name == 'setofconduits':
            f.write('\n Hidden layer size = ' + str(args.hidden_dim))
            f.write('\n Number of cycles = ' + str(args.n_cycles))
            f.write('\n Number of networks = ' + str(args.n_networks))

        elif args.model_name == 'cnpbasic':
            f.write('\n Encoder hidden dimensions = ' + str(args.d_encoder_dims))
            f.write('\n Latent variable size = ' + str(args.z_dim))
            f.write('\n Decoder hidden dimensions = ' + str(args.decoder_dims))

        elif args.model_name == 'npbasic':
            f.write('\n Encoder hidden dimensions = ' + str(args.d_encoder_dims))
            f.write('\n Latent variable size = ' + str(args.z_dim))
            f.write('\n Decoder hidden dimensions = ' + str(args.decoder_dims))

        elif args.model_name == 'npfilm':
            f.write('\n Descriptor encoder hidden dimensions = ' + str(args.d_encoder_dims))
            f.write('\n Property encoder hidden dimensions = ' + str(args.p_encoder_dims))
            f.write('\n Latent variable size = ' + str(args.z_dim))
            f.write('\n Decoder hidden dimensions = ' + str(args.decoder_dims))

        else:
            f.write('Wrong model name inputted. Try conduit, npfilm.')
            raise Exception('Wrong model name inputted. Try setofconduits, npfilm, '
                            'npbasic or cnpbasic.')

        f.write('\n Maximum number of iterations = ' + str(args.epochs))
        f.write('\n Batch size = ' + str(args.batch_size))
        f.write('\n Optimiser learning rate = ' + str(args.lr))
        f.write('\n Print frequency = ' + str(args.print_freq) + '\n')
        f.flush()

        x = np.load(args.directory + args.dataname + '_x' + str(args.num) +
                    '_train_dscrpt.npy')
        x_test = np.load(args.directory + args.dataname + '_x' + str(args.num) +
                         '_test_dscrpt.npy')

        if args.pca_components > 0:
            temp = x[:, (-args.n_properties):]
            temp_test = x_test[:, (-args.n_properties):]

            pca = PCA(n_components=args.pca_components)
            x_d = pca.fit_transform(x[:, :(-args.n_properties)])
            x_test_d = pca.transform(x_test[:, :(-args.n_properties)])

            x = np.concatenate((x_d, temp), axis=1)
            x_test = np.concatenate((x_test_d, temp_test), axis=1)

            f.write('\n Using PCA. Explained variance ratio: {}'.format(
                pca.explained_variance_ratio_))


        # Transform the data: standardise to zero mean and unit variance
        x, x_test, means, stds = nan_transform_data(x, x_test)

        if args.n_descriptors > 0:
            x, x_test, means, stds = select_descriptors(x, x_test, means, stds, args.n_properties, args.n_descriptors)

        x = torch.tensor(x, dtype=torch.float64)
        x_test = torch.tensor(x_test, dtype=torch.float64)

        if args.model_name == 'baseline':
            f.write('\n ... predictions from baseline model.')

            r2_scores, mlls, rmses = baseline_metrics_calculator(x, n_properties=args.n_properties,
                                                           means=means, stds=stds)
            r2_scores = np.array(r2_scores)
            mlls = np.array(mlls)
            rmses = np.array(rmses)
            f.write('\n R^2 score (train): {:.3f}+- {:.3f}'.format(
                np.mean(r2_scores), np.std(r2_scores)))
            f.write('\n MLL (train): {:.3f}+- {:.3f} \n'.format(
                np.mean(mlls), np.std(mlls)))
            f.write('\n RMSE (train): {:.3f}+- {:.3f} \n'.format(
                np.mean(rmses), np.std(rmses)))
            #f.write(str(r2_scores))
            r2_scores, mlls, rmses = baseline_metrics_calculator(x_test, n_properties=args.n_properties,
                                                           means=means, stds=stds)
            r2_scores = np.array(r2_scores)
            mlls = np.array(mlls)
            rmses = np.array(rmses)
            f.write('\n R^2 score (test): {:.3f}+- {:.3f}'.format(
                np.mean(r2_scores), np.std(r2_scores)))
            f.write('\n MLL (test): {:.3f}+- {:.3f} \n'.format(
                np.mean(mlls), np.std(mlls)))
            f.write('\n RMSE (test): {:.3f}+- {:.3f} \n'.format(
                np.mean(rmses), np.std(rmses)))

            f.write('\n R2 scores: \n')
            f.write(str(r2_scores))
            f.write('\n MLLs: \n')
            f.write(str(mlls))

            dir_name = os.path.dirname(f.name)
            pts = f.name[len(dir_name) + 1:-4]

            path_to_save = dir_name + '/' + pts
            np.save(path_to_save + 'r2_scores.npy', r2_scores)
            np.save(path_to_save + 'mll_scores.npy', mlls)

        elif args.model_name == 'setofconduits':
            f.write('\n ... building set of Conduit models.')

            model = SetofConduitModels(in_dim=x.shape[1], hidden_dim=args.hidden_dim,
                                       n_properties=args.n_properties, n_networks=args.n_networks,
                                       lr=args.lr)

            f.write('\n ... training. \n')
            f.flush()

            # Train the model(NB can replace x_test, y_test with x_valid and y_valid if planning to use
            # a cross validation set)
            model.train_models(x=x, total_epochs=args.epochs, n_cycles=args.n_cycles, batch_size=args.batch_size,
                        file=f, print_freq=args.print_freq, x_test=x_test, means=means, stds=stds)

        elif args.model_name == 'cnpbasic':
            f.write('\n ... building basic CNP.')

            model = CNPBasic(in_dim=x.shape[1], out_dim=1, z_dim=args.z_dim,
                           n_properties=args.n_properties,
                           encoder_dims=args.d_encoder_dims,
                           decoder_dims=args.decoder_dims)

            f.write('\n ... training. \n')
            f.flush()

            model.train_model(x=x, epochs=args.epochs, batch_size=args.batch_size, file=f,
                        print_freq=args.print_freq, x_test=x_test, means=means, stds=stds, lr=args.lr)

        elif args.model_name == 'npbasic':
            f.write('\n ... building basic NP.')

            model = NPBasic(in_dim=x.shape[1], out_dim=1, z_dim=args.z_dim,
                           n_properties=args.n_properties,
                           encoder_dims=args.d_encoder_dims,
                           decoder_dims=args.decoder_dims)

            f.write('\n ... training. \n')
            f.flush()

            model.train(x=x, epochs=args.epochs, batch_size=args.batch_size, file=f,
                        print_freq=args.print_freq, x_test=x_test, means=means, stds=stds, lr=args.lr)

        elif args.model_name == 'npfilm':
            f.write('\n ... building NPFiLM model.')

            model = NPFiLM(in_dim=x.shape[1], out_dim=1, z_dim=args.z_dim,
                           n_properties=args.n_properties,
                           d_encoder_dims=args.d_encoder_dims,
                           p_encoder_dims=args.p_encoder_dims,
                           decoder_dims=args.decoder_dims)

            f.write('\n ... training. \n')
            f.flush()

            model.train(x=x, epochs=args.epochs, batch_size=args.batch_size, file=f,
                        print_freq=args.print_freq, x_test=x_test, means=means, stds=stds, lr=args.lr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='data/raw_data/',
                        help='Directory where the training and test data is stored.')
    parser.add_argument('--dataname', default='Adrenergic',
                        help='Name of dataset.')
    parser.add_argument('--run_number', type=int, default=12, help='Run number.')
    parser.add_argument('--num', type=int, default=1,
                        help='The train/test split number. 1 '
                             'for Kinase, between 1 and 5 for '
                             'Adrenergic.')
    parser.add_argument('--n_properties', type=int, default=5,
                        help='The number of properties.')
    parser.add_argument('--model_name', default='npfilm',
                        help='Model to use.')
    parser.add_argument('--epochs', type=int, default=251,
                        help='Number of training iterations.')
    parser.add_argument('--n_descriptors', type=int, default=0,
                        help='The number of desired descriptors.')
    parser.add_argument('--pca_components', type=int, default=0,
                        help='The number of pca components.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of function samples per iteration.')
    parser.add_argument('--z_dim', type=int, default=16,
                        help='Dimensionality of context encoding, r.')
    parser.add_argument('--d_encoder_dims', nargs='+', type=int,
                        default=[64, 64],
                        help='Dimensionality of descriptor latent encoder hidden layers.')
    parser.add_argument('--p_encoder_dims', nargs='+', type=int,
                        default=[16, 16],
                        help='Dimensionality of property latent encoder hidden layers.')
    parser.add_argument('--decoder_dims', nargs='+', type=int,
                        default=[25,],
                        help='Dimensionality of decoder hidden layers.')
    parser.add_argument('--hidden_dim', type=int, default=25,
                        help='Dimensionality of hidden layer in the Conduit model.')
    parser.add_argument('--n_networks', type=int, default=12,
                        help='The number of networks in the ensemble (Conduit model).')
    parser.add_argument('--n_cycles', type=int, default=5,
                        help='The number of iteration cycles.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Optimiser learning rate.')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print information every print_freq epochs.')
    args = parser.parse_args()

    main(args)
