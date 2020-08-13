"""
Script for training various models for the goal of data imputation.
In particular, two variants on the neural process are implemented, both including FiLM layers
but one allowing for the possibility of also including skip connections. A competing model
(previously applied by Conduit et al.) is also implemented.
"""
import sys
sys.path.append('../../')
import warnings
import argparse
import pdb
import numpy as np
import torch
from sklearn.decomposition import PCA

#from models.imputation_models.conduit_model import ConduitModel
from models.imputation_models.conduit_models import SetofConduitModels
from models.imputation_models.np_basic import NPBasic
from models.imputation_models.np_film import NPFiLM
from utils.data_utils import nan_transform_data, select_descriptors, parse_boolean
#from utils.imputation_utils import baseline_metrics_calculator

#from npfilm import NPFiLM
#from npfilm2 import NPFiLM2
#from npfilm3 import NPFiLM3
#from npfilm4 import NPFiLM4




def main(args):
    """
    :return:
    """

    warnings.filterwarnings('ignore')
    torch.set_default_dtype(torch.float64)

    filename = args.dataname + str(args.num) + '_' + args.model_name + '_' + str(args.run_number)

    with open('results/{}/{}.txt'.format(args.dataname, filename), 'w+') as f:
        f.write('\n Input data:')
        f.write('\n Dataname = ' + args.dataname)
        f.write('\n Split = ' + str(args.num))
        f.write('\n Number of properties = ' + str(args.n_properties))
        if args.n_descriptors > 0:
            f.write('\n Number of descriptors = ' + str(args.n_descriptors))
        else:
            f.write('\n Using all descriptors.')

        f.write('\n Model architecture:')
        f.write('\n Model name = ' + args.model_name)
        if args.model_name == 'baseline':
            f.write('\n Imputing mean and standard deviation of each property. \n')
        elif args.model_name == 'conduit':
            f.write('\n Hidden layer size = ' + str(args.hidden_dim))
            f.write('\n Number of cycles = ' + str(args.n_cycles))
        elif args.model_name == 'setofconduits':
            f.write('\n Hidden layer size = ' + str(args.hidden_dim))
            f.write('\n Number of cycles = ' + str(args.n_cycles))
            f.write('\n Number of networks = ' + str(args.n_networks))
        elif args.model_name == 'npbasic':
            f.write('\n Encoder hidden dimensions = ' + str(args.d_encoder_dims))
            f.write('\n Latent variable size = ' + str(args.z_dim))
            f.write('\n Decoder hidden dimensions = ' + str(args.decoder_dims))
        elif args.model_name == 'npfilm':
            f.write('\n Descriptor encoder hidden dimensions = ' + str(args.d_encoder_dims))
            f.write('\n Property encoder hidden dimensions = ' + str(args.p_encoder_dims))
            f.write('\n Latent variable size = ' + str(args.z_dim))
            f.write('\n Decoder hidden dimensions = ' + str(args.decoder_dims))
            f.write('\n Initial d sigma = ' + str(args.initial_d_sigma))
            f.write('\n Initial p sigma = ' + str(args.initial_p_sigma))
        elif (args.model_name == 'npfilm2') or (args.model_name ==
                                                'npfilm3') or (args.model_name ==
                                                'npfilm4'):
            f.write('\n Activation function = ' + args.nonlinearity)
            f.write('\n Descriptor encoder hidden dimensions = ' + str(args.d_encoder_hidden_dims))
            f.write('\n Property encoder hidden dimensions = ' + str(args.p_encoder_hidden_dims))
            f.write('\n Latent variable size = ' + str(args.z_dim))
            f.write('\n Decoder hidden dimensions = ' + str(args.decoder_hidden_dims))
            f.write('\n Number of cycles = ' + str(args.n_cycles))
            f.write('\n Number of samples = ' + str(args.num_samples))
        else:
            f.write('Wrong model name inputted. Try conduit, npfilm, or npfilm2.')
            raise Exception('Wrong model name inputted. Try conduit, npfilm, or npfilm2.')

        f.write('\n Maximum number of iterations = ' + str(args.epochs))
        f.write('\n Batch size = ' + str(args.batch_size))
        f.write('\n Optimiser learning rate = ' + str(args.lr))
        f.write('\n Print frequency = ' + str(args.print_freq) + '\n')
        f.flush()

        use_latent_mean = parse_boolean(args.use_latent_mean)

        x = np.load(args.directory + args.dataname + '_x' + str(args.num) +
                    '_train_dscrpt.npy')
        x_test = np.load(args.directory + args.dataname + '_x' + str(args.num) +
                         '_test_dscrpt.npy')

        # Transform the data: standardise to zero mean and unit variance
        x, x_test, means, stds = nan_transform_data(x, x_test)

        if args.n_descriptors > 0:
            x, x_test, means, stds = select_descriptors(x, x_test, means, stds, args.n_properties, args.n_descriptors)

        x = torch.tensor(x, dtype=torch.float64)
        x_test = torch.tensor(x_test, dtype=torch.float64)

        if args.model_name == 'baseline':
            f.write('\n ... predictions from baseline model.')

            r2_scores, mlls = baseline_metrics_calculator(x, n_properties=args.n_properties,
                                                           means=means, stds=stds)

            r2_scores = np.array(r2_scores)
            mlls = np.array(mlls)
            f.write('\n R^2 score (train): {:.3f}+- {:.3f}'.format(
                np.mean(r2_scores), np.std(r2_scores)))
            f.write('\n MLL (train): {:.3f}+- {:.3f} \n'.format(
                np.mean(mlls), np.std(mlls)))

            #f.write(str(r2_scores))

            r2_scores, mlls = baseline_metrics_calculator(x_test, n_properties=args.n_properties,
                                                           means=means, stds=stds)

            r2_scores = np.array(r2_scores)
            mlls = np.array(mlls)
            f.write('\n R^2 score (test): {:.3f}+- {:.3f}'.format(
                np.mean(r2_scores), np.std(r2_scores)))
            f.write('\n MLL (test): {:.3f}+- {:.3f} \n'.format(
                np.mean(mlls), np.std(mlls)))

            f.write('\n R2 scores: \n')
            f.write(str(r2_scores))
            f.write('\n MLLs: \n')
            f.write(str(mlls))

        elif args.model_name == 'conduit':
            f.write('\n ... building Conduit model.')

            model = ConduitModel(in_dim=x.shape[1], hidden_dim=args.hidden_dim, n_properties=args.n_properties)

            f.write('\n ... training. \n')
            f.flush()

            # Train the model(NB can replace x_test, y_test with x_valid and y_valid if planning to use
            # a cross validation set)
            model.train(x=x, epochs=args.epochs, n_cycles=args.n_cycles, batch_size=args.batch_size,
                        file=f, print_freq=args.print_freq, x_test=x_test, means=means, stds=stds,
                        lr=args.lr)
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
            f.write('\n ... building NPFiLM model (no skip connections).')

            model = NPFiLM(in_dim=x.shape[1], out_dim=1, z_dim=args.z_dim,
                           n_properties=args.n_properties,
                           d_encoder_dims=args.d_encoder_dims,
                           p_encoder_dims=args.p_encoder_dims,
                           decoder_dims=args.decoder_dims)

            f.write('\n ... training. \n')
            f.flush()

            model.train(x=x, epochs=args.epochs, batch_size=args.batch_size, file=f,
                        print_freq=args.print_freq, x_test=x_test, means=means, stds=stds, lr=args.lr)

        elif args.model_name == 'npfilm2':
            f.write('\n ... building NPFiLM model (skip connections).')

            model = NPFiLM2(in_dim=x.shape[1], out_dim=1, z_dim=args.z_dim,
                            n_properties=args.n_properties,
                            d_encoder_hidden_dims=args.d_encoder_hidden_dims,
                            p_encoder_hidden_dims=args.p_encoder_hidden_dims,
                            decoder_hidden_dims=args.decoder_hidden_dims,
                            nonlinearity=args.nonlinearity,
                            initial_d_sigma=args.initial_d_sigma,
                            initial_p_sigma=args.initial_p_sigma)

            f.write('\n ... training. \n')
            f.flush()

            model.train(x=x, epochs=args.epochs, batch_size=args.batch_size,
                        file=f, print_freq=args.print_freq,
                        x_test=x_test, means=means, stds=stds, lr=args.lr,
                        n_cycles=args.n_cycles, num_samples=args.num_samples)

        elif args.model_name == 'npfilm3':
            f.write('\n ... building NPFiLM model (skip connections, only cycle descriptor once).')

            model = NPFiLM3(in_dim=x.shape[1], out_dim=1, z_dim=args.z_dim,
                            n_properties=args.n_properties,
                            d_encoder_hidden_dims=args.d_encoder_hidden_dims,
                            p_encoder_hidden_dims=args.p_encoder_hidden_dims,
                            decoder_hidden_dims=args.decoder_hidden_dims,
                            nonlinearity=args.nonlinearity,
                            initial_d_sigma=args.initial_d_sigma,
                            initial_p_sigma=args.initial_p_sigma)

            f.write('\n ... training. \n')
            f.flush()

            model.train(x=x, epochs=args.epochs, batch_size=args.batch_size,
                        file=f, print_freq=args.print_freq,
                        x_test=x_test, means=means, stds=stds, lr=args.lr,
                        n_cycles=args.n_cycles, num_samples=args.num_samples)

        elif args.model_name == 'npfilm4':
            f.write('\n ... building NPFiLM model (skip connections, only cycle descriptor once).')

            model = NPFiLM4(in_dim=x.shape[1], out_dim=1, z_dim=args.z_dim,
                            n_properties=args.n_properties,
                            d_encoder_hidden_dims=args.d_encoder_hidden_dims,
                            p_encoder_hidden_dims=args.p_encoder_hidden_dims,
                            decoder_hidden_dims=args.decoder_hidden_dims,
                            nonlinearity=args.nonlinearity,
                            initial_d_sigma=args.initial_d_sigma,
                            initial_p_sigma=args.initial_p_sigma)

            f.write('\n ... training. \n')
            f.flush()

            model.train(x=x, epochs=args.epochs, batch_size=args.batch_size,
                        file=f, print_freq=args.print_freq,
                        x_test=x_test, means=means, stds=stds, lr=args.lr,
                        n_cycles=args.n_cycles,
                        num_samples=args.num_samples,
                        use_latent_mean=use_latent_mean)
        else:
            raise Exception('Wrong model name inputted. Try conduit, npfilm, or npfilm2.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--directory', default='data/raw_data/',
                        help='Directory where the training and test data is stored.')
    parser.add_argument('--dataname', default='Adrenergic',
                        help='Name of dataset.')
    parser.add_argument('--run_number', type=int, default=0, help='Run number.')
    parser.add_argument('--num', type=int, default=1,
                        help='The train/test split number. 1 '
                             'for Kinase, between 1 and 5 for '
                             'Adrenergic.')
    parser.add_argument('--n_properties', type=int, default=5,
                        help='The number of properties.')
    parser.add_argument('--model_name', default='setofconduits',
                        help='Model to use.')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='Number of training iterations.')
    parser.add_argument('--n_descriptors', type=int, default=0,
                        help='The number of desired descriptors.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of function samples per iteration.')
    parser.add_argument('--z_dim', type=int, default=16,
                        help='Dimensionality of context encoding, r.')
    parser.add_argument('--d_encoder_dims', nargs='+', type=int,
                        default=[64, 64],
                        help='Dimensionality of descriptor latent encoder hidden layers.')
    parser.add_argument('--p_encoder_dims', nargs='+', type=int,
                        default=[16, 16, 16],
                        help='Dimensionality of property latent encoder hidden layers.')
    parser.add_argument('--decoder_dims', nargs='+', type=int,
                        default=[32, 32],
                        help='Dimensionality of decoder hidden layers.')
    parser.add_argument('--hidden_dim', type=int, default=25,
                        help='Dimensionality of hidden layer in the Conduit model.')
    parser.add_argument('--n_networks', type=int, default=12,
                        help='The number of networks in the ensemble (Conduit model).')
    parser.add_argument('--n_cycles', type=int, default=5,
                        help='The number of iteration cycles.')
    parser.add_argument('--initial_d_sigma', type=float, default=None,
                        help='The initial sigma for the descriptor encoder.')
    parser.add_argument('--initial_p_sigma', type=float, default=None,
                        help='The initial sigma for the property encoders.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Optimiser learning rate.')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print information every print_freq epochs.')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Print information every print_freq epochs.')
    parser.add_argument('--non_linearity', default='relu',
                        help='Non linear activation function.')
    parser.add_argument('--use_latent_mean', default='t',
                        help='Whether to use the latent mean values for all '
                             'but the final cycle.')
    args = parser.parse_args()

    main(args)
