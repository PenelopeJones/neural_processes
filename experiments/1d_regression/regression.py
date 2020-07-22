import sys
sys.path.append('../')

import time
import warnings
import argparse

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from models.conditional_np import CNP
from utils.data_utils import transform_data, x_generator, noisy_function, nlpd


def main(dataname, learning_rate, iterations, r_size,
         encoder_hidden_size, encoder_n_hidden, decoder_hidden_size, decoder_n_hidden,
         testing, plotting):
    """
    :param context_set_samples: Integer, describing the number of times we should sample the set
                                of context points used to form the aggregated embedding during
                                training, given the number of context points to be sampled
                                N_context. When testing this is set to 1
    :param learning_rate: A float number, describing the optimiser's learning rate
    :param iterations: An integer, describing the number of iterations. In this case it also
                       corresponds to the number of times we sample the number of context points
                       N_context
    :param r_size: An integer describing the dimensionality of the embedding / context vector r
    :param encoder_hidden_size: An integer describing the number of nodes per hidden layer in the
                                encoder neural network
    :param encoder_n_hidden: An integer describing the number of hidden layers in the encoder neural
                             network
    :param decoder_hidden_size: An integer describing the number of nodes per hidden layer in the
                                decoder neural network
    :param decoder_n_hidden: An integer describing the number of hidden layers in the decoder neural
                             network
    :param testing: A Boolean variable; if true, during testing the RMSE on test and train data '
                             'will be printed after a specific number of iterations.
    :param plotting: A Boolean variable; if true, during testing the context points and predicted mean '
                             'and variance will be plotted after a specific number of iterations.
    :return:
    """
    warnings.filterwarnings('ignore')

    r2_list = []
    rmse_list = []
    mae_list = []
    time_list = []
    print('\nBeginning training loop...')
    j = 0
    for i in range(5,6):
        start_time = time.time()

        #Generate values of x in the range [min_x, max_x], to be used for training
        X_train = x_generator(min_x, max_x, n_points)
        y_train = noisy_function(X_train, std)
        np.save('xtrain_1dreg' + str(i) + '.npy', X_train)
        np.save('ytrain_1dreg' + str(i) + '.npy', y_train)

        #Generate target values of x and y for plotting later on
        X_test = np.expand_dims(np.linspace(min_x - 0.5, max_x + 0.5, 250), axis = 1)
        y_test = noisy_function(X_test, std)

        np.save('xtest_1dreg' + str(i) + '.npy', X_test)
        np.save('ytest_1dreg' + str(i) + '.npy', y_test)

        X_train, y_train, X_test, y_test, x_scaler, y_scaler = transform_data(X_train, y_train, X_test,
                                                                    y_test)
        #Convert the data for use in PyTorch.
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()

        print('... building model.')

        #Build the Conditional Neural Process model, with the following architecture:
        #(x, y)_i --> encoder --> r_i
        #r = average(r_i)
        #(x*, r) --> decoder --> y_mean*, y_var*
        #The encoder and decoder functions are neural networks, with size and number of layers being
        # hyperparameters to be selected.
        cnp = CNP(x_size=X_train.shape[1], y_size=y_train.shape[1], r_size=r_size,
                  encoder_hidden_size=encoder_hidden_size, encoder_n_hidden=encoder_n_hidden,
                  decoder_hidden_size=decoder_hidden_size, decoder_n_hidden=decoder_n_hidden)

        print('... training.')

        #Train the model(NB can replace x_test, y_test with x_valid and y_valid if planning to use
        # a cross validation set)
        cnp.train(x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test, x_scaler=x_scaler,
                  y_scaler=y_scaler, context_set_samples=context_set_samples, lr=learning_rate,
                  iterations=iterations, testing=testing, plotting=plotting)

        #Testing: the 'context points' when testing are the entire training set, and the 'target
        # points' are the entire test set.
        x_context = torch.unsqueeze(X_train, dim=0)
        y_context = torch.unsqueeze(y_train, dim=0)
        x_test = torch.unsqueeze(X_test, dim=0)

        #Predict mean and error in y given the test inputs x_test
        _, predict_test_mean, predict_test_var = cnp.predict(x_context, y_context, x_test)

        predict_test_mean = np.squeeze(predict_test_mean.data.numpy(), axis=0)
        predict_test_var = np.squeeze(predict_test_var.data.numpy(), axis=0)

        # We transform the standardised predicted and actual y values back to the original data
        # space
        y_mean_pred = y_scaler.inverse_transform(predict_test_mean)
        y_var_pred = y_scaler.var_ * predict_test_var
        y_test = y_scaler.inverse_transform(y_test)

        #Calculate relevant metrics
        score = r2_score(y_test, y_mean_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_mean_pred))
        mae = mean_absolute_error(y_test, y_mean_pred)
        nlpd_test = nlpd(y_mean_pred, y_var_pred, y_test)
        time_taken = time.time() - start_time

        np.save('ytest_mean_pred_1dreg' + str(i) + '.npy', y_mean_pred)
        np.save('ytest_var_pred_1dreg' + str(i) + '.npy', y_var_pred)

        print("\nR^2: {:.3f}".format(score))
        print("RMSE: {:.3f}".format(rmse))
        print("MAE: {:.3f}".format(mae))
        print("NLPD: {:.4f}".format(nlpd_test))
        print("Execution time: {:.3f}".format(time_taken))
        r2_list.append(score)
        rmse_list.append(rmse)
        mae_list.append(mae)
        time_list.append(time_taken)

        if plotting:
            x_context = np.load('xtrain_1dreg' + str(i) + '.npy')
            y_context = np.load('ytrain_1dreg' + str(i) + '.npy')
            x_test = np.load('xtest_1dreg' + str(i) + '.npy')
            y_test = np.load('ytest_1dreg' + str(i) + '.npy')

            plt.figure(figsize = (9, 9))
            plt.scatter(x_context, y_context, color = 'black', s = 10, marker = '+', label = "Context points")
            plt.plot(x_test, y_test, linewidth = 1, color = 'red', label = "Target function")
            plt.plot(x_test, y_mean_pred, color='darkcyan', linewidth=1, label='Predicted mean')
            plt.fill_between(x_test[:, 0], y_mean_pred[:, 0] - 1.96 * np.sqrt(y_var_pred[:, 0]),
                             y_mean_pred[:, 0] + 1.96 * np.sqrt(y_var_pred[:, 0]),
                             color='cyan', alpha=0.2)
            plt.legend()
            plt.savefig('results/cnp_1d_reg' + str(i) + '.png')

        j += 1

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)
    time_list = np.array(time_list)

    print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list),
                                                np.std(r2_list)/np.sqrt(len(r2_list))))
    print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list),
                                               np.std(rmse_list) / np.sqrt(len(rmse_list))))
    print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list),
                                                np.std(mae_list) / np.sqrt(len(mae_list))))
    print("mean Execution time: {:.3f} +- {:.3f}\n".format(np.mean(time_list),
                                                           np.std(time_list)/
                                                           np.sqrt(len(time_list))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--encoder_dims', type=int, nargs='+',
                        default=[16, 16],
                        help='Dimensions of encoder network.')
    parser.add_argument('--decoder_dims', type=int, nargs='+',
                        default=[16, 16],



    parser.add_argument('--context_set_samples', type=int, default=1,
                        help='The number of samples to take of the context set, given the number of'
                             ' context points that should be selected.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='The training learning rate.')
    parser.add_argument('--iterations', type=int, default=5000,
                        help='Number of training iterations.')
    parser.add_argument('--r_size', type=int, default=2,
                        help='Dimensionality of context encoding, r.')
    parser.add_argument('--encoder_hidden_size', type=int, default=4,
                        help='Dimensionality of encoder hidden layers.')
    parser.add_argument('--encoder_n_hidden', type=int, default=2,
                        help='Number of encoder hidden layers.')
    parser.add_argument('--decoder_hidden_size', type=int, default=4,
                        help='Dimensionality of decoder hidden layers.')
    parser.add_argument('--decoder_n_hidden', type=int, default=2,
                        help='Number of decoder hidden layers.')
    parser.add_argument('--testing', default=False,
                        help='If true, during testing the RMSE on test and train data '
                             'will be printed after specific numbers of iterations.')
    parser.add_argument('--plotting', default=False,
                        help='If true, at the end of training a plot will be produced.')
    args = parser.parse_args()

    main(args.context_set_samples, args.learning_rate,
         args.iterations, args.r_size, args.encoder_hidden_size, args.encoder_n_hidden,
         args.decoder_hidden_size, args.decoder_n_hidden, args.testing, args.plotting)