import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font',family='Arial')

from scipy.stats import pearsonr

import pdb

from sklearn.metrics import r2_score, mean_squared_error

def select_descriptors(x, x_test, means, stds, n_properties, n_descriptors):
    n_descriptors_old = x.shape[1] - n_properties
    coeffs = np.zeros((n_descriptors_old, n_properties))
    for d in range(n_descriptors_old):
        xd = x[:, d]
        for q in range(n_properties):
            xq = x[:, -n_properties + q]
            ids = np.where(np.isfinite(xd) & np.isfinite(xq))
            ids = np.array(ids).reshape(-1)
            if np.std(xd[ids]) > 1.0e-9:
                coeffs[d, q] = pearsonr(xd[ids], xq[ids])[0]
            else:
                coeffs[d, q] = 0.0
    coeff_means = np.mean(np.abs(coeffs), axis=1)
    idx = np.flip(np.argsort(coeff_means))[0:n_descriptors]

    xd = x[:, idx]
    xp = x[:, -n_properties:]

    xd_test = x_test[:, idx]
    xp_test = x_test[:, -n_properties:]

    d_means = means[idx]
    p_means = means[-n_properties:]

    d_stds = stds[idx]
    p_stds = stds[-n_properties:]

    x_new = np.concatenate((xd, xp), axis=1)
    x_test_new = np.concatenate((xd_test, xp_test), axis=1)
    means_new = np.concatenate((d_means, p_means), axis=0)
    stds_new = np.concatenate((d_stds, p_stds), axis=0)
    return x_new, x_test_new, means_new, stds_new

def nan_transform_data(x_train, x_test):
    """
    Standardise data which has nan values. Return the standardised data and the
    means and standard deviations used to standardise.

    :param x_train: input train data
    :param x_test: input test data
    :return: x_train_scaled, x_test_scaled, means, stds
    """
    stds = np.nanstd(x_train, axis=0)
    x_train = x_train[:, np.where(stds > 1.0e-11)].reshape(x_train.shape[0], -1)
    x_test = x_test[:, np.where(stds > 1.0e-11)].reshape(x_test.shape[0], -1)
    means = np.nanmean(x_train, axis=0)
    stds = np.nanstd(x_train, axis=0)

    x_train_scaled = (x_train - means) / stds
    x_test_scaled = (x_test - means) / stds

    return x_train_scaled, x_test_scaled, means, stds

def parse_boolean(b):
    if len(b) < 1:
        raise ValueError('Cannot parse empty string into boolean.')
    b = b[0].lower()
    if b == 't' or b == 'y' or b == '1':
        return True
    if b == 'f' or b == 'n' or b == '0':
        return False
    raise ValueError('Cannot parse string into boolean.')

def torch_from_numpy_list(x_list):
    x_torch = []
    for i in range(len(x_list)):
        x_torch.append(torch.tensor(x_list[i]))
    return x_torch


def batch_sampler(x, y, batch_size):
    n_functions = len(x)

    # Sample the function from the set of functions
    idx_function = np.random.randint(n_functions)
    x = x[idx_function]
    y = y[idx_function]

    max_target = x.shape[0]
    x_dim = x.shape[1]
    y_dim = y.shape[1]

    # Sample n_target points from the function, and randomly select n_context points to condition on (these
    # will be a subset of the target set).
    n_target = torch.randint(low=4, high=int(max_target), size=(1,))
    n_context = torch.randint(low=3, high=int(n_target), size=(1,))

    idx = [np.random.permutation(x.shape[0])[:n_target] for i in range(batch_size)]
    idx_context = [idx[i][:n_context] for i in range(batch_size)]

    x_target = [x[idx[i], :] for i in range(batch_size)]
    y_target = [y[idx[i], :] for i in range(batch_size)]
    x_context = [x[idx_context[i], :] for i in range(batch_size)]
    y_context = [y[idx_context[i], :] for i in range(batch_size)]

    x_target = torch.stack(x_target).view(-1, x_dim)  # [batch_size*n_target, x_dim]
    y_target = torch.stack(y_target).view(-1, y_dim)  # [batch_size*n_target, y_dim]
    x_context = torch.stack(x_context).view(-1, x_dim)  # [batch_size*n_context, x_dim]
    y_context = torch.stack(y_context).view(-1, y_dim)  # [batch_size, n_context, y_dim]

    return x_context, y_context, x_target, y_target

def to_natural_params(mu, var):
    nu_1 = mu / var
    nu_2 = - 1 / (2 * var)
    return nu_1, nu_2


def from_natural_params(nu_1, nu_2):
    var = (- 1 / (2 * nu_2))
    mu = var * nu_1
    return mu, var


def nlpd(pred_mean_vec, pred_var_vec, targets):
    """
    Computes the negative log predictive density for a set of targets assuming a Gaussian noise model.
    :param pred_mean_vec: predictive mean of the model at the target input locations
    :param pred_var_vec: predictive variance of the model at the target input locations
    :param targets: target values
    :return: nlpd (negative log predictive density)
    """
    assert len(pred_mean_vec) == len(pred_var_vec)  # pred_mean_vec must have been evaluated at xs corresponding to ys.
    assert len(pred_mean_vec) == len(targets)
    nlpd = 0
    index = 0
    n = len(targets)  # number of data points
    pred_mean_vec = np.array(pred_mean_vec).reshape(n, )
    pred_var_vec = np.array(pred_var_vec).reshape(n, )
    pred_std_vec = np.sqrt(pred_var_vec)
    targets = np.array(targets).reshape(n, )
    for target in targets:
        density = scipy.stats.norm(pred_mean_vec[index], pred_std_vec[index]).pdf(target)
        nlpd += -np.log(density)
        index += 1
    nlpd /= n
    return nlpd


def plotter1d(x_train, y_train, x_test, y_test, x_uniform, mu_y, var_y, path_to_save, plot_test=False):
    x_target = x_uniform.numpy()
    std2_target = 1.96*np.sqrt(var_y)
    lb = mu_y - std2_target
    ub = mu_y + std2_target

    x_target, mu_y, lb, ub = zip(*sorted(zip(x_target, mu_y, lb, ub)))
    x_target = np.array(x_target).reshape(-1)
    mu_y = np.array(mu_y).reshape(-1)
    lb = np.array(lb).reshape(-1)
    ub = np.array(ub).reshape(-1)

    plt.figure(figsize = (7, 7))
    plt.plot(x_target, mu_y, color='darkcyan', linewidth=2.0, label='Mean prediction')
    plt.plot(x_target, lb, linestyle='-.', marker=None, color='darkcyan', linewidth=1.0)
    plt.plot(x_target, ub, linestyle='-.', marker=None, color='darkcyan', linewidth=1.0,
             label='Two standard deviations')
    plt.fill_between(x_target, lb, ub, color='cyan', alpha=0.2)
    if plot_test:
        plt.scatter(x_test, y_test, color='blue', s=20, marker='s', alpha=0.6, label="Test data")
        plt.scatter(x_train, y_train, color="red", s=25, marker = "o", alpha=0.9, label = "Training data")
    else:
        plt.scatter(x_train, y_train, color="red", s=25, marker="o", alpha=0.9, label="Observed data")
    plt.ylabel('f(x)', fontsize=24)
    plt.yticks([])
    plt.ylim(min(np.concatenate((y_train, y_test), axis=0)) - 1, max(np.concatenate((y_train, y_test), axis=0)) + 1.5)
    plt.xlim(min(x_target), max(x_target))
    plt.xlabel('x', fontsize=24)
    plt.xticks([])
    handlelength = 1.25
    handletextpad = 0.35
    loc = 'upper left'
    bbox_to_anchor = (0.0, 0.48, 0.95, 0.52)
    labelspacing = 0.4
    plt.legend(fontsize=16, loc=loc, bbox_to_anchor=bbox_to_anchor,
              frameon=True, ncol=2, handlelength=handlelength, handletextpad=handletextpad,
              labelspacing=labelspacing)
    plt.savefig(path_to_save)

    return


def metrics_calculator(model, model_name, x_trains, y_trains, x_tests, y_tests, dataname, epoch, x_scaler=None, y_scaler=None):
    directory = 'results/' + dataname + '/'
    subdirectory = model_name + '/'


    n_functions = len(x_trains)
    x_dim = x_trains[0].shape[-1]
    if x_dim == 1:
        x_uniform = torch.linspace(-4, 4, 200).reshape(-1, 1)

    r2_train_list = []
    rmse_train_list = []
    nlpd_train_list = []
    r2_test_list = []
    rmse_test_list = []
    nlpd_test_list = []

    for j in range(0, n_functions, 4):
        x_train = x_trains[j]  # N_train, x_size
        y_train = y_trains[j]
        x_test = x_tests[j]
        y_test = y_tests[j]


        # At prediction time the context points comprise the entire training set.
        if model_name == 'cnp':
            mu_y_train, var_y_train = model.forward(x_train, y_train, x_train, batch_size=1) #[n_train, y_size]
            mu_y_test, var_y_test = model.forward(x_train, y_train, x_test, batch_size=1)  #[n_test, y_size]
            if (j % (n_functions // 10) == 0) and (x_dim == 1):
                mu_y_uniform, var_y_uniform = model.forward(x_train, y_train, x_uniform, batch_size=1)
                mu_y_uniform = mu_y_uniform.reshape(-1).detach().numpy()
                var_y_uniform = var_y_uniform.reshape(-1).detach().numpy()
        elif model_name == 'vnp':
            mu_y_train, var_y_train = model.forward(x_train, y_train, x_train, nz_samples=10, ny_samples=100, batch_size=1) #[n_train, y_size]
            mu_y_test, var_y_test = model.forward(x_train, y_train, x_test, nz_samples=10, ny_samples=100, batch_size=1)  #[n_test, y_size]
            if (j % (n_functions // 10) == 0) and (x_dim == 1):
                mu_y_uniform, var_y_uniform = model.forward(x_train, y_train, x_uniform,
                                                            nz_samples=10, ny_samples=100, batch_size=1)
                mu_y_uniform = mu_y_uniform.reshape(-1).detach().numpy()
                var_y_uniform = var_y_uniform.reshape(-1).detach().numpy()
        elif model_name == 'anp':
            mu_y_train, var_y_train = model.forward(x_train, y_train, x_train, nz_samples=10, ny_samples=100, batch_size=1) #[n_train, y_size]
            mu_y_test, var_y_test = model.forward(x_train, y_train, x_test, nz_samples=10, ny_samples=100, batch_size=1)  #[n_test, y_size]
            if (j % (n_functions // 10) == 0) and (x_dim == 1):
                mu_y_uniform, var_y_uniform = model.forward(x_train, y_train, x_uniform, nz_samples=10, ny_samples=100,
                                                            batch_size=1)
                mu_y_uniform = mu_y_uniform.reshape(-1).detach().numpy()
                var_y_uniform = var_y_uniform.reshape(-1).detach().numpy()
        else:
            raise Exception('Model name should be cnp or vnp or anp.')

        mu_y_train = mu_y_train.reshape(-1).detach().numpy()
        var_y_train = var_y_train.reshape(-1).detach().numpy()
        mu_y_test = mu_y_test.reshape(-1).detach().numpy()
        var_y_test = var_y_test.reshape(-1).detach().numpy()
        y_train = y_train.reshape(-1).numpy()
        y_test = y_test.reshape(-1).numpy()

        if y_scaler is not None:
            mu_y_train = y_scaler.inverse_transform(mu_y_train)
            var_y_train = y_scaler.var_ * var_y_train
            mu_y_test = y_scaler.inverse_transform(mu_y_test)
            var_y_test = y_scaler.var_ * var_y_test
            mu_y_uniform = y_scaler.inverse_transform(mu_y_uniform)
            var_y_uniform = y_scaler.var_ * var_y_uniform
            y_train = y_scaler.inverse_transform(y_train)
            y_test = y_scaler.inverse_transform(y_test)

        r2_train_list.append(r2_score(y_train, mu_y_train))
        rmse_train_list.append(np.sqrt(mean_squared_error(y_train, mu_y_train)))
        nlpd_train_list.append(nlpd(mu_y_train, var_y_train, y_train))

        r2_test_list.append(r2_score(y_test, mu_y_test))
        rmse_test_list.append(np.sqrt(mean_squared_error(y_test, mu_y_test)))
        nlpd_test_list.append(nlpd(mu_y_test, var_y_test, y_test))


        if (j % (n_functions // 10) == 0) and (x_dim == 1) and (epoch % 20000 == 0):
            fig_name = dataname + '_f' + str(j) + '_epoch' + str(epoch) + model_name

            if x_scaler is not None:
                x_train = x_scaler.inverse_transform(x_train.reshape(-1))
                x_test = x_scaler.inverse_transform(x_test.reshape(-1))

            plotter1d(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, x_uniform=x_uniform,
                      mu_y=mu_y_uniform,
                      var_y=var_y_uniform,
                      path_to_save=directory+subdirectory + fig_name + "no_test.png", plot_test=False)
            plotter1d(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, x_uniform=x_uniform,
                      mu_y=mu_y_uniform,
                      var_y=var_y_uniform,
                      path_to_save=directory + subdirectory + fig_name + "_test.png", plot_test=True)

    r2_train_list = np.array(r2_train_list)
    rmse_train_list = np.array(rmse_train_list)
    nlpd_train_list = np.array(nlpd_train_list)
    r2_test_list = np.array(r2_test_list)
    rmse_test_list = np.array(rmse_test_list)
    nlpd_test_list = np.array(nlpd_test_list)

    print("\nR^2 score (train): {:.3f} +- {:.3f}".format(np.mean(r2_train_list),
                                                         np.std(r2_train_list) / np.sqrt(
                                                             len(r2_train_list))))
    # print("RMSE (train): {:.3f} +- {:.3f}".format(np.mean(rmse_train_list) / np.sqrt(
    # len(rmse_train_list))))
    print("NLPD (train): {:.3f} +- {:.3f}".format(np.mean(nlpd_train_list),
                                                  np.std(nlpd_train_list) / np.sqrt(
                                                      len(nlpd_train_list))))
    print("R^2 score (test): {:.3f} +- {:.3f}".format(np.mean(r2_test_list),
                                                      np.std(r2_test_list) / np.sqrt(len(r2_test_list))))
    # print("RMSE (test): {:.3f} +- {:.3f}".format(np.mean(rmse_test_list),
    # np.std(rmse_test_list) / np.sqrt(len(rmse_test_list))))
    print("NLPD (test): {:.3f} +- {:.3f}\n".format(np.mean(nlpd_test_list),
                                                   np.std(nlpd_test_list) / np.sqrt(len(nlpd_test_list))))

    return

