import numpy as np
import scipy
import matplotlib.pyplot as plt
import pdb

from sklearn.metrics import r2_score, mean_squared_error

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


def plotter1d(x_train, y_train, x_test, y_test, mu_y, var_y, path_to_save):
    x_target = np.concatenate((x_train, x_test), axis=0)
    std2_target = 1.96*np.sqrt(var_y)
    lb = mu_y - std2_target
    ub = mu_y + std2_target

    x_target, mu_y, lb, ub = zip(*sorted(zip(x_target, mu_y, lb, ub)))
    x_target = np.array(x_target).reshape(-1)
    mu_y = np.array(mu_y).reshape(-1)
    lb = np.array(lb).reshape(-1)
    ub = np.array(ub).reshape(-1)

    plt.figure(figsize = (7, 7))
    plt.scatter(x_train, y_train, color="red", s=3, marker = "o", label = "Training data: context points")
    plt.scatter(x_test, y_test, color='blue', s=3, marker='o', label="Test data: targets")
    plt.plot(x_target, mu_y, color='darkcyan', linewidth=1, label='Mean prediction')
    plt.plot(x_target, lb, linestyle='-.', marker=None, color='darkcyan', linewidth=0.5)
    plt.plot(x_target, ub, linestyle='-.', marker=None, color='darkcyan', linewidth=0.5,
             label='Two standard deviations')
    plt.fill_between(x_target, lb, ub, color='cyan', alpha=0.2)
    plt.title('Predictive distribution')
    plt.ylabel('f(x)')
    plt.yticks([])
    plt.ylim(min(np.concatenate((y_train, y_test), axis=0)) - 1, max(np.concatenate((y_train, y_test), axis=0)) + 1)
    plt.xlim(min(x_target), max(x_target))
    plt.xlabel('x')
    plt.xticks([])
    plt.legend()
    plt.savefig(path_to_save)

    return


def metrics_calculator(model, model_name, x_trains, y_trains, x_tests, y_tests, dataname, epoch, x_scaler=None, y_scaler=None):
    directory = 'results/'

    n_functions = len(x_trains)
    x_dim = x_trains[0].shape[-1]

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
        elif model_name == 'vnp':
            mu_y_train, var_y_train = model.forward(x_train, y_train, x_train, nz_samples=10, ny_samples=50, batch_size=1) #[n_train, y_size]
            mu_y_test, var_y_test = model.forward(x_train, y_train, x_test, nz_samples=10, ny_samples=50, batch_size=1)  #[n_test, y_size]
        else:
            raise Exception('Model name should be cnp or vnp.')
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
            y_train = y_scaler.inverse_transform(y_train)
            y_test = y_scaler.inverse_transform(y_test)

        r2_train_list.append(r2_score(y_train, mu_y_train))
        rmse_train_list.append(np.sqrt(mean_squared_error(y_train, mu_y_train)))
        nlpd_train_list.append(nlpd(mu_y_train, var_y_train, y_train))

        r2_test_list.append(r2_score(y_test, mu_y_test))
        rmse_test_list.append(np.sqrt(mean_squared_error(y_test, mu_y_test)))
        nlpd_test_list.append(nlpd(mu_y_test, var_y_test, y_test))


        if (j % (n_functions // 10) == 0) and (x_dim == 1):
            fig_name = dataname + '_f' + str(j) + '_epoch' + str(epoch) + model_name + '.png'

            if x_scaler is not None:
                x_train = x_scaler.inverse_transform(x_train.reshape(-1))
                x_test = x_scaler.inverse_transform(x_test.reshape(-1))

            plotter1d(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                      mu_y=np.concatenate((mu_y_train, mu_y_test), axis=0),
                      var_y=np.concatenate((var_y_train, var_y_test), axis=0),
                      path_to_save=directory+fig_name)

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

