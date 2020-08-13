import copy
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.gridspec as gridspec

mpl.rc('font',family='Times New Roman')

from collections import OrderedDict

linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])



def mll(mean, variance, target):
    """
    Computes the mean log likelihood assuming Gaussian noise.
    :param mean:
    :param variance:
    :param target:
    :return:
    """
    assert len(mean) == len(variance)
    assert len(mean) == len(target)

    n = len(target)
    mean = np.array(mean).reshape(n)
    variance = np.array(variance).reshape(n)

    ll = - 0.5 * np.log(2 * np.pi * variance) - 0.5 * (mean - target) ** 2 / variance
    return ll.mean()

def r2_confidence_curve(mean, variance, target, filename, figsize=(8, 8),
                     linewidth=3.0, fontsize=24):
    """
    Plot confidence curve.
    :param mean:
    :param variance:
    :param target:
    :param filename:
    :return:
    """
    assert len(mean) == len(variance)
    assert len(mean) == len(target)

    # Actual error
    errors = np.absolute(mean-target)

    R2 = np.zeros(len(target)-5)
    R2_oracle = np.zeros(len(target) - 5)
    conf_percentile = np.linspace(100, 0, len(target)-5)

    mean_conf = copy.deepcopy(mean)
    mean_oracle = copy.deepcopy(mean)
    target_conf = copy.deepcopy(target)
    target_oracle = copy.deepcopy(target)

    for i in range(len(mean) - 5):
        # Order values according to level of uncertainty
        idx = variance.argmax()

        # Remove the least confident prediction
        target_conf = np.delete(target_conf, idx)
        mean_conf = np.delete(mean_conf, idx)
        variance = np.delete(variance, idx)

        # Compute the RMSE using our predictions, using only the X% most confident prediction.
        # The RMSE should go down as X decreases
        R2[i] = r2_score(target_conf, mean_conf)

        # Compute the curve for the oracle, which correctly orders the predictions in order
        # of confidence.
        idx = errors.argmax()

        # Remove least confident prediction
        target_oracle = np.delete(target_oracle, idx)
        mean_oracle = np.delete(mean_oracle, idx)
        errors = np.delete(errors, idx)

        # Compute the RMSE using our predictions, using only the X% most confident prediction.
        # The RMSE should go down as X decreases.
        R2_oracle[i] = r2_score(target_oracle, mean_oracle)

    fig, ax = plt.subplots(figsize=figsize)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(linewidth)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)

    ax.plot(conf_percentile, R2_oracle, color="C0", linestyle=linestyles['densely dashed'],
            linewidth=linewidth, label = "Oracle")
    ax.plot(conf_percentile, R2, color="C1", linestyle=linestyles['densely dotted'],
            linewidth=linewidth, label="Model")

    ax.set_xlabel("Percentage missing data imputed", fontsize=fontsize)
    ax.set_ylabel("R2 score", fontsize=fontsize)

    ax.legend(loc='lower left', fontsize=fontsize)
    plt.savefig(filename, frameon=False, dpi=400)



def rmse_confidence_curve(mean, variance, target, filename, figsize=(8, 8),
                     linewidth=3.0, fontsize=24):
    """
    Plot confidence curve.
    :param mean:
    :param variance:
    :param target:
    :param filename:
    :return:
    """
    assert len(mean) == len(variance)
    assert len(mean) == len(target)

    # Actual error
    errors = np.absolute(mean-target)

    RMSE = np.zeros(len(target)-5)
    RMSE_oracle = np.zeros(len(target) - 5)
    conf_percentile = np.linspace(100, 0, len(target)-5)

    mean_conf = copy.deepcopy(mean)
    mean_oracle = copy.deepcopy(mean)
    target_conf = copy.deepcopy(target)
    target_oracle = copy.deepcopy(target)

    for i in range(len(mean) - 5):
        # Order values according to level of uncertainty
        idx = variance.argmax()

        # Remove the least confident prediction
        target_conf = np.delete(target_conf, idx)
        mean_conf = np.delete(mean_conf, idx)
        variance = np.delete(variance, idx)

        # Compute the RMSE using our predictions, using only the X% most confident prediction.
        # The RMSE should go down as X decreases.
        RMSE[i] = np.sqrt(mean_squared_error(target_conf, mean_conf))

        # Compute the curve for the oracle, which correctly orders the predictions in order
        # of confidence.
        idx = errors.argmax()

        # Remove least confident prediction
        target_oracle = np.delete(target_oracle, idx)
        mean_oracle = np.delete(mean_oracle, idx)
        errors = np.delete(errors, idx)

        # Compute the RMSE using our predictions, using only the X% most confident prediction.
        # The RMSE should go down as X decreases.
        RMSE_oracle[i] = np.sqrt(mean_squared_error(target_oracle, mean_oracle))

    fig, ax = plt.subplots(figsize=figsize)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(linewidth)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)

    ax.plot(conf_percentile, RMSE_oracle, color="C0", linestyle=linestyles['densely dashed'],
            linewidth=linewidth, label = "Oracle")
    ax.plot(conf_percentile, RMSE, color="C1", linestyle=linestyles['densely dotted'],
            linewidth=linewidth, label="Model")

    ax.set_xlabel("Percentage missing data imputed", fontsize=fontsize)
    ax.set_ylabel("RMSE", fontsize=fontsize)

    ax.legend(loc='lower left', fontsize=fontsize)
    plt.savefig(filename, frameon=False, dpi=400)

    #ax.set_xticks(xticks)
    #ax.set_xticklabels(xticks, fontsize=fontsize)








