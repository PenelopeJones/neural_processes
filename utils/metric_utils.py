import copy
import numpy as np
import torch
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


def confidence_curve(mean, variance, target, filename, metric='rmse', figsize=(8, 8),
                     linewidth=3.0, fontsize=24, means=None, stds=None):
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

    if metric == 'rmse':
        n_min = 5
    elif metric == 'r2':
        n_min = 20

    # Actual error
    errors = np.absolute(mean-target)

    metric_model = np.zeros(len(target)-n_min)
    metric_oracle = np.zeros(len(target) - n_min)
    conf_percentile = np.linspace(100, 100*n_min/(len(target)), len(target)-n_min)

    mean_model = copy.deepcopy(mean)
    mean_oracle = copy.deepcopy(mean)
    target_model = copy.deepcopy(target)
    target_oracle = copy.deepcopy(target)

    for i in range(len(mean) - n_min):
        # Order values according to level of uncertainty
        idx_model = variance.argmax()
        idx_oracle = errors.argmax()

        # Compute the metric using our predictions, using only the X% most confident prediction.
        # The metric should systematically go down (RMSE) or up (R2) as X decreases.
        if metric == 'rmse':
            metric_model[i] = np.sqrt(mean_squared_error(target_model, mean_model))
            metric_oracle[i] = np.sqrt(mean_squared_error(target_oracle, mean_oracle))
        elif metric == 'r2':
            metric_model[i] = r2_score(target_model, mean_model)
            metric_oracle[i] = r2_score(target_oracle, mean_oracle)
        else:
            raise Exception('Metric should be rmse or r2.')

        # Remove least confident prediction of model
        target_model = np.delete(target_model, idx_model)
        mean_model = np.delete(mean_model, idx_model)
        variance = np.delete(variance, idx_model)

        # Remove least confident prediction of oracle
        target_oracle = np.delete(target_oracle, idx_oracle)
        mean_oracle = np.delete(mean_oracle, idx_oracle)
        errors = np.delete(errors, idx_oracle)

    fig, ax = plt.subplots(figsize=figsize)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(linewidth)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)

    ax.plot(conf_percentile, metric_oracle, color="C0", linestyle=linestyles['densely dashed'],
            linewidth=linewidth, label = "Oracle")
    ax.plot(conf_percentile, metric_model, color="C1", linestyle=linestyles['densely dotted'],
            linewidth=linewidth, label="Model")

    ymin = min(np.min(metric_oracle), np.min(metric_model))
    ymax = max(np.max(metric_oracle), np.max(metric_model))

    ax.set_ylim(ymin, ymax)

    yticks = np.arange(np.round(ymin, decimals=1), np.round(ymax+0.2, decimals=1), step=0.2)
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.round(yticks, decimals=1), fontsize=fontsize)

    xticks = np.linspace(0, 100, 6)
    ax.set_xlim(0, 100)
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.round(xticks, decimals=0), fontsize=fontsize)
    ax.legend(fontsize=fontsize)

    ax.set_xlabel("Percentage missing data imputed", fontsize=fontsize)
    if metric == 'rmse':
        ax.set_ylabel("RMSE", fontsize=fontsize)
    elif metric == 'r2':
        ax.set_ylabel("R2 score", fontsize=fontsize)

    plt.savefig(filename, frameon=False, dpi=400)


def baseline_metrics_calculator(x, n_properties, means=None, stds=None):
    mask = torch.isnan(x[:, -n_properties:])
    r2_scores = []
    mlls = []
    for p in range(0, n_properties, 1):
        p_idx = torch.where(~mask[:, p])[0]
        predict_mean = torch.zeros(len(p_idx))
        predict_std = torch.ones(len(p_idx))
        target = x[p_idx][:, (-n_properties + p)]

        if (means is not None) and (stds is not None):
            predict_mean = (predict_mean.numpy() * stds[-n_properties + p] +
                            means[-n_properties + p])
            predict_std = predict_std.numpy() * stds[-n_properties + p]
            target = (target.numpy() * stds[-n_properties + p] +
                      means[-n_properties + p])
            r2_scores.append(r2_score(target, predict_mean))
            mlls.append(mll(predict_mean, predict_std ** 2, target))
        else:
            r2_scores.append(r2_score(target.numpy(), predict_mean.numpy()))
            mlls.append(mll(predict_mean, predict_std ** 2, target))

    return r2_scores, mlls

