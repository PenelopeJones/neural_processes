import sys

sys.path.append('../../')
import warnings
import argparse
import pdb
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

# from models.imputation_models.conduit_model import ConduitModel
from models.imputation_models.conduit_models import SetofConduitModels
from models.imputation_models.np_basic import NPBasic
from models.imputation_models.cnp_basic import CNPBasic
from models.imputation_models.np_film import NPFiLM
from utils.data_utils import nan_transform_data, select_descriptors, parse_boolean
from utils.metric_utils import metric_ordering, confidence_curve

from collections import OrderedDict

linestyles = OrderedDict(
    [('solid', (0, ())),
     ('loosely dotted', (0, (1, 10))),
     ('dotted', (0, (1, 5))),
     ('densely dotted', (0, (1, 1))),

     ('loosely dashed', (0, (5, 10))),
     ('dashed', (0, (5, 5))),
     ('densely dashed', (0, (5, 1))),

     ('loosely dashdotted', (0, (3, 10, 1, 10))),
     ('dashdotted', (0, (3, 5, 1, 5))),
     ('densely dashdotted', (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def main():
    """
    :return:
    """

    warnings.filterwarnings('ignore')
    torch.set_default_dtype(torch.float64)

    # run_number = 9
    # epochs = 250
    run_number = 10
    epochs = 50

    figsize = (8, 8)
    linewidth = 3.0
    fontsize = 24

    epoch_mapping = {'npbasic': 2000, 'cnpbasic': 2000, 'setofconduits': 50}
    label_mapping = {'npbasic': 'NP', 'cnpbasic': 'CNP', 'setofconduits': 'Conduit'}

    dataname = 'Kinase'
    n_properties = 159
    model_names = ['cnpbasic']
    colors = ['C0', 'C1', 'C2']
    markers = ['o', 's', 'v']
    pts = 'results/{}/summary/r2scores.png'.format(dataname)

    arrays = np.linspace(1, n_properties, n_properties)

    r2means = []
    r2stds = []

    indices_r2 = None
    for model_name in model_names:
        if model_name == 'npbasic':
            ptf = 'results/{}/{}/not_restrict_var/{}1_{}_'.format(dataname, model_name, dataname, model_name)
        else:
            ptf = 'results/{}/{}/{}1_{}_'.format(dataname, model_name, dataname, model_name)

        r2_scores = []
        mlls = []

        for j in range(run_number):
            if model_name == 'npbasic':
                filestart = ptf + '{}_{}'.format(j, epoch_mapping[model_name])
            elif model_name == 'cnpbasic':
                filestart = ptf + '{}_{}'.format(j, epoch_mapping[model_name])
            elif model_name == 'setofconduits':
                filestart = ptf + '{}_{}_'.format(j, epoch_mapping[model_name])
            r2_score = np.load(filestart + 'r2_scores.npy')
            mll = np.load(filestart + 'mll_scores.npy')
            r2_scores.append(r2_score)
            mlls.append(mll)

        r2_scores = np.array(r2_scores)
        r2_score_mn = np.mean(r2_scores, axis=0)
        r2_score_std = np.std(r2_scores, axis=0)
        mlls = np.array(mlls)
        mll_mn = np.mean(mlls, axis=0)
        mll_std = np.std(mlls, axis=0)

        #if indices_r2 is None:
        indices_r2 = np.argsort(r2_score_mn)[::-1]
        indices_mll = np.argsort(mll_mn)[::-1]
        r2means.append(r2_score_mn[indices_r2])
        r2stds.append(r2_score_std[indices_r2])


    fig, ax = plt.subplots(figsize=figsize)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(linewidth)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)

    for mn, std, color, marker, model_name in zip(r2means, r2stds, colors, markers, model_names):
        ax.errorbar(arrays, mn, yerr=std, capsize=2.0, color=color,
                    marker=marker, linestyle='', markersize=4,
                    elinewidth=1.5, label = label_mapping[model_name])

    ax.legend(fontsize=fontsize)

    ax.set_xlabel("Assay number", fontsize=fontsize)
    ax.set_ylabel("$R^2$ score", fontsize=fontsize)

    if dataname == 'Adrenergic':
        ymin = 0.4
        ymax = 0.8
        xmin = 0.8
        xmax = 5.2
        yticks = [0.4, 0.5, 0.6, 0.7, 0.8]
        yticklabels = yticks
        xticks = [1, 2, 3, 4, 5]
        xticklabels = xticks

    if dataname == 'Kinase':
        ymin = -1
        ymax = 1
        xmin = 0.0
        xmax = 160
        yticks = [-0.8, -0.4, 0, 0.4, 0.8]
        yticklabels = yticks
        xticks = [0, 40, 80, 120, 160]
        xticklabels = xticks
        ax.plot(np.linspace(xmin, xmax, 10), 0.3*np.ones(10),
                linestyle=linestyles['densely dotted'], color='grey',
                linewidth=0.5*linewidth)
        ax.plot(np.linspace(xmin, xmax, 10), 0.0 * np.ones(10),
                linestyle=linestyles['densely dotted'], color='red',
                linewidth=0.5 * linewidth)

    ax.set_ylim(ymin, ymax)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=fontsize)

    ax.set_xlim(xmin, xmax)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=fontsize)

    plt.tight_layout()

    plt.savefig(pts, frameon=False, dpi=400)


if __name__ == '__main__':
    main()
