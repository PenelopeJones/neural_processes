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
from scipy.stats import pearsonr

# from models.imputation_models.conduit_model import ConduitModel
from models.imputation_models.conduit_models import SetofConduitModels
from models.imputation_models.np_basic import NPBasic
from models.imputation_models.cnp_basic import CNPBasic
from models.imputation_models.np_film import NPFiLM
from utils.data_utils import nan_transform_data, select_descriptors, parse_boolean
from utils.metric_utils import metric_ordering, confidence_curve


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
    epochs = 250

    dataname = 'Adrenergic'
    n_properties = 5
    model_name = 'setofconduits'
    metric = 'rmse'

    if model_name == 'npbasic':
        ptf = 'results/{}/{}/not_restrict_var/{}1_{}_'.format(dataname, model_name, dataname, model_name)
    else:
        ptf = 'results/{}/{}/{}1_{}_'.format(dataname, model_name, dataname, model_name)
    pts = ptf + metric + '_confidence_curve.png'

    percentiles = np.arange(100, 4, -5)

    metric_model_mns = []
    metric_oracle_mns = []

    for j in range(run_number):
        metric_models = []
        metric_oracles = []
        for i in range(n_properties):
            if model_name == 'npbasic':
                filestart = ptf + '{}_{}_250_'.format(j, i)
            elif model_name == 'cnpbasic':
                filestart = ptf + '{}{}_'.format(j, i)
            elif model_name == 'setofconduits':
                filestart = ptf + '{}_{}_'.format(j, i)
            mean = np.load(filestart + 'mean.npy')
            std = np.load(filestart + 'std.npy')
            target = np.load(filestart + 'target.npy')

            conf_percentile, metric_model, metric_oracle = metric_ordering(mean, std**2, target, metric)
            indices = []
            for percentile in percentiles:
                indices.append(find_nearest(conf_percentile, percentile))
            indices = np.array(indices)

            metric_models.append(metric_model[indices])
            metric_oracles.append(metric_oracle[indices])

        metric_models = np.array(metric_models)
        metric_oracles = np.array(metric_oracles)

        metric_model = np.mean(metric_models, axis=0)
        metric_oracle = np.mean(metric_oracles, axis=0)
        metric_model_mns.append(metric_model)
        metric_oracle_mns.append(metric_oracle)

    metric_model_mns = np.array(metric_model_mns)
    metric_model_mn = np.mean(metric_model_mns, axis=0)
    metric_model_std = np.std(metric_model_mns, axis=0)
    metric_oracle_mns = np.array(metric_oracle_mns)
    metric_oracle_mn = np.mean(metric_oracle_mns, axis=0)
    metric_oracle_std = np.std(metric_oracle_mns, axis=0)

    confidence_curve(percentiles, metric_model_mn, metric_oracle_mn, pts,
                     metric_model_std, metric_oracle_std, metric)


if __name__ == '__main__':
    main()
