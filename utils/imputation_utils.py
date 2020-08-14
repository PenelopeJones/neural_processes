import copy

import numpy as np
import pandas as pd
import torch
import scipy
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils.metric_utils import mll, confidence_curve
import time
import pdb





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


def conduit_r2_calculator(model, x, n_properties, means, stds, n_cycles):
    mask = torch.isnan(x[:, -n_properties:])
    r2_scores = []
    input = copy.deepcopy(x)
    input[torch.where(torch.isnan(input))] = 0.0
    for p in range(0, n_properties):
        p_idx = torch.where(~mask[:, p])[0]
        if p_idx.shape[0] > 40:
            input_batch = copy.deepcopy(input[p_idx, :])
            input_batch[:, -n_properties + p] = 0.0
            target_batch = input[p_idx, -n_properties + p]

            output_batch = model.forward(input_batch, n_cycles)
            output_batch = output_batch[:, -n_properties + p]

            if (means is not None) and (stds is not None):
                output_batch = (output_batch.data.numpy() * stds[-n_properties + p] +
                                means[-n_properties + p])
                target_batch = (target_batch.data.numpy() * stds[-n_properties + p] +
                                means[-n_properties + p])
                r2_scores.append(r2_score(target_batch, output_batch))

            else:
                r2_scores.append(r2_score(target_batch.data.numpy(), output_batch.data.numpy()))

    return r2_scores
