import copy

import numpy as np
import pandas as pd
import torch
import scipy
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time
import pdb


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


def npfilm_metrics_calculator(model, x, n_properties, num_samples=1,
                              means=None, stds=None):
    mask = torch.isnan(x[:, -n_properties:])
    r2_scores = []
    nlpds = []
    for p in range(0, n_properties, 1):
        p_idx = torch.where(~mask[:, p])[0]
        if p_idx.shape[0] > 40:
            x_p = x[p_idx]
            target = x_p[:, (-n_properties + p)]

            mask_context = copy.deepcopy(mask[p_idx, :])
            mask_context[:, p] = True
            mask_p = torch.zeros_like(mask_context).fill_(True)
            mask_p[:, p] = False

            # [test_size, n_properties, z_dim]
            mu_priors, sigma_priors = model.encoder(x_p[:, :-n_properties],
                                                    x_p[:, -n_properties:],
                                                    mask_context)

            samples = []
            for i in range(num_samples):
                z = mu_priors + sigma_priors * torch.randn_like(mu_priors)
                recon_mu, recon_sigma = model.decoder(z, mask_p)
                recon_mu = recon_mu.detach()
                recon_sigma = recon_sigma.detach()
                recon_mu = recon_mu[:, p]
                recon_sigma = recon_sigma[:, p]
                sample = recon_mu + recon_sigma * torch.randn_like(recon_mu)
                samples.append(sample.transpose(0, 1))

            samples = torch.cat(samples)
            predict_mean = torch.mean(samples, dim=0)
            predict_std = torch.std(samples, dim=0)

            if (means is not None) and (stds is not None):
                predict_mean = (predict_mean.numpy() * stds[-n_properties + p] +
                                means[-n_properties + p])
                predict_std = predict_std.numpy() * stds[-n_properties + p]
                target = (target.numpy() * stds[-n_properties + p] +
                          means[-n_properties + p])
                r2_scores.append(r2_score(target, predict_mean))
                nlpds.append(nlpd(predict_mean, predict_std ** 2, target))
            else:
                r2_scores.append(r2_score(target.numpy(), predict_mean.numpy()))
                nlpds.append(nlpd(predict_mean, predict_std ** 2, target))

    return r2_scores, nlpds


def npbasic_metrics_calculator(model, x, n_properties, num_samples=1,
                               means=None, stds=None):
    mask = torch.isnan(x[:, -n_properties:])
    r2_scores = []
    nlpds = []
    for p in range(0, n_properties, 1):
        p_idx = torch.where(~mask[:, p])[0]
        if p_idx.shape[0] > 40:
            x_p = x[p_idx]

            input_p = copy.deepcopy(x_p)
            input_p[:, -n_properties:][mask[p_idx]] = 0.0
            input_p[:, (-n_properties + p)] = 0.0
            mask_p = torch.zeros_like(mask[p_idx, :]).fill_(True)
            mask_p[:, p] = False
            mu_priors, var_priors = model.encoder(input_p)

            samples = []
            for i in range(num_samples):
                z = mu_priors + var_priors ** 0.5 * torch.randn_like(mu_priors)
                recon_mus, recon_vars = model.decoder.forward(z, mask_p)
                recon_mu = recon_mus[p].detach().reshape(-1)
                recon_sigma = (recon_vars[p] ** 0.5).detach().reshape(-1)
                sample = recon_mu + recon_sigma * torch.randn_like(recon_mu)
                samples.append(sample)
            samples = torch.stack(samples)
            predict_mean = torch.mean(samples, dim=0)
            predict_std = torch.std(samples, dim=0)
            target = x_p[:, (-n_properties + p)]

            if (means is not None) and (stds is not None):
                predict_mean = (predict_mean.numpy() * stds[-n_properties + p] +
                                means[-n_properties + p])
                predict_std = predict_std.numpy() * stds[-n_properties + p]
                target = (target.numpy() * stds[-n_properties + p] +
                          means[-n_properties + p])
                r2_scores.append(r2_score(target, predict_mean))
                nlpds.append(nlpd(predict_mean, predict_std ** 2, target))
            else:
                r2_scores.append(r2_score(target.numpy(), predict_mean.numpy()))
                nlpds.append(nlpd(predict_mean, predict_std ** 2, target))

    return r2_scores, nlpds
