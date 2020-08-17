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
from sklearn.metrics import r2_score, mean_squared_error
from utils.metric_utils import mll, metric_ordering, confidence_curve, find_nearest
from scipy.stats import pearsonr


#from models.imputation_models.conduit_model import ConduitModel
from models.imputation_models.conduit_models import SetofConduitModels
from models.imputation_models.np_basic import NPBasic
from models.imputation_models.cnp_basic import CNPBasic
from models.imputation_models.np_film import NPFiLM
from utils.data_utils import nan_transform_data, select_descriptors, parse_boolean
from utils.metric_utils import baseline_metrics_calculator


def main(args):
    """
    :return:
    """

    warnings.filterwarnings('ignore')
    torch.set_default_dtype(torch.float64)

    extra = ''

    filename = args.dataname + '_' + args.model_name + '_' + extra
    #run_number = 9
    #epochs = 250
    run_number = 10
    epochs = 250
    n_properties = 5

    if args.model_name == 'npbasic':
        extra_dir = 'not_restrict_var/'
    else:
        extra_dir = ''

    with open('results/{}/summary/{}_ensemble.txt'.format(args.dataname, filename), 'w+') as f:
        r2_scores_list = []
        mlls_list = []
        rmses_list = []
        metric = 'rmse'
        percentiles = np.arange(100, 4, -5)
        metric_models = []
        metric_oracles = []
        fig_pts = 'results/{}/summary/{}_{}_{}_ensemble_confidence_curve.png'.format(args.dataname, args.dataname, args.model_name, metric)
        for p in range(n_properties):
            mns = []
            stds = []
            targets = []
            for i in range(run_number):
                if args.model_name == 'npbasic':
                    filestart = '{}{}_{}_{}_{}_250_'.format(args.dataname, args.num, args.model_name, i, p)
                elif args.model_name == 'cnpbasic':
                    filestart = '{}{}_{}_{}{}_'.format(args.dataname, args.num, args.model_name, i, p)
                else:
                    filestart = '{}{}_{}_{}_{}_'.format(args.dataname, args.num, args.model_name, i, p)
                mn = np.load('results/{}/{}/{}{}mean.npy'.format(args.dataname, args.model_name, extra_dir, filestart))
                std = np.load('results/{}/{}/{}{}std.npy'.format(args.dataname, args.model_name, extra_dir, filestart))
                target = np.load('results/{}/{}/{}{}target.npy'.format(args.dataname, args.model_name, extra_dir, filestart))
                mns.append(mn)
                stds.append(std)
                targets.append(target)

            mean = np.mean(np.array(mns), axis=0)
            var = np.mean(np.array(stds)**2, axis=0)
            target = np.mean(np.array(targets), axis=0)

            conf_percentile, metric_model, metric_oracle = metric_ordering(mean, var, target, metric)
            indices = []
            for percentile in percentiles:
                indices.append(find_nearest(conf_percentile, percentile))
            indices = np.array(indices)

            metric_models.append(metric_model[indices])
            metric_oracles.append(metric_oracle[indices])
            r2_scores_list.append(r2_score(target, mean))
            mlls_list.append(mll(mean, var, target))
            rmses_list.append(np.sqrt(mean_squared_error(target, mean)))

        r2_scores_list = np.array(r2_scores_list)
        mlls_list = np.array(mlls_list)
        rmses_list = np.array(rmses_list)
        f.write('\n R^2 score: {:.4f}+- {:.4f}'.format(np.mean(r2_scores_list), np.std(r2_scores_list)))
        f.write('\n MLL: {:.4f}+- {:.4f} \n'.format(np.mean(mlls_list), np.std(mlls_list)))
        f.write('\n RMSE: {:.4f}+- {:.4f} \n'.format(np.mean(rmses_list), np.std(rmses_list)))
        f.flush()
        metric_models = np.array(metric_models)
        metric_oracles = np.array(metric_oracles)

        metric_model = np.mean(metric_models, axis=0)
        metric_oracle = np.mean(metric_oracles, axis=0)

        confidence_curve(percentiles, metric_model, metric_oracle, fig_pts, metric)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='data/raw_data/',
                        help='Directory where the training and test data is stored.')
    parser.add_argument('--dataname', default='Adrenergic',
                        help='Name of dataset.')
    parser.add_argument('--num', type=int, default=1,
                        help='The train/test split number. 1 '
                             'for Kinase, between 1 and 5 for '
                             'Adrenergic.')
    parser.add_argument('--n_properties', type=int, default=5,
                        help='The number of properties.')
    parser.add_argument('--model_name', default='setofconduits',
                        help='Model to use.')
    args = parser.parse_args()

    main(args)