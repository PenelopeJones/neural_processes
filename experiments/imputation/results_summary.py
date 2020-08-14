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

    extra = 'restrict_var'

    filename = args.dataname + '_' + args.model_name + '_' + extra
    #run_number = 9
    #epochs = 250
    run_number = 10
    epochs = 250

    extra_dir = extra + '/'

    with open('results/{}/summary/{}.txt'.format(args.dataname, filename), 'w+') as f:
        r2_scores_list = []
        mlls_list = []
        for i in range(run_number):
            filestart = '{}{}_{}_{}_{}'.format(args.dataname, args.num, args.model_name, i, epochs)
            r2_scores = np.load('results/{}/{}/{}{}r2_scores.npy'.format(args.dataname, args.model_name, extra_dir, filestart))
            mll_scores = np.load('results/{}/{}/{}{}mll_scores.npy'.format(args.dataname, args.model_name, extra_dir, filestart))
            f.write(str(r2_scores) + '\n')
            f.write(str(mll_scores) + '\n')
            r2_scores_list.append(np.mean(r2_scores))
            mlls_list.append(np.mean(mll_scores))

        r2_scores_list = np.array(r2_scores_list)
        mlls_list = np.array(mlls_list)
        f.write('\n R^2 score: {:.4f}+- {:.4f}'.format(np.mean(r2_scores_list), np.std(r2_scores_list)))
        f.write('\n MLL: {:.4f}+- {:.4f} \n'.format(np.mean(mlls_list), np.std(mlls_list)))




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