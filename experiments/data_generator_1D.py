"""
Script for generating 1-dimensional train and test toy regression data from a noisy function.
"""
import sys
sys.path.insert(1, '/Users/penelopejones/PycharmProjects/~ml_physics/neural_processes/')

import numpy as np
import matplotlib.pyplot as plt

from gpflow.kernels import SquaredExponential

from utils.gp_sampler import GPDataGenerator

import pdb

def main():

    dataname = "1DGP_2SE_"
    ptr = "data/toy_data/"
    n_functions = 1000
    lengthscales = 1.0
    data_generator = GPDataGenerator(kernel=SquaredExponential(lengthscales=lengthscales))

    x_trains = []
    y_trains = []
    x_tests = []
    y_tests = []

    #Generate n_functions sets of values of x in the range [min_x, max_x], to be used for training and testing
    plt.figure()
    for i in range(n_functions):
        n_train = np.random.randint(low = 25, high = 100)
        n_test = int(0.2*n_train)
        print(n_train)
        print(n_test)

        x_train, y_train, x_test, y_test = data_generator.sample(train_size=n_train, test_size=n_test, x_min=-3, x_max=3)

        x_trains.append(x_train)
        y_trains.append(y_train)
        x_tests.append(x_test)
        y_tests.append(y_test)

        if i ==0:
            plt.scatter(x_train, y_train, c='r', s=1, label="train")
            plt.scatter(x_test, y_test, c="magenta", s=1, label="test")
        elif i == 1:
            plt.scatter(x_train, y_train, c='black', s=1, label="train")
            plt.scatter(x_test, y_test, c="yellow", s=1, label="test")
        elif i == 2:
            plt.scatter(x_train, y_train, c='b', s=1, label="train")
            plt.scatter(x_test, y_test, c="g", s=1, label="test")
    plt.legend()
    plt.xlabel("x")
    plt.xticks([])
    plt.ylabel('f(x)')
    plt.yticks([])
    plt.show()

    x_trains = np.array(x_trains)
    y_trains = np.array(y_trains)
    x_tests = np.array(x_tests)
    y_tests = np.array(y_tests)

    np.save(ptr + dataname + "X_trains.npy", x_trains)
    np.save(ptr + dataname + "y_trains.npy", y_trains)
    np.save(ptr + dataname + "X_tests.npy", x_tests)
    np.save(ptr + dataname + "y_tests.npy", y_tests)

if __name__ == '__main__':
    main()