from sklearn.linear_model import LinearRegression
import numpy as np


def filter_by_charge(mz, charge, ccs, state):
    """
    extract data by a specific charge state
    :param mz: list of mz values
    :param charge: list of charge states
    :param ccs: list of ccs values
    :param state: charge state to filter by
    :return: a filtered list only containing items where charge == state
    """
    candidates = zip(charge, zip(mz, ccs))
    states = list(filter(lambda x: x[0] == state, candidates))
    return np.array([x[1][0] for x in states]), np.array([x[1][1] for x in states])


def get_slopes_and_intercepts(mz, charge, ccs):
    """
    will calculate a set of linear regression lines for charges 1, 2, 3, 4
    :param mz: array of masses
    :param charge: array of charges
    :param ccs: array of ccs values
    :return: two arrays, first is slopes per charge state, second is intercepts per charge state
    """

    fit_samples = [filter_by_charge(mz, charge, ccs, c) for c in range(1, 4)]

    if fit_samples[0][0].shape[0] == 0:
        slopes, intercepts = [0.0], [0.0]
        start_with = 1

    else:
        slopes, intercepts = [], []
        start_with = 0

    for i in range(start_with, len(fit_samples)):
        x, y = fit_samples[i][0].reshape(-1, 1), fit_samples[i][1].reshape(-1, 1)
        reg = LinearRegression().fit(x, y)
        slope = reg.coef_[0][0]
        intercept = reg.intercept_[0]

        slopes.append(slope)
        intercepts.append(intercept)

    return np.array(slopes, np.float32), np.array(intercepts, np.float32)
