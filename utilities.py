import numpy as np
from scipy.stats import norm
import math
import scipy.linalg as la

def analytical_normal_measures(alpha, weights, delta, portfolio_value, returns):
    x = np.log(returns)  # log return vector
    mu = x.mean()
    sigma = x.cov()
    norm_inv = norm.ppf(alpha)
    mu_port = np.dot(weights, mu)
    sigma_port = np.dot(weights, np.dot(sigma, weights))

    # Compute the Var and the ES
    var = (-delta * mu_port + np.sqrt(delta) * norm_inv * np.sqrt(sigma_port)) * portfolio_value
    es_std = 1 / ((1 - alpha) * np.sqrt(2 * np.pi)) * np.exp(-(norm_inv ** 2) / 2)
    es = (-delta * mu_port + np.sqrt(delta) * es_std * np.sqrt(sigma_port)) * portfolio_value
    return var, es


def hs_measurements(returns, alpha, weights):
    x = np.log(returns)  # log return vector
    loses_values = -np.multiply(x, weights)
    loses_values = loses_values.sum(axis=1)
    sorted_loses_values = loses_values.sort_values(ascending=False)

    # Compute the Var and the ES
    n = len(sorted_loses_values)
    position = math.floor(n * (1 - alpha))  # position is the largest integer not exceeding n*(1-alpha)
    loses_for_es = sorted_loses_values[position:]

    var = sorted_loses_values[position]
    es = loses_for_es.mean()
    return var, es


def whs_measurements(returns, alpha, weights, lambda_portfolio):
    x = np.log(returns)  # log return vector
    loses_values = -np.multiply(x, weights)
    loses_values = loses_values.sum(axis=1)

    n2 = len(loses_values)
    c = (1 - lambda_portfolio) / (1 - lambda_portfolio ** n2)
    loses_weights = c * (np.power(lambda_portfolio, list(range(0, n2))))
    loses_values = np.multiply(loses_weights, loses_values)
    sorted_loses_values = loses_values.sort_values(ascending=False)

    # Compute the Var and the ES
    n = len(sorted_loses_values)
    position = math.floor(n * (1 - alpha))  # position is the largest integer not exceeding n*(1-alpha)
    loses_for_es = sorted_loses_values[position:]

    var = sorted_loses_values[position]
    es = loses_for_es.mean()
    return var, es

'''
def princ_comp_analysis(yearly_covariance,):
    yearly_covariance = yearly_covariance.T
    yearly_covariance_mean = yearly_covariance.mean(axis=1)
    yearly_covariance_norm = yearly_covariance - yearly_covariance_mean[:, None]

 '''