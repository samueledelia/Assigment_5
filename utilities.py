import numpy as np
import pandas as pd
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


def calculate_portfolio_metrics(n_shares, portfolio_df, delta):
    n = len(portfolio_df) - 1
    stocks_at_t = portfolio_df.iloc[[n]]
    value_portfolio_at_t = np.dot(stocks_at_t, n_shares).copy()
    weights = (stocks_at_t * n_shares) / value_portfolio_at_t[0]
    shares = portfolio_df.copy()  # Consider only the shares of the last delta periods
    returns = shares / shares.shift(delta)  # Compute the return for each company's shares
    returns = returns[1:]                     # Drop the last row
    return weights, returns, value_portfolio_at_t


def calculate_portfolio_metrics_whs( n, portfolio_df, delta):
    weights = np.ones((1, n)) / n  # equally weighted portfolio
    shares = portfolio_df.copy()
    returns = shares / shares.shift(delta)  # Compute the return for each company shares
    returns = returns[1:]                     # Drop the last row
    return weights, returns


def hs_measurements(returns, alpha, weights, value_portfolio_at_t):
    x = np.log(returns)  # log return vector
    loses_values = -np.multiply(x, np.array(weights))
    loses_values = np.multiply(loses_values.sum(axis=1), value_portfolio_at_t[0])
    sorted_loses_values = loses_values.sort_values(ascending=False)

    # Compute the Var and the ES
    n = len(sorted_loses_values)
    position = math.floor(n * (1 - alpha))  # position is the largest integer not exceeding n*(1-alpha)
    loses_for_es = sorted_loses_values[:position]

    var = sorted_loses_values[position]
    es = loses_for_es.mean()
    return var, es


def bootstrap_statistical(portfolio_size, numberOfSamplesToBootstrap,):
    index = [np.random.randint(0, portfolio_size[0] - 1) for x in range(numberOfSamplesToBootstrap)]
    return index


def whs_measurements(returns, alpha, weights, lambda_portfolio, value_portfolio):
    x = np.log(returns)  # log return vector
    losses_values = -np.multiply(x, weights)
    losses_values = np.multiply(losses_values.sum(axis=1), value_portfolio)
    n2 = len(losses_values)
    c = (1 - lambda_portfolio) / (1 - lambda_portfolio ** n2)
    losses_weights = c * (np.power(lambda_portfolio, np.linspace(n2, 0, n2)))
    losses_values = losses_values.to_frame(name="losses")
    loses_df = pd.concat([losses_values, pd.DataFrame({"weight": losses_weights}, index=losses_values.index)], axis=1)  # create a df containing loses_values and losses_weights
    sorted_losses_df = loses_df.sort_values(by=loses_df.columns[0], ascending=False).copy()

    # Compute the Var and the ES
    n = len(sorted_losses_df)
    i = 0
    sum_weight = 0
    while True:
        sum_weight += sorted_losses_df.weight[i]
        if sum_weight > 1-alpha:
            break
        i += 1

    position = i-1
    var = sorted_losses_df.losses[position]
    es = np.dot(sorted_losses_df.weight[:position], sorted_losses_df.losses[:position]) / np.sum(sorted_losses_df.weight[:position])
    return var, es


def princ_comp_analysis(yearly_covariance, ):
    yearly_covariance = yearly_covariance.T
    yearly_covariance_mean = yearly_covariance.mean(axis=1)
    yearly_covariance_norm = yearly_covariance - yearly_covariance_mean[:, None]
    U, s, VT = np.linalg.svd(yearly_covariance_norm, full_matrices=True)
    S = la.diagsvd(s, yearly_covariance.shape[0], yearly_covariance.shape[1])
    Phi = np.matmul(U.transpose(), yearly_covariance_norm)


def plausibility_check(returns, weights, alpha, delta, value_portfolio):
    x = np.log(returns)
    mu = x.mean()
    mu_port = np.dot(weights, mu)
    n = len(x)
    n_shares = len(weights)
    ordered_x = np.zeros((n, n_shares))
    for i in np.arange(n_shares):
        ordered_x[:, i] = x.iloc[:, i].sort_values().copy()
    position_l = math.floor(n * (1 - alpha))
    position_u = math.floor(n * alpha)
    l = ordered_x[position_l, :]
    u = ordered_x[position_u, :]
    s_var = np.array(weights) * (np.abs(l) + np.abs(u)) / 2
    correlation_matrix = x.corr()

    var_approx = np.sqrt(np.dot(s_var, np.dot(correlation_matrix, s_var.T)))
    var_check = (-delta * mu_port[0] + np.sqrt(delta) * var_approx) * value_portfolio
    return var_check
