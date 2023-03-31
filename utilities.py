import numpy as np
import pandas as pd
from scipy.stats import norm
import math


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
    returns = returns[1:]  # Drop the last row
    return weights, returns, value_portfolio_at_t


def calculate_portfolio_metrics_whs(n, portfolio_df, delta):
    weights = np.ones((1, n)) / n  # equally weighted portfolio
    shares = portfolio_df.copy()
    returns = shares / shares.shift(delta)  # Compute the return for each company shares
    returns = returns[1:]  # Drop the last row
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


def bootstrap_statistical(portfolio_size, numberOfSamplesToBootstrap, ):
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
    loses_df = pd.concat([losses_values, pd.DataFrame({"weight": losses_weights}, index=losses_values.index)],
                         axis=1)  # create a df containing loses_values and losses_weights
    sorted_losses_df = loses_df.sort_values(by=loses_df.columns[0], ascending=False).copy()

    # Compute the Var and the ES
    n = len(sorted_losses_df)
    i = 0
    sum_weight = 0
    while True:
        sum_weight += sorted_losses_df.weight[i]
        if sum_weight > 1 - alpha:
            break
        i += 1

    position = i - 1
    var = sorted_losses_df.losses[position]
    es = np.dot(sorted_losses_df.weight[:position], sorted_losses_df.losses[:position]) / np.sum(
        sorted_losses_df.weight[:position])
    return var, es


def PCA_VaR_ES(nu_projected, weights_projected, eigenvalues, k, delta, alpha_std, Notional):
    nu_red = np.dot(weights_projected[0:k, 0], nu_projected[0:k, 0])
    sigma_red = np.dot(weights_projected[0:k, 0] ** 2, eigenvalues[0:k, 0])

    VaR_alpha_std = norm.ppf(alpha_std)
    VaR = Notional * (-delta * nu_red + np.sqrt(delta) * np.sqrt(sigma_red) * VaR_alpha_std)

    ES_alpha_std = 1 / (1 - alpha_std) * norm.pdf(norm.ppf(alpha_std))
    ES = nu_red * delta + np.sqrt(sigma_red) * np.sqrt(delta) * ES_alpha_std

    Total_ES = ES * Notional
    return VaR, Total_ES


def plausibility_check(returns, weights, alpha, delta, value_portfolio):
    x = np.log(returns)
    n = len(x)
    mu = x.mean()
    mu_port = np.dot(weights, mu)
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
    var_check = (-delta * mu_port + np.sqrt(delta) * var_approx) * value_portfolio
    return var_check

#
# def plausibility_check(returns, weights, alpha, delta, value_portfolio):
#     mu = returns.mean()
#     mu_port = np.dot(weights, mu)
#
#     n = returns.shape[0]
#     n_shares = weights.shape[0]
#     ordered_x = np.zeros((n, n_shares))
#     for i in np.arange(n_shares):
#         ordered_x[:, i] = returns.iloc[:, i].sort_values().copy()
#
#     position_l = math.floor(n * (1 - alpha))
#     position_u = math.floor(n * alpha)
#
#     l = ordered_x[position_l, :]
#     u = ordered_x[position_u, :]
#
#     s_var = np.array(weights) * (np.abs(l) + np.abs(u)) / 2
#
#     correlation_matrix = returns.corr()
#
#     var_approx = np.sqrt(np.dot(s_var, np.dot(correlation_matrix, s_var.T)))
#     var_check = (-delta * mu_port + np.sqrt(delta) * var_approx) * value_portfolio
#     return var_check

def putprice_BS(s0, K, sigma, r, q, deltaTime):
    d1 = (np.log(s0 / K) + (r - q + sigma ** 2 / 2) * (deltaTime)) / (np.sqrt(deltaTime) * sigma)
    d2 = d1 - sigma * np.sqrt(deltaTime)
    return K * np.exp(-r * deltaTime) * norm.cdf(-d2) - s0 * np.exp(- q * deltaTime) * norm.cdf(-d1)


def putDeltaBS(s0, K, sigma, r, q, deltaTime):
    d1 = (np.log(s0 / K) + (r - q + sigma ** 2 / 2) * (deltaTime)) / (np.sqrt(deltaTime) * sigma)
    return -norm.cdf(-d1) * np.exp(-q * deltaTime)


def putGammaBS(s0, K, sigma, r, q, deltaTime):
    d1 = (np.log(s0 / K) + (r - q + sigma ** 2 / 2) * deltaTime) / (np.sqrt(deltaTime) * sigma)
    return (norm.pdf(d1) * np.exp(-q * deltaTime)) / (sigma * s0 * np.sqrt(deltaTime))


def FullMonteCarloVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,
                      volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears):
    n = len(logReturns)
    Stock_simulation = stockPrice * np.exp(logReturns)
    Stock_simulation = Stock_simulation.to_numpy().T
    putPrice1 = putprice_BS(stockPrice, strike, volatility, rate, dividend, timeToMaturityInYears).to_numpy()
    deltaT = (
                     timeToMaturityInYears * NumberOfDaysPerYears - riskMeasureTimeIntervalInYears * NumberOfDaysPerYears) / NumberOfDaysPerYears
    putPrice2 = [putprice_BS(stock, strike, volatility, rate, dividend, deltaT) for stock in Stock_simulation]
    lossDerivative = -(putPrice2[0] - putPrice1[0] * np.ones(n)) * numberOfPuts[0]
    lossStock = -(Stock_simulation - stockPrice.to_numpy() * np.ones(n)) * numberOfShares[0]
    totalLoss = lossStock + lossDerivative
    totalLoss = -np.sort(-totalLoss)
    index = math.floor(n * (1 - alpha))
    return totalLoss[0, index]


def DeltaNormalVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,
                   volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears):
    deltaPut = putDeltaBS(stockPrice, strike, volatility, rate, dividend, timeToMaturityInYears)
    sensPut = numberOfPuts * stockPrice * deltaPut
    sensStock = numberOfShares * stockPrice
    n = len(logReturns)
    losses = (-(sensPut * logReturns + sensStock * logReturns)).T
    losses = -np.sort(-losses)
    index = math.floor(n * (1 - alpha))
    return losses[0, index]


def DeltaGammaNormalVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,
                        volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears):
    deltaPut = putDeltaBS(stockPrice, strike, volatility, rate, dividend, timeToMaturityInYears)
    gammaPut = putGammaBS(stockPrice, strike, volatility, rate, dividend, timeToMaturityInYears)
    sensPut = numberOfPuts * stockPrice * deltaPut
    sensStock = numberOfShares * stockPrice
    gammaterm = numberOfPuts * gammaPut * (stockPrice ** 2)
    losses = (-(sensPut * logReturns + sensStock * logReturns + gammaterm / 2 * logReturns ** 2)).T
    losses = -np.sort(-losses)
    n = len(logReturns)
    index = math.floor(n * (1 - alpha))
    return losses[0, index]


def expectedPaymentI(s0, rate, volatility, yearOfThePayment, distPayment):
    meanstock = s0 * np.exp(rate * (yearOfThePayment - 1))
    cost = np.exp(rate * distPayment)
    d1 = (rate + volatility ** 2 / 2) * distPayment / (volatility * np.sqrt(distPayment))
    d2 = d1 - volatility * np.sqrt(distPayment)
    integral1 = norm.cdf(d1)
    integral2 = norm.cdf(d2)
    return meanstock * (cost * integral1 - integral2)
