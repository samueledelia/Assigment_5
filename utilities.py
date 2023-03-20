import numpy as np
from scipy.stats import norm

def analytical_normal_measures(alpha, weights, returns):
    x = np.log(returns)  # log return vector

    # Compute the Var and the ES
    mu = x.mean()
    sigma = x.cov()
    norm_inv = norm.ppf(alpha)

    mu_port = np.dot(weights, mu)
    sigma_port = np.dot(weights, np.dot(sigma, weights))

    var = mu_port + norm_inv * sigma_port
    return var