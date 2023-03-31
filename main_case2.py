import numpy as np
import utilities as ut


volatility = 0.25
N = 50e6

s0 = 1            # suppose the stock price equal to 1

Discounts = [0.968559215016457, 0.939091142108513, 0.914758332352477, 0.891754412977887]
SurvProb = [0.995024875621891, 0.988722369482137, 0.981650050291347, 0.974274315232432]
rate= [-1 / (i + 1) * np.log(Discounts[i]) for i in range(4)]

price = np.sum([Discounts[i] * ut.expectedPaymentI(s0, rate[i], volatility, i + 1, 1) for i in range(4)]) * N
priceDefaultCase = np.sum([Discounts[i] * SurvProb[i] * ut.expectedPaymentI(s0, rate[i], volatility, i + 1, 1) for i in range(4)]) * N

print("The Pricing in presence of counterparty risk are:")
print('Price analytical formula: {:.5f}'.format(price))
print('Price considering default: {:.5f}'.format(priceDefaultCase))