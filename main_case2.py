import numpy as np
import utilities as ut

# Setting parameters

volatility = 0.25
N = 50e6
s0 = 1            # suppose the stock price equal to 1

# Extract the following parameter from previous assignments

Discounts = [1, 0.968559215016457, 0.939091142108513, 0.914758332352477, 0.891754412977887]       # we get the discount from the bootstrap of assig 2
SurvProb = [0.995024875621891, 0.988722369482137, 0.981650050291347, 0.974274315232432]         # we get the surv prob fron assig 3
rate = [-1 / (i + 1) * np.log(Discounts[i]) for i in range(4)]

# Compute the prices

price = np.sum([Discounts[i+1]*ut.expectedPaymentI(s0, Discounts[i], Discounts[i+1], volatility, 1) for i in range(4)])*N
priceDefaultCase = np.sum([Discounts[i+1]*SurvProb[i]*ut.expectedPaymentI(s0, Discounts[i], Discounts[i+1], volatility, 1) for i in range(4)])*N

print("The Pricing in presence of counterparty risk are:")
print('Price of Cliquet with analytical formula: {:.5f}'.format(price))
print('Price of Cliquet with considering default: {:.5f}'.format(priceDefaultCase))