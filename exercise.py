
"""
Created on Mon Mar 22 15:26:26 2021

@author: Joshua D. Ricci

Python Test - Validus
"""

import pandas as pd
import numpy as np
import random
from math import exp, sqrt
import matplotlib.pyplot as plt
import datetime

notionals = pd.read_excel('C:/Users/joshu/OneDrive/Desktop/Applications/Validus/Quantitative Analyst Case Study 2021 - Cashflow Model.xlsx')

#a) Using the Monte Carlo method, simulate 1000 paths across time for the GBPUSD FX spot
# rate, using the Geometric Brownian Motion (GBM) model. See below for model assumptions.

# Parameter Definitions

# GBM equation: S_T = S_0 * exp((mu - 0.5 * sigma ** 2) * T + sigma * z * sqrt(T))
#                   = S_0 * exp((mu - 0.5 * sigma ** 2) * T + sigma * z * sqrt(T))
# vectorized: 
#               ST = S0 * np.exp((mu - 0.5 * sigma ** 2) * T + sigma * np.random.standard_normal(n_simulations) * np.sqrt(T))

# Establishing rows as business days

rows = pd.date_range(start='31/3/2021', end='31/3/2026', freq=pd.offsets.BDay(1))    

##Note! change 252 to 261 as that is the more accurate number of days

S0 = 1.3925
r = 0
mu = 0 
dt = 1
T = int(len(rows) / 5)
sigma = 0.1/sqrt(T)
n_simulations = 1000

#drift = 0 -> mu = 0
ST = S0 * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.standard_normal((rows.shape[0], n_simulations)), axis=0))

#drift = 0 -> (mu - 0.5 * sigma ** 2)
ST = S0 * np.exp(np.cumsum(sigma * np.sqrt(dt) * np.random.standard_normal((rows.shape[0], n_simulations)), axis=0))


ST_df = pd.DataFrame(ST, rows)

ST_df = ST_df.shift(1)
ST_df.iloc[0,:] = 1.3925


# b)    Using the simulated FX spot rate paths, convert the GBP cashflows into fund currency (USD)
#       and calculate the corresponding IRR (internal rate of return) for each of the simulated paths. 
#       Plot the distribution of the IRR and evaluate the 95%, 50% and 5% percentiles.

#Note! '2024-03-31' is not a business day, hence, the 1st April was chosen as a replacement

cash_dates = [ST_df.index.get_loc('2021-03-31'), ST_df.index.get_loc('2022-03-31'), ST_df.index.get_loc('2023-03-31'), ST_df.index.get_loc('2024-04-01'), ST_df.index.get_loc('2025-03-31'), ST_df.index.get_loc('2026-03-31')]

cashflows = ST_df.iloc[[cash_dates[0], cash_dates[1], cash_dates[2], cash_dates[3], cash_dates[4], cash_dates[5]],:]

#transposing dataset
cashflows_T = cashflows.T

# multiplying by notionals in excel 
# principal -100,000,000
cashflows_T['2021-03-31'] = cashflows_T['2021-03-31'].apply(lambda x: x*-100000000)

for i in range(4):
    cashflows_T.iloc[:,i+1] = cashflows_T.iloc[:,i+1].apply(lambda x: x*15000000)
    
cashflows_T['2026-03-31'] = cashflows_T['2026-03-31'].apply(lambda x: x*115000000)

#transposing again for the purpose of simplicity
cashflows_Tb = cashflows_T.T

#calculating IRR and appending to list
irr = [] 
for i in range(len(cashflows_Tb.keys())):
    irr.append(np.irr(cashflows_Tb[i]))

#plotting distrubution of irr

plt.hist(irr)

#computing percentiles, 95%, 50% and 5%

ninenty_five_per = np.percentile(irr, 95)
fifty_per= np.percentile(irr, 50)
five_per = np.percentile(irr, 5)

print('The 95% percentile is ' + str(ninenty_five_per))
print('The 50% percentile is ' + str(fifty_per))
print('The 5% percentile is ' + str(five_per))



# c)	Assume we would like to buy a GBPUSD European Put option to hedge our FX exposure. If the USD
#       and GBP risk-free (interest) rates are 0% throughout the time horizon of the fund, how can we 
#       use the simulated FX spot rate paths to calculate the option’s fair market value (premium) on 
#       the trade date. See the option details below:
#       •	Trade Date: 31/03/2021
#       •	Expiry Date: 31/03/2026
#       •	Notional Amount: 100,000,000 GBP
#       •	Strike: 1.3925

#parameters

discount_factor = np.exp(-r * T)
K = 1.3925
notional = 100000000

put_payoffs = [max(K - ST_df.iloc[-1, i],0.0) for i in range(n_simulations)]

put_price = discount_factor * (sum(put_payoffs) / n_simulations) * notional


# d)	Calculate the IRR of the hedged portfolio, including the option premium payment you calculated 
#       in (c) and the option payoff. Looking at the distribution of the hedged portfolio IRR, assess 
#       the impact of the Put option on the portfolio FX risk.

# adding put option costs and payoffs

cashflows_T['2021-03-31'] -= put_price
for y in range(1000):
    cashflows_T.iloc[y, -1] += put_payoffs[y]*notional

#calculating IRR and appending to list

cashflows_Td = cashflows_T.T
irr_d = [] 
for i in range(len(cashflows_Td.keys())):
    irr_d.append(np.irr(cashflows_Td[i]))
 
#plotting new irr
plt.hist(irr_d)
plt.legend()

#%%

#code test
#this is another code from another random person to check differences in the price
# I am not sure if this follows a GBM

import datetime
from random import gauss
#from math import exp, sqrt
import numpy as np

def generate_asset_price(S,v,r,T):
    return S * np.exp((r - 0.5 * v**2) * T + v * np.sqrt(T) * gauss(0,1.0))

def put_payoff(S_T,K):
    return max(0.0,(S_T-K)*-1)

S = 1.3925 # underlying price
v = 0.1 # vol of 20.76%
r = 0.0 # rate of 0.14%
T = (datetime.date(2026,3,31) - datetime.date(2021,3,31)).days / 365.0
K = 1.3925
simulations = 1000
payoffs = []
discount_factor = np.exp(-r * T)

for i in range(simulations):
    S_T = generate_asset_price(S,v,r,T)
    payoffs.append(
        put_payoff(S_T, K)
    )

price = discount_factor * (sum(payoffs) / float(simulations))
print(price)

#%%



