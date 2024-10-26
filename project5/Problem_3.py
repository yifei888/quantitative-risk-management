#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from scipy.stats import t, norm, multivariate_normal
from scipy.optimize import minimize
portfolio = pd.read_csv('portfolio-3.csv')
price = pd.read_csv('DailyPrices.csv')

price = price.drop(columns = 'Date')
price = price.apply(pd.to_numeric, errors='coerce')
arithmetic_returns = price.diff().dropna()

portfolio_groups = portfolio.groupby('Portfolio')['Stock'].apply(list)
stocks_portfolio_A, stocks_portfolio_B, stocks_portfolio_C = portfolio_groups['A'], portfolio_groups['B'], portfolio_groups['C']
available_stocks = set(arithmetic_returns.columns)

stocks_portfolio_A = [stock for stock in stocks_portfolio_A if stock in available_stocks]
stocks_portfolio_B = [stock for stock in stocks_portfolio_B if stock in available_stocks]
stocks_portfolio_C = [stock for stock in stocks_portfolio_C if stock in available_stocks]

def neg_log_likelihood_t(params, data):
    df, mean, scale = params
    if scale <= 0: return np.inf
    return -np.sum(t.logpdf(data, df=df, loc=mean, scale=scale))

params_portfolio_A, params_portfolio_B, params_portfolio_C = {}, {}, {}

for stock in stocks_portfolio_A:
    data = arithmetic_returns[stock].dropna()
    if len(data) > 1:
        initial_guess = [2, np.mean(data), np.std(data)]
        result = minimize(neg_log_likelihood_t, initial_guess, args=(data,), bounds=((1.01, None), (None, None), (0.0001, None)))
        params_portfolio_A[stock] = result.x

for stock in stocks_portfolio_B:
    data = arithmetic_returns[stock].dropna()
    if len(data) > 1:
        initial_guess = [2, np.mean(data), np.std(data)]
        result = minimize(neg_log_likelihood_t, initial_guess, args=(data,), bounds=((1.01, None), (None, None), (0.0001, None)))
        params_portfolio_B[stock] = result.x

for stock in stocks_portfolio_C:
    data = arithmetic_returns[stock].dropna()
    if len(data) > 1:
        mean, std_dev = norm.fit(data)
        params_portfolio_C[stock] = (mean, std_dev)

def calculate_var_es(simulated_returns, confidence_level):
    if simulated_returns.size == 0:
        return np.nan, np.nan
    VaR = np.percentile(simulated_returns, (1 - confidence_level) * 100)
    ES = simulated_returns[simulated_returns <= VaR].mean()
    return VaR, ES

weights_A = np.array([portfolio.set_index('Stock')['Holding'][stock] for stock in stocks_portfolio_A])
weights_B = np.array([portfolio.set_index('Stock')['Holding'][stock] for stock in stocks_portfolio_B])
weights_C = np.array([portfolio.set_index('Stock')['Holding'][stock] for stock in stocks_portfolio_C])

sim_portfolio_A = np.dot(arithmetic_returns[stocks_portfolio_A].values, weights_A) if weights_A.size > 0 else np.array([])
sim_portfolio_B = np.dot(arithmetic_returns[stocks_portfolio_B].values, weights_B) if weights_B.size > 0 else np.array([])
sim_portfolio_C = np.dot(arithmetic_returns[stocks_portfolio_C].values, weights_C) if weights_C.size > 0 else np.array([])

confidence_level = 0.95
VaR_A, ES_A = calculate_var_es(sim_portfolio_A, confidence_level)
VaR_B, ES_B = calculate_var_es(sim_portfolio_B, confidence_level)
VaR_C, ES_C = calculate_var_es(sim_portfolio_C, confidence_level)


# In[9]:


def to_uniform(data, params, dist_type='t'):
    if dist_type == 't':
        df, mean, scale = params
        return t.cdf(data, df=df, loc=mean, scale=scale)
    elif dist_type == 'norm':
        mean, std_dev = params
        return norm.cdf(data, loc=mean, scale=std_dev)

uniform_A = np.column_stack([to_uniform(arithmetic_returns[stock], params_portfolio_A[stock], 't') for stock in stocks_portfolio_A])
uniform_B = np.column_stack([to_uniform(arithmetic_returns[stock], params_portfolio_B[stock], 't') for stock in stocks_portfolio_B])
uniform_C = np.column_stack([to_uniform(arithmetic_returns[stock], params_portfolio_C[stock], 'norm') for stock in stocks_portfolio_C if stock in params_portfolio_C])

uniform_all = np.hstack((uniform_A, uniform_B, uniform_C))
correlation_matrix = np.corrcoef(uniform_all, rowvar=False)

copula_samples = multivariate_normal.rvs(mean=np.zeros(correlation_matrix.shape[0]), cov=correlation_matrix, size=10000)
copula_samples = norm.cdf(copula_samples)

def from_uniform(uniform_data, params, dist_type='t'):
    if dist_type == 't':
        df, mean, scale = params
        return t.ppf(uniform_data, df=df, loc=mean, scale=scale)
    elif dist_type == 'norm':
        mean, std_dev = params
        return norm.ppf(uniform_data, loc=mean, scale=std_dev)

sim_returns_A = np.column_stack([from_uniform(copula_samples[:, i], params_portfolio_A[stock], 't') for i, stock in enumerate(stocks_portfolio_A)])
sim_returns_B = np.column_stack([from_uniform(copula_samples[:, i + len(stocks_portfolio_A)], params_portfolio_B[stock], 't') for i, stock in enumerate(stocks_portfolio_B)])
sim_returns_C = np.column_stack([
    from_uniform(copula_samples[:, i + len(stocks_portfolio_A) + len(stocks_portfolio_B)], params_portfolio_C[stock], 'norm')
    for i, stock in enumerate(stocks_portfolio_C)
    if stock in params_portfolio_C and len(params_portfolio_C[stock]) == 2
])
sim_total_returns = np.hstack((sim_returns_A, sim_returns_B, sim_returns_C))
total_weights = np.concatenate((weights_A, weights_B, weights_C))
sim_total_portfolio = np.dot(sim_total_returns, total_weights)

VaR_total, ES_total = calculate_var_es(sim_total_portfolio, confidence_level)

print("Portfolio A VaR:", - VaR_A, "ES:", - ES_A)
print("Portfolio B VaR:", -VaR_B, "ES:", -ES_B)
print("Portfolio C VaR:", -VaR_C, "ES:", -ES_C)
print("Total Portfolio VaR (using copula):", -VaR_total, "ES:", -ES_total)


# In[ ]:




