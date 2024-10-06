#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import norm


# In[4]:


portfolio = pd.read_csv('Portfolio.csv')  
daily_prices = pd.read_csv('DailyPrices.csv')  

daily_prices['Date'] = pd.to_datetime(daily_prices['Date'])
daily_prices.set_index('Date', inplace=True)
returns = daily_prices.pct_change().dropna() 


# In[5]:


def ewma_cov(returns, lambd=0.97):
    n = len(returns)
    weights = np.array([(1 - lambd) * lambd ** i for i in range(n)])
    weights = weights[::-1] / np.sum(weights)
    weighted_returns = returns.multiply(weights, axis=0)
    cov_matrix = np.cov(weighted_returns.T)
    return cov_matrix


# In[6]:


ewma_cov_matrix = ewma_cov(returns, lambd=0.97)


# In[10]:


def calc_portfolio_var(portfolio, cov_matrix, returns, prices, alpha=0.05):
    stocks = portfolio['Stock'].values
    holdings = portfolio['Holding'].values
    available_stocks = [stock for stock in stocks if stock in prices.columns]
    if len(available_stocks) < len(stocks):
        missing_stocks = [stock for stock in stocks if stock not in prices.columns]
        print(f"exclude: {missing_stocks}")
    available_holdings = [holdings[i] for i, stock in enumerate(stocks) if stock in prices.columns]
    latest_prices = prices.loc[prices.index[-1], available_stocks].values
    portfolio_value = np.array(available_holdings) * latest_prices
    stock_indices = [returns.columns.get_loc(stock) for stock in available_stocks]
    portfolio_cov_matrix = cov_matrix[np.ix_(stock_indices, stock_indices)]
    portfolio_std = np.sqrt(np.dot(np.array(available_holdings).T, np.dot(portfolio_cov_matrix, np.array(available_holdings))))
    dollar_portfolio_std = portfolio_std * np.sum(portfolio_value)
    z_alpha = norm.ppf(alpha)
    var_value = -z_alpha * dollar_portfolio_std
    return var_value


# In[11]:


latest_prices = daily_prices.iloc[-1]


# In[12]:


portfolio_A = portfolio[portfolio['Portfolio'] == 'A']
portfolio_B = portfolio[portfolio['Portfolio'] == 'B']
portfolio_C = portfolio[portfolio['Portfolio'] == 'C']

var_A = calc_portfolio_var(portfolio_A, ewma_cov_matrix, returns, daily_prices)
var_B = calc_portfolio_var(portfolio_B, ewma_cov_matrix, returns, daily_prices)
var_C = calc_portfolio_var(portfolio_C, ewma_cov_matrix, returns, daily_prices)

print(f"Portfolio A VaR (EWMA, $): ${var_A:.2f}")
print(f"Portfolio B VaR (EWMA, $): ${var_B:.2f}")
print(f"Portfolio C VaR (EWMA, $): ${var_C:.2f}")

total_portfolio = pd.concat([portfolio_A, portfolio_B, portfolio_C])
total_var = calc_portfolio_var(total_portfolio, ewma_cov_matrix, returns, daily_prices)
print(f"Total Portfolio VaR (EWMA, $): ${total_var:.2f}")


# In[15]:


def historical_var(portfolio, returns, prices, alpha=0.05):
    stocks = portfolio['Stock'].values
    holdings = portfolio['Holding'].values
    available_stocks = [stock for stock in stocks if stock in prices.columns]
    if len(available_stocks) < len(stocks):
        missing_stocks = [stock for stock in stocks if stock not in prices.columns]
        print(f"excluded: {missing_stocks}")
    available_holdings = [holdings[i] for i, stock in enumerate(stocks) if stock in prices.columns]
    latest_prices = prices.iloc[-1][available_stocks].values
    portfolio_returns = returns[available_stocks].dot(np.array(available_holdings) * latest_prices)
    var = np.percentile(portfolio_returns, 100 * alpha)
    return -var

historical_var_A = historical_var(portfolio_A, returns, daily_prices)
historical_var_B = historical_var(portfolio_B, returns, daily_prices)
historical_var_C = historical_var(portfolio_C, returns, daily_prices)

print(f"Portfolio A VaR (Historical, $): ${historical_var_A:.2f}")
print(f"Portfolio B VaR (Historical, $): ${historical_var_B:.2f}")
print(f"Portfolio C VaR (Historical, $): ${historical_var_C:.2f}")
historical_var_total = historical_var(total_portfolio, returns, daily_prices)
print(f"Total Portfolio VaR (Historical, $): ${historical_var_total:.2f}")

