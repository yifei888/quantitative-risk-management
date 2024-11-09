#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
import datetime
from scipy import optimize
from scipy.stats import norm


# In[29]:


from datetime import datetime, timedelta
import yfinance as yf
end_date = datetime.today()
start_date = end_date - timedelta(days=10 * 252)  

stock_list = [
    'AAPL', 'META', 'UNH', 'MA', 'MSFT', 'NVDA', 'HD', 'PFE',
    'AMZN', 'BRK-B', 'PG', 'XOM', 'TSLA', 'JPM', 'V', 'DIS',
    'GOOGL', 'JNJ', 'BAC', 'CSCO'
]


price_data = yf.download(stock_list, start=start_date, end=end_date)['Adj Close']

def calculate_log_return(price_data):
    """Calculate log returns, centered by mean."""
    log_ret = np.log(price_data / price_data.shift(1)).dropna()
    return log_ret - log_ret.mean()

log_return = calculate_log_return(price_data)

fama_3_factor = fama_3_factor = pd.read_csv('F-F_Research_Data_Factors_daily.csv')
fama_3_factor['Date'] = pd.to_datetime(fama_3_factor['Date'], format='%Y%m%d', errors='coerce')
fama_3_factor.set_index('Date', inplace=True)
fama_3_factor = fama_3_factor[["Mkt-RF", "SMB", "HML", "RF"]] / 100 


momentum = pd.read_csv('F-F_Momentum_Factor_daily.csv')
momentum['Date'] = pd.to_datetime(momentum['Date'], format='%Y%m%d', errors='coerce')
momentum.set_index('Date', inplace=True)
momentum = momentum[["Mom   "]] / 100  

fama_data = fama_3_factor.join(momentum, how='inner')


intersected_dates = log_return.index.intersection(fama_data.index)
rt = log_return.loc[intersected_dates, stock_list]
excess_return = rt.sub(fama_data.loc[intersected_dates, 'RF'], axis=0)

def calculate_beta_matrix(factor_data, excess_returns, factors):
    """Calculate the beta matrix using chosen factors."""
    factor_matrix = factor_data[factors].copy()
    factor_matrix['Intercept'] = 1
    X = factor_matrix.values
    beta = np.linalg.lstsq(X, excess_returns.values, rcond=None)[0]
    return pd.DataFrame(beta, index=factor_matrix.columns, columns=excess_returns.columns)

factors_4 = ["Mkt-RF", "SMB", "HML", "Mom   "]
beta_4 = calculate_beta_matrix(fama_data.loc[intersected_dates], excess_return, factors_4)

=
def calculate_expected_returns(factor_data, beta_matrix, factors, rf_rate):
    """Calculate expected returns based on past factor returns and risk-free rate."""
    avg_factors = factor_data[factors].mean()
    print("Average Factor Values:\n", avg_factors) 
    avg_factors['Intercept'] = 1  
    factor_based_return = avg_factors @ beta_matrix  
    print("Factor-Based Expected Returns (before adding RF):\n", factor_based_return)
    return (rf_rate + factor_based_return) * 252 


rf_rate = 0.05
expected_return_4 = calculate_expected_returns(fama_data.loc[intersected_dates], beta_4, factors_4, rf_rate)

cov_matrix = rt.cov() * 252  


def optimize_portfolio(expected_returns, cov_matrix, rf=0.05):
    """Optimize portfolio weights for maximal Sharpe ratio."""
    def sharpe_ratio(weights):
        port_return = weights @ expected_returns - rf
        port_volatility = np.sqrt(weights @ cov_matrix @ weights)
        return -(port_return / port_volatility)  

    num_assets = len(expected_returns)
    initial_weights = np.ones(num_assets) / num_assets
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(num_assets)]

    result = optimize.minimize(sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


opt_weights_4 = optimize_portfolio(expected_return_4, cov_matrix).x

def display_portfolio(weights, model_name):
    weights_df = pd.DataFrame(weights, index=stock_list, columns=['Weight']).T
    print(f"\n{model_name} Portfolio Weights:\n{weights_df.round(3)}")

display_portfolio(opt_weights_4, "4-Factor Model")

opt_return_4 = -optimize_portfolio(expected_return_4, cov_matrix).fun
print(f"\nOptimal Portfolio Return (4-Factor Model): {opt_return_4:.4f}")


# In[28]:


expected_return_4


# In[ ]:




