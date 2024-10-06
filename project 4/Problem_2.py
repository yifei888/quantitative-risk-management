#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import norm, t
from statsmodels.tsa.ar_model import AutoReg


# In[4]:


daily_prices = pd.read_csv('DailyPrices.csv')
daily_prices


# In[5]:


def return_calculate(prices, method="DISCRETE", date_column="Date"):
    
    cols = list(prices.columns)
    if date_column not in cols:
        raise ValueError(f"dateColumn: {date_column} not in DataFrame: {cols}")
    
    vars = [col for col in cols if col != date_column]
    
    p = prices[vars].values
    n, m = p.shape  
    
    p2 = np.empty((n - 1, m), dtype=np.float64)
    
    for i in range(n - 1):
        p2[i, :] = p[i + 1, :] / p[i, :]
    
    if method.upper() == "DISCRETE":
        p2 = p2 - 1.0  
    elif method.upper() == "LOG":
        p2 = np.log(p2)  
    else:
        raise ValueError(f"method: {method} must be 'DISCRETE' or 'LOG'")
    
    
    dates = prices[date_column].iloc[1:].values 
    out = pd.DataFrame({date_column: dates})
    
    for i, var in enumerate(vars):
        out[var] = p2[:, i]
    
    return out


# In[11]:


returns = return_calculate(daily_prices, method="DISCRETE", date_column="Date")
returns


# In[12]:


returns["META"] = returns["META"] - returns["META"].mean()
returns = returns["META"]
returns


# In[17]:


def calculate_var_normal(returns, alpha=0.05):
    mean = returns.mean()
    std_dev = returns.std()
    return -(norm.ppf(alpha)) * std_dev + mean

def calculate_var_ewma(returns, alpha=0.05, lambd=0.94):
    weights = np.array([(1 - lambd) * lambd ** i for i in range(len(returns))])
    weights = weights[::-1] / np.sum(weights)
    weighted_mean = np.sum(weights * returns)
    weighted_var = np.sum(weights * (returns - weighted_mean) ** 2)
    weighted_std = np.sqrt(weighted_var)
    return -(norm.ppf(alpha)) * weighted_std + weighted_mean

def calculate_var_t(returns, alpha=0.05):
    params = t.fit(returns)
    df, loc, scale = params
    return -(t.ppf(alpha, df, loc, scale))

def calculate_var_ar1(returns, alpha=0.05):
    model = AutoReg(returns, lags=1).fit()
    ar1_mean = model.params[0] 
    ar1_std = model.bse[0] 
    return -(norm.ppf(alpha)) * ar1_std + ar1_mean

def calculate_var_historical(returns, alpha=0.05):
    return -(np.percentile(returns, alpha * 100))


var_normal = calculate_var_normal(returns)

var_ewma = calculate_var_ewma(returns)

var_t = calculate_var_t(returns)

var_ar1 = calculate_var_ar1(returns)

var_historical = calculate_var_historical(returns)

var_values = {
    "Normal VaR": var_normal,
    "EWMA VaR": var_ewma,
    "T-distribution VaR": var_t,
    "AR(1) Model VaR": var_ar1,
    "Historical Simulation VaR": var_historical
}

print("Value at Risk (VaR) for META using different methods:")
for method, value in var_values.items():
    print(f"{method}: {value:.6f}")


# In[ ]:




