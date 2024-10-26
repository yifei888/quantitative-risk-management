#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t
from scipy.optimize import minimize
df = pd.read_csv('problem1.csv')
df


# # EWMA 

# In[13]:


def ewma_variance(returns, lambda_ewma=0.97):
    
    squared_returns = returns ** 2
    n = len(returns)
    
    ewma_var = np.zeros(n)
    ewma_var[0] = squared_returns[0] 
    
    for i in range(1, n):
        ewma_var[i] = lambda_ewma * ewma_var[i - 1] + (1 - lambda_ewma) * squared_returns[i]
    
    return ewma_var


# In[14]:


returns = df['x'].values 
ewma_var_result = ewma_variance(returns)
ewma_var_result


# In[52]:


current_volatility = np.sqrt(ewma_var_result[-1])
mean = returns.mean()
VaR_normal = (mean * (-1)) + norm.ppf(1 - 0.95) * current_volatility
ES_normal = (mean * (-1)) + (norm.pdf(norm.ppf(1 - 0.95)) / (1 - 0.95)) * current_volatility
print('EWMA VaR: ', - VaR_normal)
print('EWMA ES: ', ES_normal)


# # MLE t-ditribution

# In[35]:


def neg_log_likelihood(params):
    df, loc, sd = params[0], params[1], params[2]
    if sd <= 0:
        return np.inf
    nll = -np.sum(t.logpdf(returns, df=df, loc=loc, scale=sd))
    return nll


# In[51]:


initial_guess = [2, np.mean(returns), np.std(returns)]
result = minimize(neg_log_likelihood, initial_guess, method='Powell', bounds=((1.01, None), (None, None), (0.0001, None)))
df_mle, loc_mle, sd_mle = result.x
confidence_level = 0.95
VaR_t = (mean * (-1)) + sd_mle * t.ppf(1 - confidence_level, df=df_mle)
ES_t = (mean * (-1)) + (t.pdf(t.ppf(1 - confidence_level, df=df_mle), df=df_mle) / (1 - confidence_level)) * sd_mle
print('MLE t-distribution VaR: ', - VaR_t)
print('MLE t-distribution ES: ', ES_t)


# # Historic Simulation

# In[53]:


return_sorted = np.sort(returns)
var_index = int((1 - confidence_level) * len(return_sorted))
VaR_historic = return_sorted[var_index]
ES_historic = return_sorted[:var_index].mean()
print('Historic Simulation VaR: ', -VaR_historic)
print('Historic Simulation ES: ', -ES_historic)


# In[ ]:




