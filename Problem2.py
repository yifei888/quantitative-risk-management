#!/usr/bin/env python
# coding: utf-8

# In[88]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
df = pd.read_csv('problem2.csv')
df


# # OLS Fitting

# In[94]:


Y = df['y'].to_numpy()  
X = df['x'].to_numpy().reshape(-1, 1) 
olsmodel = LinearRegression()
olsmodel.fit(X, Y)
beta = olsmodel.coef_[0]
intercept = olsmodel.intercept_
predict_y = ols_model.predict(X)
residuals = Y - predict_y
sd = np.std(residuals)
print(f"Beta: {beta}")
print(f"Intercept: {intercept}")
print(f"Standard Error: {sd}")


# # MLE Fitting with normal distribution

# In[76]:


X = np.array(df['x'])
Y = np.array(df['y'])
def myll(parameters):
    beta = parameters[0]  
    a = parameters[1] 
    sd = parameters[2]
    if sd <= 0:
        return np.inf 
    predict_y = beta * X.flatten() + a
    residuals = Y - predict_y
    n = len(X)
    ll = (n / 2) * np.log(2 * np.pi * sd**2) + (1 / (2 * sd**2)) * np.sum(residuals**2)
    return ll 
initial_guess = [0, 0, 1]
result = minimize(myll, initial_guess, method='Powell')


# In[96]:


print('Beta: ',result.x[0])
print('Intercept: ',result.x[1])
print('Standard error: ',result.x[2])


# # MLE Fitting with T-distribution

# In[106]:


from scipy.stats import t
def myllt(parameters):
    beta = parameters[0]  
    a = parameters[1] 
    sd = parameters[2]
    dfree = parameters[3]
    if sd <= 0:
        return np.inf 
    predict_y = beta * X.flatten() + a
    residuals = Y - predict_y
    n = len(X)
    ll = -np.sum(t.logpdf(residuals / sd, df=dfree)) + n * np.log(sd)
    return ll 
initial_guesst = [0, 0, 1, 2]
resultt = minimize(myllt, initial_guesst, method='Powell')
print('Beta: ',resultt.x[0])
print('Intercept: ',resultt.x[1])
print('Standard error: ',resultt.x[2])
print('Degree of freedom: ',resultt.x[3])

