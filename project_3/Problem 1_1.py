#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
data = pd.read_csv('DailyReturn.csv')


# In[5]:


data


# In[16]:


def exp_weighted_cov_matrix(data, lambda_value):
    stocks_number = data.shape[1]
    cov_matrix = np.zeros((stocks_number, stocks_number)) 
    mean = np.mean(data, axis=0)
    
    for i in range(1, len(data)):
        returns_previous = data.iloc[i-1, :]
        diff = returns_previous - mean
        cov_matrix = lambda_value * cov_matrix + (1 - lambda_value) * np.outer(diff, diff)
    return cov_matrix


# In[30]:


def pca(cov_matrix, lambda_value):
    pca = PCA()
    pca.fit(cov_matrix) 
    explained = np.cumsum(pca.explained_variance_ratio_)  
    plt.plot(np.arange(1, len(explained) + 1), explained, label=f'λ = {lambda_value:.2f}')


lambda_value = [0.1, 0.3, 0.9, 0.97, 0.99]
for value in lambda_value:
    cov_matrix = exp_weighted_cov_matrix(data, value)
    pca(cov_matrix, value)

plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.show()


# In[ ]:




