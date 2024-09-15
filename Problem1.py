#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd
from scipy import stats
df = pd.read_csv('problem1.csv')
df = df.to_numpy().flatten()
df


# # Calculation using note

# In[23]:


mean = np.mean(df)
variance = np.var(df)
skewness = np.mean(((df - mean) / np.sqrt(variance))**3)
kurtosis = np.mean(((df - mean) / np.sqrt(variance))**4)
print('mean: ',mean)
print('variance: ', variance)
print('skewness', skewness)
print('kurtosis:', kurtosis)


# # Calculation using package

# In[25]:


mean_p = np.mean(df)
variance_p = np.var(df)
skewness_p = pd.Series(df).skew()
kurtosis_p = pd.Series(df).kurt()
print('mean: ',mean_p)
print('variance: ', variance_p)
print('skewness', skewness_p)
print('kurtosis:', kurtosis_p)


# # Hypothesis Testing

# In[31]:


t_skew, p_skew = stats.ttest_1samp((df - np.mean(df))**3 / (np.std(df)**3), 0)
t_kur, p_kur = stats.ttest_1samp((df - np.mean(df))**4 / (np.std(df)**4), 3)
print('t test for skewness: ', t_skew)
print('t test for kurtosis: ', t_kur)
print('P value for skewness: ', p_skew)
print('P value for kurtosis: ', p_kur)


# In[ ]:





# In[ ]:




