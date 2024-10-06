#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[11]:


sd = np.sqrt(0.01)
p_tminus1 = 1000
r_t = np.random.normal(0, sd, 10000)
P_t_brownian = p_tminus1 + r_t
P_t_arithmetic = p_tminus1 * (1 + r_t)
P_t_logarithmic = p_tminus1 * np.exp(r_t)


# In[12]:


mean_brownian = np.mean(P_t_brownian)
std_brownian_sim = np.std(P_t_brownian)
print(" Mean for classic brownian: ", mean_brownian)
print("Standard Deviation for classic brownian: ", std_brownian_sim)


# In[13]:


mean_arithmetic = np.mean(P_t_arithmetic)
std_arithmetic_sim = np.std(P_t_arithmetic)
print(" Mean for arithemtic: ", mean_arithmetic)
print("Standard Deviation for arithemtic: ", std_arithmetic_sim)


# In[15]:


mean_logarithmic = np.mean(P_t_logarithmic)
std_logarithmic_sim = np.std(P_t_logarithmic)
print(" Mean for logrithemtic: ", mean_logarithmic)
print("Standard Deviation for logbrithemtic: ", std_logarithmic_sim)


# In[ ]:




