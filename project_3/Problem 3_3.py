#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import pandas as pd
import math
import time
data = pd.read_csv('DailyReturn.csv')


# In[53]:


corr = data.corr().values 
var = data.var().values    


# In[54]:


def compute_exp_weighted_var(data, lambda_val=0.97):
    exp_weighted_var = np.zeros(len(data.columns))
    
    for i in range(1, len(data)):
        returns_previous = data.iloc[i-1, :]
        exp_weighted_var = lambda_val * exp_weighted_var + (1 - lambda_val) * returns_previous**2
    
    return exp_weighted_var

exp_weighted_var = compute_exp_weighted_var(data)



def compute_exp_weighted_corr(data, lambda_val=0.97):
    n = len(data.columns)  
    exp_weighted_cov = np.zeros((n, n))  
    weighted_mean_diff = np.zeros(n)
    mean = data.mean().values  
    
    for i in range(1, len(data)):
        returns_previous = data.iloc[i-1, :] - mean  
        
        for j in range(n):
            for k in range(n):
                exp_weighted_cov[j][k] = lambda_val * exp_weighted_cov[j][k] + (1 - lambda_val) * (returns_previous[j] * returns_previous[k])

    std_dev = np.sqrt(np.diag(exp_weighted_cov))
    
    exp_weighted_corr = exp_weighted_cov / np.outer(std_dev, std_dev)
    
    return exp_weighted_corr
exp_weighted_corr = compute_exp_weighted_corr(data, lambda_val=0.97)


# In[55]:


def covariance_matrix(corr, var):
    n = len(var)
    cov_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            cov_matrix[i][j] = corr[i][j] * math.sqrt(var[i] * var[j])
    
    return cov_matrix


# # 4 matrices

# In[56]:


cov_pearson_var = covariance_matrix(corr, var)              
cov_pearson_exp_var = covariance_matrix(corr, exp_weighted_var) 
cov_exp_corr_var = covariance_matrix(exp_weighted_corr, var) 
cov_exp_corr_exp_var = covariance_matrix(exp_weighted_corr, exp_weighted_var)


# # Direct Method

# In[57]:


def cholesky_decomposition(matrix):
    n = len(matrix)
    L = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1):
            sum_val = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(matrix[i][i] - sum_val)
            else:
                L[i][j] = (matrix[i][j] - sum_val) / L[j][j]
    
    return L

def simulate_multivariate_normal(cov_matrix, n_draws=25000):
    n = len(cov_matrix)
    L = cholesky_decomposition(cov_matrix)
    
    simulated_data = np.zeros((n_draws, n))
    
    for i in range(n_draws):
        Z = np.random.normal(0, 1, n) 
        simulated_data[i] = L @ Z
    
    return simulated_data


simulated_pearson_var = simulate_multivariate_normal(cov_pearson_var, 25000)
simulated_pearson_exp_var = simulate_multivariate_normal(cov_pearson_exp_var, 25000)
simulated_exp_corr_var = simulate_multivariate_normal(cov_exp_corr_var, 25000)
simulated_exp_corr_exp_var = simulate_multivariate_normal(cov_exp_corr_exp_var, 25000)


# # PCA Method

# In[58]:


def eigen_decomposition(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvalues, eigenvectors


def simulate_pca(cov_matrix, variance_explained=1.0, n_draws=25000):
    eigenvalues, eigenvectors = eigen_decomposition(cov_matrix)
    
   
    sorted_indices = np.argsort(eigenvalues)[::-1]  
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
  
    total_variance = np.sum(eigenvalues)
    cumulative_variance = 0
    n_components = 0
    
    for ev in eigenvalues:
        cumulative_variance += ev
        n_components += 1
        if cumulative_variance / total_variance >= variance_explained:
            break
    
  
    simulated_data = np.zeros((n_draws, cov_matrix.shape[0]))
    
    for i in range(n_draws):
        Z = np.random.normal(0, 1, n_components)  
        for j in range(cov_matrix.shape[0]):
            simulated_data[i, j] = np.dot(eigenvectors[j, :n_components], Z)
    
    return simulated_data


cov_matrices = [cov_pearson_var, cov_pearson_exp_var, cov_exp_corr_var, cov_exp_corr_exp_var]

variance_levels = [1.0, 0.75, 0.50]
simulated_pca_data = {}
cov_matrix_names = ['pearson_var', 'pearson_exp_var', 'exp_corr_var', 'exp_corr_exp_var']

for idx, cov_matrix in enumerate(cov_matrices):
    matrix_name = cov_matrix_names[idx] 
    for var_level in variance_levels:
        key_name = f'simulated_pca_{int(var_level * 100)}_{matrix_name}'
        simulated_pca_data[key_name] = simulate_pca(cov_matrix, variance_explained=var_level, n_draws=25000)


# In[59]:


simulated_direct_data = {}
for idx, cov_matrix in enumerate(cov_matrices):
    matrix_name = cov_matrix_names[idx]
    key_name = f'simulated_direct_{matrix_name}'
    simulated_direct_data[key_name] = simulate_multivariate_normal(cov_matrix, 25000)


# In[60]:


def compute_covariance(data):
    return np.cov(data.T)

# Calculate covariance matrices for the simulated PCA data
cov_simulated_pca = {}
for key in simulated_pca_data:
    cov_simulated_pca[key] = compute_covariance(simulated_pca_data[key])

# Calculate covariance matrices for the direct simulated data
cov_simulated_direct = {}
for key in simulated_direct_data:
    cov_simulated_direct[key] = compute_covariance(simulated_direct_data[key])


# In[66]:


def frobenius_norm(matrix1, matrix2):
    # Calculate the Frobenius norm
    diff = matrix1 - matrix2
    norm = np.sqrt(np.sum(diff**2))
    return norm

frobenius_results = {}
for idx, cov_matrix in enumerate(cov_matrices):
    matrix_name = cov_matrix_names[idx]
    frobenius_results[f'direct_{matrix_name}'] = frobenius_norm(cov_matrix, cov_simulated_direct[f'simulated_direct_{matrix_name}'])

    for var_level in variance_levels:
        key_name = f'simulated_pca_{int(var_level * 100)}_{matrix_name}'
        frobenius_results[key_name] = frobenius_norm(cov_matrix, cov_simulated_pca[key_name])

# Print Frobenius results for comparison
for key, result in frobenius_results.items():
    print(f"Frobenius Norm for {key}: {result}")


# In[63]:


timing_results = {}
for idx, cov_matrix in enumerate(cov_matrices):
    matrix_name = cov_matrix_names[idx]
    
    # Timing for direct simulation
    start_time = time.time()
    simulate_multivariate_normal(cov_matrix, 25000)
    timing_results[f'direct_{matrix_name}'] = time.time() - start_time
    
    # Timing for PCA simulation (100%)
    start_time = time.time()
    simulate_pca(cov_matrix, variance_explained=1.0, n_draws=25000)
    timing_results[f'pca_100_{matrix_name}'] = time.time() - start_time
    
    # Timing for PCA simulation (75%)
    start_time = time.time()
    simulate_pca(cov_matrix, variance_explained=0.75, n_draws=25000)
    timing_results[f'pca_75_{matrix_name}'] = time.time() - start_time
    
    # Timing for PCA simulation (50%)
    start_time = time.time()
    simulate_pca(cov_matrix, variance_explained=0.50, n_draws=25000)
    timing_results[f'pca_50_{matrix_name}'] = time.time() - start_time

# Print timing results
for key, result in timing_results.items():
    print(f"Runtime for {key}: {result} seconds")

