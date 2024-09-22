#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import pandas as pd
import time


# In[2]:


def near_psd(a, epsilon=0.0):
    n = a.shape[0]
    invSD = None
    out = a.copy()

    # Standardize if covariance matrix (not correlation matrix)
    if np.count_nonzero(np.isclose(np.diag(out), 1.0)) != n:
        invSD = np.diag(1 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD

    # Eigen decomposition
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    
    # Rebuild the matrix with modified eigenvalues
    T = 1 / (vecs * vecs @ vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T

    # Restore variance if necessary
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        out = invSD @ out @ invSD

    return out


# In[3]:


def chol_psd(root, a):
    n = a.shape[0]
    root.fill(0.0)

    for j in range(n):
        s = 0.0
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])

        temp = a[j, j] - s
        if 0 >= temp >= -1e-8:
            temp = 0.0
        root[j, j] = np.sqrt(temp)

        if root[j, j] != 0.0:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir


# In[67]:


def higham_psd(A, max_iters=500, tol=1e-8):
    """
    Higham's 2002 Nearest Positive Semi-Definite Matrix Algorithm.
    Parameters:
        A : np.ndarray
            The input matrix (must be symmetric) that we want to project to a PSD correlation matrix.
        max_iters : int
            Maximum number of iterations.
        tol : float
            Tolerance for stopping criterion.
    Returns:
        X : np.ndarray
            The nearest PSD matrix.
    """
    n = A.shape[0]
    Y = A.copy()
    dS = np.zeros((n, n))
    X = A.copy()

    for k in range(max_iters):
        # Projection onto the space of symmetric matrices
        R = X - dS
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals = np.maximum(eigvals, 0)  # Ensure non-negative eigenvalues
        X_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T  # Reconstruct the PSD matrix

        # Projection onto the space of correlation matrices
        X_psd[np.diag_indices(n)] = 1  # Set diagonal elements to 1 (correlation matrix)
        dS = X_psd - R  # Difference for the next iteration
        X = X_psd

        # Check convergence
        if np.linalg.norm(X - A, ord='fro') < tol:
            break

    return X


# In[86]:


n = 100
sigma = np.full((n, n), 0.9)
np.fill_diagonal(sigma, 1.0)
sigma[0, 1] = sigma[1, 0] = 0.7357


# In[69]:


sigma_near_psd = near_psd(sigma)
sigma_higham_psd = higham_psd(sigma)


# In[84]:


eigenvalues_near_psd = np.linalg.eigvalsh(sigma_near_psd)
eigenvalues_higham_psd = np.linalg.eigvalsh(sigma_higham_psd)
eigenvalues_near_psd_sorted = np.sort(eigenvalues_near_psd)
eigenvalues_higham_psd_sorted = np.sort(eigenvalues_higham_psd)

# Print the smallest 10 eigenvalues
print("Smallest eigenvalues from near_psd:", eigenvalues_near_psd_sorted[:1])
print("Smallest eigenvalues from Higham's method:", eigenvalues_higham_psd_sorted[:1])


# In[73]:


sigma_higham_psd = higham_psd(sigma_higham_psd)
eigenvalues_near_psd = np.linalg.eigvalsh(sigma_near_psd)
eigenvalues_higham_psd = np.linalg.eigvalsh(sigma_higham_psd)
eigenvalues_near_psd_sorted = np.sort(eigenvalues_near_psd)
eigenvalues_higham_psd_sorted = np.sort(eigenvalues_higham_psd)

# Print the smallest 10 eigenvalues
print("Smallest 10 eigenvalues from near_psd:", eigenvalues_near_psd_sorted[:1])
print("Smallest 10 eigenvalues from Higham's method:", eigenvalues_higham_psd_sorted[:1])


# In[75]:


sigma_higham_psd = higham_psd(sigma_higham_psd)
eigenvalues_near_psd = np.linalg.eigvalsh(sigma_near_psd)
eigenvalues_higham_psd = np.linalg.eigvalsh(sigma_higham_psd)
eigenvalues_near_psd_sorted = np.sort(eigenvalues_near_psd)
eigenvalues_higham_psd_sorted = np.sort(eigenvalues_higham_psd)

# Print the smallest 10 eigenvalues
print("Smallest 10 eigenvalues from near_psd:", eigenvalues_near_psd_sorted[:1])
print("Smallest 10 eigenvalues from Higham's method:", eigenvalues_higham_psd_sorted[:1])


# In[76]:


sigma_higham_psd = higham_psd(sigma_higham_psd)
eigenvalues_near_psd = np.linalg.eigvalsh(sigma_near_psd)
eigenvalues_higham_psd = np.linalg.eigvalsh(sigma_higham_psd)
eigenvalues_near_psd_sorted = np.sort(eigenvalues_near_psd)
eigenvalues_higham_psd_sorted = np.sort(eigenvalues_higham_psd)

# Print the smallest 10 eigenvalues
print("Smallest 10 eigenvalues from near_psd:", eigenvalues_near_psd_sorted[:1])
print("Smallest 10 eigenvalues from Higham's method:", eigenvalues_higham_psd_sorted[:1])


# In[77]:


sigma_higham_psd = higham_psd(sigma_higham_psd)
eigenvalues_near_psd = np.linalg.eigvalsh(sigma_near_psd)
eigenvalues_higham_psd = np.linalg.eigvalsh(sigma_higham_psd)
eigenvalues_near_psd_sorted = np.sort(eigenvalues_near_psd)
eigenvalues_higham_psd_sorted = np.sort(eigenvalues_higham_psd)

# Print the smallest 10 eigenvalues
print("Smallest 10 eigenvalues from near_psd:", eigenvalues_near_psd_sorted[:1])
print("Smallest 10 eigenvalues from Higham's method:", eigenvalues_higham_psd_sorted[:1])


# In[78]:


sigma_higham_psd = higham_psd(sigma_higham_psd)
eigenvalues_near_psd = np.linalg.eigvalsh(sigma_near_psd)
eigenvalues_higham_psd = np.linalg.eigvalsh(sigma_higham_psd)
eigenvalues_near_psd_sorted = np.sort(eigenvalues_near_psd)
eigenvalues_higham_psd_sorted = np.sort(eigenvalues_higham_psd)

# Print the smallest 10 eigenvalues
print("Smallest 10 eigenvalues from near_psd:", eigenvalues_near_psd_sorted[:1])
print("Smallest 10 eigenvalues from Higham's method:", eigenvalues_higham_psd_sorted[:1])


# In[79]:


sigma_higham_psd = higham_psd(sigma_higham_psd)
eigenvalues_near_psd = np.linalg.eigvalsh(sigma_near_psd)
eigenvalues_higham_psd = np.linalg.eigvalsh(sigma_higham_psd)
eigenvalues_near_psd_sorted = np.sort(eigenvalues_near_psd)
eigenvalues_higham_psd_sorted = np.sort(eigenvalues_higham_psd)

# Print the smallest 10 eigenvalues
print("Smallest 10 eigenvalues from near_psd:", eigenvalues_near_psd_sorted[:1])
print("Smallest 10 eigenvalues from Higham's method:", eigenvalues_higham_psd_sorted[:1])


# In[80]:


sigma_higham_psd = higham_psd(sigma_higham_psd)
eigenvalues_near_psd = np.linalg.eigvalsh(sigma_near_psd)
eigenvalues_higham_psd = np.linalg.eigvalsh(sigma_higham_psd)
eigenvalues_near_psd_sorted = np.sort(eigenvalues_near_psd)
eigenvalues_higham_psd_sorted = np.sort(eigenvalues_higham_psd)

# Print the smallest 10 eigenvalues
print("Smallest 10 eigenvalues from near_psd:", eigenvalues_near_psd_sorted[:1])
print("Smallest 10 eigenvalues from Higham's method:", eigenvalues_higham_psd_sorted[:1])


# In[81]:


sigma_higham_psd = higham_psd(sigma_higham_psd)
eigenvalues_near_psd = np.linalg.eigvalsh(sigma_near_psd)
eigenvalues_higham_psd = np.linalg.eigvalsh(sigma_higham_psd)
eigenvalues_near_psd_sorted = np.sort(eigenvalues_near_psd)
eigenvalues_higham_psd_sorted = np.sort(eigenvalues_higham_psd)

# Print the smallest 10 eigenvalues
print("Smallest 10 eigenvalues from near_psd:", eigenvalues_near_psd_sorted[:1])
print("Smallest 10 eigenvalues from Higham's method:", eigenvalues_higham_psd_sorted[:1])


# In[85]:


sigma_higham_psd = higham_psd(sigma_higham_psd)
eigenvalues_near_psd = np.linalg.eigvalsh(sigma_near_psd)
eigenvalues_higham_psd = np.linalg.eigvalsh(sigma_higham_psd)
eigenvalues_near_psd_sorted = np.sort(eigenvalues_near_psd)
eigenvalues_higham_psd_sorted = np.sort(eigenvalues_higham_psd)

# Print the smallest 10 eigenvalues
print("Smallest eigenvalues from near_psd:", eigenvalues_near_psd_sorted[:1])
print("Smallest eigenvalues from Higham's method:", eigenvalues_higham_psd_sorted[:1])


# In[87]:


start_time = time.time()
sigma_near_psd = near_psd(sigma)
near_psd_time = time.time() - start_time

frobenius_near_psd = np.linalg.norm(sigma_near_psd - sigma, 'fro')

# Measure runtime and Frobenius norm for Higham's method
start_time = time.time()
sigma_higham_psd = higham_psd(sigma)
higham_psd_time = time.time() - start_time

frobenius_higham_psd = np.linalg.norm(sigma_higham_psd - sigma, 'fro')

# Print comparison
print(f"near_psd() runtime: {near_psd_time:.6f} seconds, Frobenius norm: {frobenius_near_psd:.6f}")
print(f"Higham's method runtime: {higham_psd_time:.6f} seconds, Frobenius norm: {frobenius_higham_psd:.6f}")


# In[ ]:




