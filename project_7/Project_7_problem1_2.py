#!/usr/bin/env python
# coding: utf-8

# In[101]:


import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from datetime import datetime
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
import yfinance as yf
get_ipython().system('pip install cvxpy')
import cvxpy as cp


# # Problem 1

# In[12]:


S0 = 151.03               
K = 165                  、
T = (33 / 365)            
r = 0.0425               
q = 0.0053               
div_date = "2022-04-11"  
div_amount = 0.88        
sigma = 0.2          


# In[6]:


def d1(S, K, T, r, sigma, q):
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma, q):
    return d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


# In[7]:


def black_scholes_call(S, K, T, r, sigma, q):
    d1_val = d1(S, K, T, r, sigma, q)
    d2_val = d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)

def black_scholes_put(S, K, T, r, sigma, q):
    d1_val = d1(S, K, T, r, sigma, q)
    d2_val = d2(S, K, T, r, sigma, q)
    return K * np.exp(-r * T) * norm.cdf(-d2_val) - S * np.exp(-q * T) * norm.cdf(-d1_val)


# In[8]:


def delta(S, K, T, r, sigma, q, option_type="call"):
    d1_val = d1(S, K, T, r, sigma, q)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1_val)
    else:
        return -np.exp(-q * T) * norm.cdf(-d1_val)

def gamma(S, K, T, r, sigma, q):
    d1_val = d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1_val) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma, q):
    d1_val = d1(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1_val) * np.sqrt(T)

def theta(S, K, T, r, sigma, q, option_type="call"):
    d1_val = d1(S, K, T, r, sigma, q)
    d2_val = d2(S, K, T, r, sigma, q)
    term1 = -S * np.exp(-q * T) * norm.pdf(d1_val) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        term2 = q * S * np.exp(-q * T) * norm.cdf(d1_val)
        term3 = r * K * np.exp(-r * T) * norm.cdf(d2_val)
        return term1 - term2 - term3
    else:
        term2 = q * S * np.exp(-q * T) * norm.cdf(-d1_val)
        term3 = r * K * np.exp(-r * T) * norm.cdf(-d2_val)
        return term1 + term2 + term3

def rho(S, K, T, r, sigma, q, option_type="call"):
    d2_val = d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2_val)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2_val)


# In[9]:


def finite_difference_delta(S, K, T, r, sigma, q, epsilon=0.01, option_type="call"):
    price_up = black_scholes_call(S + epsilon, K, T, r, sigma, q) if option_type == "call" else black_scholes_put(S + epsilon, K, T, r, sigma, q)
    price_down = black_scholes_call(S - epsilon, K, T, r, sigma, q) if option_type == "call" else black_scholes_put(S - epsilon, K, T, r, sigma, q)
    return (price_up - price_down) / (2 * epsilon)

def finite_difference_gamma(S, K, T, r, sigma, q, epsilon=0.01, option_type="call"):
    price_up = black_scholes_call(S + epsilon, K, T, r, sigma, q) if option_type == "call" else black_scholes_put(S + epsilon, K, T, r, sigma, q)
    price = black_scholes_call(S, K, T, r, sigma, q) if option_type == "call" else black_scholes_put(S, K, T, r, sigma, q)
    price_down = black_scholes_call(S - epsilon, K, T, r, sigma, q) if option_type == "call" else black_scholes_put(S - epsilon, K, T, r, sigma, q)
    return (price_up - 2 * price + price_down) / (epsilon**2)

def finite_difference_vega(S, K, T, r, sigma, q, epsilon=0.01, option_type="call"):
    price_up = black_scholes_call(S, K, T, r, sigma + epsilon, q) if option_type == "call" else black_scholes_put(S, K, T, r, sigma + epsilon, q)
    price_down = black_scholes_call(S, K, T, r, sigma - epsilon, q) if option_type == "call" else black_scholes_put(S, K, T, r, sigma - epsilon, q)
    return (price_up - price_down) / (2 * epsilon)

def finite_difference_theta(S, K, T, r, sigma, q, epsilon=1/365, option_type="call"):
    price = black_scholes_call(S, K, T, r, sigma, q) if option_type == "call" else black_scholes_put(S, K, T, r, sigma, q)
    price_down = black_scholes_call(S, K, T - epsilon, r, sigma, q) if option_type == "call" else black_scholes_put(S, K, T - epsilon, r, sigma, q)
    return (price_down - price) / epsilon

def finite_difference_rho(S, K, T, r, sigma, q, epsilon=0.0001, option_type="call"):
    price_up = black_scholes_call(S, K, T, r + epsilon, sigma, q) if option_type == "call" else black_scholes_put(S, K, T, r + epsilon, sigma, q)
    price_down = black_scholes_call(S, K, T, r - epsilon, sigma, q) if option_type == "call" else black_scholes_put(S, K, T, r - epsilon, sigma, q)
    return (price_up - price_down) / (2 * epsilon)


# In[18]:


def binomial_tree_american_option(S, K, T, r, sigma, N, option_type="call", dividend_amount=0.0, dividend_date=None):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - 0) * dt) - d) / (u - d)  

    
    stock_tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            stock_tree[j, i] = S * (u ** (i - j)) * (d ** j)
            if dividend_date and (i * dt) >= ((33 - 4) / 365):  
                stock_tree[j, i] -= dividend_amount * np.exp(-r * ((33 - 4) / 365))

   
    option_tree = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        if option_type == "call":
            option_tree[j, N] = max(0, stock_tree[j, N] - K)
        else:
            option_tree[j, N] = max(0, K - stock_tree[j, N])

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            hold = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
            exercise = max(0, (stock_tree[j, i] - K) if option_type == "call" else (K - stock_tree[j, i]))
            option_tree[j, i] = max(hold, exercise)

    return option_tree[0, 0]


# In[19]:


def binomial_tree_delta(S, K, T, r, sigma, N, option_type, h=1.0):
    price_up = binomial_tree_american_option(S + h, K, T, r, sigma, N, option_type)
    price_down = binomial_tree_american_option(S - h, K, T, r, sigma, N, option_type)
    return (price_up - price_down) / (2 * h)


def binomial_tree_gamma(S, K, T, r, sigma, N, option_type, h=1.0):
    price_up = binomial_tree_american_option(S + h, K, T, r, sigma, N, option_type)
    price_down = binomial_tree_american_option(S - h, K, T, r, sigma, N, option_type)
    price = binomial_tree_american_option(S, K, T, r, sigma, N, option_type)
    return (price_up - 2 * price + price_down) / (h ** 2)

def binomial_tree_theta(S, K, T, r, sigma, N, option_type, h=1/365):
    price_now = binomial_tree_american_option(S, K, T, r, sigma, N, option_type)
    price_later = binomial_tree_american_option(S, K, T - h, r, sigma, N, option_type)
    return (price_later - price_now) / h


def binomial_tree_rho(S, K, T, r, sigma, N, option_type, h=0.0001):
    price_up = binomial_tree_american_option(S, K, T, r + h, sigma, N, option_type)
    price_down = binomial_tree_american_option(S, K, T, r - h, sigma, N, option_type)
    return (price_up - price_down) / (2 * h)


N = 100 


call_price_binomial = binomial_tree_american_option(S0, K, T, r, sigma, N, "call", div_amount, div_date)
put_price_binomial = binomial_tree_american_option(S0, K, T, r, sigma, N, "put", div_amount, div_date)


call_delta = binomial_tree_delta(S0, K, T, r, sigma, N, "call")
put_delta = binomial_tree_delta(S0, K, T, r, sigma, N, "put")

call_gamma = binomial_tree_gamma(S0, K, T, r, sigma, N, "call")
put_gamma = binomial_tree_gamma(S0, K, T, r, sigma, N, "put")

call_theta = binomial_tree_theta(S0, K, T, r, sigma, N, "call")
put_theta = binomial_tree_theta(S0, K, T, r, sigma, N, "put")

call_rho = binomial_tree_rho(S0, K, T, r, sigma, N, "call")
put_rho = binomial_tree_rho(S0, K, T, r, sigma, N, "put")


print("\nBinomial Tree American Option Prices (with dividend):")
print(f"Call Price: {call_price_binomial:.4f}")
print(f"Put Price: {put_price_binomial:.4f}")

print("\nBinomial Tree Greeks for American Options:")
print(f"Delta (Call): {call_delta:.4f}, Delta (Put): {put_delta:.4f}")
print(f"Gamma (Call): {call_gamma:.4f}, Gamma (Put): {put_gamma:.4f}")
print(f"Theta (Call): {call_theta:.4f}, Theta (Put): {put_theta:.4f}")
print(f"Rho (Call): {call_rho:.4f}, Rho (Put): {put_rho:.4f}")


# In[23]:


for new_div in [0.5, 0.88, 1.2]:
    call_price = binomial_tree_american_option(S0, K, T, r, sigma, N, "call", new_div, div_date)
    put_price = binomial_tree_american_option(S0, K, T, r, sigma, N, "put", new_div, div_date)
    print(f"Dividend: {new_div:.2f}, Call Price: {call_price:.4f}, Put Price: {put_price:.4f}")


# In[15]:


for div, call_price, put_price in dividend_sensitivities:
    print(f"Dividend: {div}, Call Price: {call_price:.2f}, Put Price: {put_price:.2f}")

print("\nBlack-Scholes Greeks (Closed-form):")
print(f"Delta (call): {delta(S0, K, T, r, sigma, q, 'call'):.4f}")
print(f"Delta (put): {delta(S0, K, T, r, sigma, q, 'put'):.4f}")
print(f"Gamma: {gamma(S0, K, T, r, sigma, q):.4f}")
print(f"Vega: {vega(S0, K, T, r, sigma, q):.4f}")
print(f"Theta (call): {theta(S0, K, T, r, sigma, q, 'call'):.4f}")
print(f"Theta (put): {theta(S0, K, T, r, sigma, q, 'put'):.4f}")
print(f"Rho (call): {rho(S0, K, T, r, sigma, q, 'call'):.4f}")
print(f"Rho (put): {rho(S0, K, T, r, sigma, q, 'put'):.4f}")


# In[27]:


epsilon = 0.01  

delta_fd_call = finite_difference_delta(S0, K, T, r, sigma, q, epsilon, "call")
delta_fd_put = finite_difference_delta(S0, K, T, r, sigma, q, epsilon, "put")
gamma_fd = finite_difference_gamma(S0, K, T, r, sigma, q, epsilon)
vega_fd = finite_difference_vega(S0, K, T, r, sigma, q, epsilon)
theta_fd_call = finite_difference_theta(S0, K, T, r, sigma, q, 1/365, "call")
theta_fd_put = finite_difference_theta(S0, K, T, r, sigma, q, 1/365, "put")
rho_fd_call = finite_difference_rho(S0, K, T, r, sigma, q, 0.0001, "call")
rho_fd_put = finite_difference_rho(S0, K, T, r, sigma, q, 0.0001, "put")

print("\nComparison of Closed-form and Finite Difference Greeks:")
print(f"Delta - Finite Difference(call): {delta_fd_call:.4f}")
print(f"Delta - Finite Difference(put): {delta_fd_put:.4f}")
print(f"Gamma - Closed-form: {gamma(S0, K, T, r, sigma, q):.4f}, Finite Difference: {gamma_fd:.4f}")
print(f"Vega - Closed-form: {vega(S0, K, T, r, sigma, q):.4f}, Finite Difference: {vega_fd:.4f}")
print(f"Theta - Finite Difference(call): {theta_fd_call:.4f}")
print(f"Theta - Finite Difference(put): {theta_fd_put:.4f}")
print(f"Rho - Finite Difference(call): {rho_fd_call:.4f}")
print(f"Rho - Finite Difference(put): {rho_fd_put:.4f}")


# # Problem 2

# In[28]:


daily_prices_df = pd.read_csv('DailyPrices-2.csv')
options_portfolio_df = pd.read_csv('problem2.csv')


# In[111]:


current_price = 165  
risk_free_rate = 0.0425  
dividend_amount = 1.0  
dividend_day = 8  
days_ahead = 10 
simulations = 10000  


# In[112]:


daily_prices_df['AAPL_Return'] = daily_prices_df['AAPL'].pct_change()
aapl_returns = daily_prices_df['AAPL_Return'].dropna()
std_dev = np.std(aapl_returns)  

daily_returns = np.random.normal(0, std_dev, (simulations, days_ahead))
simulated_prices = np.zeros_like(daily_returns)
simulated_prices[:, 0] = current_price * (1 + daily_returns[:, 0])
for day in range(1, days_ahead):
    simulated_prices[:, day] = simulated_prices[:, day - 1] * (1 + daily_returns[:, day])
simulated_prices[:, dividend_day:] -= dividend_amount
ending_prices = simulated_prices[:, -1]


# In[37]:


def construct_binomial_tree(price, days, up_factor, down_factor, dividend_day, dividend_amount):
    tree = [[price]]
    for day in range(1, days + 1):
        prev_prices = tree[-1]
        current_prices = []
        for prev_price in prev_prices:
            adjusted_price = prev_price - dividend_amount if day == dividend_day else prev_price
            current_prices.append(adjusted_price * up_factor) 
            current_prices.append(adjusted_price * down_factor)  
        tree.append(current_prices)
    return tree


def american_option_value(price_tree, strike, option_type='call'):
    option_tree = []
    for final_price in price_tree[-1]:
        if option_type == 'call':
            option_tree.append(max(final_price - strike, 0))
        elif option_type == 'put':
            option_tree.append(max(strike - final_price, 0))
    
    discount_factor = np.exp(-risk_free_rate / 252)
    for day in range(len(price_tree) - 2, -1, -1):
        new_option_tree = []
        for i in range(0, len(price_tree[day])):
            intrinsic_value = max(price_tree[day][i] - strike, 0) if option_type == 'call' else max(strike - price_tree[day][i], 0)
            continuation_value = discount_factor * (0.5 * option_tree[2 * i] + 0.5 * option_tree[2 * i + 1])
            new_option_tree.append(max(intrinsic_value, continuation_value))
        option_tree = new_option_tree
    return option_tree[0]


# In[113]:


def calculate_portfolio_value(price, portfolio, days):
    up_factor = np.exp(std_dev)
    down_factor = 1 / up_factor
    binomial_tree = construct_binomial_tree(price, days, up_factor, down_factor, dividend_day, dividend_amount)
    
    portfolio_value = 0
    for _, row in portfolio.iterrows():
        if row['Type'] == 'Option':
            option_value = american_option_value(binomial_tree, row['Strike'], option_type=row['OptionType'].lower())
            portfolio_value += row['Holding'] * option_value
        elif row['Type'] == 'Stock':
            portfolio_value += row['Holding'] * binomial_tree[-1][0]
    return portfolio_value

portfolio_var_es = {}

for portfolio_name, portfolio_data in options_portfolio_df.groupby('Portfolio'):
    portfolio_values = []
    for final_price in ending_prices:
        portfolio_value = calculate_portfolio_value(final_price, portfolio_data, days_ahead)
        portfolio_values.append(portfolio_value)
    
    current_portfolio_value = calculate_portfolio_value(current_price, portfolio_data, days_ahead)
    
    mean_value = np.mean(portfolio_values)

    mean_return = (mean_value - current_portfolio_value) / current_portfolio_value
    
    pnl = [current_portfolio_value - value for value in portfolio_values]
    var_95 = np.percentile(pnl, 95)  
    es_95 = np.mean([loss for loss in pnl if loss >= var_95])  
    
    portfolio_var_es[portfolio_name] = {
        'Current Value': current_portfolio_value,
        'Mean Return': mean_return,
        'VaR (95%)': var_95,
        'ES (95%)': es_95
    }

for portfolio_name, metrics in portfolio_var_es.items():
    print(f"Portfolio: {portfolio_name}")
    print(f"Current Value: ${metrics['Current Value']:.2f}")
    print(f"Mean Return: {metrics['Mean Return']:.2%}")  
    print(f"VaR (95%): ${metrics['VaR (95%)']:.2f}")
    print(f"ES (95%): ${metrics['ES (95%)']:.2f}")
    print("\n")


# In[115]:


portfolio_metrics_df = pd.DataFrame([
    {
        "Portfolio": portfolio_name,
        "Current Value": f"${metrics['Current Value']:.2f}",
        "Mean Return": f"{metrics['Mean Return']:.2%}",
        "VaR (95%)": f"${metrics['VaR (95%)']:.2f}",
        "ES (95%)": f"${metrics['ES (95%)']:.2f}"
    }
    for portfolio_name, metrics in portfolio_var_es.items()
])

# Display the DataFrame
portfolio_metrics_df


# In[114]:


daily_prices_df['AAPL_Return'] = daily_prices_df['AAPL'].pct_change()
aapl_returns = daily_prices_df['AAPL_Return'].dropna()
std_dev = np.std(aapl_returns)

def calculate_portfolio_delta(portfolio, current_price):
    portfolio_delta = 0
    for _, row in portfolio.iterrows():
        if row['Type'] == 'Option':
            delta = 0.5 if row['OptionType'].lower() == 'call' else -0.5
            portfolio_delta += row['Holding'] * delta
        elif row['Type'] == 'Stock':
            portfolio_delta += row['Holding']  
    return portfolio_delta * current_price

delta_normal_results = {}

for portfolio_name, portfolio_data in options_portfolio_df.groupby('Portfolio'):
    portfolio_delta = calculate_portfolio_delta(portfolio_data, current_price)
    portfolio_std = abs(portfolio_delta) * std_dev * current_price

    # VaR and ES (95% confidence level)
    var_95_delta_normal = 1.65 * portfolio_std
    es_95_delta_normal = 2.06 * portfolio_std
    
    delta_normal_results[portfolio_name] = {
        'Portfolio Delta': portfolio_delta,
        'Delta-Normal VaR (95%)': var_95_delta_normal,
        'Delta-Normal ES (95%)': es_95_delta_normal
    }

# Output the results
for portfolio_name, metrics in delta_normal_results.items():
    print(f"Portfolio: {portfolio_name}")
    print(f"Portfolio Delta: {metrics['Portfolio Delta']}")
    print(f"Delta-Normal VaR (95%): ${metrics['Delta-Normal VaR (95%)']:.2f}")
    print(f"Delta-Normal ES (95%): ${metrics['Delta-Normal ES (95%)']:.2f}")
    print("\n")


# In[120]:


portfolio_metrics_df = pd.DataFrame([
    {
        print(f"Portfolio: {portfolio_name}"),
        print(f"Portfolio Delta: {metrics['Portfolio Delta']}"),
        print(f"Delta-Normal VaR (95%): ${metrics['Delta-Normal VaR (95%)']:.2f}"),
        print(f"Delta-Normal ES (95%): ${metrics['Delta-Normal ES (95%)']:.2f}"),
        print("\n")
    }
    for portfolio_name, metrics in delta_normal_results.items()
])

# Display the DataFrame
portfolio_metrics_df


# In[ ]:




