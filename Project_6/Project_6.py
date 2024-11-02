#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from datetime import datetime
from scipy.optimize import minimize_scalar
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA


# # Problem 1

# Assume you a call and a put option with the following
# ● Current Stock Price $165
# ● Current Date 03/03/2023
# ● Options Expiration Date 03/17/2023
# ● Risk Free Rate of 5.25%
# ● Continuously Compounding Coupon of 0.53%
# Calculate the time to maturity using calendar days (not trading days).
# For a range of implied volatilities between 10% and 80%, plot the value of the call and the put. Discuss these graphs. How does the supply and demand affect the implied volatility?

# In[4]:


#time to maturity
ttm = round(14 / 365, 4)
print('ttm is: ', ttm)


# In[5]:


def black_scholes_option_value(S, K, T, r, q, vol_range):
    
    def call_price(S, K, T, r, q, sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    def put_price(S, K, T, r, q, sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    call_prices = [call_price(S, K, T, r, q, sigma) for sigma in vol_range]
    put_prices = [put_price(S, K, T, r, q, sigma) for sigma in vol_range]

    return call_prices, put_prices


# In[14]:


S = 165                
K = 165              
T = ttm            
r = 0.0525             
q = 0.0053 
implied_vols = np.linspace(0.1, 0.8, 100) 
call_prices, put_prices = black_scholes_option_value(S, K, T, r, q, implied_vols)


# In[15]:


plt.figure(figsize=(12, 6))
plt.plot(implied_vols, call_prices, label="Call Option Price", color="blue")
plt.plot(implied_vols, put_prices, label="Put Option Price", color="orange")
plt.xlabel("Implied Volatility")
plt.ylabel("Option Price")
plt.title("Option Prices vs Implied Volatility")
plt.legend()
plt.grid(True)
plt.show()


# # Problem 2

# Use the options found in AAPL_Options.csv
# ● Current AAPL price is 170.15
# ● Current Date: 10/30/2023
# ● Risk Free Rate: 5.25%
# ● Dividend Rate: 0.57%.
# Calculate the implied volatility for each option.
# Plot the implied volatility vs the strike price for Puts and Calls. Discuss the shape of these graphs. What market dynamics could make these graphs?

# In[16]:


option = pd.read_csv('AAPL_Options.csv')


# In[17]:


option


# In[45]:


def black_scholes_price(S, K, T, r, q, sigma, option_type):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    
    


# In[67]:


def implied_volatility(target_price, S, K, T, r, q, option_type):
    def objective_function(sigma):
        return black_scholes_price(S, K, T, r, q, sigma, option_type) - target_price
    try:
        return brentq(objective_function, 1e-10, 20.0)
    except ValueError:
        print(f"All attempts failed for target price: {target_price}, strike: {K}. Using approximate solution.")
            
        result = minimize_scalar(
            lambda sigma: abs(objective_function(sigma)),
            bounds=(1e-6, 20.0),
            method='bounded'
        )
        return result.x if result.success else np.nan
    
    
    


# In[106]:


S = 170.15 
current_date = datetime(2023, 10, 30)
r = 0.0525  
q = 0.0057

implied_vols = []
for index, row in option.iterrows():
    K = row['Strike']
    last_price = row['Last Price']
    option_type = row['Type'].lower()
    
    expiration_date = datetime.strptime(row['Expiration'], "%m/%d/%Y")
    T = (expiration_date - current_date).days / 365 
    imp_vol = implied_volatility(last_price, S, K, T, r, q, option_type)
    implied_vols.append(imp_vol)
    
implied_vols


# In[107]:


strike_prices = [140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190]

# Sample implied volatilities for calls and puts (replace these with actual values from your calculations)
call_implied_vols = [0.0414, 0.3427, 0.3743, 0.3456, 0.3141, 0.2994, 0.2759, 0.2612, 0.2462, 0.2353, 0.2295]
put_implied_vols = [0.3784, 0.3606, 0.3420, 0.3240, 0.3101, 0.2943, 0.2801, 0.2710, 0.2640, 0.2443, 0.2865]

# Plot Implied Volatility vs Strike Price for Calls
plt.figure(figsize=(12, 6))
plt.plot(strike_prices, call_implied_vols, label="Calls", marker='o', linestyle='-', color="blue")
plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")
plt.title("Implied Volatility vs Strike Price (Calls)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Implied Volatility vs Strike Price for Puts
plt.figure(figsize=(12, 6))
plt.plot(strike_prices, put_implied_vols, label="Puts", marker='o', linestyle='-', color="orange")
plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")
plt.title("Implied Volatility vs Strike Price (Puts)")
plt.legend()
plt.grid(True)
plt.show()


# # Problem 3 (1)

# In[76]:


portfolios_df = pd.read_csv('problem3.csv')
portfolios_df


# In[83]:


current_price = 170.15  
risk_free_rate = 5.25 / 100  
dividend_rate = 0.57 / 100 
current_date = datetime.strptime("10/30/2023", "%m/%d/%Y")
expiration_date = datetime.strptime("12/15/2023", "%m/%d/%Y")
days_to_expiration = (expiration_date - current_date).days
T = days_to_expiration / 365 


# In[84]:


def black_scholes_price(S, K, T, r, q, sigma, option_type):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        return np.nan  # Handle unexpected option types

# Define function to calculate implied volatility
def implied_volatility(target_price, S, K, T, r, q, option_type):
    def objective_function(sigma):
        return black_scholes_price(S, K, T, r, q, sigma, option_type) - target_price
    
    try:
        # Use Brent's method to find implied volatility
        return brentq(objective_function, 1e-10, 20.0)
    except ValueError:
        # If brentq fails, fall back to minimize_scalar for approximate solution
        result = minimize_scalar(
            lambda sigma: abs(objective_function(sigma)),
            bounds=(1e-6, 20.0),
            method='bounded'
        )
        return result.x if result.success else np.nan


# In[85]:


portfolios_df['ImpliedVolatility'] = portfolios_df.apply(
    lambda row: implied_volatility(
        row['CurrentPrice'], 
        current_price, 
        row['Strike'], 
        T, 
        risk_free_rate, 
        dividend_rate, 
        row['OptionType']
    ) if pd.notna(row['OptionType']) else np.nan,
    axis=1
)


# In[86]:


price_range = np.linspace(100, 250, 100)  

portfolio_values_over_range = {portfolio: [] for portfolio in portfolios_df['Portfolio'].unique()}

for S in price_range:
    portfolios_df['PositionValue'] = portfolios_df.apply(
        lambda row: S * row['Holding'] if row['Type'] == 'Stock' else (
            black_scholes_price(
                S=S, 
                K=row['Strike'], 
                T=T, 
                r=risk_free_rate, 
                q=dividend_rate, 
                sigma=row['ImpliedVolatility'], 
                option_type=row['OptionType']
            ) * row['Holding'] if row['Type'] == 'Option' else np.nan
        ),
        axis=1
    )
    
    portfolio_values_at_price = portfolios_df.groupby('Portfolio')['PositionValue'].sum()
    
    for portfolio in portfolio_values_at_price.index:
        portfolio_values_over_range[portfolio].append(portfolio_values_at_price[portfolio])
plt.figure(figsize=(12, 8))
for portfolio, values in portfolio_values_over_range.items():
    plt.plot(price_range, values, label=portfolio)

plt.title("Portfolio Values Over Range of AAPL Prices")
plt.xlabel("AAPL Stock Price")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True)
plt.show()


# # Problem 3 (2)

# In[12]:


data = pd.read_csv('DailyPrices.csv')
data


# In[13]:


aapl_log_returns = np.log(data['AAPL'] / data['AAPL'].shift(1)).dropna()
aapl_log_returns_demeaned = aapl_log_returns - aapl_log_returns.mean()
aapl_log_returns_demeaned


# In[25]:


model = ARIMA(aapl_log_returns_demeaned, order=(1, 0, 0))
ar1_fit = model.fit()


simulated_returns = ar1_fit.simulate(nsimulations=10)


last_price = 170.15


simulated_prices = last_price * np.exp(np.cumsum(simulated_returns))


mean_return = simulated_returns.mean()
var_95 = np.percentile(simulated_returns, 5)  
es_95 = simulated_returns[simulated_returns <= var_95].mean() 

print("Mean Return:", mean_return)
print("Value at Risk (VaR) at 95%:", -var_95 * 170.15)
print("Expected Shortfall (ES) at 95%:",-es_95 * 170.15)


# In[ ]:





# In[ ]:




