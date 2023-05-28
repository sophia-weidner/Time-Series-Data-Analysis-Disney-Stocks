#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[46]:


df = pd.read_csv('/Users/sophiaweidner/Downloads/dis_stocks.csv')


# In[47]:


df.head(10)


# In[48]:


# Plotting time series data, using Closing stock prices, for DIS from 03/2020-05/2023.
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Close'], label='Stock Price (USD)')
plt.xlabel('Date')
plt.ylabel('Closing DIS Stock Price')
plt.title('Time Series Data Graph')
plt.legend()
plt.grid(True)
plt.show()


# In[49]:


# Creating graph for when Chapek was announced as CEO in February 2020.

# Filtering data for March 2020 - April 2020
start_date = '2020-03-02'
end_date = '2020-04-30'
chapek = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# Limiting x-labels for graph readability.
num_labels_1 = 3

total_data_points_1 = len(chapek['Date'])
step_size_1 = total_data_points_1 // num_labels_1

# Create the line plot
plt.plot(chapek['Date'], chapek['Close'])
plt.xlabel('Date')
plt.ylabel('Stock Value at Close')
plt.title('Stock Values from 03/2020 - 04/2020, Chapek Announced as CEO')
plt.xticks(chapek['Date'][::step_size_1], rotation=45)
plt.grid(True)
plt.show()


# In[50]:


# Creating graph for when Chapek was removed as CEO on November 20th, 2022 and Iger was announced as his replacement.

# Filtering data for November 2022 - January 2023
start_date_2 = '2022-11-01'
end_date_2 = '2023-01-31'
iger = df[(df['Date'] >= start_date_2) & (df['Date'] <= end_date_2)]

# Limiting x-labels for graph readability.
num_labels_2 = 3

total_data_points_2 = len(iger['Date'])
step_size_2 = total_data_points_2 // num_labels_2

# Create the line plot
plt.plot(iger['Date'], iger['Close'])
plt.xlabel('Date')
plt.ylabel('Stock Value at Close')
plt.title('Stock Values from 11/2022 - 01/2023, Iger Announced as CEO')
plt.xticks(iger['Date'][::step_size_2], rotation=45)
plt.grid(True)
plt.show()


# In[51]:


# Creating graph for when Disney Genie+ was launched at Walt Disney World.

# Filtering data for October 18th, 2021 - November 2021
start_date_3 = '2021-10-18'
end_date_3 = '2021-11-30'
genie = df[(df['Date'] >= start_date_3) & (df['Date'] <= end_date_3)]

# Limiting x-labels for graph readability.
num_labels_3 = 5

total_data_points_3 = len(genie['Date'])
step_size_3 = total_data_points_3 // num_labels_3

# Create the line plot
plt.plot(genie['Date'], genie['Close'])
plt.xlabel('Date')
plt.ylabel('Stock Value at Close')
plt.title('Stock Values from 10/2021 - 11/2021, Genie+ Launched at WDW')
plt.xticks(genie['Date'][::step_size_2], rotation=45)
plt.grid(True)
plt.show()


# In[52]:


# Creating graph for when Disney+ and Hulu "One-App Experience" was announced.

# Filtering data for May 1st, 2023 through May 24th, 2023
start_date_4 = '2023-05-01'
end_date_4 = '2023-05-24'
stream = df[(df['Date'] >= start_date_4) & (df['Date'] <= end_date_4)]

# Create the line plot
plt.plot(stream['Date'], stream['Close'])
plt.xlabel('Date')
plt.ylabel('Stock Value at Close')
plt.title('Stock Values from 05/01/2023 - 05/24/2023, Disney+ and Hulu Announcement')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# ### Time Series Data Analysis

# In[53]:


import statsmodels.api as sm


# In[54]:


# Converting 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' column as the index
df.set_index('Date', inplace=True)


# In[56]:


# Checking current frequency of the index. 
print(df.index.freq)  # Output: None

# Resampling data to frequency "Daily"
df = df.asfreq('D')


# In[57]:


df.head(10)


# In[59]:


# After changing the frequency to "Daily", we now have NaN values. Checking how many rows have NaN values.
print(df.isna().sum())


# In[60]:


# Checking the Mean and replacing NaN values with the Mean.
mean_value = df.mean()
print(mean_value)


# In[61]:


# Replace NaN values with mean
df_new = df.fillna(mean_value)


# In[62]:


df_new.head(10)


# In[63]:


# Decomposing DIS data into trend, seasonality, and residuals.
decomposition = sm.tsa.seasonal_decompose(df_new['Close'], model='additive')
trend = decomposition.trend
seasonality = decomposition.seasonal
residuals = decomposition.resid


# In[64]:


# Plotting decomposed components
plt.figure(figsize=(10, 8))
plt.subplot(411)
plt.plot(df_new.index, df_new['Close'], label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(df_new.index, trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(df_new.index, seasonality, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(df_new.index, residuals, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# In[65]:


# Fitting ARIMA model.
model = sm.tsa.ARIMA(df_new['Close'], order=(1, 1, 1))  # Specify the order of the ARIMA model
results = model.fit()


# In[66]:


# Printing results.
print(results.summary())


# In[ ]:




