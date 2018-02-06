#coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import fbprophet 

df = pd.read_csv('2330_v1.csv', header=0, sep='\t', delimiter='\t', encoding = 'utf-8')
df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
df.set_index('date')

# The adjusted close accounts for stock splits, so that is what we should graph
plt.plot(df.index, df['close'])
plt.title('TW 2330 Stock Price')
plt.ylabel('Price ($)');
plt.show()

df.columns = ['ds','y']
df.tail()

df_new = df.iloc[:,:]
m = fbprophet.Prophet(changepoint_prior_scale=0.95)
m.fit(df_new)
future = m.make_future_dataframe(periods = 90)
forecast = m.predict(future)
m.plot(forecast)
#m.plot_components(forecast)
plt.show()

