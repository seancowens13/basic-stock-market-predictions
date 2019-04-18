import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('sphist.csv')

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date', ascending=True)

df['day_5'] = df['Close'].rolling(5).mean().shift()
df['day_30'] = df['Close'].rolling(30).mean().shift()
df['day_365'] = df['Close'].rolling(365).mean().shift()

df = df.dropna(axis=0)

train = df[df['Date'] < datetime(year=2013, month=1, day=1)]
test = df[df['Date'] > datetime(year=2013, month=1, day=1)]

model = LinearRegression()
model.fit(train[['day_5', 'day_30', 'day_365']], train[['Close']])
predictions = model.predict(test[['day_5', 'day_30', 'day_365']])

mse = mean_squared_error(predictions, test['Close'])

print(mse)