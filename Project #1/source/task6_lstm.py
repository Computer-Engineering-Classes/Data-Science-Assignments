# Task 6 - Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

# United States solar data
energy_data = pd.read_csv('filtered-energy-data.csv')
countries = ['United States', 'Canada', 'Brazil', 'Mexico']
solar_data: pd.DataFrame = energy_data.loc[
    (energy_data.country == countries[0]) & (energy_data.solar_consumption.notnull()),
    ['year', 'solar_consumption']]

solar_data.set_index('year', inplace=True)

y = solar_data['solar_consumption'].fillna(method='ffill')
y = y.to_numpy().reshape(-1, 1)

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(y)
y = scaler.transform(y)

# generate the input and output sequences
n_lookback = 3  # length of input sequences (lookback period)
n_forecast = 1  # length of output sequences (forecast period)

X = []
Y = []

for i in range(n_lookback, len(y) - n_forecast + 1):
    X.append(y[i - n_lookback: i])
    Y.append(y[i: i + n_forecast])

X = np.array(X)
Y = np.array(Y)

# fit the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
model.add(LSTM(units=50))
model.add(Dense(n_forecast))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=500, batch_size=50)

# generate the forecasts
X_ = y[- n_lookback:]  # last available input sequence
X_ = X_.reshape(1, n_lookback, 1)
Y_ = model.predict(X_).reshape(-1, 1)
Y_ = scaler.inverse_transform(Y_)

# organize the results in a data frame
df_past = solar_data[['solar_consumption']].reset_index()
df_past.rename(columns={'index': 'year', 'solar_consumption': 'Actual'}, inplace=True)
df_past['year'] = pd.date_range(start=str(solar_data.index[0]),
                                periods=len(solar_data), freq='AS')
df_past['Forecast'] = np.nan
df_past.at[df_past.index[-1], 'Forecast'] = df_past.at[df_past.index[-1], 'Actual']

df_future = pd.DataFrame(columns=['year', 'Actual', 'Forecast'])
df_future['year'] = pd.date_range(start=df_past.at[df_past.index[-1], 'year'] + pd.DateOffset(months=12),
                                  periods=n_forecast, freq='AS')
df_future['Forecast'] = Y_.flatten()
df_future['Actual'] = np.nan
results = pd.concat([df_past, df_future]).set_index('year')
# plot the results
results.plot(title=f'Solar consumption of {countries[0]}')
plt.show()

# Prediction for next year
print(f'Prediction for {results.index[-1].year}: {results.at[results.index[-1], "Forecast"]}')
