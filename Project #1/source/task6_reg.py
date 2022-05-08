# Task 6 - Using simple regression models
import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt


def mono_exp(x, m, t, b):
    return m * np.exp(t * x) + b


energy_data = pd.read_csv('filtered-energy-data.csv')
countries = ['United States', 'Canada', 'Brazil', 'Mexico']
solar_data: pd.DataFrame = energy_data.loc[
    (energy_data.country == countries[0]) & (energy_data.solar_consumption.notnull()),
    ['year', 'solar_consumption']]

xs = solar_data['year'].to_numpy()
ys = solar_data['solar_consumption'].to_numpy()

xs_shifted = xs - xs[0]

# perform the fit
p0 = (1, 1e-6, 0)  # start with values near those we expect
params, cv = scipy.optimize.curve_fit(mono_exp, xs_shifted, ys, p0)
m, t, b = params
sampleRate = 20_000  # Hz
tauSec = (1 / t) / sampleRate

# determine quality of the fit
squaredDiffs = np.square(ys - mono_exp(xs_shifted, m, t, b))
squaredDiffsFromMean = np.square(ys - np.mean(ys))
rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
print(f'R² = {rSquared}')

# plot the results
plt.plot(xs_shifted, ys, '.', label='data')
plt.plot(xs_shifted, mono_exp(xs_shifted, m, t, b), '-', label='fitted')
plt.title(f'Solar consumption of {countries[0]}')
xlocs, _ = plt.xticks()
plt.xticks(xlocs, xlocs.astype(int) + xs[0])
plt.xlabel('year')
plt.legend(['Actual', 'Forecast'])
plt.show()

# inspect the parameters
print(f'Y = {m} * e^(-{t} * x) + {b}')
print(f'Tau = {tauSec * 1e6} µs')

next_year = xs[-1] + 1
print(f'Prediction for {next_year}: {mono_exp(next_year - xs[0], m, t, b)}')
