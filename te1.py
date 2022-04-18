import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from torch import nn

from model import LSTM
from utils import *

energy_data: pd.DataFrame


def highest_nuclear_consumption(country: str) -> None:
    country_data = energy_data.loc[(energy_data.country == country),
                                   ['year', 'nuclear_consumption']]
    i = country_data['nuclear_consumption'].idxmax()
    year = country_data.loc[i, 'year']
    consumption = country_data.loc[i, 'nuclear_consumption']
    print(f'Nuclear consumption of {country} in {year} was {consumption}')


def main() -> None:
    global energy_data
    countries = ['United States', 'Canada', 'Brazil', 'Mexico']
    # 1.
    owid_energy_data = pd.read_csv('owid-energy-data.csv')
    filtered_index = owid_energy_data.country.isin(countries)
    energy_data = owid_energy_data[filtered_index]
    energy_data.to_csv('filtered-energy-data.csv')
    # 2.
    plt.figure()
    colors = ['blue', 'red', 'green', 'yellow']
    for country, color in zip(countries, colors):
        data = energy_data[(energy_data.country == country)
                           & (energy_data['oil_electricity'].notnull())]
        plt.plot(data.year, data['oil_electricity'], 'o', color=color, label=country)
    plt.xlabel('year')
    plt.ylabel('oil electricity')
    plt.title('Electricity production from oil in certain countries')
    plt.legend()
    plt.show()
    # 3.
    usa_energy_data = energy_data[(energy_data.country == 'United States')
                                  & (energy_data.year == 2010)]
    elec_sources = ['coal_electricity', 'biofuel_electricity',
                    'fossil_electricity', 'gas_electricity',
                    'hydro_electricity', 'nuclear_electricity',
                    'oil_electricity']
    usa_electricity = usa_energy_data[elec_sources].values.flatten().tolist()

    plt.figure()
    plt.title('Electricity production from the United States in 2010')
    elec_sources = [s.replace('_', ' ').capitalize() for s in elec_sources]
    plt.pie(usa_electricity, labels=elec_sources)
    plt.show()
    # 4.
    print('Highest nuclear consumption year per country:')
    [highest_nuclear_consumption(country) for country in countries]
    # 5.
    sns.regplot(data=energy_data, x='gas_consumption', y='nuclear_consumption')
    plt.xlabel('Gas consumption')
    plt.ylabel('Nuclear consumption')
    plt.title(f'Gas consumption / Nuclear consumption')
    # grid = sns.FacetGrid(energy_data, col="country", hue="country", col_wrap=2)
    # grid.map(sns.scatterplot, 'gas_consumption', 'nuclear_consumption')
    # grid.add_legend()
    plt.show()
    # 6.
    solar_data: dict[str, pd.DataFrame] = {}
    for country in countries:
        solar_data[country] = energy_data.loc[(energy_data['solar_consumption'].notnull())
                                              & (energy_data.country == country),
                                              ['year', 'solar_consumption']]
        solar_data[country].set_index('year', inplace=True, drop=True)

    train_data = solar_data['Mexico'].values
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    train_scaled = torch.FloatTensor(train_scaled).view(-1)
    x_train, y_train = get_x_y_pairs(train_scaled, 10, 5)

    model = LSTM(input_size=1, hidden_size=50, output_size=5)
    model.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 300

    model.train()
    for i in range(epochs):
        loss = 0
        for x, y in zip(x_train, y_train):
            x = x.cuda()
            y = y.cuda()
            y_hat, _ = model(x, None)
            optimizer.zero_grad()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
        if i % 25 == 0:
            print(f'epoch: {i:4} loss:{loss.item():10.8f}')

    # next 5 years
    future_years = np.arange(2021, 2026)
    model.eval()
    future = scaler.fit_transform(future_years.reshape(-1, 1))
    future = torch.FloatTensor(future).view(-1)
    with torch.no_grad():
        future = future.cuda()
        predictions, _ = model(future, None)
    # -- Apply inverse transform to undo scaling
    predictions = scaler.inverse_transform(np.array(predictions.reshape(-1, 1).cpu()))

    x = solar_data['United States'].index.to_numpy()
    x = np.concatenate((x, future_years))
    plt.figure()
    plt.grid(visible=True)
    plt.plot(x[:-len(predictions)],
             solar_data['Mexico']['solar_consumption'], "b-")
    plt.plot(x[-len(predictions):],
             predictions,
             "r-",
             label='Predicted Values')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
