import matplotlib.pyplot as plt
import pandas as pd

energy_data = pd.DataFrame()


def main():
    global energy_data
    # 1.
    countries = ['United States', 'Canada', 'Brazil', 'Mexico']
    energy_data = pd.read_csv('owid-energy-data.csv')
    filtered_index = energy_data.country.isin(countries)
    energy_data = energy_data[filtered_index]
    print(type(energy_data))
    energy_data.to_csv('filtered-energy-data.csv')

    # 2.
    plt.figure()
    colors = ['blue', 'red', 'green', 'yellow']
    for country, color in zip(countries, colors):
        data = energy_data[(energy_data.country == country)
                           & (energy_data['oil_electricity'].notnull())]
        y = data['oil_electricity']
        x = data.year
        plt.plot(x, y, 'o', color=color, label=country)

    plt.xlabel('year')
    plt.ylabel('oil electricity')
    plt.title('Electricity production from oil in certain countries')
    plt.legend()
    plt.show()

    # 3.
    year = 2010
    usa_energy_data = energy_data[(energy_data.country == countries[0])
                                  & (energy_data.year == year)]
    elec_sources = ['coal_electricity', 'biofuel_electricity',
                    'fossil_electricity', 'gas_electricity',
                    'hydro_electricity', 'nuclear_electricity',
                    'oil_electricity']
    usa_electricity = usa_energy_data[elec_sources].values.flatten().tolist()

    plt.figure()
    plt.title(f'Electricity production from the {countries[0]} in {year}')
    elec_sources = [s.replace('_', ' ').capitalize() for s in elec_sources]
    plt.pie(usa_electricity, labels=elec_sources)
    plt.show()

    # 4.
    [highest_nuclear_consumption(country) for country in countries]

    # 5.


def highest_nuclear_consumption(country: str):
    country_data = energy_data.loc[(energy_data.country == country),
                                   ['year', 'nuclear_consumption']]
    i = country_data['nuclear_consumption'].idxmax()
    year = country_data.loc[i, 'year']
    consumption = country_data.loc[i, 'nuclear_consumption']
    print(f'Nuclear consumption of {country} in {year} was {consumption}')


if __name__ == '__main__':
    main()
