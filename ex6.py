import matplotlib.axes as axes
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Ex. 6
from warnings import simplefilter
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

# Exercício 1

owid_energy_data = pd.read_csv('owid-energy-data.csv')
countries = ['United States', 'Canada', 'Brazil', 'Mexico']
filtered_index = owid_energy_data.country.isin(countries)
energy_data = owid_energy_data[filtered_index]
energy_data.to_csv('filtered-energy-data.csv')

# Exercício 6

simplefilter("ignore")

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
)


def main():
    # Exercício 6
    solar_data: pd.DataFrame = energy_data.loc[
        (energy_data.country == countries[0]) & (energy_data.solar_consumption.notnull()),
        ['year', 'solar_consumption']]
    solar_data.set_index('year', inplace=True, drop=True)
    y = solar_data.solar_consumption.copy()
    x = make_lags(y, lags=2).fillna(0.0)

    # 4-years forecast
    y = make_multistep_target(y, steps=4).dropna()

    # Shifting has created indexes that don't match. Only keep times for
    # which we have both targets and features.
    y, x = y.align(x, join='inner', axis=0)

    # Create splits
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=False)

    model = MultiOutputRegressor(XGBRegressor())
    model.fit(x_train, y_train)

    y_fit = pd.DataFrame(model.predict(x_train), index=x_train.index, columns=y.columns)
    y_pred = pd.DataFrame(model.predict(x_test), index=x_test.index, columns=y.columns)

    # Plotting
    train_rmse = mean_squared_error(y_train, y_fit, squared=False)
    test_rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Train RMSE: {train_rmse:.2f}\n" f"Test RMSE: {test_rmse:.2f}")

    palette = dict(palette='husl', n_colors=64)
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6))

    train_tmp = solar_data['solar_consumption'][y_fit.index]
    train_tmp.index = pd.period_range(start=train_tmp.index[0], periods=len(train_tmp), freq='Y')
    ax1 = train_tmp.plot(**plot_params, ax=ax1)
    ax1 = plot_multistep(y_fit, ax=ax1, palette_kwargs=palette)
    _ = ax1.legend(['Solar consumption (train)', 'Forecast'])

    test_tmp = solar_data['solar_consumption'][y_pred.index]
    test_tmp.index = pd.period_range(start=test_tmp.index[0], periods=len(test_tmp), freq='Y')
    ax2 = test_tmp.plot(**plot_params, ax=ax2)
    ax2 = plot_multistep(y_pred, ax=ax2, palette_kwargs=palette)
    _ = ax2.legend(['Solar consumption (test)', 'Forecast'])
    plt.show()


def plot_multistep(y: pd.DataFrame, every: int = 1, ax: axes.Axes = None, palette_kwargs: dict = None):
    palette_kwargs_ = dict(palette='husl', n_colors=16, desat=None)
    if palette_kwargs is not None:
        palette_kwargs_.update(palette_kwargs)
    palette = sns.color_palette(**palette_kwargs_)
    if ax is None:
        _, ax = plt.subplots()
    ax.set_prop_cycle(plt.cycler('color', palette))
    for date, preds in y[::every].iterrows():
        preds.index = pd.period_range(start=date, periods=len(preds), freq='Y')
        preds.plot(ax=ax)
        # print(preds)
    return ax


def make_lags(ts: pd.DataFrame, lags: int, lead_time: int = 1) -> pd.DataFrame:
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(lead_time, lags + lead_time)
        },
        axis=1)


def make_multistep_target(ts: pd.DataFrame, steps: int):
    return pd.concat(
        {f'y_step_{i + 1}': ts.shift(-i)
         for i in range(steps)},
        axis=1)


if __name__ == '__main__':
    main()
