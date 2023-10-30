import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# def get_covid_datasets(countries: list, start: int, mid: int, end: int, key: str) -> list:
#     df = pd.read_csv('crosslearning/data/owid-covid-data.csv')
#     datasets_train = {}
#     datasets_test = {}

#     for ele in countries:
#         datasets_train[ele] = df[df["iso_code"] == ele][start:mid]
#         datasets_test[ele]= df[df["iso_code"] == ele][mid:end]


#     return datasets_train, datasets_test

def get_covid_datasets(countries: list, start: int, mid: int, end: int, key: str) -> list:
    df = pd.read_csv('crosslearning/data/owid-covid-data.csv')
    datasets = {}

    for ele in countries:
        thisCountry = {}
        thisCountry['train'] = df[df["iso_code"] == ele][key].loc[start:mid]
        thisCountry['test'] = df[df["iso_code"] == ele][key].loc[mid:end]
        datasets[ele] = thisCountry

    return datasets

def get_SIR_covid_datasets(countries: list, start: int, mid: int, end: int) -> list:
    df = pd.read_csv('crosslearning/data/owid-covid-data.csv')
    datasets = {}

    for ele in countries:
        thisCountryTrain = {}
        thisCountryTrain['S'] = df[df["iso_code"] == ele]['population'].to_numpy()[start:mid] - df[df["iso_code"] == ele]['total_cases'].fillna(0).to_numpy()[start:mid]
        thisCountryTrain['I'] = df[df["iso_code"] == ele]['new_cases'].fillna(0).rolling(14).sum().fillna(0).to_numpy()[start:mid]
        thisCountryTrain['R'] = df[df["iso_code"] == ele]['population'].to_numpy()[start:mid]-thisCountryTrain['S']-thisCountryTrain['I']
        thisCountryTrain['array'] = np.array([thisCountryTrain['S'],thisCountryTrain['I'],thisCountryTrain['R']])
        thisCountryTest = {}
        thisCountryTest['S'] = df[df["iso_code"] == ele]['population'].to_numpy()[mid:end] - df[df["iso_code"] == ele]['total_cases'].fillna(0).to_numpy()[mid:end]
        thisCountryTest['I'] = df[df["iso_code"] == ele]['new_cases'].fillna(0).rolling(14).sum().fillna(0).to_numpy()[mid:end]
        thisCountryTest['R'] = df[df["iso_code"] == ele]['population'].to_numpy()[mid:end]-thisCountryTest['S']-thisCountryTest['I']
        thisCountryTest['array'] = np.array([thisCountryTest['S'],thisCountryTest['I'],thisCountryTest['R']])

        datasets[ele] = {'train':thisCountryTrain,
                         'test':thisCountryTest,
                         'population': df[df["iso_code"] == ele]['population'].to_numpy()[0]}

    return datasets

def plot_SIR_covid_datasets(countries: dict, num_cols: int = 3, fold: str = 'train'):
    """
    Plot a list of figures side by side.

    Parameters:
    - figures: A list of figure objects (e.g., created using plt.figure()).
    """
    num_figures = len(countries)
    num_rows = (num_figures + 1) // num_cols 
    # Create a single figure with subplots based on the number of figures
    fig = plt.figure(figsize=(15, 5))  # You can adjust figsize as needed

    # Loop through each figure and plot it in a separate subplot
    for i, key in enumerate(countries):
        # ax = fig.add_subplot(1, num_figures, i + 1)
        ax = fig.add_subplot(num_rows, num_cols, i + 1)

        x = np.arange(len(countries[key][fold]['S']))

        ax.plot(x,  countries[key][fold]['S'])  # You may need to adjust this based on your specific figures
        ax.plot(x,  countries[key][fold]['I'])  # You may need to adjust this based on your specific figures
        ax.plot(x,  countries[key][fold]['R'])  # You may need to adjust this based on your specific figures
        ax.set_title(key)
        ax.legend(['S','I','R'])
        
        # [i].axis('off') 
    plt.savefig('saved.pdf')
    pass
