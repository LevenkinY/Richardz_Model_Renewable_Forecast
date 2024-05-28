import os
import pandas as pd
import numpy as np

directory = 'AllEnergyData\\CapacityByCountryTop30A'
csv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
country_NS_solarPV = ['Belgium', 'Germany', 'Greece', 'Italy', 'United Kingdom of Great Britain and Northern Ireland (the)']
scope_solarPV = {
    'Belgium': [2012, 2016],
    'Germany': [2012, 2017],
    'Greece': [2013, 2020],
    'Italy': [2013, 2020],
    'United Kingdom of Great Britain and Northern Ireland (the)': [2017, 2020]
}
country_NS_Wind = ['Austria', 'Canada', 'Chinese Taipei', 'Poland', 'Spain']
scope_Wind = {
    'Austria': [2006, 2010],
    'Canada': [2015, 2019],
    'Chinese Taipei': [2010, 2018],
    'Poland': [2016, 2019],
    'Spain': [2012, 2018]
}

for file in csv_files:
    filename = os.path.basename(file)
    name, extension = os.path.splitext(filename)
    CountryInfo = name
    if CountryInfo not in country_NS_Wind:
        continue
    data = pd.read_csv(file)
    years = data['Year']
    capacity = data['Wind energy']
    first_derivative = np.gradient(capacity)
    inflection_point = np.argmin(first_derivative[scope_Wind[CountryInfo][0]-2000:scope_Wind[CountryInfo][1]-2000]) + scope_Wind[CountryInfo][0]
    print(CountryInfo, inflection_point)

