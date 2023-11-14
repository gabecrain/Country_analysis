import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import trim_mean, iqr, pearsonr

#initial data inspection
countries = pd.read_csv('/Users/gabrielcrain/Desktop/countries.csv')
print(countries.head())
print(countries.info())

#clean up column names and drop extra columns
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
dropped_columns = ['Abbreviation', 'Calling Code', 'Latitude', 'Longitude', 'Official language', 'Largest city', 'CPI', 'CPI Change (%)', 'Capital/Major City', 'Currency-Code']
countries.drop(columns=dropped_columns, inplace=True)

countries.rename(columns={
    'Land Area(Km2)': 'land_area_km2',
    'Armed Forces size': 'military_size',
    'Birth Rate': 'birth_rate',
    'Co2-Emissions': 'carbon_emissions',
    'Minimum wage': 'min_wage',
    'Gasoline Price': 'gas_price',
    'Gross primary education enrollment (%)': 'primary_enrollment',
    'Gross tertiary education enrollment (%)': 'tertiary_enrollment',
    'Population: Labor force participation (%)': 'labor_force',
    'Unemployment rate': 'unemployment_rate',
    'Tax revenue (%)': 'tax_revenue',
    'Physicians per thousand': 'doctors_per_1000',
    'Out of pocket health expenditure': 'health_expenditure',
    'Fertility Rate': 'fertility_rate',
    'Forested Area (%)': 'percent_forested',
    'Agricultural Land( %)': 'percent_ag_land',
    'Density\n(P/Km2)': 'population_density',
    'Country': 'country',
    'GDP': 'gdp',
    'Infant mortality': 'infant_mortality',
    'Life expectancy': 'life_expectancy',
    'Maternal mortality ratio': 'maternal_mortality',
    'Population': 'population',
    'Total tax rate': 'total_tax_percent'
}, inplace=True)


print(countries.describe)
print(countries.columns)

#use regex to parse column data
countries.birth_rate = countries['birth_rate'].replace('[\%,]', '', regex=True)
countries.birth_rate = pd.to_numeric(countries.birth_rate)
countries.birth_rate = countries['birth_rate'] / 100

countries.labor_force = countries['labor_force'].replace('[\%,]', '', regex=True)
countries.labor_force = pd.to_numeric(countries.labor_force)
countries.labor_force = countries['labor_force'] / 100

countries.tax_revenue = countries['tax_revenue'].replace('[\%,]', '', regex=True)
countries.tax_revenue = pd.to_numeric(countries.tax_revenue)
countries.tax_revenue = countries['tax_revenue'] / 100

countries.percent_ag_land = countries['percent_ag_land'].replace('[\%,]', '', regex=True)
countries.percent_ag_land = pd.to_numeric(countries.percent_ag_land)
countries.percent_ag_land = countries['percent_ag_land'] / 100

countries.total_tax_percent = countries['total_tax_percent'].replace('[\%,]', '', regex=True)
countries.total_tax_percent = pd.to_numeric(countries.total_tax_percent)
countries.total_tax_percent = countries['total_tax_percent'] / 100

countries.unemployment_rate = countries['unemployment_rate'].replace('[\%,]', '', regex=True)
countries.unemployment_rate = pd.to_numeric(countries.unemployment_rate)
countries.unemployment_rate = countries['unemployment_rate'] / 100

countries.percent_forested = countries['percent_forested'].replace('[\%,]', '', regex=True)
countries.percent_forested = pd.to_numeric(countries.percent_forested)
countries.percent_forested = countries['percent_forested'] / 100

countries.health_expenditure = countries['health_expenditure'].replace('[\%,]', '', regex=True)
countries.health_expenditure = pd.to_numeric(countries.health_expenditure)
countries.health_expenditure = countries['health_expenditure'] / 100

countries.gdp = countries['gdp'].replace('[\$,]', '', regex=True)
countries.gdp = pd.to_numeric(countries.gdp)

countries.population = countries['population'].replace('[\,,]', '', regex=True)
countries.population = pd.to_numeric(countries.population)

countries.military_size = countries['military_size'].replace('[\,,]', '', regex=True)
countries.military_size = pd.to_numeric(countries.military_size)

countries.Urban_population = countries['Urban_population'].replace('[\,,]', '', regex=True)
countries.Urban_population = pd.to_numeric(countries.Urban_population)

countries.land_area_km2 = countries['land_area_km2'].replace('[\,,]', '', regex=True)
countries.land_area_km2 = pd.to_numeric(countries.land_area_km2)

countries.carbon_emissions = countries['carbon_emissions'].replace('[\,,]', '', regex=True)
countries.carbon_emissions = pd.to_numeric(countries.carbon_emissions)

print(countries.dtypes)
print(countries.head())

#investigate univariate variables
population_mean = countries.population.mean()
print(population_mean)

countries = countries.dropna(subset=['population'])
print(countries.population.nunique)
countries.population = countries['population'].astype(int)
population_trim_mean = trim_mean(countries.population, proportiontocut=.1)
print(population_trim_mean)

percent_forested_mean = countries.percent_forested.mean()
print(percent_forested_mean)

greater_80_percent_forested = countries[countries.percent_forested >= .8]
print(greater_80_percent_forested)

#visualize different variables with seaborn
gdp_lower_quantile = countries['gdp'].quantile(.05)
gdp_upper_quantile = countries['gdp'].quantile(.95)
gdp_trim = countries[(countries.gdp >= gdp_lower_quantile) & (countries.gdp <= gdp_upper_quantile)]
# sns.boxplot(x='gdp', data=gdp_trim)
# plt.show()
# plt.clf()

countries = countries.dropna(subset=['unemployment_rate'])
# sns.histplot(x='unemployment_rate', data=countries)
# plt.show()
# plt.clf()

#compare military size vs carbon emissions
trimmed_countries = countries[countries.carbon_emissions < 200000]
trimmed_countries = countries[countries.military_size < 600000]

# plt.scatter(x=trimmed_countries.carbon_emissions, y=trimmed_countries.military_size)
# plt.xlabel('Carbon Emissions (Tons)')
# plt.ylabel('Military Size')
# plt.show()

#compare percent forested vs life expectancy
# plt.scatter(x=countries.life_expectancy, y=countries.percent_forested)
# plt.xlabel('Life Expectancy (Years)')
# plt.ylabel('Percent Forested (%)')
# plt.show()
#there appears to be no correlation between a countries average life expectancy and the amount of the land that is forested.

#compare percent forested vs percent agriculture land
# plt.scatter(x=countries.percent_ag_land, y=countries.percent_forested)
# plt.xlabel('Agriculture Land (%)')
# plt.ylabel('Forested Land (%)')
# plt.show()

#explore covariance and pearson correlation between percent forested vs percent agriculture land
countries = countries.dropna(subset=['percent_ag_land', 'percent_forested', 'health_expenditure'])
cov_forested_agricultre = np.cov(countries.percent_ag_land, countries.percent_forested)
print(cov_forested_agricultre)

corr_forested_agriculture, p = pearsonr(countries.percent_ag_land, countries.percent_forested)
print(corr_forested_agriculture)
#we can safely conclude there is a negative correlation between the amount of forested land in a country vs the amount of land being used for agriculture

#study relationship between life expectancy and health expenditure
# print(countries.health_expenditure.isna().sum())
# plt.scatter(x=countries.health_expenditure, y=countries.life_expectancy)
# plt.xlabel('Health Expenditure (%)')
# plt.ylabel('Life Expectancy (Years)')
# plt.show()
#initial inspection shows what appears to be little relationship between life expectancy and health expenditure

#explore covariance and pearson correlation between life expectancy and health expenditure
cov_lifeyears_healthcost = np.cov(countries.life_expectancy, countries.health_expenditure)
print(cov_lifeyears_healthcost)

corr_lifeyears_healthcost, p = pearsonr(countries.life_expectancy, countries.health_expenditure)
print(corr_lifeyears_healthcost)
#upon further inspection it appears there is a small negative correlation between life expectancy and health expenditure
