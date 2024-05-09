import numpy as np
import pandas as pd
import os
import pymc as pm
import matplotlib.pyplot as plt
from pymc.stats import r2_score
from datetime import timedelta
ErrorData = []
Allenergy = ['Total Renewable','Solar photovoltaic','Solar thermal energy','Onshore wind energy',
                            'Offshore wind energy','Wind energy','Renewable hydropower','Mixed Hydro Plants','Marine energy',
                           'Solid biofuels','Renewable municipal waste','Liquid biofuels','Biogas','Geothermal energy']
EnergyUse = ['Wind energy']
# Function to save parameters
def save_parameters(paras, country, energy_type, file_path='Top30Results/parameters.csv'):
    paras_df = pd.DataFrame([paras])
    paras_df['Country'] = country
    paras_df['EnergyType'] = energy_type
    paras_df.to_csv(file_path, mode='a', header=False, index=False)

def Richardz(t, L, k, t0, m):
    return L * (1 + (m - 1) * np.exp(-k * (t - t0))) ** (1 / (1 - m))

print(f"Running on PyMC3 v{pm.__version__}")

# Process each CSV file for different energy types
directory = 'AllEnergyData\CapacityByCountryTop10A'
csv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]

for file in csv_files:
    filename = os.path.basename(file)
    name, extension = os.path.splitext(filename)
    CountryInfo = name
    df = pd.read_csv(file)
    # Loop through each type of energy data
    for energy_type in EnergyUse:
        print(CountryInfo, energy_type)
        try:
            df_filtered = df[df[energy_type] > 0]
            if len(df_filtered['Year']) <= 5:
                ErrorData.append(f'Insufficient historical data for {energy_type} of {CountryInfo}')
                continue

            ModelData = df_filtered[energy_type]
            endyear = df_filtered['Year'].iloc[-1]
            startyear = df_filtered['Year'].iloc[0]
            years = np.arange(startyear, endyear + 1)

            # Bayesian modeling
            with pm.Model():
                L = pm.TruncatedNormal('L', mu=np.max(ModelData) * 3, sigma=np.std(ModelData) * 5, 
                                    lower=0, upper=np.max(ModelData) * 20)
                k = pm.Uniform('k', lower=0, upper=10)
                t0 = pm.Uniform('t0', lower=2000, upper=2060)
                m = pm.TruncatedNormal('m', mu=2, sigma=2, lower=0.001, upper=1000)
                sigma = pm.HalfCauchy("sigma", beta=10)
                pm.Deterministic("richardz", Richardz(years, L, k, t0, m))
                ModelData_observed = pm.Normal('ModelData_observed', mu=Richardz(years, L, k, t0, m), 
                                            sigma=sigma, observed=ModelData)
                trace = pm.sample(draws=1000, tune=1000, chains=4, cores=1, discard_tuned_samples=True)
                ppc = pm.sample_posterior_predictive(trace, var_names=["ModelData_observed"])
            # Calculate and save predictions with confidence intervals

            pred_samples = ppc.posterior_predictive["ModelData_observed"].values
            lower_bound = np.percentile(pred_samples, 2.5, axis=0)
            upper_bound = np.percentile(pred_samples, 97.5, axis=0)
            lower_present = np.mean(lower_bound, axis=0)
            upper_present = np.mean(upper_bound, axis=0)
            pred_samples_ChainMean = np.mean(pred_samples, axis=0)
            Modeldata_post = np.mean(pred_samples_ChainMean, axis=0)
            bayesian_r2 = r2_score(ModelData, Modeldata_post)['r2']
            bayes_show = round(bayesian_r2, 4)

            # Extract results and save parameters
            summary = pm.summary(trace)
            paras = {'L': summary.loc['L', 'mean'], 'k': summary.loc['k', 'mean'], 't0': summary.loc['t0', 'mean'], 'm': summary.loc['m', 'mean'], 'R2':bayesian_r2}
            save_parameters(paras, CountryInfo, energy_type)

            future_years = np.arange(endyear + 1, 2031)
            all_predictions = []
            L_samples = trace.posterior["L"].values.flatten()
            k_samples = trace.posterior["k"].values.flatten()
            t0_samples = trace.posterior["t0"].values.flatten()
            m_samples = trace.posterior["m"].values.flatten()
            for L, k, t0, m in zip(L_samples, k_samples, t0_samples, m_samples):
                predictions = Richardz(future_years, L, k, t0, m)
                all_predictions.append(predictions)
            lower_predic = np.percentile(all_predictions, 2.5, axis=0)
            upper_predic = np.percentile(all_predictions, 97.5, axis=0)
            mean_predictions = np.mean(all_predictions, axis=0)

            forecast_df = pd.DataFrame({
                'Year': future_years,
                'Prediction': mean_predictions,
                'lower_predic': lower_predic,
                'upper_predic': upper_predic
            })

            forecast_df.to_csv(f'Top30Results\\forecasts\\{energy_type}\\{CountryInfo}.csv', index=False)

            # Plotting and saving the figure
            plt.figure(figsize=(8, 6))
            original_dates = pd.date_range(start=str(startyear), end=str(endyear), freq='YS')
            post_dates = pd.date_range(start=str(endyear + 1), end='2030', freq='YS')
            
            plt.plot(original_dates, ModelData, 'o-', color='darkblue', label=f'Original data ({energy_type})', markerfacecolor='blue', markersize=8)
            plt.plot(original_dates, Modeldata_post, 'x--', color='green', label=f'Simulated, R2 = {bayes_show}')
            plt.fill_between(original_dates, lower_present, upper_present, color='green', alpha=0.5)
            plt.plot(post_dates, mean_predictions, '.--', color='red', label='Predicted')
            plt.fill_between(post_dates, lower_predic, upper_predic, color='gray', alpha=0.5)
            
            inflection_year_float = paras['t0']
            if inflection_year_float <= 2030:
                year_part = int(inflection_year_float)
                fraction_part = inflection_year_float - year_part
                days_in_year = 365.25  # Rough approximation accounting for leap years
                days_offset = int(fraction_part * days_in_year)  
                inflection_point = pd.to_datetime(f'{year_part}-01-01') + timedelta(days=days_offset)
                # inflection_value = paras[0] * paras[3] ** (1/(1-paras[3]))
                plt.plot(inflection_point, 0,  '*', color='gold', markersize=8, label='Inflection Point')

            plt.xlabel('Year')
            plt.ylabel('Capacity (MW)')
            plt.title(f'Forecast of {energy_type} Capacity in {CountryInfo}')
            plt.xticks(rotation=45, fontsize = 8)            
            plt.yticks(fontsize=8)

            plt.text(0.05, 0.75, f'L = {paras['L']:.2f}\n'
                                f'k = {paras['k']:.2f}\n'
                                f't0 = {paras['t0']:.2f}\n'
                                f'm = {paras['m']:.2f}',
                    verticalalignment='top', horizontalalignment='left',
                    transform=plt.gca().transAxes, fontsize=8, 
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
            plt.text(0.05, 0.55, f'Triple target: {(3*ModelData[22]):.0f}\n2030 Forecast: {(mean_predictions[-1]):.0f}',
                verticalalignment='bottom', horizontalalignment='left',
                transform=plt.gca().transAxes, fontsize=8,
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=1'))
            
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.tight_layout() 
            plt.savefig(f'Top30Results\\pictures\\{energy_type}\\{CountryInfo}.png', format='png')
            # plt.show()
            plt.close()

        except Exception as e:
            ErrorData.append(f'Failed to process {file}: {e}')

# Save ErrorData to a file for review
with open('Top30Results\\error_log.txt', 'w') as error_file:
    for error in ErrorData:
        error_file.write(f'{error}\n')
