"""
Energy Forecast Model with Policy Sensitivity Analysis

A comprehensive single-file implementation for renewable energy forecasting
with Bayesian growth curve modeling (Richards, Logistic, Gompertz) and 
policy scenario analysis.

Optimized based on user feedback:
1.  Updated policy scenarios (S0-S6).
2.  Changed posterior sampling from median to mean.
3.  Merged distribution and sensitivity plots into a single a/b figure.
4.  Implemented a more academic color scheme.
5.  Added support for Logistic and Gompertz models.
6.  Included a new plot to compare predictions from different models.
"""

import numpy as np
import pymc as pm
from pymc.stats import r2_score
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import timedelta
import os
from io import StringIO
import seaborn as sns

# ==================== Growth Forecast Models ====================

class GrowthForecastModel:
    """
    A class to perform Bayesian forecasting of energy capacity using various growth curves.
    Supports Richards, Logistic, and Gompertz models.
    """

    def __init__(self, model_type='Richards', output_dir='output'):
        """
        Initialize the forecast model.

        Parameters:
        -----------
        model_type : str
            Type of growth model ('Richards', 'Logistic', or 'Gompertz').
        output_dir : str
            Directory to save output files.
        """
        if model_type not in ['Richards', 'Logistic', 'Gompertz']:
            raise ValueError("model_type must be 'Richards', 'Logistic', or 'Gompertz'")
            
        self.model_type = model_type
        self.output_dir = output_dir
        self.model = None
        self.trace = None
        self.idata = None
        self.ppc = None
        self.parameters = None
        self.forecast_df = None
        self.r2 = None
        self.energy_type = None
        self.curve_function = self.get_curve_function()

    def get_curve_function(self):
        """Returns the appropriate curve function based on model_type."""
        if self.model_type == 'Richards':
            return self.richards_curve
        elif self.model_type == 'Logistic':
            return self.logistic_curve
        elif self.model_type == 'Gompertz':
            return self.gompertz_curve

    @staticmethod
    def richards_curve(t, L, k, t0, m):
        """Richards growth curve function."""
        # Add a small epsilon to (1 - m) to avoid division by zero if m is close to 1
        return L * (1 + (m - 1) * np.exp(-k * (t - t0))) ** (1 / (1 - m + 1e-9))

    @staticmethod
    def logistic_curve(t, L, k, t0, m=None): # m is ignored, for compatibility
        """Logistic growth curve function."""
        return L / (1 + np.exp(-k * (t - t0)))

    @staticmethod
    def gompertz_curve(t, L, k, t0, m=None): # m is ignored, for compatibility
        """Gompertz growth curve function."""
        return L * np.exp(-np.exp(-k * (t - t0)))

    def fit(self, years, data, energy_type=None):
        """
        Fit the selected growth model to historical data using Bayesian inference.
        """
        self.energy_type = energy_type

        with pm.Model() as self.model:
            # --- Priors (common for all models) ---
            L = pm.TruncatedNormal('L', mu=np.max(data) * 3, sigma=np.std(data) * 5,
                                    lower=np.max(data), upper=np.max(data) * 20)
            k = pm.Uniform('k', lower=0, upper=5)
            t0 = pm.Uniform('t0', lower=min(years) - 10, upper=max(years) + 40)
            sigma = pm.HalfCauchy("sigma", beta=10)

            # --- Model-specific likelihood ---
            if self.model_type == 'Richards':
                m = pm.TruncatedNormal('m', mu=2, sigma=2, lower=0.01, upper=100)
                mu = pm.Deterministic("mu", self.richards_curve(years, L, k, t0, m))
            elif self.model_type == 'Logistic':
                mu = pm.Deterministic("mu", self.logistic_curve(years, L, k, t0))
            elif self.model_type == 'Gompertz':
                mu = pm.Deterministic("mu", self.gompertz_curve(years, L, k, t0))

            data_observed = pm.Normal('data_observed', mu=mu, sigma=sigma, observed=data)

            # --- Sample from posterior ---
            self.trace = pm.sample(draws=2000, tune=2000, chains=4, cores=1,
                                   target_accept=0.9, discard_tuned_samples=True)
            self.ppc = pm.sample_posterior_predictive(self.trace, var_names=["data_observed"])
            self.idata = pm.to_inference_data(trace=self.trace, posterior_predictive=self.ppc, model=self.model)

        # --- Extract results ---
        pred_samples = self.ppc.posterior_predictive["data_observed"].values
        model_data_post = np.mean(pred_samples, axis=(0, 1))
        self.r2 = r2_score(data, model_data_post)['r2']
        
        posterior = self.idata.posterior
        self.parameters = {
            'L': float(posterior['L'].mean()),
            'k': float(posterior['k'].mean()),
            't0': float(posterior['t0'].mean()),
            'R2': self.r2
        }
        if self.model_type == 'Richards':
            self.parameters['m'] = float(posterior['m'].mean())
            
        return self

    def save_posterior(self, filename=None):
        """Save the full posterior distribution to a netCDF file."""
        if self.idata is None:
            raise ValueError("Model must be fitted before saving posterior")

        posterior_dir = os.path.join(self.output_dir, 'posteriors')
        os.makedirs(posterior_dir, exist_ok=True)

        if filename is None:
            if self.energy_type is None:
                raise ValueError("Either filename or energy_type must be provided")
            # Include model type in the filename
            filename = f"{self.energy_type.replace(' ', '_')}_{self.model_type}.nc"

        filepath = os.path.join(posterior_dir, filename)
        self.idata.to_netcdf(filepath)
        return filepath

    @classmethod
    def load_posterior(cls, filepath, model_type, output_dir='output'):
        """Load a saved posterior distribution and create a model instance."""
        model = cls(model_type=model_type, output_dir=output_dir)
        model.idata = az.from_netcdf(filepath)
        model.trace = model.idata.posterior

        # Extract parameters using MEAN
        model.parameters = {
            'L': float(model.trace['L'].mean()),
            'k': float(model.trace['k'].mean()),
            't0': float(model.trace['t0'].mean())
        }
        if model.model_type == 'Richards' and 'm' in model.trace:
            model.parameters['m'] = float(model.trace['m'].mean())
            
        return model

    def predict(self, future_years):
        """Generate forecasts for future years."""
        if self.trace is None:
            raise ValueError("Model must be fitted or loaded before making predictions")

        posterior = self.idata.posterior
        L_samples = posterior["L"].values.flatten()
        k_samples = posterior["k"].values.flatten()
        t0_samples = posterior["t0"].values.flatten()
        m_samples = posterior.get("m", None)
        if m_samples is not None:
            m_samples = m_samples.values.flatten()
        else:
            # Create a dummy array if m is not in the model
            m_samples = [None] * len(L_samples)

        all_predictions = [self.curve_function(future_years, L, k, t0, m)
                           for L, k, t0, m in zip(L_samples, k_samples, t0_samples, m_samples)]

        mean_predictions = np.mean(all_predictions, axis=0)
        lower_predic = np.percentile(all_predictions, 2.5, axis=0)
        upper_predic = np.percentile(all_predictions, 97.5, axis=0)

        self.forecast_df = pd.DataFrame({
            'Year': future_years,
            'Prediction': mean_predictions,
            'lower_predic': lower_predic,
            'upper_predic': upper_predic
        })
        return self.forecast_df

# ==================== Policy Scenario Analyzer ====================

class PolicyScenarioAnalyzer:
    """
    Analyzes policy scenarios by loading saved posteriors and applying parameter modifications.
    """

    def __init__(self, posterior_dir='output/posteriors', base_year=2022, target_year=2030):
        self.posterior_dir = posterior_dir
        self.base_year = base_year
        self.target_year = target_year
        self.energy_types = ['Solar photovoltaic', 'Wind energy', 'Hydropower',
                             'Bioenergy', 'Geothermal energy']
        self.posteriors = {} # This will now hold posteriors for a specific model type
        self.historical_data = None
        self.colors = sns.color_palette('crest', 7)
        # Store the curve function with the loaded posterior for correct simulation
        self.curve_functions = {} 

    def load_posteriors_for_model(self, model_type):
        """Load all saved posterior distributions for a given model type."""
        self.posteriors.clear()
        self.curve_functions.clear()
        print(f"--- Loading posteriors for {model_type} model ---")
        for energy_type in self.energy_types:
            filename = f"{energy_type.replace(' ', '_')}_{model_type}.nc"
            filepath = os.path.join(self.posterior_dir, filename)
            if os.path.exists(filepath):
                self.posteriors[energy_type] = az.from_netcdf(filepath)
                # Store the correct curve function for this energy type
                if model_type == 'Richards':
                    self.curve_functions[energy_type] = GrowthForecastModel.richards_curve
                elif model_type == 'Logistic':
                    self.curve_functions[energy_type] = GrowthForecastModel.logistic_curve
                elif model_type == 'Gompertz':
                    self.curve_functions[energy_type] = GrowthForecastModel.gompertz_curve
            else:
                print(f"Warning: Posterior file not found for {energy_type} ({model_type}) at {filepath}")
        if not self.posteriors:
            raise FileNotFoundError(f"No posterior files found for model type '{model_type}' in {self.posterior_dir}")


    def set_historical_data(self, historical_data_df):
        """Set historical data for base year calculations."""
        self.historical_data = historical_data_df

    def get_base_year_total(self):
        """Get total renewable capacity in base year."""
        if self.historical_data is None: raise ValueError("Historical data not set")
        base_data = self.historical_data[self.historical_data['Year'] == self.base_year]
        if base_data.empty: raise ValueError(f"No data found for base year {self.base_year}")
        return base_data[self.energy_types].sum(axis=1).values[0]

    def define_scenarios(self):
        """Defines policy scenarios based on the updated S0-S6 framework."""
        # Scenario definitions remain the same
        scenarios = {
            'S0': {'name': 'BAU', 'modifications': {}},
            'S1': {'name': 'Solar Acceleration', 'modifications': {'Solar photovoltaic': {'k': 1.35, 'L': 1.25}}},
            'S2': {'name': 'Wind Acceleration', 'modifications': {'Wind energy': {'k': 1.30, 'L': 1.50}}},
            'S3': {'name': 'Capacity Expansion Focus', 'modifications': {
                'Solar photovoltaic': {'k': 1.10, 'L': 1.30}, 'Wind energy': {'k': 1.10, 'L': 1.35},
                'Hydropower': {'k': 1.05, 'L': 1.15}, 'Bioenergy': {'k': 1.05, 'L': 1.20},
                'Geothermal energy': {'k': 1.05, 'L': 1.20}}},
            'S4': {'name': 'Growth Rate Enhancement', 'modifications': {
                'Solar photovoltaic': {'k': 1.35, 'L': 1.10}, 'Wind energy': {'k': 1.30, 'L': 1.10},
                'Hydropower': {'k': 1.15, 'L': 1.05}, 'Bioenergy': {'k': 1.25, 'L': 1.05},
                'Geothermal energy': {'k': 1.25, 'L': 1.05}}},
            'S5': {'name': 'Balanced Enhancement', 'modifications': {
                'Solar photovoltaic': {'k': 1.25, 'L': 1.20}, 'Wind energy': {'k': 1.25, 'L': 1.25},
                'Hydropower': {'k': 1.10, 'L': 1.10}, 'Bioenergy': {'k': 1.15, 'L': 1.15},
                'Geothermal energy': {'k': 1.15, 'L': 1.10}}},
            'S6': {'name': 'Ambitious Target', 'modifications': {
                'Solar photovoltaic': {'k': 1.45, 'L': 1.35}, 'Wind energy': {'k': 1.40, 'L': 1.45},
                'Hydropower': {'k': 1.15, 'L': 1.15}, 'Bioenergy': {'k': 1.25, 'L': 1.20},
                'Geothermal energy': {'k': 1.20, 'L': 1.15}}}
        }
        return scenarios

    def apply_modifications(self, posterior, modifications):
        """Apply parameter modifications to posterior samples."""
        modified_params = {}
        # Support 'm' for Richards model if present
        param_list = ['L', 'k', 't0', 'm'] if 'm' in posterior.posterior else ['L', 'k', 't0']
        
        for param in param_list:
            original_values = posterior.posterior[param].values.flatten()
            if param in modifications:
                modified_params[param] = original_values * modifications[param]
            else:
                modified_params[param] = original_values
        return modified_params

    def simulate_scenario(self, scenario_modifications, n_simulations=2000, return_timeseries=False):
        """Simulate a scenario by sampling from modified posteriors."""
        future_years = np.arange(self.base_year + 1, self.target_year + 1)
        total_predictions = np.zeros((n_simulations, len(future_years)))

        for energy_type, posterior_data in self.posteriors.items():
            n_posterior_samples = len(posterior_data.posterior.draw) * len(posterior_data.posterior.chain)
            sample_indices = np.random.choice(n_posterior_samples, n_simulations, replace=True)

            mods = scenario_modifications.get(energy_type, {})
            modified_params = self.apply_modifications(posterior_data, mods)

            L_samples = modified_params['L'][sample_indices]
            k_samples = modified_params['k'][sample_indices]
            t0_samples = modified_params['t0'][sample_indices]
            # Handle 'm' for Richards model
            m_samples = modified_params.get('m', [None]*n_simulations)
            if 'm' in modified_params:
                m_samples = m_samples[sample_indices]

            curve_func = self.curve_functions[energy_type]
            
            for i in range(n_simulations):
                prediction = curve_func(future_years, L_samples[i], k_samples[i], t0_samples[i], m_samples[i])
                total_predictions[i, :] += prediction

        if return_timeseries:
            return total_predictions
        else:
            # Return only the target year's predictions
            return total_predictions[:, -1]

    def calculate_triple_probability(self, predictions, base_total):
        """Calculate probability of achieving triple target."""
        triple_target = 11000000 # base_total * 3
        return np.mean(predictions >= triple_target)

    def run_all_scenarios(self, n_simulations=2000):
        """Run all defined scenarios and calculate results."""
        scenarios = self.define_scenarios()
        base_total = self.get_base_year_total()
        results = {}

        for scenario_key, scenario_config in scenarios.items():
            print(f"  - Running scenario: {scenario_config['name']}")
            predictions = self.simulate_scenario(scenario_config['modifications'], n_simulations)
            
            results[scenario_key] = {
                'name': scenario_config['name'], 'predictions': predictions,
                'mean': np.mean(predictions), 'median': np.median(predictions),
                'std': np.std(predictions), 'lower_95': np.percentile(predictions, 2.5),
                'upper_95': np.percentile(predictions, 97.5),
                'triple_probability': self.calculate_triple_probability(predictions, base_total),
                'growth_ratio': np.mean(predictions) / base_total,
                'modifications': scenario_config['modifications']
            }
        return results

    def plot_combined_scenario_analysis(self, results, model_type, output_dir='output'):
        """Creates a combined plot showing scenario distributions and parameter sensitivity."""
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 18), dpi=300, gridspec_kw={'height_ratios': [2, 1.5]})
        # fig.suptitle(f'Policy Scenario Analysis ({model_type} Model) for Reaching 2030 Triple Target',
        #             fontsize=20, fontweight='bold', y=0.98)
    
        # 使用新的配色系统
        better_rose_colors = ['#ecefcd', '#b5cdb5', '#86aaa0', '#5c898e', '#39687d', '#21455c', '#1f2531']
        
        # --- Part (a): Distribution of 2030 Capacity Predictions ---
        scenarios = list(results.keys())
        names = [results[s]['name'] for s in scenarios]
        predictions_data = [results[s]['predictions'] / 1000 for s in scenarios] # to GW
        probabilities = [results[s]['triple_probability'] for s in scenarios]
        violin_parts = ax1.violinplot(predictions_data, positions=np.arange(len(scenarios)), showmeans=False, showmedians=False, showextrema=False)
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(better_rose_colors[i % len(better_rose_colors)])
            pc.set_edgecolor('#1f2531')  # 使用最深的颜色作为边框
            pc.set_alpha(0.8)
        quartile1, medians, quartile3 = np.percentile(predictions_data, [25, 50, 75], axis=1)
        whiskers_min, whiskers_max = np.percentile(predictions_data, [2.5, 97.5], axis=1)
        ax1.scatter(np.arange(len(scenarios)), medians, marker='o', color='white', s=30, zorder=3, edgecolor='#1f2531', linewidth=1.5)
        ax1.vlines(np.arange(len(scenarios)), quartile1, quartile3, color='#1f2531', linestyle='-', lw=5)
        ax1.vlines(np.arange(len(scenarios)), whiskers_min, whiskers_max, color='#1f2531', linestyle='-', lw=1)
        ax1.set_xticks(np.arange(len(scenarios)))
        # 设置y轴label字号
        ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.set_ylabel('Total Renewable Capacity in 2030 (GW)', fontsize=16)
        ax1.set_title('(a) Distribution of 2030 Renewable Capacity Predictions by Scenario', fontsize=16, pad=20)
        # ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.grid(False)
        for i, prob in enumerate(probabilities):
            ax1.text(i, ax1.get_ylim()[1] * 0.93, f'P(3x)={prob:.1%}', ha='center', fontsize=14, fontweight='bold', 
                    color='#1f2531', bbox=dict(facecolor='white', alpha=0.8, edgecolor='#39687d', boxstyle='round,pad=0.3'))
        base_total_gw = self.get_base_year_total() / 1000
        ax1.axhline(y=11000, color='#39687d', linestyle='--', linewidth=2.5, label=f'Triple Target ({11000:.0f} GW)')
        ax1.legend(fontsize=14, loc='upper left')
        
        # --- Part (b): Unified Parameter Sensitivity Analysis ---
        self._plot_sensitivity_on_ax(ax2)
        ax2.set_title('(b) Sensitivity to Uniform Changes in Growth (k) and Potential (L)', fontsize=16, pad=20)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filepath = os.path.join(output_dir, f'combined_scenario_analysis_{model_type}.png')
        plt.savefig(filepath, dpi = 300, bbox_inches='tight')
        print(f"Combined analysis plot for {model_type} saved to {filepath}")
        plt.close()

    def _plot_sensitivity_on_ax(self, ax):
        """Helper to create the sensitivity heatmap on a given axes object."""
        k_multipliers = np.linspace(1.0, 1.5, 6)
        L_multipliers = np.linspace(1.0, 1.5, 6)
        sensitivity_matrix = np.zeros((len(L_multipliers), len(k_multipliers)))
        base_total = self.get_base_year_total()
        for i, L_mult in enumerate(L_multipliers):
            for j, k_mult in enumerate(k_multipliers):
                scenario_mod = {energy_type: {'k': k_mult, 'L': L_mult} for energy_type in self.energy_types}
                predictions = self.simulate_scenario(scenario_mod, n_simulations=500)
                probability = self.calculate_triple_probability(predictions, base_total)
                sensitivity_matrix[i, j] = probability
        
        # 使用新的配色创建自定义colormap
        from matplotlib.colors import LinearSegmentedColormap
        better_rose_cmap = LinearSegmentedColormap.from_list('better_rose', 
                                                            ['#ecefcd', '#b5cdb5', '#86aaa0', '#5c898e', '#39687d', '#21455c'])
        
        im = ax.imshow(sensitivity_matrix, aspect='auto', origin='lower', cmap=better_rose_cmap, vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(k_multipliers)))
        ax.set_xticklabels([f'{m:.2f}x' for m in k_multipliers], fontsize=14)
        ax.set_yticks(np.arange(len(L_multipliers)))
        ax.set_yticklabels([f'{m:.2f}x' for m in L_multipliers], fontsize=14)
        ax.set_xlabel('Uniform Growth Rate (k) Multiplier', fontsize=16)
        ax.set_ylabel('Uniform Capacity Limit (L) Multiplier', fontsize=16)
        ax.grid(False)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Probability of Achieving Triple Target', fontsize=16)
        cbar.ax.tick_params(labelsize=14)
        for i in range(len(L_multipliers)):
            for j in range(len(k_multipliers)):
                text_color = 'white' if sensitivity_matrix[i, j] > 0.5 else '#1f2531'
                ax.text(j, i, f'{sensitivity_matrix[i, j]:.2f}', ha="center", va="center", 
                    color=text_color, fontweight='bold', fontsize=14)

    def plot_model_comparison(self, all_model_bau_timeseries, output_dir='output'):
        """
        Plots a comparison of Solar PV and Wind energy forecasts from different models,
        including historical fits for validation.
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=300)
        # fig.suptitle('Model Comparison: Historical Fit and Future Projections', 
        #             fontsize=18, fontweight='bold', y=0.98)
        
        # Energy types to plot
        energy_types_to_plot = ['Solar photovoltaic', 'Wind energy']
        axes = [ax1, ax2]
        titles = ['(a) Solar Photovoltaic', '(b) Wind Energy']
        
        # Model colors
        model_colors = {'Richards': '#444577', 'Logistic': '#c65861', 'Gompertz': '#ffa725'}
        
        # Store R² and m values for each energy type and model
        r2_values = {energy: {} for energy in energy_types_to_plot}
        m_values = {energy: {} for energy in energy_types_to_plot}
        L_values = {energy: {} for energy in energy_types_to_plot}
        k_values = {energy: {} for energy in energy_types_to_plot}
        t0_values = {energy: {} for energy in energy_types_to_plot}

        for idx, (energy_type, ax, title) in enumerate(zip(energy_types_to_plot, axes, titles)):
            # Create inset axes for long-term trend
            # Position: [left, bottom, width, height] in axes coordinates
            # if idx == 0:  # Solar PV - position in upper right
            #     inset_ax = ax.inset_axes([0.30, 0.30, 0.35, 0.35])
            # else:  # Wind - position in upper right
            #     inset_ax = ax.inset_axes([0.30, 0.40, 0.35, 0.35])
            
            # Plot historical data for this energy type
            hist_years = self.historical_data['Year'].values
            hist_data = self.historical_data[energy_type] / 1000  # to GW
            ax.scatter(hist_years, hist_data, color='#1f2531', label='Historical Data', 
                    s=40, zorder=5, alpha=0.8, edgecolors='white', linewidth=1)
            
            # Also plot in inset
            # inset_ax.scatter(hist_years, hist_data, color='#1f2531', s=15, zorder=5, 
            #                 alpha=0.8, edgecolors='white', linewidth=0.5)
            
            # Define future years
            future_years = np.arange(self.base_year + 2, self.target_year + 1)
            extended_years = np.arange(2000, 2061)  # For inset plot
            
            # For each model, calculate and plot historical fits and predictions
            for model_type in all_model_bau_timeseries.keys():
                # Skip Gompertz for Wind energy completely
                if model_type == 'Gompertz' and energy_type == 'Wind energy':
                    continue
                    
                color = model_colors.get(model_type, '#be588d')
                
                # Load posteriors for this model type
                self.load_posteriors_for_model(model_type)
                
                if energy_type in self.posteriors:
                    posterior = self.posteriors[energy_type].posterior
                    
                    # Get mean parameters
                    L_mean = float(posterior['L'].mean())
                    k_mean = float(posterior['k'].mean())
                    t0_mean = float(posterior['t0'].mean())
                    
                    # Get m value based on model type
                    if model_type == 'Richards' and 'm' in posterior:
                        m_mean = float(posterior['m'].mean())
                        m_values[energy_type][model_type] = m_mean
                    elif model_type == 'Logistic':
                        m_values[energy_type][model_type] = 2.0
                    elif model_type == 'Gompertz':
                        m_values[energy_type][model_type] = 1.0
                        
                    L_values[energy_type][model_type] = L_mean
                    k_values[energy_type][model_type] = k_mean
                    t0_values[energy_type][model_type] = t0_mean

                    # Get the appropriate curve function
                    curve_func = self.curve_functions[energy_type]
                    
                    # Calculate historical fit
                    if model_type == 'Richards' and 'm' in posterior:
                        m_mean = float(posterior['m'].mean())
                        hist_fit = curve_func(hist_years, L_mean, k_mean, t0_mean, m_mean) / 1000  # to GW
                    else:
                        hist_fit = curve_func(hist_years, L_mean, k_mean, t0_mean, None) / 1000  # to GW
                    
                    # Calculate extended fit for inset (2000-2060)
                    if model_type == 'Richards' and 'm' in posterior:
                        extended_fit = curve_func(extended_years, L_mean, k_mean, t0_mean, m_mean) / 1000
                    else:
                        extended_fit = curve_func(extended_years, L_mean, k_mean, t0_mean, None) / 1000
                    
                    # Plot extended fit in inset
                    # inset_ax.plot(extended_years, extended_fit, color=color, lw=1.5, 
                    #             alpha=0.8, label=model_type)
                    
                    # Calculate R²
                    ss_res = np.sum((hist_data.values - hist_fit) ** 2)
                    ss_tot = np.sum((hist_data.values - np.mean(hist_data.values)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    r2_values[energy_type][model_type] = r2
                    
                    # Plot historical fit with dashed line
                    ax.plot(hist_years, hist_fit, color=color, lw=2, linestyle='--', 
                        alpha=0.7, label=f'{model_type} Historical Fit')
                    
                    # Get future predictions for this specific energy type
                    n_simulations = all_model_bau_timeseries[model_type].shape[0]
                    energy_predictions = np.zeros((n_simulations, len(future_years)))
                    
                    # Simulate predictions for this specific energy type
                    sample_indices = np.random.choice(
                        len(posterior.draw) * len(posterior.chain), 
                        n_simulations, 
                        replace=True
                    )
                    
                    L_samples = posterior['L'].values.flatten()[sample_indices]
                    k_samples = posterior['k'].values.flatten()[sample_indices]
                    t0_samples = posterior['t0'].values.flatten()[sample_indices]
                    
                    if model_type == 'Richards' and 'm' in posterior:
                        m_samples = posterior['m'].values.flatten()[sample_indices]
                    else:
                        m_samples = [None] * n_simulations

                    for i in range(n_simulations):
                        pred = curve_func(future_years, L_samples[i], k_samples[i], 
                                        t0_samples[i], m_samples[i])
                        energy_predictions[i, :] = pred / 1000  # to GW
                    
                    mean_pred = np.mean(energy_predictions, axis=0)
                    
                    # Connect last historical fit point to first prediction point
                    transition_years = [hist_years[-1], future_years[0]]
                    transition_values = [hist_fit[-1], mean_pred[0]]
                    ax.plot(transition_years, transition_values, color=color, lw=2, alpha=0.5)
                    
                    # Plot future forecast
                    ax.plot(future_years, mean_pred, label=f'{model_type} Forecast', 
                            color=color, lw=3, solid_capstyle='round')
                    
                    # Add confidence interval for Richards model
                    if model_type == 'Richards':
                        lower_ci = np.percentile(energy_predictions, 2.5, axis=0)
                        upper_ci = np.percentile(energy_predictions, 97.5, axis=0)
                        ax.fill_between(future_years, lower_ci, upper_ci, alpha=0.2, 
                                    color=color, label='Richards 95% CI')
            
            # Style the inset
            # inset_ax.set_xlim(2000, 2060)
            # inset_ax.set_ylim(0, inset_ax.get_ylim()[1] * 1.1)
            # inset_ax.set_xlabel('Year', fontsize=8)
            # inset_ax.set_ylabel('Capacity (GW)', fontsize=8)
            # inset_ax.tick_params(axis='both', which='major', labelsize=7)
            # inset_ax.grid(True, alpha=0.3, linewidth=0.5)
            
            # Add a subtle border to the inset
            # for spine in inset_ax.spines.values():
            #     spine.set_edgecolor('#39687d')
            #     spine.set_linewidth(1.5)
            
            # Add inflection point indicators if applicable
            # for model_type in t0_values[energy_type].keys():
            #     if model_type == 'Gompertz' and energy_type == 'Wind energy':
            #         continue
            #     t0 = t0_values[energy_type][model_type]
            #     if 2000 <= t0 <= 2060:
            #         inset_ax.axvline(x=t0, color=model_colors[model_type], 
            #                     linestyle=':', alpha=0.5, linewidth=1)
            
            # # Add vertical line at 2023 and 2030 in inset
            # inset_ax.axvline(x=2023, color='gray', linestyle='--', alpha=0.4, lw=1)
            # inset_ax.axvline(x=2030, color='gray', linestyle='--', alpha=0.4, lw=1)
            
            # Add R² and m values text box with white background
            info_text = "Model Performance:\n"
            info_text += "─" * 20 + "\n"
            for model_type in r2_values[energy_type].keys():
                r2 = r2_values[energy_type][model_type]
                m = m_values[energy_type].get(model_type, 'N/A')
                L = L_values[energy_type].get(model_type, 'N/A')
                k = k_values[energy_type].get(model_type, 'N/A')
                t0 = t0_values[energy_type].get(model_type, 'N/A')
                # convert L to GW
                if isinstance(L, float):
                    L = L / 1000

                if isinstance(m, float):
                    info_text += f"{model_type}:\n  R² = {r2:.3f}  m = {m:.2f} "
                else:
                    info_text += f"{model_type}:\n  R² = {r2:.3f}  m = {m} "
                
                if isinstance(L, float):
                    info_text += f"  L = {L:.0f} "
                else:
                    info_text += f"  L = {L} "
                
                if isinstance(k, float):
                    info_text += f"  k = {k:.2f} "
                else:
                    info_text += f"  k = {k} "
                
                if isinstance(t0, float):
                    info_text += f"  t0 = {t0:.0f}\n"
                else:
                    info_text += f"  t0 = {t0}\n"
                
            props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, 
                        edgecolor='#39687d', linewidth=1.5)
            ax.text(0.02, 0.98, info_text.strip(), transform=ax.transAxes, 
                    fontsize=14, verticalalignment='top', bbox=props, 
                    family='monospace')  # Use monospace for better alignment
            
            # Styling
            ax.set_xlabel('Year', fontsize=16, fontweight='bold')
            ax.set_ylabel('Capacity (GW)', fontsize=16, fontweight='bold')
            ax.set_title(f'{title} Capacity: Historical Fit and Projections', 
                        fontsize=16, pad=10, fontweight='bold')
            
            # Grid styling
            ax.grid(False)
            ax.minorticks_on()
            
            # Set axis limits
            ax.set_xlim(hist_years.min() - 1, self.target_year + 1)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
            
            # Legend
            ax.legend(fontsize=14, loc = 'center left', frameon=True, fancybox=True, 
                    shadow=False, ncol=1)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'solar_wind_model_comparison.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"\nSolar PV and Wind model comparison plot saved to {filepath}")
        plt.close()


# ==================== Main Execution Functions ====================

HISTORICAL_DATA_CSV = """Year,Solar photovoltaic,Wind energy,Hydropower,Bioenergy,Geothermal energy
2000,1224.691,16956.675,697169.998,28330.12,8272.7
2001,1476.414,23947.906,709059.493,31489.125,7960.91
2002,1822.521,30711.984,720271.816,33077.425,8113.45
2003,2349.945,38645.057,740387.163,35302.886,8188.35
2004,3436.187,47657.213,761767.442,38259.582,8208.88
2005,4944.92,58382.656,784161.201,43528.635,8556.116
2006,6499.567,73135.614,805564.569,47468.337,8758.696
2007,8976.022,91523.785,832413.899,50623.092,9012.936
2008,15250.752,115540.989,861919.028,54572.964,9318.816
2009,23591.166,150120.756,891731.386,61509.109,9754.316
2010,41576.957,181075.824,925877.098,66226.215,9914.316
2011,73964.989,220205.239,953921.047,72652.922,10058.666
2012,104228.953,267271.174,984711.803,77618.447,10517.666
2013,141411.963,299914.327,1029604.688,85014.713,10786.366
2014,180759.166,349417.734,1067333.698,90843.273,11248.716
2015,229058.367,416334.765,1099509.822,96842.107,11846.966
2016,301186.358,466956.459,1130040.822,105674.616,12172.974
2017,396316.204,515044.819,1151053.663,111410.236,12729.613
2018,492640.787,563839.619,1173728.032,118607.736,13196.361
2019,595492.499,622773.377,1192459.1,125244.338,13824.041
2020,728405.331,733719.062,1212924.784,133199.851,14157.341
2021,873858.454,824601.993,1235882.207,139510.667,14432.038
2022,1073135.531,901230.776,1260882.706,145896.296,14652.955
2023,1418968.982,1017198.786,1267902.92,150261.17,14845.935
2024,1865489.954,1132836.624,1283040.695,150762.863,15427.487
"""

def fit_all_models(output_dir='output'):
    """Fit models for all energy types and save posteriors."""
    df = pd.read_csv(StringIO(HISTORICAL_DATA_CSV))
    energy_types = ['Solar photovoltaic', 'Wind energy', 'Hydropower',
                    'Bioenergy', 'Geothermal energy']
    
    # Define which models to run. Richards is commented out as requested.
    models_to_run = []
    # models_to_run = ['Logistic', 'Gompertz']
    # To run the Richards model as well, uncomment the following line:
    models_to_run.append('Richards') 

    for model_type in models_to_run:
        print(f"\n{'='*20} FITTING {model_type.upper()} MODELS {'='*20}")
        for energy_type in energy_types:
            print(f"\nFitting model for {energy_type}...")
            years, data = df['Year'].values, df[energy_type].values
            mask = data > 0
            years_filtered, data_filtered = years[mask], data[mask]

            if len(data_filtered) < 5:
                print(f"Skipping {energy_type} due to insufficient data points.")
                continue
                
            model = GrowthForecastModel(model_type=model_type, output_dir=output_dir)
            model.fit(years_filtered, data_filtered, energy_type=energy_type)
            filepath = model.save_posterior()
            print(f"Saved posterior to: {filepath}")
            print(f"Parameters (mean): {model.parameters}")
            print(f"R² = {model.r2:.4f}")

def run_full_analysis_and_comparison(output_dir='output'):
    """
    Run the complete scenario analysis for multiple models and generate comparison plots.
    """
    df = pd.read_csv(StringIO(HISTORICAL_DATA_CSV))
    # Run analysis for all available models
    models_to_analyze = ['Richards', 'Logistic', 'Gompertz']
    all_results = {}
    all_bau_timeseries = {}

    analyzer = PolicyScenarioAnalyzer(
        posterior_dir=os.path.join(output_dir, 'posteriors'),
        base_year=2022, target_year=2030
    )
    analyzer.set_historical_data(df)

    base_total = analyzer.get_base_year_total()
    print(f"\nBase year (2022) total capacity: {base_total:,.0f} MW ({base_total/1000:.1f} GW)")
    print(f"Triple target for 2030: {base_total * 3:,.0f} MW ({base_total * 3 / 1000:.1f} GW)")

    for model_type in models_to_analyze:
        try:
            print(f"\n{'='*80}")
            print(f"STARTING ANALYSIS FOR MODEL: {model_type.upper()}")
            print(f"{'='*80}")
            
            analyzer.load_posteriors_for_model(model_type)
            
            print("\nRunning scenario analysis...")
            results = analyzer.run_all_scenarios(n_simulations=5000)
            all_results[model_type] = results

            print("\nGenerating visualizations...")
            analyzer.plot_combined_scenario_analysis(results, model_type, output_dir)
            
            # Simulate BAU timeseries for the final comparison plot
            bau_scen = analyzer.define_scenarios()['S0']
            all_bau_timeseries[model_type] = analyzer.simulate_scenario(
                bau_scen['modifications'], n_simulations=5000, return_timeseries=True)

            # Generate and print summary table for this model
            summary_data = [{'Scenario': r['name'], 'Mean Capacity (GW)': r['mean']/1000,
                             '95% CI (GW)': f"[{r['lower_95']/1000:.1f} - {r['upper_95']/1000:.1f}]",
                             'P(Achieve 3x Target)': f"{r['triple_probability']:.1%}",
                             'Growth vs. Base': f"{r['growth_ratio']:.2f}x"} for r in results.values()]
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(output_dir, f'scenario_summary_{model_type}.csv'), index=False)
            
            print(f"\nSCENARIO ANALYSIS RESULTS ({model_type} Model)")
            print("-" * 50)
            print(summary_df.to_string(index=False))
            print("-" * 50)

        except FileNotFoundError as e:
            print(f"\nCould not run analysis for {model_type}: {e}")
            print("Please ensure you have run 'fit_all_models()' for this model type first.")
    
    # After analyzing all models, create the final comparison plot
    if len(all_bau_timeseries) > 1:
        print(f"\n{'='*80}")
        print("GENERATING FINAL MODEL COMPARISON PLOT")
        print(f"{'='*80}")
        analyzer.plot_model_comparison(all_bau_timeseries, output_dir)
    else:
        print("\nSkipping model comparison plot: only one model's results were available.")

# ==================== Main Entry Point ====================

if __name__ == "__main__":
    print("Energy Forecast Model with Policy Sensitivity Analysis")
    print("="*60)
    print("\nPlease uncomment the desired function below to run:")

    # --- Step 1: Fit models and save posteriors (run this first) ---
    # This will run the Logistic and Gompertz models as requested.
    # To run the Richards model as well, change the 'models_to_run' list inside the function.
    # It can take a significant amount of time.
    fit_all_models(output_dir='output')

    # --- Step 2: Run full analysis and create comparison plots ---
    # This requires the '.nc' posterior files generated by Step 1 for
    # Richards, Logistic, and Gompertz models to be present.
    # run_full_analysis_and_comparison(output_dir='output')
