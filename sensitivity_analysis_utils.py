import numpy as np 
import pandas as pd 
import statistics
import matplotlib.pyplot as plt
import os 

example = pd.read_csv('sensitivity_bond/sensitivity_1725286982144.csv')
print(example.head())

def get_normalized_sensitivity(features, df):
    '''input df -> contains sensitivity for each feature
    features -> list of features to use 
    return a new df with column features and sensitivity_normalized
    '''
    sensitivity_sum = df['0'].sum()
    sensitivities = []
    normalized_sensitivities = []
    for feature in features:
        row = df[df['Unnamed: 0']==feature]
        if not row.empty:
            sens = row['0'].values[0]
            sensitivities.append(sens)
            normalized_sensitivities.append(sens/sensitivity_sum)
        else: 
            raise ValueError(f'feauture not found: {feature}')
    return pd.DataFrame({
        "features": features,
        "sensitivity": sensitivities, 
        "sensitivity_normalized": normalized_sensitivities
    })
def plot_box_chart(features, medians, lower_percentiles, upper_percentiles, targets=None, order = None, title='', save=False):
        if not order:
            order = np.argsort(medians)[::-1]
        features = [features[i] for i in order]
        medians = [medians[i] for i in order]
        lower_percentiles = [lower_percentiles[i] for i in order]
        upper_percentiles = [upper_percentiles[i] for i in order]
        lower_errors = [mean - lower for mean, lower in zip(medians, lower_percentiles)]
        upper_errors = [upper - mean for mean, upper in zip(medians, upper_percentiles)]
        plt.figure()
        colors = ['blue', 'darkgreen', 'blue', 'blue', 'pink', 'pink', 'pink', 'lightgreen', 'pink', 'lightgreen', 'pink', 'lightgreen', 'lightgreen', 'lightgreen']
        bar_plot = plt.barh(features, medians, xerr=[lower_errors, upper_errors], capsize=5, color=colors)
        if targets: 
            for i, bar in enumerate(bar_plot):
                plt.scatter(targets[i], bar.get_y() + bar.get_height()/2, color='red', marker='o', zorder=5)
        plt.xlabel('Sensitivity')
        plt.title('Sensitivity Box Chart '+title)
        plt.gca().invert_yaxis()
        if save:
            plt.savefig("individual_variable_box_chart{}.png".format(title), dpi=300, bbox_inches='tight')
        plt.show()


def get_stats_for_feature(feature, selected_models, all_models_csv):
    sensitivity_list = [] 
    for model in selected_models:
        curr_df = all_models_csv[model]
        if feature in curr_df['features'].values:
            sensitivity = curr_df[curr_df['features'] == feature]['sensitivity_normalized'].values[0]
            sensitivity_list.append(sensitivity)
    if sensitivity_list:
        mean = np.mean(sensitivity_list)
        p25 = np.percentile(sensitivity_list, 25)
        p75 = np.percentile(sensitivity_list, 75)
        return mean, p25, p75

feature_names = ['flow', 'aum', 'dividend_yield', 'exp_ratio_net', 'fee_rate',
                       'turnover_ratio',
                       'st_reversal', 'st_momentum', 'momentum', 'int_momentum', 'lt_momentum', 'lt_reversal',
                       'age', 'R_squared', 'fund_st_reversal', 'fund_st_momentum', 'fund_momentum', 'family_flow',
                       'family_aum', 'family_age', 'family_fund_momentum', 'no_funds']
__all__ = ['feature_names', 'plot_box_chart', 'get_stats_for_feature', 'get_normalized_sensitivity']