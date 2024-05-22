import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import math
import os

def merge_and_encode(household_df, person_df, household_vars, person_vars):
    """
    Merges person-level data into household-level data, counts occurrences of specified person variables,
    and one-hot encodes specified household variables, then returns an enriched household DataFrame.

    Parameters:
    - household_df: DataFrame containing household-level data.
    - person_df: DataFrame containing person-level data.
    - household_vars: List of columns from household_df to keep and potentially encode.
    - person_vars: List of columns from person_df to count and merge into household_df.

    Returns:
    - DataFrame: The enriched household DataFrame with person variable counts and one-hot encoded variables.
    """

    # Count occurrences of specified person variables
    person_counts = person_df.groupby(['hid'] + person_vars).size().unstack(fill_value=0)
    person_counts.columns = [f'{var}_{int(col)}' for var in person_vars for col in person_counts.columns]

    # Select specified columns from household_df and merge with person_counts
    household_selected = household_df[['hid'] + household_vars].copy()
    household_enriched = pd.merge(household_selected, person_counts, on='hid', how='left')

    # One-hot encode specified variables in household_vars
    for var in household_vars:
        if household_enriched[var].dtype == 'object' or len(household_enriched[var].unique()) > 2:
            household_enriched = pd.get_dummies(household_enriched, columns=[var], prefix=var, dtype=int)

    return household_enriched

def read_marginal_data(path, header_rows=[0, 1], index_col=0):
    """Reads CSV file with multi-level header and sets MultiIndex for columns."""
    df = pd.read_csv(path, header=header_rows, index_col=index_col)
    df.columns = pd.MultiIndex.from_tuples([(col[0], int(col[1])) for col in df.columns])
    return df

def plot_variables_in_subplots(variable_names_list, synthetic_df, marginal_df, geo, category_title):
    num_variables = len(variable_names_list)
    num_cols = 2
    num_rows = math.ceil(num_variables / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 6))
    axes = axes.flatten() if num_variables > 1 else [axes]

    for i, variable in enumerate(variable_names_list):
        if i >= len(axes):
            break
        if variable in synthetic_df.columns and variable in marginal_df.columns.get_level_values(0):
            geo_synthetic_subset = synthetic_df[synthetic_df['geo'] == geo]
            marginal_values = marginal_df.xs(geo).loc[variable]
            synthetic_values = geo_synthetic_subset[variable].value_counts()

            categories = marginal_values.index.intersection(synthetic_values.index).tolist()
            marginal_counts = [marginal_values.loc[cat] for cat in categories]
            synthetic_counts = [synthetic_values.loc[cat] for cat in categories]

            x = np.arange(len(categories))
            bar_width = 0.4

            ax = axes[i]
            bars_marginal = ax.barh(x - bar_width / 2, marginal_counts, height=bar_width, label='Marginal')
            bars_synthetic = ax.barh(x + bar_width / 2, synthetic_counts, height=bar_width, label='Synthetic')

            ax.set_title(f'{variable}')
            ax.set_xlabel('Counts')
            ax.set_ylabel('Categories')
            ax.set_yticks(x)
            ax.set_yticklabels(categories)
            ax.legend()

            # Add text on top of the bars
            for bar in bars_marginal:
                width = bar.get_width()
                label_x_pos = bar.get_x() + width + 1
                ax.text(label_x_pos, bar.get_y(), f'{width:.0f}', va='center')

            for bar in bars_synthetic:
                width = bar.get_width()
                label_x_pos = bar.get_x() + width + 1
                ax.text(label_x_pos, bar.get_y(), f'{width:.0f}', va='center')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'{category_title} Variables for Geo {geo}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def group_marginal_geo_by_region(household_marginal, geo_region_mapping):
    """
    Groups household data by region using a geo-region mapping DataFrame,
    without altering the original household_marginal DataFrame.

    Parameters:
    - household_marginal: DataFrame containing household marginal data, indexed by 'geo'.
    - geo_region_mapping: DataFrame that maps 'geo' to 'region'.

    Returns:
    - DataFrame grouped by 'region' with summed variables.
    """
    geo_to_region = dict(zip(geo_region_mapping['geo'], geo_region_mapping['region']))
    household_marginal_copy = household_marginal.reset_index().copy()
    household_marginal_copy['region'] = household_marginal_copy['geo'].map(geo_to_region)
    household_marginal_grouped_by_region = household_marginal_copy.groupby('region').sum()
    return household_marginal_grouped_by_region

def plot_marginal_distribution(household_marginal, region_household_marginal):
    choice = input("Please enter 'geo' or 'region': ").strip().lower()

    if choice == 'geo':
        geo = int(input("Please enter the geographical area code: "))
        hsize_filter = input("Please enter the control variable: ")
        geo_row = household_marginal.loc[geo]
        hsize_columns = [col for col in household_marginal.columns if hsize_filter in col]
        hsize_data = geo_row[hsize_columns]

        plt.figure(figsize=(8, 3))
        plt.bar(range(len(hsize_data)), hsize_data.values, color='skyblue')
        plt.title(f'Distribution of {hsize_filter} at Geo Level {geo}')
        plt.xlabel('Categories')
        plt.ylabel('Count')
        plt.xticks(range(len(hsize_data)), hsize_data.index, rotation=45)
        plt.show()

    elif choice == 'region':
        region_code = int(input("Please enter the geographical area code: "))
        rhsize_filter = input("Please enter the control variable: ")
        region_row = region_household_marginal.loc[region_code]
        rhsize_columns = [col for col in region_household_marginal.columns if rhsize_filter in col]
        rhsize_data = region_row[rhsize_columns]

        plt.figure(figsize=(8, 3))
        plt.bar(range(len(rhsize_data)), rhsize_data.values, color='skyblue')
        plt.title(f'Distribution of {rhsize_filter} at Region Level {region_code}')
        plt.xlabel('Categories')
        plt.ylabel('Count')
        plt.xticks(range(len(rhsize_data)), rhsize_data.index, rotation=45)
        plt.show()

    else:
        print("Invalid input, please enter 'geo' or 'region'.")

def map_sample_geo_to_geo(household_sample, geo_sample_mapping):
    return household_sample.merge(geo_sample_mapping, on='sample_geo')

def map_geo_to_region(geo_mapped_sample, region_geo_mapping):
    return geo_mapped_sample.merge(region_geo_mapping, on='geo')

def plot_sample_distribution(household_sample, geo_sample_mapping, region_geo_mapping):
    choice = input("Please enter 'geo' or 'region': ").strip().lower()

    if choice == 'geo':
        geo_id = int(input("Please enter geographical area code: "))
        variable = input("Please enter the control variable (e.g., 'hsize'): ")
        sample_geo_ids = geo_sample_mapping[geo_sample_mapping['geo'] == geo_id]['sample_geo'].unique()
        data_to_plot = household_sample[household_sample['sample_geo'].isin(sample_geo_ids)]

        plt.figure(figsize=(8, 3))
        data_to_plot[variable].value_counts().sort_index().plot(kind='bar', color='skyblue')
        plt.title(f'Sample Distribution of {variable} for GEO ID {geo_id}')
        plt.xlabel('Categories')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

    elif choice == 'region':
        region_id = int(input("Please enter geographical area code: "))
        variable = input("Please enter the control variable (e.g., 'rhsize'): ")
        geo_ids = region_geo_mapping[region_geo_mapping['region'] == region_id]['geo'].unique()
        sample_geo_ids = geo_sample_mapping[geo_sample_mapping['geo'].isin(geo_ids)]['sample_geo'].unique()
        data_to_plot = household_sample[household_sample['sample_geo'].isin(sample_geo_ids)]

        plt.figure(figsize=(8, 3))
        data_to_plot[variable].value_counts().sort_index().plot(kind='bar', color='skyblue')
        plt.title(f'Sample Distribution of {variable} for REGION ID {region_id}')
        plt.xlabel('Categories')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

    else:
        print("Invalid input, please enter 'geo' or 'region'.")
