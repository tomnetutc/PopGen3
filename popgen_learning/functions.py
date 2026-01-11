import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import math
import os


def merge_and_encode(household_df, person_df, household_vars, person_vars):
    """
    Performs one-hot encoding on specified household variables and independently counts occurrences of specified person-level variables (without interactions).
    Combines these into a single household-level DataFrame, with household variables appearing first, followed by person-level count variables.

    Parameters:
    - household_df: DataFrame containing household-level data.
    - person_df: DataFrame containing person-level data.
    - household_vars: List of household-level variables to encode using one-hot encoding.
    - person_vars: List of person-level variables whose counts are independently added to the household data.

    Returns:
    - household_enriched: DataFrame enriched with one-hot encoded household variables and counts of person-level variables.
    """
    # Select household variables for processing
    household_selected = household_df[['hid'] + household_vars].copy()

    # Perform one-hot encoding on specified household variables
    for var in household_vars:
        if household_selected[var].dtype == 'object' or household_selected[var].nunique() > 2:
            household_selected = pd.get_dummies(household_selected, columns=[var], prefix=var, dtype=int)

    # Independently count occurrences for each person-level variable
    person_counts = household_selected[['hid']].copy()
    for var in person_vars:
        counts = person_df.groupby(['hid', var]).size().unstack(fill_value=0)
        counts.columns = [f'{var}_{col}' for col in counts.columns]
        person_counts = person_counts.merge(counts, on='hid', how='left')

    # Replace missing values with zeros (for households with no corresponding persons)
    person_counts.fillna(0, inplace=True)

    # Merge household-level data with person-level counts
    household_enriched = household_selected.merge(person_counts, on='hid', how='left')

    # Arrange columns: household variables first, then person-level count variables
    ordered_cols = ['hid'] + [col for col in household_enriched.columns if col != 'hid' and col not in person_counts.columns] + \
                   [col for col in person_counts.columns if col != 'hid']

    return household_enriched[ordered_cols]

def merge_and_encode_joint(household_df, person_df, household_vars, person_vars):
    """
    Household variables: joint one-hot encoding
    Person variables: joint combination counts

    Parameters:
    - household_df: Household-level DataFrame.
    - person_df: Person-level DataFrame.
    - household_vars: List of household variables for joint one-hot encoding.
    - person_vars: List of person-level variables for joint combination counting.

    Returns:
    - DataFrame with joint-encoded household variables and joint-combination person-level counts.
    """
    # Step 1: Household联合编码
    household_selected = household_df[['hid'] + household_vars].copy()
    household_joint_col = '_'.join(household_vars)
    household_selected[household_joint_col] = household_selected[household_vars].astype(str).agg('_'.join, axis=1)

    # Household联合one-hot编码
    household_encoded = pd.get_dummies(household_selected[['hid', household_joint_col]],
                                       columns=[household_joint_col], dtype=int)

    # Step 2: Person联合变量计数
    person_selected = person_df[['hid'] + person_vars].copy()
    person_joint_col = '_'.join(person_vars)
    person_selected[person_joint_col] = person_selected[person_vars].astype(str).agg('_'.join, axis=1)

    # 对每个家庭计数person联合变量组合的出现次数
    person_counts = person_selected.groupby(['hid', person_joint_col]).size().unstack(fill_value=0)

    # 更新person_counts的列名
    person_counts.columns = [f'{person_joint_col}_{col}' for col in person_counts.columns]

    # Step 3: Household 和 Person数据合并
    final_df = household_encoded.merge(person_counts, on='hid', how='left')

    # 填充缺失值（没有person的家庭）
    final_df.fillna(0, inplace=True)

    return final_df

def read_marginal_data(path, header_rows=[0, 1], index_col=0):
    """Reads CSV file with multi-level header and sets MultiIndex for columns."""
    df = pd.read_csv(path, header=header_rows, index_col=index_col)
    df.columns = pd.MultiIndex.from_tuples([(col[0], int(col[1])) for col in df.columns])
    return df

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

def map_sample_geo_to_geo(household_sample, geo_sample_mapping):
    return household_sample.merge(geo_sample_mapping, on='sample_geo')

def map_geo_to_region(geo_mapped_sample, region_geo_mapping):
    return geo_mapped_sample.merge(region_geo_mapping, on='geo')

def plot_marginal_distribution(household_marginal, region_household_marginal):
    choice = input("Please enter 'region' or 'geo': ").strip().lower()

    if choice == 'geo':
        geo = int(input("Please enter the geographical area code (e.g., '4013010101'): "))
        var_filter = input("Please enter the control variable (e.g., 'hsize'): ").strip()
        if not var_filter:
            print("Invalid input: control variable name cannot be empty.")
            return

        geo_row = household_marginal.loc[geo]
        filtered_cols = [col for col in household_marginal.columns if var_filter in str(col)]
        data = geo_row[filtered_cols]

        # 提取类别数字
        categories = [int(col[1]) if isinstance(col, tuple) else col for col in data.index]
        total = data.sum()
        percentages = data.values / total * 100

        plt.figure(figsize=(8, 4))
        bars = plt.bar(categories, percentages, color='steelblue', edgecolor='black', linewidth=0.6)

        plt.title(f'{var_filter.upper()} Distribution at Geo {geo}', fontsize=14, weight='bold')
        plt.xlabel('Categories', fontsize=11)
        plt.ylabel('Percentage (%)', fontsize=11)
        plt.xticks(categories)
        plt.ylim(0, 100)

        # 添加总数信息
        plt.text(0.98, 0.95, f"Marginal: N = {int(total):,}",
                 ha='right', va='top', transform=plt.gca().transAxes,
                 fontsize=10, color='gray')

        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

    elif choice == 'region':
        region_code = int(input("Please enter the geographical area code (e.g., '8'): "))
        var_filter = input("Please enter the control variable (e.g., 'rhsize'): ").strip()
        if not var_filter:
            print("Invalid input: control variable name cannot be empty.")
            return

        region_row = region_household_marginal.loc[region_code]
        filtered_cols = [col for col in region_household_marginal.columns if var_filter in str(col)]
        data = region_row[filtered_cols]

        # 提取类别数字
        categories = [int(col[1]) if isinstance(col, tuple) else col for col in data.index]
        total = data.sum()
        percentages = data.values / total * 100

        plt.figure(figsize=(8, 4))
        bars = plt.bar(categories, percentages, color='steelblue', edgecolor='black', linewidth=0.6)

        plt.title(f'{var_filter.upper()} Distribution at Region {region_code}', fontsize=14, weight='bold')
        plt.xlabel('Categories', fontsize=11)
        plt.ylabel('Percentage (%)', fontsize=11)
        plt.xticks(categories)
        plt.ylim(0, 100)

        # 添加总数信息
        plt.text(0.98, 0.95, f"Marginal: N = {int(total):,}",
                 ha='right', va='top', transform=plt.gca().transAxes,
                 fontsize=10, color='gray')

        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

    else:
        print("Invalid input, please enter 'geo' or 'region'.")

def plot_distribution_comparison(household_marginal, region_household_marginal, household_sample=None):
    choice = input("Please enter 'region' or 'geo': ").strip().lower()

    if choice not in ['geo', 'region']:
        print("Invalid input, please enter 'region' or 'geo'.")
        return

    area_code = int(input("Please enter the geographical area code (e.g., '8' or '4013010101'): "))
    var_filter = input("Please enter the control variable (e.g., 'rhsize' or 'hsize'): ").strip()
    if not var_filter:
        print("Invalid input: control variable name cannot be empty.")
        return

    if choice == 'geo':
        row = household_marginal.loc[area_code]
        filtered_cols = [col for col in household_marginal.columns if var_filter in str(col)]
    else:
        row = region_household_marginal.loc[area_code]
        filtered_cols = [col for col in region_household_marginal.columns if var_filter in str(col)]

    marginal_data = row[filtered_cols]
    marginal_categories = [int(col[1]) if isinstance(col, tuple) else col for col in marginal_data.index]
    marginal_total = marginal_data.sum()
    marginal_pct = marginal_data.values / marginal_total * 100

    sample_pct = None
    sample_total = None
    sample_categories = None

    if household_sample is not None:
        sample_value_counts = household_sample[var_filter].value_counts().sort_index()
        sample_total = sample_value_counts.sum()
        sample_categories = sample_value_counts.index.tolist()
        sample_pct = sample_value_counts.values / sample_total * 100

    categories_all = sorted(set(marginal_categories) | set(sample_categories or []))
    x = np.arange(len(categories_all))
    width = 0.45

    fig, ax = plt.subplots(figsize=(8, 4))
    bar1 = ax.bar(
        x - width/2,
        [marginal_pct[marginal_categories.index(c)] if c in marginal_categories else 0 for c in categories_all],
        width=width,
        label='Marginal',
        color='#1f77b4'
    )

    if sample_pct is not None:
        bar2 = ax.bar(
            x + width/2,
            [sample_pct[sample_categories.index(c)] if c in sample_categories else 0 for c in categories_all],
            width=width,
            label='Sample',
            color='#ff7f0e'
        )

    ax.set_title(f'{var_filter.upper()} Distribution at {"Region" if choice=="region" else "Geo"} {area_code}', fontsize=14, weight='bold')
    ax.set_xlabel('Categories', fontsize=11)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(categories_all)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.4)


    # 添加每个 bar 上的百分比文本
    for i, c in enumerate(categories_all):
        if c in marginal_categories:
            value = marginal_pct[marginal_categories.index(c)]
            ax.text(x[i] - width/2, value + 0.8, f"{value:.1f}%", ha='center', va='bottom',
                    fontsize=8, color='#1f77b4')
        if sample_pct is not None and c in sample_categories:
            value = sample_pct[sample_categories.index(c)]
            ax.text(x[i] + width/2, value + 0.8, f"{value:.1f}%", ha='center', va='bottom',
                    fontsize=8, color='#ff7f0e')


    # 显示样本大小
    ax.text(0.98, 0.95, f"Marginal: N = {int(marginal_total):,}",
            transform=ax.transAxes, ha='right', va='top', fontsize=10, color='gray')
    if sample_total is not None:
        ax.text(0.98, 0.88, f"Survey: N = {int(sample_total):,}",
                transform=ax.transAxes, ha='right', va='top', fontsize=10, color='gray')

    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def build_region_to_geos(region_geo_mapping):
    """Build a dictionary mapping each region to its associated geo units."""
    return region_geo_mapping.groupby('region')['geo'].apply(list).to_dict()

def compare_all_marginal_distributions_with_person(
    household_sample,
    person_sample,
    weights,
    geo_household_marginal,
    region_household_marginal,
    geo_person_marginal,
    region_person_marginal,
    region_geo_mapping
):
    """
    Compare marginal distributions at geo and region levels with sample data.

    Parameters:
        household_sample (DataFrame): Household sample data.
        person_sample (DataFrame): Person sample data.
        weights (DataFrame): Household weights with geo columns.
        geo_household_marginal (DataFrame): Household marginals by geo.
        region_household_marginal (DataFrame): Household marginals by region.
        geo_person_marginal (DataFrame): Person marginals by geo.
        region_person_marginal (DataFrame): Person marginals by region.
        region_geo_mapping (DataFrame): Mapping of regions to geos.

    Returns:
        tuple: Two dictionaries containing comparison tables for geo and region levels.
    """
    # Standardize weights column names
    weights.columns = weights.columns.astype(str)

    # Extract variables from marginals
    household_vars_geo = sorted(set(var for var, _ in geo_household_marginal.columns))
    household_vars_region = sorted(set(var for var, _ in region_household_marginal.columns))
    person_vars_geo = sorted(set(var for var, _ in geo_person_marginal.columns))
    person_vars_region = sorted(set(var for var, _ in region_person_marginal.columns))

    region_to_geos = build_region_to_geos(region_geo_mapping)
    geo_list = [col for col in weights.columns if col != 'hid' and not col.startswith('region_')]

    all_geo_tables, all_region_tables = {}, {}

    # GEO-LEVEL COMPARISONS
    for geo in geo_list:
        geo_int = int(geo)
        rows = []
        merged_weights = weights[['hid', geo]].dropna(subset=[geo])

        # Household-level comparisons
        merged_hh = pd.merge(household_sample, merged_weights, on='hid', how='inner')
        for var in household_vars_geo:
            if var not in household_sample.columns:
                continue
            mg_counts = geo_household_marginal.loc[geo_int, geo_household_marginal.columns.get_level_values(0) == var]
            mg_dist = (mg_counts / mg_counts.sum()).droplevel(0)

            subset = merged_hh[[var, geo]].dropna()
            uw = subset[var].value_counts(normalize=True)
            wt = subset.groupby(var)[geo].sum() / subset[geo].sum()

            for cat in sorted(set(uw.index) | set(wt.index) | set(mg_dist.index)):
                rows.append({
                    "Variable": var,
                    "Category": cat,
                    "Unweighted %": round(uw.get(cat, 0) * 100, 2),
                    "Weighted %": round(wt.get(cat, 0) * 100, 2),
                    "Marginal %": round(mg_dist.get(cat, 0) * 100, 2),
                    "Delta (W - M) %": round(wt.get(cat, 0) * 100 - mg_dist.get(cat, 0) * 100, 2)
                })

        # Person-level comparisons (GEO)
        merged_person = pd.merge(person_sample, merged_weights, on='hid', how='inner')
        for var in person_vars_geo:
            if var not in person_sample.columns:
                continue
            mg_counts = geo_person_marginal.loc[geo_int, geo_person_marginal.columns.get_level_values(0) == var]
            mg_dist = (mg_counts / mg_counts.sum()).droplevel(0)

            subset = merged_person[[var, geo]].dropna()
            uw = subset[var].value_counts(normalize=True)
            wt = subset.groupby(var)[geo].sum() / subset[geo].sum()

            for cat in sorted(set(uw.index) | set(wt.index) | set(mg_dist.index)):
                rows.append({
                    "Variable": var,
                    "Category": cat,
                    "Unweighted %": round(uw.get(cat, 0) * 100, 2),
                    "Weighted %": round(wt.get(cat, 0) * 100, 2),
                    "Marginal %": round(mg_dist.get(cat, 0) * 100, 2),
                    "Delta (W - M) %": round(wt.get(cat, 0) * 100 - mg_dist.get(cat, 0) * 100, 2)
                })

        all_geo_tables[geo] = pd.DataFrame(rows)

    # REGION-LEVEL COMPARISONS
    for region, geos in region_to_geos.items():
        valid_geos = [str(g) for g in geos if str(g) in weights.columns]
        if not valid_geos:
            continue

        region_col = f"region_{region}"
        weights[region_col] = weights[valid_geos].sum(axis=1)
        rows = []

        # Household-level comparisons (REGION)
        merged_hh = pd.merge(household_sample, weights[['hid', region_col]].dropna(), on='hid', how='inner')
        for var in household_vars_region:
            if var not in household_sample.columns:
                continue
            mg_counts = region_household_marginal.loc[region, region_household_marginal.columns.get_level_values(0) == var]
            mg_dist = (mg_counts / mg_counts.sum()).droplevel(0)

            subset = merged_hh[[var, region_col]].dropna()
            uw = subset[var].value_counts(normalize=True)
            wt = subset.groupby(var)[region_col].sum() / subset[region_col].sum()

            for cat in sorted(set(uw.index) | set(wt.index) | set(mg_dist.index)):
                rows.append({
                    "Variable": var,
                    "Category": cat,
                    "Unweighted %": round(uw.get(cat, 0) * 100, 2),
                    "Weighted %": round(wt.get(cat, 0) * 100, 2),
                    "Marginal %": round(mg_dist.get(cat, 0) * 100, 2),
                    "Delta (W - M) %": round(wt.get(cat, 0) * 100 - mg_dist.get(cat, 0) * 100, 2)
                })

        # Person-level comparisons (REGION)
        merged_person = pd.merge(person_sample, weights[['hid', region_col]].dropna(), on='hid', how='inner')
        for var in person_vars_region:
            if var not in person_sample.columns:
                continue
            mg_counts = region_person_marginal.loc[region, region_person_marginal.columns.get_level_values(0) == var]
            mg_dist = (mg_counts / mg_counts.sum()).droplevel(0)

            subset = merged_person[[var, region_col]].dropna()
            uw = subset[var].value_counts(normalize=True)
            wt = subset.groupby(var)[region_col].sum() / subset[region_col].sum()

            for cat in sorted(set(uw.index) | set(wt.index) | set(mg_dist.index)):
                rows.append({
                    "Variable": var,
                    "Category": cat,
                    "Unweighted %": round(uw.get(cat, 0) * 100, 2),
                    "Weighted %": round(wt.get(cat, 0) * 100, 2),
                    "Marginal %": round(mg_dist.get(cat, 0) * 100, 2),
                    "Delta (W - M) %": round(wt.get(cat, 0) * 100 - mg_dist.get(cat, 0) * 100, 2)
                })

        all_region_tables[str(region)] = pd.DataFrame(rows)

    return all_geo_tables, all_region_tables


# def plot_synthetic_vs_marginal_comparison(
#     housing_synthetic_df,
#     housing_marginal_df,
#     person_synthetic_df,
#     person_marginal_df,
#     geo_col='geo',
#     output_pdf_path=None
# ):
#
#     def plot_comparison(subset_synthetic, marginal_counts, var, entity_type, area):
#         marginal_total = marginal_counts.sum()
#         marginal_dist = marginal_counts / marginal_total
#
#         synthetic_counts = subset_synthetic[var].value_counts()
#         synthetic_total = synthetic_counts.sum()
#         synthetic_dist = synthetic_counts / synthetic_total
#
#         categories = sorted(set(marginal_dist.index) | set(synthetic_dist.index))
#
#         marginal_vals = [marginal_dist.get(cat, 0) * 100 for cat in categories]
#         synthetic_vals = [synthetic_dist.get(cat, 0) * 100 for cat in categories]
#
#         x = np.arange(len(categories))
#         width = 0.45
#
#         fig, ax = plt.subplots(figsize=(8, 4))
#         ax.bar(x - width/2, marginal_vals, width, label='Marginal', color='#1f77b4')
#         ax.bar(x + width/2, synthetic_vals, width, label='Synthetic', color='#ff7f0e')
#
#         ax.set_xlabel('Category', fontsize=12)
#         ax.set_ylabel('Percentage (%)', fontsize=12)
#         ax.set_title(f'{entity_type.capitalize()} Variable "{var}" Distribution at Geo {area}', fontsize=14, weight='bold')
#         ax.set_xticks(x)
#         ax.set_xticklabels(categories)
#         ax.set_ylim(0, 120)
#         ax.grid(axis='y', linestyle='--', alpha=0.4)
#
#         for i, cat in enumerate(categories):
#             if marginal_vals[i] > 0:
#                 ax.text(x[i] - width/2, marginal_vals[i] + 1.0, f"{marginal_vals[i]:.1f}%", ha='center', va='bottom', fontsize=8, color='#1f77b4')
#             if synthetic_vals[i] > 0:
#                 ax.text(x[i] + width/2, synthetic_vals[i] + 1.0, f"{synthetic_vals[i]:.1f}%", ha='center', va='bottom', fontsize=8, color='#ff7f0e')
#
#         ax.text(0.98, 0.95, f"Marginal: N = {int(marginal_total):,}", transform=ax.transAxes, ha='right', va='top', fontsize=10, color='black')
#         ax.text(0.98, 0.88, f"Synthetic: N = {int(synthetic_total):,}", transform=ax.transAxes, ha='right', va='top', fontsize=10, color='black')
#
#         ax.legend(loc='upper left')
#         plt.tight_layout()
#         plt.show()
#
#     geo_areas = housing_synthetic_df[geo_col].unique()
#
#     for area in geo_areas:
#         print("\n" + "="*60)
#         print("\033[1m" + f'📍 GEO: {area}'.center(60) + "\033[0m")
#         print("="*60 + "\n")
#         housing_subset = housing_synthetic_df[(housing_synthetic_df[geo_col] == area) & (housing_synthetic_df['entity'] == 'household')]
#         person_subset = person_synthetic_df[(person_synthetic_df[geo_col] == area) & (person_synthetic_df['entity'] == 'person')]
#
#         if area in housing_marginal_df.index:
#             print("\033[1m" + f'Household Variables'.center(60) + "\033[0m")
#             for var in housing_marginal_df.columns.levels[0]:
#                 cols = [col for col in housing_marginal_df.columns if col[0] == var]
#                 marginal_counts = housing_marginal_df.loc[area, cols]
#                 marginal_counts.index = [c[1] for c in marginal_counts.index]
#                 plot_comparison(housing_subset, marginal_counts, var, 'household', area)
#
#         if area in person_marginal_df.index:
#             print("\033[1m" + f'Person Variables'.center(60) + "\033[0m")
#             for var in person_marginal_df.columns.levels[0]:
#                 cols = [col for col in person_marginal_df.columns if col[0] == var]
#                 marginal_counts = person_marginal_df.loc[area, cols]
#                 marginal_counts.index = [c[1] for c in marginal_counts.index]
#                 plot_comparison(person_subset, marginal_counts, var, 'person', area)



import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def plot_synthetic_vs_marginal_comparison(
    housing_synthetic_df,
    housing_marginal_df,
    person_synthetic_df,
    person_marginal_df,
    geo_col='geo',
    output_pdf_path=None
):
    def plot_comparison(subset_synthetic, marginal_counts, var, entity_type, area, pdf=None):
        marginal_total = marginal_counts.sum()
        marginal_dist = marginal_counts / marginal_total

        synthetic_counts = subset_synthetic[var].value_counts()
        synthetic_total = synthetic_counts.sum()
        synthetic_dist = synthetic_counts / synthetic_total

        categories = sorted(set(marginal_dist.index) | set(synthetic_dist.index))

        marginal_vals = [marginal_dist.get(cat, 0) * 100 for cat in categories]
        synthetic_vals = [synthetic_dist.get(cat, 0) * 100 for cat in categories]

        x = np.arange(len(categories))
        width = 0.45

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - width/2, marginal_vals, width, label='Marginal', color='#1f77b4')
        ax.bar(x + width/2, synthetic_vals, width, label='Synthetic', color='#ff7f0e')

        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title(f'{entity_type.capitalize()} Variable "{var}" Distribution at Geo {area}', fontsize=14, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 120)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        for i, cat in enumerate(categories):
            if marginal_vals[i] > 0:
                ax.text(x[i] - width/2, marginal_vals[i] + 1.0, f"{marginal_vals[i]:.1f}%", ha='center', va='bottom', fontsize=8, color='#1f77b4')
            if synthetic_vals[i] > 0:
                ax.text(x[i] + width/2, synthetic_vals[i] + 1.0, f"{synthetic_vals[i]:.1f}%", ha='center', va='bottom', fontsize=8, color='#ff7f0e')

        ax.text(0.98, 0.95, f"Marginal: N = {int(marginal_total):,}", transform=ax.transAxes, ha='right', va='top', fontsize=10, color='black')
        ax.text(0.98, 0.88, f"Synthetic: N = {int(synthetic_total):,}", transform=ax.transAxes, ha='right', va='top', fontsize=10, color='black')

        ax.legend(loc='upper left')
        plt.tight_layout()

        if pdf:
            pdf.savefig(fig)
            plt.close(fig)
        else:
            plt.show()

    geo_areas = housing_synthetic_df[geo_col].unique()

    if output_pdf_path:
        with PdfPages(output_pdf_path) as pdf:
            for area in geo_areas:
                housing_subset = housing_synthetic_df[
                    (housing_synthetic_df[geo_col] == area) & (housing_synthetic_df['entity'] == 'household')]
                person_subset = person_synthetic_df[
                    (person_synthetic_df[geo_col] == area) & (person_synthetic_df['entity'] == 'person')]

                if area in housing_marginal_df.index:
                    for var in housing_marginal_df.columns.levels[0]:
                        cols = [col for col in housing_marginal_df.columns if col[0] == var]
                        marginal_counts = housing_marginal_df.loc[area, cols]
                        marginal_counts.index = [c[1] for c in marginal_counts.index]
                        plot_comparison(housing_subset, marginal_counts, var, 'household', area, pdf)

                if area in person_marginal_df.index:
                    for var in person_marginal_df.columns.levels[0]:
                        cols = [col for col in person_marginal_df.columns if col[0] == var]
                        marginal_counts = person_marginal_df.loc[area, cols]
                        marginal_counts.index = [c[1] for c in marginal_counts.index]
                        plot_comparison(person_subset, marginal_counts, var, 'person', area, pdf)

        print(f"✅ All plots have been saved to: {output_pdf_path}")
    else:
        for area in geo_areas:
            print("\n" + "="*60)
            print("\033[1m" + f'📍 GEO: {area}'.center(60) + "\033[0m")
            print("="*60 + "\n")
            housing_subset = housing_synthetic_df[
                (housing_synthetic_df[geo_col] == area) & (housing_synthetic_df['entity'] == 'household')]
            person_subset = person_synthetic_df[
                (person_synthetic_df[geo_col] == area) & (person_synthetic_df['entity'] == 'person')]

            if area in housing_marginal_df.index:
                for var in housing_marginal_df.columns.levels[0]:
                    cols = [col for col in housing_marginal_df.columns if col[0] == var]
                    marginal_counts = housing_marginal_df.loc[area, cols]
                    marginal_counts.index = [c[1] for c in marginal_counts.index]
                    plot_comparison(housing_subset, marginal_counts, var, 'household', area)

            if area in person_marginal_df.index:
                for var in person_marginal_df.columns.levels[0]:
                    cols = [col for col in person_marginal_df.columns if col[0] == var]
                    marginal_counts = person_marginal_df.loc[area, cols]
                    marginal_counts.index = [c[1] for c in marginal_counts.index]
                    plot_comparison(person_subset, marginal_counts, var, 'person', area)


# def add_title_page(pdf, area):
#     fig, ax = plt.subplots(figsize=(8, 2))
#     ax.axis('off')
#     ax.text(0.5, 0.5, f'📍 GEO: {area}', fontsize=16, weight='bold', ha='center', va='center')
#     pdf.savefig(fig)
#     plt.close(fig)
