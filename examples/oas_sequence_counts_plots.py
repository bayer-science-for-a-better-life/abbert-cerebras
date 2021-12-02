from pathlib import Path
from typing import Union, Sequence

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from abbert2.oas import OAS

# --- Massage the data

# Load the nice unit metadata dataframe - including Unit objects
# This dataframe has one unit per row
units_df = OAS().nice_unit_meta_df(recompute=False, add_units=True)

# Get rid of units left with no sequences
units_df = units_df.query('num_heavy_sequences > 0 or num_light_sequences > 0')


# Let's define a function to normalize sequence counts according to the total
def normalize_counts(df) -> pd.DataFrame:
    df = df.copy()
    df['pct_heavy'] = df['num_heavy_sequences'] / df['num_heavy_sequences'].sum() * 100
    df['pct_light'] = df['num_light_sequences'] / df['num_light_sequences'].sum() * 100
    return df


# Convenience to massage the dataframes
def group_by_normalize(df: pd.DataFrame,
                       by: Union[str, Sequence[str]] = 'study_id',
                       melt: bool = False) -> pd.DataFrame:
    if isinstance(by, str):
        by = by,
    by = list(by)
    df = normalize_counts(df.groupby(by=by)[['num_heavy_sequences', 'num_light_sequences']].sum())
    df = df.drop(columns=['num_heavy_sequences', 'num_light_sequences']).reset_index()
    df = df.sort_values(['pct_heavy'], ascending=False)
    if melt:
        df = df.melt(id_vars=by,
                     value_vars=['pct_heavy', 'pct_light'],
                     var_name='chain',
                     value_name='percentage')
        df = df['chain'].rename(columns={'pct_heavy': 'heavy', 'pct_light': 'light'})
    return df


#
# We can group sequences by:
#   - study
#   - subject
#   - unit
# and always add species to the mix
#

# --- Some plots

by_unit_df = normalize_counts(units_df)
by_study_df = group_by_normalize(units_df, by=['study_id', 'species'])
by_subject_df = group_by_normalize(units_df, by=['subject', 'species'])


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(by_unit_df)
    print(by_study_df)
    print(by_subject_df)

sns.set_context('talk')

plt.figure(figsize=(12, 12))
splot = sns.scatterplot(data=by_study_df,
                        x='pct_heavy', y='pct_light',
                        hue='species', hue_order=['human', 'mouse', 'rat', 'rabbit', 'rhesus'])
splot.set(title='Percentage of Heavy and Light Chains per Study', xlim=(None, 60), ylim=(None, 60))
plt.savefig(Path.home() / 'pct_chains_per_study.jpg', dpi=200)

plt.figure(figsize=(12, 12))
splot = sns.scatterplot(data=by_study_df.query('pct_light <= 10 and pct_heavy <= 10'),
                        x='pct_heavy', y='pct_light',
                        hue='species', hue_order=['human', 'mouse', 'rat', 'rabbit', 'rhesus'])
splot.set(title='Percentage of Heavy and Light Chains per Study (no >10%)', xlim=(None, 10), ylim=(None, 10))
plt.savefig(Path.home() / 'pct_chains_per_study_no_outliers.jpg', dpi=200)

plt.figure(figsize=(12, 12))
splot = sns.scatterplot(data=by_subject_df,
                        x='pct_heavy', y='pct_light',
                        hue='species', hue_order=['human', 'mouse', 'rat', 'rabbit', 'rhesus'])
splot.set(title='Percentage of Heavy and Light Chains per Subject', xlim=(None, 10), ylim=(None, 10))
plt.savefig(Path.home() / 'pct_chains_per_subject.jpg', dpi=200)
