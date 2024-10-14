import pandas as pd
from config import CFG

def get_frontal(df):
    df['is_frontal'] = df['ViewPosition'].str.contains('AP|PA', case=False, na=False)
    return df

def position_wise_mean(df):
    grouped_df = df.groupby(['study_id', 'is_frontal'])[CFG.target_cols].mean().reset_index()
    grouped_df = grouped_df.groupby(['study_id'])[CFG.target_cols].mean().reset_index()
    merged_df = df.merge(grouped_df, on='study_id', suffixes=('', '_grouped'))
    for target_col in CFG.target_cols:
        merged_df[target_col] = merged_df[target_col + '_grouped']
        merged_df.drop(target_col + '_grouped', axis=1, inplace=True)
    return merged_df