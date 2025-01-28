import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = df = pd.read_csv('C:\\Users\\Tkeli\\Downloads\\medical_examination.csv')
df = pd.read_csv('C:\\Users\\Tkeli\\Downloads\\medical_examination.csv')

# 2
df['overweight'] = np.where((df['weight'] / df['height'] ** 2) > 25, '1', '0')
df.head()

# 3

df['cholesterol'] = np.where(df['cholesterol'] ==1, '0', '1')
df['gluc'] = np.where(df['gluc'] ==1, '0', '1')
df.head()

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],  # Keep 'cardio' as an identifier
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )
    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()
    

    # 7
    cat_plot = sns.catplot(
        x='variable', 
        y='size', 
        hue='value', 
        col='cardio', 
        data=df_cat, 
        kind='bar',
        height=5,
        aspect=1
    )

    # 8
    fig = cat_plot.fig
    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &  # Diastolic pressure ≤ Systolic pressure
        (df['height'] >= df['height'].quantile(0.025)) &  # Height ≥ 2.5th percentile
        (df['height'] <= df['height'].quantile(0.975)) &  # Height ≤ 97.5th percentile
        (df['weight'] >= df['weight'].quantile(0.025)) &  # Weight ≥ 2.5th percentile
        (df['weight'] <= df['weight'].quantile(0.975))    # Weight ≤ 97.5th percentile
    ]
    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))


    # 14
    fig, ax = plt.subplots(figsize=(12, 8))

    # 15
    sns.heatmap(
        corr,
        mask=mask,
        cmap='coolwarm',
        annot=True,
        fmt='.1f',
        square=True,
        cbar_kws={'shrink': 0.5},
        ax=ax
    )


    # 16
    fig.savefig('heatmap.png')
    return fig