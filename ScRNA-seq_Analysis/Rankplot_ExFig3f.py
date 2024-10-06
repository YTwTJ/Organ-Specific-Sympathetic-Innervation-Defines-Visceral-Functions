# CGSMG scRNA-seq Rankplot for ExFig3f
# Run Diff_gene_1.py and Diff_gene_2.py prior to this script

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Supplementary_Table_1.csv')
df['log2FoldChange'] = -df['log2FoldChange']
df = df.rename(columns={'Unnamed: 0': 'names'})
df['rank'] = df['log2FoldChange'].rank(ascending=False)
df = df[df['rank'] <= 750]
gene_list = ['Rxfp1']
plt.figure(figsize=(6, 6))
ax = sns.scatterplot(data = df[~df['names'].isin(gene_list)], x = 'rank', y = 'log2FoldChange', color='k', edgecolor=None, s=10)
ax = sns.scatterplot(data = df[df['names'].isin(gene_list)], x = 'rank', y = 'log2FoldChange', color='r', edgecolor=None, s=15)
texts = []
for name in gene_list:
    rank = df.loc[df['names'] == name, 'rank'].values[0]
    score = df.loc[df['names'] == name, 'log2FoldChange'].values[0]
    texts.append(plt.text(rank, y=score, s=name))
plt.show()

df = pd.read_csv('Supplementary_Table_1.csv')
df = df.rename(columns={'Unnamed: 0': 'names'})
df['rank'] = df['log2FoldChange'].rank(ascending=False)
df = df[df['rank'] <= 750]
gene_list = ['Shox2']
plt.figure(figsize=(6, 6))
ax = sns.scatterplot(data = df[~df['names'].isin(gene_list)], x = 'rank', y = 'log2FoldChange', color='k', edgecolor=None, s=10)
ax = sns.scatterplot(data = df[df['names'].isin(gene_list)], x = 'rank', y = 'log2FoldChange', color='r', edgecolor=None, s=15)
texts = []
for name in gene_list:
    rank = df.loc[df['names'] == name, 'rank'].values[0]
    score = df.loc[df['names'] == name, 'log2FoldChange'].values[0]
    texts.append(plt.text(rank, y=score, s=name))
plt.show()


