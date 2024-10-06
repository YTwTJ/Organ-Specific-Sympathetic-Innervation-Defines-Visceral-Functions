# Spatial transcriptomics (seqFISH) of CG-SMG all cell types

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import scanpy as sc
from anndata import AnnData
import os
from harmonypy import run_harmony
import random
from sklearn.utils import check_random_state

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
check_random_state(seed_value)

def hex_to_rgb_normalized(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

def generate_colormap(end_color, start_color="#ffffff"):
    colors = [hex_to_rgb_normalized(start_color), hex_to_rgb_normalized(end_color)]
    n_bins = [3]  # Discretizes the interpolation into bins
    cmap_name = 'custom_div_cmap'
    cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    return cm

CM = generate_colormap("#0000FF", "#808080")

def get_neuron(df):
    return df 

def get_batch_id(name):
    return '_'.join(name.split('_')[1: -1])

DIR = './seqFISH_Data_DAPI_seg/'

# pick Rcg files
Rcg_names = []
Rcg_dfs = []
for filename in os.listdir(DIR):
    if filename.endswith('.csv') and not filename.endswith('xy.csv') and filename.startswith('RCG'):
        Rcg_names.append(filename)
        Rcg_dfs.append(get_neuron(pd.read_csv(DIR + filename, index_col=0)))

Rcg_names = Rcg_names
Rcg_dfs = Rcg_dfs

# pick Lcg files
Lcg_names = []
Lcg_dfs = []
for filename in os.listdir(DIR):
    if filename.endswith('.csv') and not filename.endswith('xy.csv') and filename.startswith('LCG'):
        Lcg_names.append(filename)
        Lcg_dfs.append(get_neuron(pd.read_csv(DIR + filename, index_col=0)))

Lcg_names = Lcg_names
Lcg_dfs = Lcg_dfs

# pick smg files
smg_names = []
smg_dfs = []
for filename in os.listdir(DIR):
    if filename.endswith('.csv') and not filename.endswith('xy.csv') and filename.startswith('SMG'):
        smg_names.append(filename)
        smg_dfs.append(get_neuron(pd.read_csv(DIR + filename, index_col=0)))

smg_names = smg_names
smg_dfs = smg_dfs

Rcg_df = pd.concat(Rcg_dfs, ignore_index=True).dropna(axis=1, how='any')
Lcg_df = pd.concat(Lcg_dfs, ignore_index=True).dropna(axis=1, how='any')
smg_df = pd.concat(smg_dfs, ignore_index=True).dropna(axis=1, how='any')

combined_df = pd.concat([Rcg_df, Lcg_df, smg_df], ignore_index=True).dropna(axis=1, how='any')

adata = AnnData(combined_df)
adata.obs['label'] = ['RCG'] * len(Rcg_df) + ['LCG'] * len(Lcg_df) + ['SMG'] * len(smg_df)
file_label = []
for name, df in zip(Rcg_names, Rcg_dfs):
    file_label += [name] * len(df)
for name, df in zip(Lcg_names, Lcg_dfs):
    file_label += [name] * len(df)
for name, df in zip(smg_names, smg_dfs):
    file_label += [name] * len(df)
adata.obs['file_label'] = file_label

# get batch count
batch_label = []
for name, df in zip(Rcg_names, Rcg_dfs):
    batch_label += [get_batch_id(name)] * len(df)
for name, df in zip(Lcg_names, Lcg_dfs):
    batch_label += [get_batch_id(name)] * len(df)
for name, df in zip(smg_names, smg_dfs):
    batch_label += [get_batch_id(name)] * len(df)
adata.obs['batch_label'] = batch_label

adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
adata.obs['total_counts'] = adata.X.sum(axis=1)

sc.pp.filter_cells(adata, min_genes=5)
sc.pp.filter_cells(adata, min_counts=20)
print(adata.shape)

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.pca(adata, n_comps=50)
harmony_out = run_harmony(adata.obsm['X_pca'], adata.obs, 'batch_label', max_iter_harmony=10, kmeans_method='scipy_kmeans2')
adata.obsm['X_pca_harmony'] = harmony_out.Z_corr.T
sc.pp.neighbors(adata, n_neighbors=20, use_rep='X_pca_harmony', random_state=seed_value)
sc.tl.umap(adata, random_state=seed_value)
sc.tl.leiden(adata, resolution=1)
sc.pl.umap(adata, color=['leiden'], cmap=CM, legend_loc="on data")
sc.pl.umap(adata, color=['Th','Fabp7','Ncmap','Dcn','Rgs5','Vtn','Myh11','Cldn5','Plvap','C1qb','Ctss'], cmap=CM)

clusters_weight = ['0', '5', '2', '4', '13', '15', '9', '6', '10', '11', '14']
gene_index = np.where(adata.var_names == 'Th')[0][0]
for cluster in clusters_weight:
    cells_in_cluster = adata.obs[adata.obs['leiden'] == cluster].index
    adata[cells_in_cluster, gene_index] = 0

sc.pp.pca(adata, n_comps=50)
harmony_out = run_harmony(adata.obsm['X_pca'], adata.obs, 'batch_label', max_iter_harmony=10, kmeans_method='scipy_kmeans2')
adata.obsm['X_pca_harmony'] = harmony_out.Z_corr.T
sc.pp.neighbors(adata, n_neighbors=20, use_rep='X_pca_harmony', random_state=seed_value)
sc.tl.umap(adata, random_state=seed_value)
sc.tl.leiden(adata, resolution=1)

mapping = {
    '0': 'Neurons',
    '1': 'Fibroblasts',
    '2': 'Neurons',
    '3': 'Fibroblasts',
    '4': 'Mural cells',
    '5': 'Mural cells',
    '6': 'Vas. endo. cells',
    '7': 'Neurons',
    '8': 'Neurons',
    '9': 'Macrophages',
    '10': 'Schwann cells',
    '11': 'Mural cells',
    '12': 'Satellite glia',
    '13': 'Fibroblasts',
    '14': 'Fibroblasts',
    }

adata.obs['leiden'] = adata.obs['leiden'].map(mapping).astype('category')
adata.obs['leiden'] = adata.obs['leiden'].astype('category')
sc.pl.umap(adata, color=['leiden'], cmap=CM, legend_loc="on data")
sc.pl.umap(adata, color=['leiden'], cmap=CM)

