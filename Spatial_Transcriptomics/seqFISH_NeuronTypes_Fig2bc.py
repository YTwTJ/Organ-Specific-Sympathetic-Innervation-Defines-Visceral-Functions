# Spatial transcriptomics (seqFISH) of CG-SMG neuron types

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
    n_bins = [3]
    cmap_name = 'custom_div_cmap'
    cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    return cm

CM = generate_colormap("#0000FF", "#808080")

def get_neuron(df):
    return df

def get_batch_id(name):
    return '_'.join(name.split('_')[1: -1])

DIR = './seqFISH_Data_TH_seg/'

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
adata.raw = adata
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

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.pca(adata, n_comps=50)
harmony_out = run_harmony(adata.obsm['X_pca'], adata.obs, 'batch_label', max_iter_harmony=10, kmeans_method='scipy_kmeans2')
adata.obsm['X_pca_harmony'] = harmony_out.Z_corr.T

sc.pp.neighbors(adata, n_neighbors=20, use_rep='X_pca_harmony', random_state=seed_value)
sc.tl.umap(adata, random_state=seed_value)
sc.tl.leiden(adata, resolution=2.5)

clusters_to_r = ['4','7','16','17','18','19','21','22','23','24','25','27']
th_filtered0 = adata[~adata.obs['leiden'].isin(clusters_to_r)].copy()

sc.pp.neighbors(th_filtered0, n_neighbors=20, use_rep='X_pca_harmony', random_state=seed_value)
sc.tl.umap(th_filtered0, random_state=seed_value)
sc.tl.leiden(th_filtered0, resolution=0.2)

th_filtered1 = th_filtered0

mapping = {
    '0': 'NA1',
    '1': 'NA3',
    '2': 'NA2',
    }
th_filtered1.obs['leiden'] = th_filtered1.obs['leiden'].map(mapping).astype('category')
th_filtered1.obs['leiden'] = th_filtered1.obs['leiden'].astype('category')
th_filtered1.obs['leiden'] = th_filtered1.obs['leiden'].cat.reorder_categories(['NA1', 'NA2', 'NA3'])
custom_colors2 = ["#FF0000", "#FF7F0E", "#00AEEF"]
th_filtered1.uns['leiden_colors'] = custom_colors2

CMr = generate_colormap("#ff0000", "#808080")
CMg = generate_colormap("#00a651", "#808080")
CMo = generate_colormap("#FF7F0E", "#808080")
CMc = generate_colormap("#00AEEF", "#808080")

sc.pl.umap(th_filtered1, color=['Th'], cmap=CM, use_raw=False, vmax = 5) 
sc.pl.umap(th_filtered1, color=['Rxfp1'], cmap=CMr, use_raw=False, vmax=4, vmin=0.2) 
sc.pl.umap(th_filtered1, color=['Shox2'], cmap=CMg, use_raw=False,vmax=1.5) 

markers1 = ['Th', 'Dbh', 'Slc6a2', 'Slc18a2','Npy']
sc.pl.heatmap(th_filtered1, markers1, groupby='leiden', swap_axes=True,cmap=cm.jet, use_raw=False,vmax=6) 
markers2 = ['Rxfp1','Sctr']
sc.pl.heatmap(th_filtered1, markers2, groupby='leiden', swap_axes=True,cmap=cm.jet, use_raw=False,vmax=3.5) 
markers3 = ['Shox2','Sstr2']
sc.pl.heatmap(th_filtered1, markers3, groupby='leiden', swap_axes=True,cmap=cm.jet, use_raw=False,vmax=3.8)

sc.pl.umap(th_filtered1, color=['label'], cmap=CM, use_raw=False)

clusters_to_remove33 = ['NA1']
adata_g4 = th_filtered1[~th_filtered1.obs['leiden'].isin(clusters_to_remove33)].copy()

sc.pp.neighbors(adata_g4, n_neighbors=20, use_rep='X_pca_harmony', random_state=seed_value)
sc.tl.umap(adata_g4, random_state=seed_value)
sc.tl.leiden(adata_g4, resolution=0.1)
sc.pl.umap(adata_g4, color=['Bmp3'], cmap=CMo, use_raw=False,vmax=1.5) 
sc.pl.umap(adata_g4, color=['Dsp'], cmap=CMc, use_raw=False,vmax=1.5) 

markers4 = ['Bmp3','Dsp']
sc.pl.heatmap(adata_g4, markers4, groupby='leiden', swap_axes=True,cmap=cm.jet, use_raw=False, vmax=3) 
