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

####### normalized expression level ########

TH = th_filtered1[(th_filtered1[:, th_filtered1.var_names == "Th"].X > 0)]
th_Rxfp = th_filtered1[(th_filtered1[:, th_filtered1.var_names == "Rxfp1"].X > 0.6) & (th_filtered1[:, th_filtered1.var_names == "Th"].X > 0)]
th_Shox = th_filtered1[(th_filtered1[:, th_filtered1.var_names == "Shox2"].X > 0.6) & (th_filtered1[:, th_filtered1.var_names == "Th"].X > 0)]
th_S_R = th_filtered1[(th_filtered1[:, th_filtered1.var_names == "Rxfp1"].X > 0.6) & (th_filtered1[:, th_filtered1.var_names == "Shox2"].X > 0.6) & (th_filtered1[:, th_filtered1.var_names == "Th"].X > 0)]

th_Bmp = th_filtered1[(th_filtered1[:, th_filtered1.var_names == "Bmp3"].X > 0) & (th_filtered1[:, th_filtered1.var_names == "Shox2"].X > 0) & (th_filtered1[:, th_filtered1.var_names == "Th"].X > 0)]
th_Dsp = th_filtered1[(th_filtered1[:, th_filtered1.var_names == "Dsp"].X > 0) & (th_filtered1[:, th_filtered1.var_names == "Shox2"].X > 0) & (th_filtered1[:, th_filtered1.var_names == "Th"].X > 0)]
th_B_D = th_filtered1[(th_filtered1[:, th_filtered1.var_names == "Bmp3"].X > 0) & (th_filtered1[:, th_filtered1.var_names == "Dsp"].X > 0) & (th_filtered1[:, th_filtered1.var_names == "Shox2"].X > 0)& (th_filtered1[:, th_filtered1.var_names == "Th"].X > 0)]

filenames = th_filtered1.obs['file_label']
r = th_filtered1.X[:, th_filtered1.var_names.get_loc("Rxfp1")]
s = th_filtered1.X[:, th_filtered1.var_names.get_loc("Shox2")]
t = th_filtered1.X[:, th_filtered1.var_names.get_loc("Th")]
bb = th_filtered1.X[:, th_filtered1.var_names.get_loc("Bmp3")]
dd = th_filtered1.X[:, th_filtered1.var_names.get_loc("Dsp")]

for n in sorted(set(filenames)):
    print(n, ((r[filenames == n] > 0.6) & (t[filenames == n] > 0)).sum(), ((s[filenames == n] > 0.6) & (t[filenames == n] > 0)).sum(), (t[filenames == n] > 0).sum(), ((r[filenames == n] > 0.6) & (s[filenames == n] > 0.6) & (t[filenames == n] > 0)).sum())

# BMP & DSP quantify 
for n in sorted(set(filenames)):
    print(n, ((bb[filenames == n] > 0) & (t[filenames == n] > 0) & (s[filenames == n] > 0.6)).sum(), ((dd[filenames == n] > 0) & (t[filenames == n] > 0) & (s[filenames == n] > 0.6)).sum(), ((bb[filenames == n] > 0) & (dd[filenames == n] > 0) & (t[filenames == n] > 0) & (s[filenames == n] > 0.6)).sum())
