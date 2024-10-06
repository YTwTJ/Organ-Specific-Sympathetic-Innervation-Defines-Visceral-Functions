# CGSMG scRNA-seq differential gene analysis - part 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import scanpy as sc
from anndata import AnnData
import random
import os

seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
pd.util.testing.N = seed_value

os.environ['PYTHONHASHSEED'] = str(seed_value)

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
CMr = generate_colormap("#ff0000", "#808080")
CMg = generate_colormap("#00a651", "#808080")

# Read datasets - GSE278457
data_dir1 = './filtered_feature_bc_matrix/'
data_dir2 = './filtered_feature_bc_matrix2/'
adata1 = sc.read_10x_mtx(data_dir1, var_names='gene_symbols', cache=True)
adata2 = sc.read_10x_mtx(data_dir2, var_names='gene_symbols', cache=True)
adata = adata1.concatenate(adata2, batch_categories=['batch1', 'batch2'], index_unique='-')

mito_genes = adata.var_names.str.startswith('mt-')
adata.obs['percent.mito'] = (adata[:, mito_genes].X.sum(axis=1) / adata.X.sum(axis=1))
adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
adata.obs['total_counts'] = adata.X.sum(axis=1)

adata = adata[adata.obs['percent.mito'] < 0.1, :]

sc.pp.filter_cells(adata, min_counts=1500)
sc.pp.filter_cells(adata, max_counts=45000)
adata.raw = adata.copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable]
sc.pp.scale(adata, max_value=10)

sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_neighbors=20, use_rep='X_pca')
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=1.8)

sc.pl.umap(adata, color=['leiden','Th','Rxfp1', 'Shox2'], cmap=CMr, legend_loc="on data")

a1 = adata[adata.obs['leiden'] == '26']
pd.DataFrame(a1.raw[:, a1.var_names].X.toarray(), index=a1.obs_names, columns=a1.var_names).to_csv('Shox2.csv')
a2 = adata[adata.obs['leiden'] == '28']
pd.DataFrame(a2.raw[:, a2.var_names].X.toarray(), index=a2.obs_names, columns=a2.var_names).to_csv('Rxfp1.csv')
