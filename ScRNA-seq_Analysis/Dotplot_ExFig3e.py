# CGSMG scRNA-seq dotplots for ExFig3e

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

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata.raw = adata
adata = adata[:, adata.var.highly_variable]

sc.pp.scale(adata, max_value=10)

sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_neighbors=20, use_rep='X_pca')
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=1.8)

clusters_of_interest = ['26', '28']
adata_filtered = adata[adata.obs['leiden'].isin(clusters_of_interest)]
adata_filtered.obs['leiden'].cat.reorder_categories(['28', '26'], ordered=True, inplace=True)
adata_filtered.obs['leiden_renamed'] = adata_filtered.obs['leiden'].replace({'26': 'SHOX2+', '28': 'RXFP1+'})

markersRecep1 = ['Agtr1a','Agtr1b','Gabrb3','Gcgr','Gfra2','Gria2','Grik2','Grik5','Gpr1','Xpr1']
markersRecep2 = ['Antxr2','Chrna5','Chrna7','Csf2ra','Epha3','Grik4','Htr1f','Htr3a','Il20ra','Lingo1','Ltk','Oprm1','Prom1','Robo1','Sctr','Tenm2','Tspan8','Tyro3']
markersRecep3 = ['Asic2','Cckar','Cckbr','Csmd3','Drd2','Grik1','Lrp1b','Nkain2','Ntrk2','Ptprd','Ptprt','Slc18a1','Slc35f4','Sorl1','Sstr2']
sc.pl.dotplot(adata_filtered, markersRecep1, groupby='leiden_renamed', dendrogram=False, vmin=0, vmax=3)
sc.pl.dotplot(adata_filtered, markersRecep2, groupby='leiden_renamed', dendrogram=False, vmin=0, vmax=3)
sc.pl.dotplot(adata_filtered, markersRecep3, groupby='leiden_renamed', dendrogram=False, vmin=0, vmax=3)

markersNeuPep = ['Npy','Sst','Adcyap1','Calca','Calcb','Cbln2','Chgb','Crispld1','Nppc','Nrg1','Tac1','Tac2','Vip']
sc.pl.dotplot(adata_filtered, markersNeuPep, groupby='leiden_renamed', dendrogram=False, vmin=0, vmax=3)

markersMem = ['Brinp2','Cadm1','Cdh6','Thsd7a','Igsf11','Kirrel3','Opcml','Sdk1']
sc.pl.dotplot(adata_filtered, markersMem, groupby='leiden_renamed', dendrogram=False, vmin=0, vmax=3)

markersEny = ['Pcsk2','Tiam2','Zfp804a','Man2a1','Pde11a','Pdzrn3','Hs6st3','Sox6']
sc.pl.dotplot(adata_filtered, markersEny, groupby='leiden_renamed', dendrogram=False, vmin=0, vmax=3)

markersdotV = ['Scn2b','Scn3b','Scn4b','Scn7a','Scn9a','Kcna1','Kcna2','Kcna6','Kcnab1','Kcnab2','Kcnb1','Kcnc1','Kcnc4','Kcnd1','Kcnd2','Kcne4','Kcnh2','Kcnh7','Kcnh8','Kcnip1','Kcnip2','Kcnip3','Kcnip4','Kcnk1','Kcnk3','Kcnk13','Kcnmb4','Kcns3','Kcnt1','Hcn1','Cachd1','Cacna1b','Cacna2d1','Cacna2d3','Cacnb1','Cacnb3','Cacng2','Cacng3']
sc.pl.dotplot(adata_filtered, markersdotV, groupby='leiden_renamed', dendrogram=False, vmin=0, vmax=3)
