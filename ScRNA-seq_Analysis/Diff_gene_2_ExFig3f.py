# CGSMG scRNA-seq  differential gene analysis - part 2

import numpy as np
import pandas as pd

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data

from adjustText import adjust_text

a = pd.read_csv('Shox2.csv', index_col=0)
b = pd.read_csv('Rxfp1.csv', index_col=0)
c = pd.concat([a, b], ignore_index=True)

m = ['b'] * a.shape[0] + ['a'] * b.shape[0]
m = pd.DataFrame({'condition': m}, index=c.index)

counts_df = c
metadata = m

genes_to_keep = counts_df.columns[counts_df.sum(axis=0) >= 10]
counts_df = counts_df[genes_to_keep]

inference = DefaultInference(n_cpus=8)
dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design_factors="condition",
    refit_cooks=True,
    inference=inference,
)

dds.deseq2()
stat_res = DeseqStats(dds, inference=inference)
stat_res.summary()

df = stat_res.results_df

df = df.dropna()
df['nlog10'] = -np.log10(df.padj)

df.to_csv('Supplementary_Table_1.csv')
