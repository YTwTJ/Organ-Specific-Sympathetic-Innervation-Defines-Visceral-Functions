# CGSMG scRNA-seq analysis

Datasets available at GSE278457

## Dotplots
```
# set up python dependencies for dotplots
conda env create -f environment-sc.yml

# make sure to activate conda environment before running dotplots
conda activate sc

python Dotplot_ExFig3e.py
python Dotplot_ExFig4a.py
```

## Differential Gene Analysis
```
# set up python dependencies for differential gene analysis
conda env create -f environment-pydeseq2.yml

# make sure to activate conda environment before running differential gene analysis
conda activate pydeseq2

python Diff_gene_1_ExFig3f.py
python Diff_gene_2_ExFig3f.py
python Rankplot_ExFig3f.py
```
