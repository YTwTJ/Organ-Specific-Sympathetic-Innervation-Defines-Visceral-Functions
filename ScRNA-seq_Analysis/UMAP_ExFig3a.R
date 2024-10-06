# CGSMG transcriptomic cell types for ExFig3a
# R (4.1.1) and Seurat (4.2.1) were used

library(Seurat)
library(dplyr)
library(Matrix)

#### Load the dataset and initialize Seurat object
Sample1 = Read10X(data.dir = "./filtered_feature_bc_matrix")
Sample1_dbase <- CreateSeuratObject(Sample1, min.features = 0, min.cells = 0, project = "sample1")
Sample2 = Read10X(data.dir = "./filtered_feature_bc_matrix2")
Sample2_dbase <- CreateSeuratObject(Sample2, min.features = 0, min.cells = 0, project = "sample2")

CGSMG_dbase <- merge(Sample1_dbase, Sample2_dbase, add.cell.ids = c("sample1", "sample2")) 

#### Basic QC and thresholding
mito.genes <- grep(pattern = "^mt-", x = rownames(x = GetAssayData(object = CGSMG_dbase)), value = TRUE)
percent.mito <- Matrix::colSums(GetAssayData(object = CGSMG_dbase, slot = "counts")[mito.genes, ])/Matrix::colSums(GetAssayData(object = CGSMG_dbase, slot = "counts"))
CGSMG_dbase$percent.mito = percent.mito

# Accessing data in Seurat object
CGSMG_dbase[[]] # feature fields for cells
GetAssayData(object = CGSMG_dbase, slot = "counts")
GetAssayData(object = CGSMG_dbase, slot = "scale.data")
GetAssayData(object = CGSMG_dbase, slot = "counts")

# Filter cells
CGSMG_dbase = subset(x = CGSMG_dbase, subset = percent.mito < 0.1 & nCount_RNA > 1500 & nCount_RNA < 45000 ) 

#### Data Normalization
CGSMG_dbase <- NormalizeData(object = CGSMG_dbase, normalization.method = "LogNormalize", scale.factor = 10000)

#### Detect variable genes across the single cells
CGSMG_dbase <- FindVariableFeatures(object = CGSMG_dbase, selection.method = "vst", nfeatures = 1000, loess.span = 0.3, clip.max = "auto") 

#### Scale the data
CGSMG_dbase <- ScaleData(object = CGSMG_dbase, do.scale = TRUE, do.center = TRUE)

#### Carry out PCA and evaluate PC dimensions
CGSMG_dbase <- RunPCA(object = CGSMG_dbase, features = VariableFeatures(object=CGSMG_dbase), verbose = TRUE, ndims.print = 1:5, nfeatures.print = 5, npcs=40, seed.use = 1)

#### Find clusters (save.SNN = T saves the SNN so that the clustering algorithm can be rerun using the same graph but with a different resolution value (see docs for full details)
CGSMG_dbase <- FindNeighbors(object = CGSMG_dbase, reduction = "pca", dims = 1:19, compute.SNN = TRUE, verbose = TRUE)
CGSMG_dbase <- FindClusters(object = CGSMG_dbase, modularity.fxn = 1, resolution = 1.2, algorithm = 1, n.start = 10, n.iter = 10, random.seed = 1, verbose = TRUE)

# Visualize clustering results
CGSMG_dbase <- RunUMAP(CGSMG_dbase, reduction = "pca", dims = 1:19, seed.use = 1, n.neighbors = 5, min.dist = 1,  spread = 1) 
current.cluster.ids <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)
new.cluster.ids <- c('Satellite glia', 'Fibroblasts', 'Satellite glia', 'Satellite glia', 'Macrophages', 'Fibroblasts', 'Satellite glia', 'Fibroblasts', 'Neurons', 'Satellite glia', 'Fibroblasts', 'Macrophages', 'Vas. endo. cells', 'Mural cells', 'Vas. endo. cells', 'Vas. endo. cells', 'Mural cells', 'Schwann cells', 'Macrophages', 'Neurons', 'Macrophages', 'Satellite glia', 'Fibroblasts', 'Macrophages', 'Macrophages', 'Neurons', 'Neurons', 'Macrophages', 'Macrophages', 'Vas. endo. cells', 'Macrophages')
Idents(object = CGSMG_dbase) = plyr::mapvalues(Idents(object = CGSMG_dbase), from = current.cluster.ids, to = new.cluster.ids)  
Idents(object = CGSMG_dbase) = factor(Idents(object = CGSMG_dbase), levels = c('Neurons','Satellite glia','Fibroblasts','Macrophages','Vas. endo. cells','Mural cells','Schwann cells'))
DimPlot(object = CGSMG_dbase, reduction = "umap", label = TRUE) + NoLegend()
FeaturePlot(CGSMG_dbase, c("Th"), cols = c("lightgrey","blue"), min.cutoff = 1, max.cutoff = 2) 
FeaturePlot(CGSMG_dbase, c("Rxfp1"), cols = c("lightgrey","#FF2400")) 
FeaturePlot(CGSMG_dbase, c("Shox2"), cols = c("lightgrey","#00a651"), max.cutoff = 2) 

