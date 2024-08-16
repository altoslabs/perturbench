## Install packages
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")


if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}

## Install libcairo and libxt-dev using apt-get
# system2(
#   "apt-get", 
#   c("install", "libcairo2-dev", "libxt-dev")
# )

remotes::install_version("Matrix", version = "1.6.5", repos = "http://cran.us.r-project.org")
install.packages("Seurat")
remotes::install_github("mojaveazure/seurat-disk")
BiocManager::install(c('BiocGenerics', 'DelayedArray', 'DelayedMatrixStats',
                       'limma', 'lme4', 'S4Vectors', 'SingleCellExperiment',
                       'SummarizedExperiment', 'batchelor', 'HDF5Array',
                       'terra', 'ggrastr'))
remotes::install_github('cole-trapnell-lab/monocle3')

## Load libraries
library(Seurat)
library(SingleCellExperiment)
library(SeuratDisk)
library(Matrix)

## Set your working directory to the current file (only works on Rstudio)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

## Data cache directory
data_cache_dir = '../perturbench_data' ## Change this to your local data directory

## GXE1
gxe1_cds_path = paste0(data_cache_dir, "/GSM7056148_sciPlexGxE_1_preprocessed_cds.rds.gz", sep="")
print(gxe1_cds_path)

system2(
  "wget",
  c(
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM7056nnn/GSM7056148/suppl/GSM7056148_sciPlexGxE_1_preprocessed_cds.rds.gz",
    "-O",
    gxe1_cds_path
  )
)
system2(
  "gzip", 
  c("-d", gxe1_cds_path)
)

gxe1 = readRDS(gsub(".gz", "", gxe1_cds_path))
gxe1_counts = gxe1@assays@data$counts
gxe1_gene_meta = data.frame(gxe1@rowRanges@elementMetadata@listData)
gxe1_cell_meta = data.frame(gxe1@colData)
head(gxe1_cell_meta)

colnames(gxe1_counts) = rownames(gxe1_cell_meta)
rownames(gxe1_counts) = gxe1_gene_meta$id
gxe1_seurat = CreateSeuratObject(gxe1_counts, meta.data = gxe1_cell_meta)
gxe1_seurat@assays$RNA@meta.features <- gxe1_gene_meta
gxe1_seurat$cell_type = 'A172'
for (col in colnames(gxe1_seurat@meta.data)) {
  if (is.factor(gxe1_seurat@meta.data[[col]])) {
    print(col)
    gxe1_seurat@meta.data[[col]] = as.character(gxe1_seurat@meta.data[[col]])
  }
}

SaveH5Seurat(gxe1_seurat, filename = paste0(data_cache_dir, "/gxe1.h5Seurat"), overwrite = T)
Convert(paste0(data_cache_dir, "/gxe1.h5Seurat"), dest = "h5ad", overwrite = T)

## GXE2
gxe2_cds_path = paste0(data_cache_dir, "/GSM7056149_sciPlexGxE_2_preprocessed_cds.list.RDS.gz")
system2(
  "wget",
  c(
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM7056nnn/GSM7056149/suppl/GSM7056149%5FsciPlexGxE%5F2%5Fpreprocessed%5Fcds.list.RDS.gz",
    "-O",
    gxe2_cds_path
  )
)
system2(
  "gzip", 
  c("-d", gxe2_cds_path)
)
gxe2_list = readRDS(gsub(".gz", "", gxe2_cds_path))

base_path = data_cache_dir
gxe2_seurat_list = lapply(1:length(gxe2_list), function(i) {
  sce = gxe2_list[[i]]
  counts = sce@assays@.xData$data$counts
  gene_meta = data.frame(sce@rowRanges@elementMetadata@listData)
  cell_meta = data.frame(sce@colData)
  head(cell_meta)
  
  colnames(counts) = rownames(cell_meta)
  rownames(counts) = gene_meta$id
  seurat_obj = CreateSeuratObject(counts, meta.data = cell_meta)
  seurat_obj@assays$RNA@meta.features <- gene_meta
  for (col in colnames(seurat_obj@meta.data)) {
    if (is.factor(seurat_obj@meta.data[[col]])) {
      print(col)
      seurat_obj@meta.data[[col]] = as.character(seurat_obj@meta.data[[col]])
    }
  }
  cl = names(gxe2_list)[[i]]
  seurat_obj$cell_type = cl
  
  out_h5Seurat = paste0(base_path, "/gxe2_", cl, ".h5Seurat")
  SaveH5Seurat(seurat_obj, filename = out_h5Seurat, overwrite = T)
  Convert(out_h5Seurat, dest = "h5ad", overwrite = T)
  
  seurat_obj
})