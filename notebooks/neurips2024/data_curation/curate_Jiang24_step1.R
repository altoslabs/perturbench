if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}

remotes::install_version("Matrix", version = "1.6.5", repos = "http://cran.us.r-project.org")
install.packages("Seurat")
remotes::install_github("mojaveazure/seurat-disk")

library(Seurat)
library(SeuratDisk)

## Set your working directory to the current file (only works on Rstudio)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

## Data cache directory
data_cache_dir = '/cluster/scratch/fluebeck/perturbench_data' ## Change this to your local data directory

## Define paths to Seurat objects and download from Zenodo
ifng_seurat_path = paste0(data_cache_dir, "/Seurat_object_IFNG_Perturb_seq.rds", sep="")
system2(
  "wget",
  c(
    "https://zenodo.org/records/10520190/files/Seurat_object_IFNG_Perturb_seq.rds?download=1",
    "-O",
    ifng_seurat_path
  )
)

ifnb_seurat_path = paste0(data_cache_dir, "/Seurat_object_IFNB_Perturb_seq.rds", sep="")
system2(
  "wget",
  c(
    "https://zenodo.org/records/10520190/files/Seurat_object_IFNB_Perturb_seq.rds?download=1",
    "-O",
    ifnb_seurat_path
  )
)

ins_seurat_path = paste0(data_cache_dir, "/Seurat_object_INS_Perturb_seq.rds", sep="")
system2(
  "wget",
  c(
    "https://zenodo.org/records/10520190/files/Seurat_object_INS_Perturb_seq.rds?download=1",
    "-O",
    ins_seurat_path
  )
)

tgfb_seurat_path = paste0(data_cache_dir, "/Seurat_object_TGFB_Perturb_seq.rds", sep="")
system2(
  "wget",
  c(
    "https://zenodo.org/records/10520190/files/Seurat_object_TGFB_Perturb_seq.rds?download=1",
    "-O",
    tgfb_seurat_path
  )
)

tnfa_seurat_path = paste0(data_cache_dir, "/Seurat_object_TNFA_Perturb_seq.rds", sep="")
system2(
  "wget",
  c(
    "https://zenodo.org/records/10520190/files/Seurat_object_TNFA_Perturb_seq.rds?download=1",
    "-O",
    tnfa_seurat_path
  )
)


seurat_files = c(
  ifng_seurat_path,
  ifnb_seurat_path,
  ins_seurat_path,
  tgfb_seurat_path,
  tnfa_seurat_path
)

for (seurat_path in seurat_files) {
  print(seurat_path)
  obj = readRDS(seurat_path)
  print(obj)
  
  out_h5seurat_path = gsub(".rds", ".h5Seurat", seurat_path)
  SaveH5Seurat(obj, filename = out_h5seurat_path)
  Convert(out_h5seurat_path, dest = "h5ad")
}

