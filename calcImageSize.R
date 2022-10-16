rm(list=ls())
options(stringsAsFactors=F)

library(dplyr)
library(data.table)
library(ggplot2)
library(argparse)


# create parser object
parser <- ArgumentParser()

parser$add_argument("--dataDir", default=paste0("/home/",Sys.getenv("USER"),"/DeepSpaCE/data"),
                    help="data directory [default %(default)s]")
parser$add_argument("--sampleName", default="Human_Breast_Cancer_Block_A_Section_1",
                    help="sample name [default %(default)s]")

args <- parser$parse_args()


dataDir <- args$dataDir
sampleName <- args$sampleName


filePath <- paste0(dataDir,"/",sampleName,"/SpaceRanger/spatial/tissue_positions_list.csv")


print(filePath)

tissue_pos <- fread(filePath, data.table=F)

colnames(tissue_pos) <- c("barcode","in_tissue","array_row","array_col","pxl_row_in_fullres","pxl_col_in_fullres")

g <- ggplot(tissue_pos, aes(x=pxl_col_in_fullres, y=pxl_row_in_fullres)) + geom_point()
g <- g + scale_y_reverse()
#plot(g)

res <- NULL

for(i in 0:77){
  tissue_pos_tmp <- tissue_pos[tissue_pos$array_row == i,]
  
  for(j in 2:nrow(tissue_pos_tmp)){
    res <- append(res, tissue_pos_tmp$pxl_col_in_fullres[j] - tissue_pos_tmp$pxl_col_in_fullres[j-1])
  }
}
table(res)
mean(res)

print(paste0("Radius: ", mean(res) / 100 * 27.5))

