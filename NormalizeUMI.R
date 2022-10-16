#!/usr/bin/env Rscript

.libPaths("")
rm(list=ls())
options(stringsAsFactors=F)

library(dplyr)
library(data.table)

library(Seurat)
library(ggplot2)
library(patchwork)

library(Matrix)
library(cowplot)
library(RColorBrewer)
library(grid)
library(readbitmap)
library(argparse)


# create parser object
parser <- ArgumentParser()

parser$add_argument("--dataDir", default=paste0("/home/",Sys.getenv("USER"),"/DeepSpaCE/data"),
                    help="Data directory [default %(default)s]")
parser$add_argument("--sampleName", default="Breast_Cancer_Block_A_Section_1",
                    help="Sample name [default %(default)s]")
parser$add_argument("--threshold_count", type="integer", default=1000,
                    help="threshold: number of counts [default %(default)s]")
parser$add_argument("--threshold_gene", type="integer", default=1000,
                    help="threshold: number of genes [default %(default)s]")

args <- parser$parse_args()


dataDir <- args$dataDir
sampleName <- args$sampleName
threshold_count <- args$threshold_count
threshold_gene <- args$threshold_gene


print(paste0("dataDir: ",dataDir))
print(paste0("sampleName: ",sampleName))
print(paste0("threshold_count: ",threshold_count))
print(paste0("threshold_gene: ",threshold_gene))


system(paste0("mkdir ",dataDir,"/",sampleName,"/NormUMI/"))


# load Visium data
visiumData <- Load10X_Spatial(paste0(dataDir,"/",sampleName,"/SpaceRanger/"),
                              filename = paste0("filtered_feature_bc_matrix.h5"),
                              assay = "Spatial",
                              slice = "slice1",
                              filter.matrix = TRUE,
                              to.upper = FALSE)

# violin plot (before SCTransform)
plot1 <- VlnPlot(visiumData, features = "nCount_Spatial", pt.size = 0.1) + NoLegend()
plot2 <- SpatialFeaturePlot(visiumData, features = "nCount_Spatial") + theme(legend.position = "right")
p <- wrap_plots(plot1, plot2)
ggsave(file = paste0(dataDir,"/",sampleName,"/NormUMI/plot_Vln.png"), plot = p, dpi = 200, width = 12, height = 6)

# SCTransform
visiumData_SCT <- SCTransform(visiumData, assay = "Spatial", verbose = FALSE)

# violin plot (after SCTransform)
plot1 <- VlnPlot(visiumData_SCT, features = "nCount_SCT", pt.size = 0.1) + NoLegend()
plot2 <- SpatialFeaturePlot(visiumData_SCT, features = "nCount_SCT") + theme(legend.position = "right")
p <- wrap_plots(plot1, plot2)
ggsave(file = paste0(dataDir,"/",sampleName,"/NormUMI/plot_Vln_SCT.png"), plot = p, dpi = 200, width = 12, height = 6)

# make exp matrix
mat <- data.frame(GetAssayData(visiumData, slot="counts"))
mat_SCT <- data.frame(GetAssayData(visiumData_SCT, slot="counts"))

mat_log10 <- log10(mat+1)
mat_SCT_log10 <- log10(mat_SCT+1)

mat$symbol <- rownames(mat)
mat <- mat[,c(ncol(mat),1:(ncol(mat)-1))]

mat_SCT$symbol <- rownames(mat_SCT)
mat_SCT <- mat_SCT[,c(ncol(mat_SCT),1:(ncol(mat_SCT)-1))]

write.table(mat,paste0(dataDir,"/",sampleName,"/NormUMI/exp_mat.txt"),quote=F,sep="\t",row.names=F,col.names=T)
write.table(mat_SCT,paste0(dataDir,"/",sampleName,"/NormUMI/exp_mat_SCT.txt"),quote=F,sep="\t",row.names=F,col.names=T)

# make exp matrix (log10)
mat_log10$symbol <- rownames(mat_log10)
mat_log10 <- mat_log10[,c(ncol(mat_log10),1:(ncol(mat_log10)-1))]

mat_SCT_log10$symbol <- rownames(mat_SCT_log10)
mat_SCT_log10 <- mat_SCT_log10[,c(ncol(mat_SCT_log10),1:(ncol(mat_SCT_log10)-1))]

write.table(mat_log10,paste0(dataDir,"/",sampleName,"/NormUMI/exp_mat_log10.txt"),quote=F,sep="\t",row.names=F,col.names=T)
write.table(mat_SCT_log10,paste0(dataDir,"/",sampleName,"/NormUMI/exp_mat_SCT_log10.txt"),quote=F,sep="\t",row.names=F,col.names=T)



### Filtering low expression ###
visiumData_fil <- subset(visiumData, subset = nCount_Spatial >= threshold_count & nFeature_Spatial >= threshold_gene)

# violin plot (before SCTransform)
plot1 <- VlnPlot(visiumData_fil, features = "nCount_Spatial", pt.size = 0.1) + NoLegend()
plot2 <- SpatialFeaturePlot(visiumData_fil, features = "nCount_Spatial") + theme(legend.position = "right")
p <- wrap_plots(plot1, plot2)
ggsave(file = paste0(dataDir,"/",sampleName,"/NormUMI/plot_Vln_fil.png"), plot = p, dpi = 200, width = 12, height = 6)

# SCTransform
visiumData_fil_SCT <- SCTransform(visiumData_fil, assay = "Spatial", verbose = FALSE)

# violin plot (after SCTransform)
plot1 <- VlnPlot(visiumData_fil_SCT, features = "nCount_SCT", pt.size = 0.1) + NoLegend()
plot2 <- SpatialFeaturePlot(visiumData_fil_SCT, features = "nCount_SCT") + theme(legend.position = "right")
p <- wrap_plots(plot1, plot2)
ggsave(file = paste0(dataDir,"/",sampleName,"/NormUMI/plot_Vln_fil_SCT.png"), plot = p, dpi = 200, width = 12, height = 6)

# make exp matrix
mat <- data.frame(GetAssayData(visiumData_fil, slot="counts"))
mat_SCT <- data.frame(GetAssayData(visiumData_fil_SCT, slot="counts"))

mat_log10 <- log10(mat+1)
mat_SCT_log10 <- log10(mat_SCT+1)

mat$symbol <- rownames(mat)
mat <- mat[,c(ncol(mat),1:(ncol(mat)-1))]

mat_SCT$symbol <- rownames(mat_SCT)
mat_SCT <- mat_SCT[,c(ncol(mat_SCT),1:(ncol(mat_SCT)-1))]

write.table(mat,paste0(dataDir,"/",sampleName,"/NormUMI/exp_mat_fil.txt"),quote=F,sep="\t",row.names=F,col.names=T)
write.table(mat_SCT,paste0(dataDir,"/",sampleName,"/NormUMI/exp_mat_fil_SCT.txt"),quote=F,sep="\t",row.names=F,col.names=T)

# make exp matrix (log10)
mat_log10$symbol <- rownames(mat_log10)
mat_log10 <- mat_log10[,c(ncol(mat_log10),1:(ncol(mat_log10)-1))]

mat_SCT_log10$symbol <- rownames(mat_SCT_log10)
mat_SCT_log10 <- mat_SCT_log10[,c(ncol(mat_SCT_log10),1:(ncol(mat_SCT_log10)-1))]

write.table(mat_log10,paste0(dataDir,"/",sampleName,"/NormUMI/exp_mat_fil_log10.txt"),quote=F,sep="\t",row.names=F,col.names=T)
write.table(mat_SCT_log10,paste0(dataDir,"/",sampleName,"/NormUMI/exp_mat_fil_SCT_log10.txt"),quote=F,sep="\t",row.names=F,col.names=T)


###################################################################################################
print(paste0("Number of genes (before SCT): ",nrow(mat_log10)))
print(paste0("Number of genes (after SCT): ",nrow(mat_SCT_log10)))


###################################################################################################
# total UMIcount and geneNum (before filtering)
sumstat <- data.frame(sample=sampleName, totalUMI=visiumData$nCount_Spatial, geneNum=visiumData$nFeature_Spatial)

# violin plot (before filtering)
p1 <- ggplot(sumstat,aes(x=sample,y=log10(totalUMI)))
p1 <- p1 + geom_violin(trim=F,fill="#999999",alpha=I(1/3)) + geom_jitter(size=0.3) + geom_hline(yintercept=log10(threshold_count), linetype="dashed", color = "red", size=2)
p1 <- p1 + theme_linedraw() + theme(axis.text=element_text(size=16), axis.title=element_text(size=20,face="bold"), axis.title.x = element_blank())

p2 <- ggplot(sumstat,aes(x=sample,y=log10(geneNum)))
p2 <- p2 + geom_violin(trim=F,fill="#999999",alpha=I(1/3)) + geom_jitter(size=0.3) + geom_hline(yintercept=log10(threshold_gene), linetype="dashed", color = "red", size=2)
p2 <- p2 + theme_linedraw() + theme(axis.text=element_text(size=16), axis.title=element_text(size=20,face="bold"), axis.title.x = element_blank())

p <- wrap_plots(p1, p2)
ggsave(file = paste0(dataDir,"/",sampleName,"/NormUMI/plot_violin_totalUMI_totalGene.png"), plot = p, dpi = 200, width = 12, height = 6)

visiumData_fil <- subset(visiumData, subset = nCount_Spatial >= threshold_count & nFeature_Spatial >= threshold_gene)


# total UMIcount and geneNum (after filtering)
sumstat <- data.frame(sample=sampleName, totalUMI=visiumData_fil$nCount_Spatial, geneNum=visiumData_fil$nFeature_Spatial)

# violin plot (after filtering)
p1 <- ggplot(sumstat,aes(x=sample,y=log10(totalUMI)))
p1 <- p1 + geom_violin(trim=F,fill="#999999",alpha=I(1/3)) + geom_jitter(size=0.3) + geom_hline(yintercept=log10(threshold_count), linetype="dashed", color = "red", size=2)
p1 <- p1 + theme_linedraw() + theme(axis.text=element_text(size=16), axis.title=element_text(size=20,face="bold"), axis.title.x = element_blank())

p2 <- ggplot(sumstat,aes(x=sample,y=log10(geneNum)))
p2 <- p2 + geom_violin(trim=F,fill="#999999",alpha=I(1/3)) + geom_jitter(size=0.3) + geom_hline(yintercept=log10(threshold_gene), linetype="dashed", color = "red", size=2)
p2 <- p2 + theme_linedraw() + theme(axis.text=element_text(size=16), axis.title=element_text(size=20,face="bold"), axis.title.x = element_blank())

p <- wrap_plots(p1, p2)
ggsave(file = paste0(dataDir,"/",sampleName,"/NormUMI/plot_violin_totalUMI_totalGene_fil.png"), plot = p, dpi = 200, width = 12, height = 6)



### keep spot name ###
keep_spot_name <- colnames(visiumData_fil)
length(keep_spot_name)


### Function ###
geom_spatial <-  function(mapping = NULL,
                          data = NULL,
                          stat = "identity",
                          position = "identity",
                          na.rm = FALSE,
                          show.legend = NA,
                          inherit.aes = FALSE,
                          ...) {
  
  GeomCustom <- ggproto(
    "GeomCustom",
    Geom,
    setup_data = function(self, data, params) {
      data <- ggproto_parent(Geom, self)$setup_data(data, params)
      data
    },
    
    draw_group = function(data, panel_scales, coord) {
      vp <- grid::viewport(x=data$x, y=data$y)
      g <- grid::editGrob(data$grob[[1]], vp=vp)
      ggplot2:::ggname("geom_spatial", g)
    },
    
    required_aes = c("grob","x","y")
    
  )
  
  layer(
    geom = GeomCustom,
    mapping = mapping,
    data = data,
    stat = stat,
    position = position,
    show.legend = show.legend,
    inherit.aes = inherit.aes,
    params = list(na.rm = na.rm, ...)
  )
}


## define paths
sample_names <- c(sampleName)

image_paths <- paste0(dataDir,"/",sample_names,"/SpaceRanger/spatial/tissue_lowres_image.png")

scalefactor_paths <- paste0(dataDir,"/",sample_names,"/SpaceRanger/spatial/scalefactors_json.json")

tissue_paths <- paste0(dataDir,"/",sample_names,"/SpaceRanger/spatial/tissue_positions_list.csv")

cluster_paths <- paste0(dataDir,"/",sample_names,"/SpaceRanger/analysis/clustering/graphclust/clusters.csv")

matrix_paths <- paste0(dataDir,"/",sample_names,"/SpaceRanger/filtered_feature_bc_matrix/")

## Read in Down Sampled Images
images_cl <- list()

for (i in 1:length(sample_names)) {
  images_cl[[i]] <- read.bitmap(image_paths[i])
}

height <- list()

for (i in 1:length(sample_names)) {
  height[[i]] <-  data.frame(height = nrow(images_cl[[i]]))
}

height <- bind_rows(height)

width <- list()

for (i in 1:length(sample_names)) {
  width[[i]] <- data.frame(width = ncol(images_cl[[i]]))
}

width <- bind_rows(width)

# Convert the Images to Grobs
grobs <- list()
for (i in 1:length(sample_names)) {
  grobs[[i]] <- rasterGrob(images_cl[[i]], width=unit(1,"npc"), height=unit(1,"npc"))
}

images_tibble <- tibble(sample=factor(sample_names), grob=grobs)
images_tibble$height <- height$height
images_tibble$width <- width$width

scales <- list()

for (i in 1:length(sample_names)) {
  scales[[i]] <- rjson::fromJSON(file = scalefactor_paths[i])
}


## Read in Clusters
clusters <- list()
for (i in 1:length(sample_names)) {
  clusters[[i]] <- read.csv(cluster_paths[i])
}


## Combine Clusters and Tissue Information for Easy Plotting
bcs <- list()

for (i in 1:length(sample_names)) {
  bcs[[i]] <- read.csv(tissue_paths[i],col.names=c("barcode","tissue","row","col","imagerow","imagecol"), header = FALSE)
  bcs[[i]]$imagerow <- bcs[[i]]$imagerow * scales[[i]]$tissue_lowres_scalef    # scale tissue coordinates for lowres image
  bcs[[i]]$imagecol <- bcs[[i]]$imagecol * scales[[i]]$tissue_lowres_scalef
  bcs[[i]]$tissue <- as.factor(bcs[[i]]$tissue)
  bcs[[i]] <- merge(bcs[[i]], clusters[[i]], by.x = "barcode", by.y = "Barcode", all = TRUE)
  bcs[[i]]$height <- height$height[i]
  bcs[[i]]$width <- width$width[i]
}

names(bcs) <- sample_names


## Read in the Matrix, Barcodes, and Genes
matrix <- list()

mat <- fread(paste0(dataDir,"/",sample_names,"/NormUMI/exp_mat.txt"), data.table=F)

rownames(mat) <- mat$symbol
mat <- mat[,-1]
mat <- t(as.matrix(mat))
mat <- data.frame(mat)
rownames(mat) <- gsub("\\.","-",rownames(mat))

mat_sparse <- as(as.matrix(mat), "dgCMatrix")
matrix[[1]] <- mat_sparse



## Make Summary data.frames
# Total UMI per spot
umi_sum <- list() 

for (i in 1:length(sample_names)) {
  umi_sum[[i]] <- data.frame(barcode =  row.names(matrix[[i]]),
                             sum_umi = Matrix::rowSums(matrix[[i]]))
  
}
names(umi_sum) <- sample_names

umi_sum <- bind_rows(umi_sum, .id = "sample")


# Total Genes per Spot
gene_sum <- list() 

for (i in 1:length(sample_names)) {
  gene_sum[[i]] <- data.frame(barcode =  row.names(matrix[[i]]),
                              sum_gene = Matrix::rowSums(matrix[[i]] != 0))
  
}
names(gene_sum) <- sample_names

gene_sum <- bind_rows(gene_sum, .id = "sample")


## Merge All the Necessary Data
bcs_merge <- bind_rows(bcs, .id = "sample")
bcs_merge <- merge(bcs_merge,umi_sum, by = c("barcode", "sample"))
bcs_merge <- merge(bcs_merge,gene_sum, by = c("barcode", "sample"))


### Plotting ###
xlim(0,max(bcs_merge %>% 
             filter(sample == sample_names[i]) %>% 
             select(width)))

# Define our color palette for plotting
myPalette <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))


## Total UMI per Tissue Covered Spot
plots <- list()

for (i in 1:length(sample_names)) {
  
  plots[[1]] <- bcs_merge %>% 
    filter(sample ==sample_names[i]) %>% 
    ggplot(aes(x=imagecol,y=imagerow,fill=sum_umi)) +
    geom_spatial(data=images_tibble[i,], aes(grob=grob), x=0.5, y=0.5)+
    geom_point(shape = 21, size = 1.3, stroke = 0.0, alpha=1.0)+
    coord_cartesian(expand=FALSE)+
    scale_fill_gradientn(colours = myPalette(100))+
    xlim(0,max(bcs_merge %>% 
                 filter(sample ==sample_names[i]) %>% 
                 select(width)))+
    ylim(max(bcs_merge %>% 
               filter(sample ==sample_names[i]) %>% 
               select(height)),0)+
    xlab("") +
    ylab("") +
    ggtitle(sample_names[i])+
    labs(fill = "Total UMI")+
    theme_set(theme_bw(base_size = 10))+
    theme(panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          panel.background = element_blank(), 
          #          axis.line = element_line(colour = "black"),
          axis.text = element_blank(),
          axis.ticks = element_blank())
}

p <- plot_grid(plotlist = plots)
ggsave(file = paste0(dataDir,"/",sample_names,"/NormUMI/plot_totalUMI.png"), plot = p, dpi = 200, width = 6, height = 6)


## Total Genes per Tissue Covered Spot

for (i in 1:length(sample_names)) {
  
  plots[[2]] <- bcs_merge %>% 
    filter(sample ==sample_names[i]) %>% 
    ggplot(aes(x=imagecol,y=imagerow,fill=sum_gene)) +
    geom_spatial(data=images_tibble[i,], aes(grob=grob), x=0.5, y=0.5)+
    geom_point(shape = 21, size = 1.3, stroke = 0.0, alpha=1.0)+
    coord_cartesian(expand=FALSE)+
    scale_fill_gradientn(colours = myPalette(100))+
    xlim(0,max(bcs_merge %>% 
                 filter(sample ==sample_names[i]) %>% 
                 select(width)))+
    ylim(max(bcs_merge %>% 
               filter(sample ==sample_names[i]) %>% 
               select(height)),0)+
    xlab("") +
    ylab("") +
    ggtitle(sample_names[i])+
    labs(fill = "Total Genes")+
    theme_set(theme_bw(base_size = 10))+
    theme(panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          panel.background = element_blank(), 
          #          axis.line = element_line(colour = "black"),
          axis.text = element_blank(),
          axis.ticks = element_blank())
}

p <- plot_grid(plotlist = plots)
ggsave(file = paste0(dataDir,"/",sample_names,"/NormUMI/plot_totalUMI_totalGene.png"), plot = p, dpi = 200, width = 12, height = 6)


## plot remove spots
bcs_merge$remove_spot <- NA
idx <- bcs_merge$barcode %in% keep_spot_name
bcs_merge$remove_spot[idx] <- 0
bcs_merge$remove_spot[!idx] <-1

plots <- list()

for (i in 1:length(sample_names)) {
  
  plots[[1]] <- bcs_merge %>% 
    filter(sample ==sample_names[i]) %>% 
    ggplot(aes(x=imagecol,y=imagerow,fill=remove_spot)) +
    geom_spatial(data=images_tibble[i,], aes(grob=grob), x=0.5, y=0.5)+
    geom_point(shape = 21, size = 1.3, stroke = 0.0, alpha=1.0)+
    coord_cartesian(expand=FALSE)+
    scale_fill_gradientn(colours=c("white","red"))+
    xlim(0,max(bcs_merge %>% 
                 filter(sample ==sample_names[i]) %>% 
                 select(width)))+
    ylim(max(bcs_merge %>% 
               filter(sample ==sample_names[i]) %>% 
               select(height)),0)+
    xlab("") +
    ylab("") +
    ggtitle(sample_names[i])+
    labs(fill = "Remove spot")+
    theme_set(theme_bw(base_size = 10))+
    theme(panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          panel.background = element_blank(), 
          #          axis.line = element_line(colour = "black"),
          axis.text = element_blank(),
          axis.ticks = element_blank(),
          legend.position = 'none')
}

p <- plot_grid(plotlist = plots)
ggsave(file = paste0(dataDir,"/",sample_names,"/NormUMI/plot_remove_spots.png"), plot = p, dpi = 200, width = 6, height = 6)

