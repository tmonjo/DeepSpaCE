rm(list=ls())
#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
options(stringsAsFactors=F)

library(dplyr)
library(data.table)
library(ggplot2)
library(hrbrthemes)

library(Seurat)

library(Matrix)
library(rjson)
library(cowplot)
library(grid)
library(readbitmap)
library(argparse)


### Min-max scaling ###
minmaxscale <- function(x)
{
  return((x- min(x)) /(max(x)-min(x)))
}

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


# create parser object
parser <- ArgumentParser()

parser$add_argument("--dataDir", default=paste0("/home/",Sys.getenv("USER"),"/DeepSpaCE/data"),
                    help="data directory [default %(default)s]")
parser$add_argument("--sampleName", default="Human_Breast_Cancer_Block_A_Section_1",
                    help="sample name [default %(default)s]")
parser$add_argument("--outDir", default=paste0("/home/",Sys.getenv("USER"),"/DeepSpaCE/out"),
                    help="out directory [default %(default)s]")
parser$add_argument("--geneSymbol", default="SPARC",
                    help="gene symbol [default %(default)s]")
parser$add_argument("--extraSize", type="integer", default=150,
                    help="additional image size [%] [default %(default)s]")
parser$add_argument("--resolution", choices=c("low", "high"), default='low',
                    help="resolution (low or high) [default %(default)s]")

args <- parser$parse_args()


dataDir <- args$dataDir
sampleName <- args$sampleName
outDir <- args$outDir
geneSymbol <- args$geneSymbol
extraSize <- args$extraSize
resolution <- args$resolution


print(paste0("dataDir: ",dataDir))
print(paste0("sampleName: ",sampleName))
print(paste0("outDir: ",outDir))
print(paste0("geneSymbol: ",geneSymbol))
print(paste0("extraSize: ",extraSize))
print(paste0("resolution: ",resolution))


## Define Your Paths
if(resolution == "low"){
  image_path <- paste0(dataDir,"/",sampleName,"/SpaceRanger/spatial/tissue_hires_image.png")
}else if (resolution == "high"){
  image_path <- paste0(dataDir,"/",sampleName,"/CropImage/size_",extraSize,"/transpose_image.tif")
}

scalefactor_path <- paste0(dataDir,"/",sampleName,"/SpaceRanger/spatial/scalefactors_json.json")

tissue_path <- paste0(dataDir,"/",sampleName,"/SpaceRanger/spatial/tissue_positions_list.csv")

cluster_path <- paste0(dataDir,"/",sampleName,"/SpaceRanger/analysis/clustering/graphclust/clusters.csv")

matrix_path <- paste0(dataDir,"/",sampleName,"/SpaceRanger/filtered_feature_bc_matrix/")


## Read in Down Sampled Images
images_cl <- list()
images_cl[[1]] <- read.bitmap(image_path)

height <- list()
height <- data.frame(height = nrow(images_cl[[1]]))
height <- bind_rows(height)

width <- list()
width <- data.frame(width = ncol(images_cl[[1]]))
width <- bind_rows(width)


# Convert the Images to Grobs
grobs <- list()
grobs[[1]] <- rasterGrob(images_cl[[1]], width=unit(1,"npc"), height=unit(1,"npc"))

images_tibble <- tibble(sample=factor(sampleName), grob=grobs)
images_tibble$height <- height$height
images_tibble$width <- width$width

scales <- list()
scales[[1]] <- rjson::fromJSON(file = scalefactor_path)


## Read in Clusters
clusters <- list()
clusters[[1]] <- read.csv(cluster_path)



## Combine Clusters and Tissue Information for Easy Plotting
bcs <- list()

bcs[[1]] <- read.csv(tissue_path,col.names=c("barcode","tissue","row","col","imagerow","imagecol"), header = FALSE)
if(resolution == "low"){
  bcs[[1]]$imagerow <- bcs[[1]]$imagerow * scales[[1]]$tissue_hires_scalef    # scale tissue coordinates for lowres image
  bcs[[1]]$imagecol <- bcs[[1]]$imagecol * scales[[1]]$tissue_hires_scalef
}
bcs[[1]]$tissue <- as.factor(bcs[[1]]$tissue)
bcs[[1]] <- merge(bcs[[1]], clusters[[1]], by.x = "barcode", by.y = "Barcode", all = TRUE)
bcs[[1]]$height <- height$height[1]
bcs[[1]]$width <- width$width[1]

names(bcs)[1] <- sampleName

scale_factor <- scales[[1]]$tissue_hires_scalef


## Read in the Matrix, Barcodes, and Genes
matrix <- list()


## set expression value
mat <- fread(paste0(dataDir,"/",sampleName,"/NormUMI/exp_mat_log10.txt"), data.table=F)

rownames(mat) <- mat$symbol
mat <- mat[,-1]
mat <- t(as.matrix(mat))
mat <- data.frame(mat)
rownames(mat) <- gsub("\\.","-",rownames(mat))

mat <- mat[,geneSymbol,drop=F]

# Min-max scaling
for(i in 1:ncol(mat)){
  mat[,i] <- minmaxscale(mat[,i])
}

mat_sparse <- as(as.matrix(mat), "dgCMatrix")
matrix[[1]] <- mat_sparse



## Make Summary data.frames
# Total UMI per spot
umi_sum <- list() 

umi_sum[[1]] <- data.frame(barcode =  row.names(matrix[[1]]),
                           sum_umi = Matrix::rowSums(matrix[[1]]))

names(umi_sum)[1] <- sampleName

umi_sum <- bind_rows(umi_sum, .id = "sample")


# Total Genes per Spot
gene_sum <- list() 

gene_sum[[1]] <- data.frame(barcode =  row.names(matrix[[1]]),
                            sum_gene = Matrix::rowSums(matrix[[1]] != 0))
  
names(gene_sum)[1] <- sampleName

gene_sum <- bind_rows(gene_sum, .id = "sample")


## Merge All the Necessary Data
bcs_merge <- bind_rows(bcs, .id = "sample")
bcs_merge <- merge(bcs_merge,umi_sum, by = c("barcode", "sample"))
bcs_merge <- merge(bcs_merge,gene_sum, by = c("barcode", "sample"))


### Plotting ###
xlim(0,max(bcs_merge %>% 
             filter(sample == sampleName) %>% 
             select(width)))

# Define our color palette for plotting
myPalette <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))



### load Super-resolution images
image_list <- fread(paste0(outDir,"/image_list_pred.txt"), data.table=F)

colnames(image_list)[1] <- "sample"
colnames(image_list)[3] <- "imagecol"
colnames(image_list)[4] <- "imagerow"


#bcs_merge <- rbind(bcs_merge, pixcel_res)
image_list$barcode <- NA
image_list$tissue <- "1"
image_list$row <- NA
image_list$col <- NA
image_list$height <- bcs_merge$height[1]
image_list$width <- bcs_merge$width[1]
image_list$sum_umi <- 1
image_list$sum_gene <- NA
image_list$Cluster <- NA

if(resolution == "low"){
  image_list$imagerow <- image_list$imagerow * scale_factor
  image_list$imagecol <- image_list$imagecol * scale_factor
}

image_list <- image_list[,c("barcode","sample","tissue","row","col","imagerow","imagecol","Cluster","height","width","sum_umi","sum_gene",paste0(geneSymbol,"_pred"))]


# Min-max scaling
image_list[,paste0(geneSymbol,"_pred")] <- minmaxscale(image_list[,paste0(geneSymbol,"_pred")])


idx <- match(bcs_merge$barcode, rownames(matrix[[1]]))
sum(is.na(idx))

bcs_merge$exp_pred <- matrix[[1]][idx, geneSymbol]



colnames(bcs_merge)[colnames(bcs_merge) == "exp_pred"] <- paste0(geneSymbol,"_pred")

bcs_merge$type <- "original"
image_list$type <- "interpolate"

bcs_merge <- rbind(bcs_merge, image_list)




## Gene of Interest
plots <- list()

plots[[1]] <- bcs_merge %>% 
  filter(sample == sampleName) %>%
  filter(type == "interpolate") %>%
  ggplot(aes_string(x="imagecol",y="imagerow",color=paste0(geneSymbol,"_pred"))) +
  geom_spatial(data=images_tibble[i,], aes(grob=grob), x=0.5, y=0.5)+
  coord_cartesian(expand=FALSE)+
  scale_fill_gradientn(colours = myPalette(100))+
  scale_color_gradientn(colours = myPalette(100))+
  xlim(0,max(bcs_merge %>% 
               filter(sample == sampleName) %>% 
               select(width)))+
  ylim(max(bcs_merge %>% 
             filter(sample == sampleName) %>% 
             select(height)),0)+
  xlab("") +
  ylab("") +
  theme_set(theme_bw(base_size = 10))+
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        legend.position = 'none')


## Gene of Interest
plots[[2]] <- bcs_merge %>% 
  filter(sample == sampleName) %>%
  filter(type == "original") %>%
  ggplot(aes_string(x="imagecol",y="imagerow",color=paste0(geneSymbol,"_pred"))) +
  geom_point(aes_string(x="imagecol",y="imagerow",fill=paste0(geneSymbol,"_pred"),shape="type"), size = 2.1, stroke = 0.0, alpha=1)+
  coord_cartesian(expand=FALSE)+
  scale_fill_gradientn(colours = myPalette(100))+
  scale_color_gradientn(colours = myPalette(100))+
  xlim(0,max(bcs_merge %>% 
               filter(sample == sampleName) %>% 
               select(width)))+
  ylim(max(bcs_merge %>% 
             filter(sample == sampleName) %>% 
             select(height)),0)+
  xlab("") +
  ylab("") +
  #  ggtitle(sample_names[i])+
  theme_set(theme_bw(base_size = 10))+
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = 'black'),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        legend.position = 'none')



## Gene of Interest
plots[[3]] <- bcs_merge %>% 
  filter(sample == sampleName) %>%
  filter(type == "original") %>%
  ggplot(aes_string(x="imagecol",y="imagerow",color=paste0(geneSymbol,"_pred"))) +
  geom_spatial(data=images_tibble[i,], aes(grob=grob), x=0.5, y=0.5)+
  geom_point(aes_string(x="imagecol",y="imagerow",fill=paste0(geneSymbol,"_pred"),shape="type"), size = 2.1, stroke = 0.0, alpha=0.5)+
  coord_cartesian(expand=FALSE)+
  scale_fill_gradientn(colours = myPalette(100))+
  scale_color_gradientn(colours = myPalette(100))+
  xlim(0,max(bcs_merge %>% 
               filter(sample == sampleName) %>% 
               select(width)))+
  ylim(max(bcs_merge %>% 
             filter(sample == sampleName) %>% 
             select(height)),0)+
  xlab("") +
  ylab("") +
  #  ggtitle(sample_names[i])+
  theme_set(theme_bw(base_size = 10))+
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        legend.position = 'none')

ggsave(plots[[3]], file = paste0(outDir,"/superResolution__",sampleName,"__",geneSymbol,"_org.png"), dpi=400, width = 11, height = 11)


plots[[4]] <- bcs_merge %>% 
  filter(sample == sampleName) %>%
  filter(type == "interpolate") %>%
  ggplot(aes_string(x="imagecol",y="imagerow",color=paste0(geneSymbol,"_pred"))) +
  geom_spatial(data=images_tibble[i,], aes(grob=grob), x=0.5, y=0.5)+
  coord_cartesian(expand=FALSE)+
  scale_fill_gradientn(colours = myPalette(100))+
  scale_color_gradientn(colours = myPalette(100))+
  xlim(0,max(bcs_merge %>% 
               filter(sample == sampleName) %>% 
               select(width)))+
  ylim(max(bcs_merge %>% 
             filter(sample == sampleName) %>% 
             select(height)),0)+
  xlab("") +
  ylab("") +
  #  ggtitle(sample_names[i])+
  theme_set(theme_bw(base_size = 10))+
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        legend.position = 'none')


plots[[5]] <- bcs_merge %>% 
  filter(sample == sampleName) %>%
  filter(type == "interpolate") %>%
  ggplot(aes_string(x="imagecol",y="imagerow",color=paste0(geneSymbol,"_pred"))) +
  geom_point(aes_string(x="imagecol",y="imagerow",fill=paste0(geneSymbol,"_pred"),shape="type"), size = 2.1, stroke = 0.0, alpha=1)+
  coord_cartesian(expand=FALSE)+
  scale_fill_gradientn(colours = myPalette(100))+
  scale_color_gradientn(colours = myPalette(100))+
  xlim(0,max(bcs_merge %>% 
               filter(sample == sampleName) %>% 
               select(width)))+
  ylim(max(bcs_merge %>% 
             filter(sample == sampleName) %>% 
             select(height)),0)+
  xlab("") +
  ylab("") +
  theme_set(theme_bw(base_size = 10))+
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = 'black'),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        legend.position = 'none')


plots[[6]] <- bcs_merge %>% 
  filter(sample == sampleName) %>%
  filter(type == "interpolate") %>%
  ggplot(aes_string(x="imagecol",y="imagerow",color=paste0(geneSymbol,"_pred"))) +
  geom_spatial(data=images_tibble[i,], aes(grob=grob), x=0.5, y=0.5)+
  geom_point(aes_string(x="imagecol",y="imagerow",fill=paste0(geneSymbol,"_pred"),shape="type"), size = 2.1, stroke = 0.0, alpha=0.5)+
  coord_cartesian(expand=FALSE)+
  scale_fill_gradientn(colours = myPalette(100))+
  scale_color_gradientn(colours = myPalette(100))+
  xlim(0,max(bcs_merge %>% 
               filter(sample == sampleName) %>% 
               select(width)))+
  ylim(max(bcs_merge %>% 
             filter(sample == sampleName) %>% 
             select(height)),0)+
  xlab("") +
  ylab("") +
  theme_set(theme_bw(base_size = 10))+
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        legend.position = 'none')

ggsave(plots[[6]], file = paste0(outDir,"/superResolution__",sampleName,"__",geneSymbol,"_pred.png"), dpi=400, width = 11, height = 11)


plots[[7]] <- bcs_merge %>% 
  filter(sample == sampleName) %>%
  filter(type == "original" | type == "interpolate") %>%
  ggplot(aes_string(x="imagecol",y="imagerow",color=paste0(geneSymbol,"_pred"))) +
  geom_spatial(data=images_tibble[i,], aes(grob=grob), x=0.5, y=0.5)+
  coord_cartesian(expand=FALSE)+
  scale_fill_gradientn(colours = myPalette(100))+
  scale_color_gradientn(colours = myPalette(100))+
  xlim(0,max(bcs_merge %>% 
               filter(sample == sampleName) %>% 
               select(width)))+
  ylim(max(bcs_merge %>% 
             filter(sample == sampleName) %>% 
             select(height)),0)+
  xlab("") +
  ylab("") +
  theme_set(theme_bw(base_size = 10))+
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        legend.position = 'none')


plots[[8]] <- bcs_merge %>% 
  filter(sample == sampleName) %>%
  filter(type == "original" | type == "interpolate") %>%
  ggplot(aes_string(x="imagecol",y="imagerow",color=paste0(geneSymbol,"_pred"))) +
  geom_point(aes_string(x="imagecol",y="imagerow",fill=paste0(geneSymbol,"_pred")), size = 2.1, stroke = 0.0, alpha=1)+
  coord_cartesian(expand=FALSE)+
  scale_fill_gradientn(colours = myPalette(100))+
  scale_color_gradientn(colours = myPalette(100))+
  xlim(0,max(bcs_merge %>% 
               filter(sample == sampleName) %>% 
               select(width)))+
  ylim(max(bcs_merge %>% 
             filter(sample == sampleName) %>% 
             select(height)),0)+
  xlab("") +
  ylab("") +
  theme_set(theme_bw(base_size = 10))+
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = 'black'),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        legend.position = 'none')


plots[[9]] <- bcs_merge %>% 
  filter(sample == sampleName) %>%
  filter(type == "original" | type == "interpolate") %>%
  ggplot(aes_string(x="imagecol",y="imagerow",color=paste0(geneSymbol,"_pred"))) +
  geom_spatial(data=images_tibble[i,], aes(grob=grob), x=0.5, y=0.5)+
  geom_point(aes_string(x="imagecol",y="imagerow",fill=paste0(geneSymbol,"_pred")), size = 2.1, stroke = 0.0, alpha=0.5)+
  coord_cartesian(expand=FALSE)+
  scale_fill_gradientn(colours = myPalette(100))+
  scale_color_gradientn(colours = myPalette(100))+
  xlim(0,max(bcs_merge %>% 
               filter(sample == sampleName) %>% 
               select(width)))+
  ylim(max(bcs_merge %>% 
             filter(sample == sampleName) %>% 
             select(height)),0)+
  xlab("") +
  ylab("") +
  theme_set(theme_bw(base_size = 10))+
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        legend.position = 'none')

ggsave(plots[[9]], file = paste0(outDir,"/superResolution__",sampleName,"__",geneSymbol,"_both.png"), dpi=400, width = 11, height = 11)


plot_grid(plotlist = plots, nrow=3)
ggsave(file = paste0(outDir,"/superResolution__",sampleName,"__",geneSymbol,".png"), dpi=400, width = 33, height = 33)

