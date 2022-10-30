# DeepSpaCE

The **Deep** learning model for **Spa**tial gene **C**lusters and **E**xpression (DeepSpaCE) is a method that predicts spatial gene-expression levels and transcriptomic cluster types from tissue section images using deep learning.

# Note

DeepSpaCE is now beta version. Please note that we shall not be responsible for any loss, damages and troubles.

# Table of Contents
- [Requirement](#requirement)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)
- [FAQ](#FAQ)
- [Release notes](#Release-notes)

# Requirement
* Singularity (verified in [v3.7](https://sylabs.io/guides/3.7/user-guide/))


# Installation
## Clone the DeepSpaCE repository

    git clone https://github.com/tmonjo/DeepSpaCE

## Build a Singularity image (option 1)
Pull an image from Dokcer Hub.

    singularity pull docker://tmonjo/deepspace:"v1.0"

## Build a Singularity image (option 2)
Build an image on your local environment since root privileges are required. Then, you can run DeepSpaCE with "DeepSpaCE.sif" on any servers.

    sudo singularity build DeepSpaCE.sif DeepSpaCE.srecipe


# Usage
## Input files (all files must be located in a same directory of sampleName)

1. Space Ranger outputs

    /home/$USER/DeepSpaCE/data/{sampleName}/SpaceRanger/analysis/
    
    /home/$USER/DeepSpaCE/data/{sampleName}/SpaceRanger/spatial/
    
    /home/$USER/DeepSpaCE/data/{sampleName}/SpaceRanger/filtered_feature_bc_matrix.h5

2. TIFF image (same directory)

    /home/$USER/DeepSpaCE/data/{sampleName}/SpaceRanger/image.tif



## Preprocessing 1: Section image files

    singularity exec DeepSpaCE.sif \
        python script/CropImage.py \
            --dataDir /home/$USER/DeepSpaCE/data \
            --sampleName Human_Breast_Cancer_Block_A_Section_1 \
            --transposeType 0 \
            --radiusPixel 75 \
            --extraSize 150 \
            --quantileRGB 80

## Preprocessing 2: Satial expression data measured by Visium

    singularity exec DeepSpaCE.sif \
        Rscript script/NormalizeUMI.R \
            --dataDir /home/$USER/DeepSpaCE/data \
            --sampleName Human_Breast_Cancer_Block_A_Section_1 \
            --threshold_count 1000 \
            --threshold_gene 1000

## Run DeepSpaCE (Training and validation) (running on GPU is recommended)

    singularity exec --nv DeepSpaCE.sif \
        python script/DeepSpaCE.py \
            --dataDir /home/$USER/DeepSpaCE/data \
            --outDir /home/$USER/DeepSpaCE/out \
            --sampleNames_train Human_Breast_Cancer_Block_A_Section_1 \
            --sampleNames_test Human_Breast_Cancer_Block_A_Section_1 \
            --sampleNames_semi None \
            --semi_option normal \
            --seed 0 \
            --threads 8 \
            --GPUs 1 \
            --cuda \
            --transfer \
            --model VGG16 \
            --batch_size 128 \
            --num_epochs 10 \
            --lr 1e-4 \
            --weight_decay 1e-4 \
            --clusteringMethod graphclust \
            --extraSize 150 \
            --quantileRGB 80 \
            --augmentation flip,crop,color,random \
            --early_stop_max 5 \
            --cross_index 0 \
            --geneSymbols ESR1,ERBB2,MKI67


## Super-resolution

### Run super-resolution

    singularity exec DeepSpaCE.sif \
        python script/SuperResolution.py \
            --dataDir /home/$USER/DeepSpaCE/data \
            --outDir /home/$USER/DeepSpaCE/out \
            --sampleName Human_Breast_Cancer_Block_A_Section_1 \
            --model VGG16 \
            --seed 0 \
            --threads 8 \
            --modelName teacher \
            --batch_size 128 \
            --extraSize 150 \
            --quantileRGB 80 \
            --geneSymbols ESR1,ERBB2,MKI67

### Plot a super-resolved image

    singularity exec DeepSpaCE.sif \
        Rscript script/PlotSuperResolution.R \
            --dataDir /home/$USER/DeepSpaCE/data \
            --outDir /home/$USER/DeepSpaCE/out \
            --sampleName Human_Breast_Cancer_Block_A_Section_1 \
            --geneSymbol ESR1 \
            --extraSize 150

## Semi-supervised learning (under development)

Prepare image_list.txt for semi-supervised learning.
image_list.txt should contain "ImageSet", "sample_No, and "image_path".

/home/$USER/DeepSpaCE/data/Visium/ImageSet/"+sampleName+"/ImageSet_0/image_list.txt


# Citation
Monjo, T., Koido, M., Nagasawa, S. et al. Efficient prediction of a spatial transcriptomics profile better characterizes breast cancer tissue sections without costly experimentation. Sci Rep 12, 4133 (2022). https://doi.org/10.1038/s41598-022-07685-4


# License
GNU General Public License v3.0

# FAQ
1. Can I install DeepSpaCE without Singularity?

    Please install Python 3.6, R >= 4.1, and libraries written in "DeepSpaCE.srecipe".

    Pipfile is also available. ([Pipenv](https://pipenv.pypa.io/))

        pipenv install Pipfile

# Release notes

* v0.1 (November 14 2021): First release
