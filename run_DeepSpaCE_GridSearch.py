#! /usr/bin/python
# coding=utf-8

import subprocess
import datetime


dt_now = datetime.datetime.now()
outDirName = "out_Gene_"+dt_now.strftime('%m%d__%H%M')

subprocess.call(['mkdir','-p','../out/'+outDirName])


model = "VGG16"
clustering = "graphclust"

pheno_train = ["Human_Breast_Cancer_Block_A_Section_2"]
pheno_test = ["Human_Breast_Cancer_Block_A_Section_2"]

gene_list = "CPB1,CXCL14,CRISP3"
    
    
#for image_filter_percent in [50,60,70,80,90,100]:
for image_filter_percent in [80]:
    #    for semi_option in ['normal','permutation','random']:
    for semi_option in ['normal']:
        #        for margin in [0, 50, 100, 150, 200]:
        for margin in [150]:
            #            for aug in ["none","flip","flip,random","flip,color","flip,crop","flip,contrast","flip,noise","flip,blur","flip,dist"]:
            for aug in ["flip,crop,color,random"]:
                #                for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
                for lr in [1e-4]:
                    #                    for weight_decay in [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
                    for weight_decay in [1e-4]:
                        #                        for seed in [0, 1, 2, 3, 4]:
#                        for cross_index in [0, 1, 2, 3, 4]:
                        for cross_index in [0]:
                            aug_name = '_'.join(aug.split(','))

                            dirName = "Train_"+'_'.join(pheno_train)+"__Test_"+'_'.join(pheno_test)+"__model_"+model+"__RGB"+str(image_filter_percent)+"__Size_"+str(margin)+"__aug_"+aug_name+"__lr_"+str(lr)+"__weightDecay_"+str(weight_decay)+"__cross_index_"+str(cross_index)

                            subprocess.call(['mkdir','-p','../out/'+outDirName+'/'+dirName+'/out/'])
                            subprocess.call(['mkdir','-p','../out/'+outDirName+'/'+dirName+'/script/'])
                            
                            
                            f = open("../out/"+outDirName+"/"+dirName+"/script/run.sh", 'w')


                            f.write("""#$ -S /usr/bin/bash
#$ -cwd
#$ -V
#$ -q gpuv.q
#$ -pe def_slot 8
#$ -l v100=1,s_vmem=125G
""")

                            f.write("""
export OMP_NUM_THREADS=8
cd ../out/"""+outDirName+"""/"""+dirName+"""/script/

module load /usr/local/package/modulefiles/singularity/3.7.0
""")

                            f.write("""
cp -rp /home/$USER/DeepSpaCE/script/BasicLib.py .
cp -rp /home/$USER/DeepSpaCE/script/DeepSpaceLib.py .
cp -rp /home/$USER/DeepSpaCE/script/DeepSpaCE.py .
                                
singularity exec --nv /home/$USER/DeepSpaCE/Singularity/DeepSpaCE.sif \\
python DeepSpaCE.py \\
--dataDir /home/$USER/DeepSpaCE/data \\
--outDir ../out \\
--sampleNames_train """+','.join(pheno_train)+""" \\
--sampleNames_test """+','.join(pheno_test)+""" \\
--seed 0 \\
--threads 8 \\
--GPUs 1 \\
--cuda \\
--model """+model+""" \\
--batch_size 128 \\
--num_epochs 50 \\
--lr """+str(lr)+""" \\
--weight_decay """+str(weight_decay)+""" \\
--extraSize """+str(margin)+""" \\
--quantileRGB """+str(image_filter_percent)+""" \\
--augmentation """+aug+""" \\
--early_stop_max 5 \\
--cross_index """+str(cross_index)+""" \\
--geneSymbols """+gene_list+"""
""")

                            f.close()
                            
                            subprocess.call(['qsub','-o','../out/'+outDirName+'/'+dirName+'/script/out/','-e','../out/'+outDirName+'/'+dirName+'/script/err/','../out/'+outDirName+'/'+dirName+'/script/run.sh'])
