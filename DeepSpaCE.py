#!/usr/bin/env python
# coding: utf-8


import glob
import os.path as osp
import random
import numpy as np
import pandas as pd
import json
import pickle
import sys
import time
import math
import subprocess
import argparse

from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from matplotlib.ticker import MaxNLocator

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import KFold

from mlxtend.plotting import plot_confusion_matrix
from mlxtend.evaluate import confusion_matrix

import csv
import gzip
import os
import scipy.io
import cv2

import albumentations as albu
from albumentations.pytorch import ToTensor

sys.path.append('./')

from DeepSpaceLib import makeDataList
from DeepSpaceLib import makeTrainDataloader
from DeepSpaceLib import make_model
from DeepSpaceLib import run_train
from DeepSpaceLib import makeTestDataloader
from DeepSpaceLib import run_test
from DeepSpaceLib import makeDataListSemi
from DeepSpaceLib import makeSemiDataloader
from DeepSpaceLib import predict_semi_label



argrequired = False



parser = argparse.ArgumentParser(description='DeepSpaCE')

parser.add_argument('--dataDir', type=str, nargs=1, default='/home/'+os.environ['USER']+'/DeepSpaCE/data', required=argrequired,
                    help='Data directory (default: '+'/home/'+os.environ['USER']+'/DeepSpaCE/data'+')')

parser.add_argument('--outDir', type=str, nargs=1, default='/home/'+os.environ['USER']+'/DeepSpaCE/',
                    help='Root directory (default: '+'/home/'+os.environ['USER']+'/DeepSpaCE/'+')')

parser.add_argument('--sampleNames_train', type=str, nargs=1, default='Human_Breast_Cancer_Block_A_Section_1',
                    help='Sample names to train (default: Human_Breast_Cancer_Block_A_Section_1)')

parser.add_argument('--sampleNames_test', type=str, nargs=1, default='Human_Breast_Cancer_Block_A_Section_1',
                    help='Sample names to test (default: Human_Breast_Cancer_Block_A_Section_1)')

parser.add_argument('--sampleNames_semi', type=str, nargs=1, default='None',
                    help='Sample names to semi-supervised learning (default: None)')

parser.add_argument('--semi_option', type=str, nargs=1, choices=['normal', 'random', 'permutation'], default='normal',
                    help='Option of semi-supervised learning (default: normal)')

parser.add_argument('--seed', type=int, nargs=1, default=0,
                    help='Random seed (default: 0)')

parser.add_argument('--threads', type=int, nargs=1, default=8,
                    help='Number of CPU threads (default: 8)')

parser.add_argument('--GPUs', type=int, nargs=1, default=1,
                    help='Number of GPUs (default: 1)')

parser.add_argument('--cuda', action='store_true',
                    help='Enables CUDA training')

parser.add_argument('--transfer', action='store_true',
                    help='Enables transfer training')

parser.add_argument('--model', type=str, nargs=1, choices=['VGG16','DenseNet121'], default='DenseNet121',
                    help='Deep learning model')

parser.add_argument('--batch_size', type=int, nargs=1, default=128,
                    help='Input batch size for training (default: 128)')

parser.add_argument('--num_epochs', type=int, nargs=1, default=10,
                    help='Number of epochs to train (default: 100)')

parser.add_argument('--lr', type=float, nargs=1, default=1e-4,
                    help='Learning rate (default: 1e-4)')

parser.add_argument('--weight_decay', type=float, nargs=1, default=1e-4,
                    help='Weight decay (default: 1e-4)')

parser.add_argument('--clusteringMethod', type=str, nargs=1, choices=['graphclust', 'kmeans_2_clusters', 'kmeans_3_clusters', 'kmeans_4_clusters', 'kmeans_5_clusters', 'kmeans_6_clusters', 'kmeans_7_clusters','kmeans_8_clusters', 'kmeans_9_clusters', 'kmeans_10_clusters'], default='graphclust',
                    help='Clustering method (default: graphclust)')

parser.add_argument('--extraSize', type=int, nargs=1, default=150,
                    help='Extra image size (default: 150)')

parser.add_argument('--quantileRGB', type=int, nargs=1, default=80,
                    help='Threshold of quantile RGB (default: 80)')

parser.add_argument('--augmentation', type=str, nargs=1, default='flip,crop,color,random',
                    help='Image augmentation methods (default: flip,crop,color,random)')

parser.add_argument('--early_stop_max', type=int, nargs=1, default=5,
                    help='How many epochs to wait for loss improvement (default: 5)')

parser.add_argument('--rm_cluster', type=str, nargs=1, default='-1',
                    help='Remove cluster name (default: None)')

parser.add_argument('--ClusterPredictionMode', action='store_true',
                    help='Enables ClusterPredictionMode')

parser.add_argument('--cross_index', type=int, nargs=1, default=0,
                    help='Index of 5-fold cross-validation (default: 0)')

parser.add_argument('--geneSymbols', type=str, nargs=1, default='ESR1,ERBB2,MKI67',
                    help='Gene symbols (default: ESR1,ERBB2,MKI67)')

args = parser.parse_args()



print(args)



dataDir = args.dataDir
print("dataDir: "+str(dataDir))

outDir = args.outDir
print("outDir: "+str(outDir))

batch_size = args.batch_size * args.GPUs
print("batch_size: "+str(batch_size))

num_epochs = args.num_epochs
print("num_epochs: "+str(num_epochs))

lr = args.lr
print("lr: "+str(lr))

weight_decay = args.weight_decay
print("weight_decay: "+str(weight_decay))

model = args.model
print("model: "+str(model))

clusteringMethod = args.clusteringMethod
print("clusteringMethod: "+str(clusteringMethod))

cuda = args.cuda and torch.cuda.is_available()
print("cuda: "+str(cuda))

transfer = args.transfer
print("transfer: "+str(transfer))

quantileRGB = args.quantileRGB
print("quantileRGB: "+str(quantileRGB))

seed = args.seed
print("seed: "+str(seed))

threads = args.threads
print("threads: "+str(threads))

early_stop_max = args.early_stop_max
print("early_stop_max: "+str(early_stop_max))

extraSize = args.extraSize
print("extraSize: "+str(extraSize))

augmentation = args.augmentation
print("augmentation: "+augmentation)

semi_option = args.semi_option
print("semi_option: "+str(semi_option))

cross_index = args.cross_index
print("cross_index: "+str(cross_index))

ClusterPredictionMode = args.ClusterPredictionMode
print("ClusterPredictionMode: "+str(ClusterPredictionMode))



if args.rm_cluster == 'None':
    rm_cluster = -1
else:
    rm_cluster = int(args.rm_cluster)

print("rm_cluster: "+str(rm_cluster))



sampleNames_train = args.sampleNames_train.split(',')
print("sampleNames_train: "+str(sampleNames_train))



sampleNames_test = args.sampleNames_test.split(',')
print("sampleNames_test: "+str(sampleNames_test))



if sampleNames_train == sampleNames_test:
    train_equals_test = True
else:
    train_equals_test = False

print("train_equals_test: "+str(train_equals_test))



sampleNames_semi = args.sampleNames_semi.split(',')
print("sampleNames_semi: "+str(sampleNames_semi))



geneSymbols = args.geneSymbols.split(',')
print(geneSymbols)



size = 224
print("size: "+str(size))

mean = (0.485, 0.456, 0.406)
print("mean: "+str(mean))

std = (0.229, 0.224, 0.225)
print("std: "+str(std))



print("### make outDir ###")
os.makedirs(outDir, exist_ok=True)



print("### Set seeds ###")
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.set_num_threads(threads)



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



print("### Check GPU availability ###")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)


# # make data_list (teacher)


data_list_teacher = makeDataList(rootDir=dataDir,
                                 sampleNames=sampleNames_train,
                                 clusteringMethod=clusteringMethod,
                                 extraSize=extraSize,
                                 geneSymbols=geneSymbols,
                                 quantileRGB=quantileRGB,
                                 seed=seed,
                                 cross_index=cross_index,
                                 train_equals_test=train_equals_test,
                                 is_test=False,
                                 rm_cluster=rm_cluster)

if train_equals_test:
    data_list_teacher_tmp = data_list_teacher.copy()
    data_list_teacher = data_list_teacher_tmp.query('phase != "test"').copy()


data_list_teacher.to_csv(outDir+"/data_list_teacher.txt", index=False, sep='\t', float_format='%.6f')

print("data_list_teacher: "+str(data_list_teacher.shape))
data_list_teacher.head()


# # make dataloader (teacher)


dataloaders_dict_teacher = makeTrainDataloader(rootDir=dataDir,
                                               data_list_df=data_list_teacher,
                                               geneSymbols=geneSymbols,
                                               size=size,
                                               mean=mean,
                                               std=std,
                                               augmentation=augmentation,
                                               batch_size=batch_size,
                                               ClusterPredictionMode=ClusterPredictionMode)

print("### save dataloader ###")
with open(outDir+"/dataloaders_dict_teacher.pickle", mode='wb') as f:
    pickle.dump(dataloaders_dict_teacher, f)






# # make network model (teacher)


print("### make model ###")
if ClusterPredictionMode:
    net, params_to_update = make_model(use_pretrained=True,
                                       num_features=len(data_list_teacher['Cluster'].unique()),
                                       transfer=transfer,
                                       model=model)
else:
    net, params_to_update = make_model(use_pretrained=True,
                                       num_features=len(geneSymbols),
                                       transfer=transfer,
                                       model=model)


# # set optimizer


print("### set optimizer ###")
optimizer = optim.Adam(params=params_to_update, lr=lr, weight_decay=weight_decay)






# # Training (teacher)


print("### run train ###")
run_train(outDir=outDir,
          net=net,
          dataloaders_dict=dataloaders_dict_teacher,
          optimizer=optimizer,
          num_epochs=num_epochs,
          device=device,
          early_stop_max=early_stop_max,
          ClusterPredictionMode=ClusterPredictionMode,
          name='teacher')






# # Test (teacher)


data_list_test = makeDataList(rootDir=dataDir,
                              sampleNames=sampleNames_test,
                              clusteringMethod=clusteringMethod,
                              extraSize=extraSize,
                              geneSymbols=geneSymbols,
                              quantileRGB=quantileRGB,
                              seed=seed,
                              cross_index=cross_index,
                              train_equals_test=True,
                              is_test=True,
                              rm_cluster=rm_cluster)

if train_equals_test:
    data_list_test = data_list_teacher_tmp.query('phase == "test"').copy()


data_list_test['phase'] = 'valid'
data_list_test.to_csv(outDir+"/data_list_test.txt", index=False, sep='\t', float_format='%.6f')

print("data_list_test: "+str(data_list_test.shape))
data_list_test.head()



dataloaders_dict_test = makeTestDataloader(rootDir=dataDir,
                                           data_list_df=data_list_test,
                                           model=model,
                                           geneSymbols=geneSymbols,
                                           size=size,
                                           mean=mean,
                                           std=std,
                                           augmentation=augmentation,
                                           batch_size=batch_size,
                                           ClusterPredictionMode=ClusterPredictionMode)

# save dataloader
with open(outDir+"/DataLoader_test.pickle", mode='wb') as f:
    pickle.dump(dataloaders_dict_test, f)



### Test ###
data_list_test_teacher, net_best = run_test(outDir=outDir,
                                            data_list_df=data_list_test,
                                            dataloaders_dict=dataloaders_dict_test,
                                            model=model,
                                            device=device,
                                            geneSymbols=geneSymbols,
                                            num_features=len(data_list_teacher['Cluster'].unique()),
                                            ClusterPredictionMode=ClusterPredictionMode,
                                            name="teacher")

data_list_test_teacher.to_csv(outDir+"/data_list_test_teacher.txt", index=False, sep='\t', float_format='%.6f')

print("data_list_test_teacher: "+str(data_list_test_teacher.shape))
data_list_test_teacher.head()


# # Semi-supervised


if sampleNames_semi == ["None"]:
    sys.exit()
else:
    print("### Semi-supervised ###")



ImageSet = [[0,1],[2,3],[4,5],[6,7],[8,9]]
print("ImageSet: "+str(ImageSet))







for i_semi in range(5):
    if sampleNames_semi == ["TCGA"]:
        data_list_semi = makeDataListSemi(rootDir=dataDir,
                                          sampleNames=sampleNames_semi,
                                          semiType="TCGA",
                                          ImageSet=ImageSet[i_semi],
                                          semiName="semi"+str(i_semi+1))
    elif sampleNames_semi == ["ImageNet"]:
        data_list_semi = makeDataListSemi(rootDir=dataDir,
                                          sampleNames=sampleNames_semi,
                                          semiType="ImageNet",
                                          ImageSet=ImageSet[i_semi],
                                          semiName="semi"+str(i_semi+1))
    else:
        data_list_semi = makeDataListSemi(rootDir=dataDir,
                                          sampleNames=sampleNames_semi,
                                          semiType="Visium",
                                          ImageSet=ImageSet[i_semi],
                                          semiName="semi"+str(i_semi+1))

    # make semi dictionary
    dataloaders_dict_semi = makeSemiDataloader(rootDir=dataDir,
                                               data_list_df=data_list_semi,
                                               size=size,
                                               mean=mean,
                                               std=std,
                                               augmentation=augmentation,
                                               batch_size=batch_size)

    # save semi dataloader
    with open("../out/dataloaders_dict_semi"+str(i_semi+1)+".pickle", mode='wb') as f:
        pickle.dump(dataloaders_dict_semi, f)

    # predict semi labels
    data_list_semi = predict_semi_label(net=net_best,
                                        data_list_semi=data_list_semi,
                                        dataloaders_dict_semi=dataloaders_dict_semi,
                                        geneSymbols=geneSymbols,
                                        ClusterPredictionMode=ClusterPredictionMode)
    
    if ClusterPredictionMode:
        if semi_option == "permutation":
            data_list_semi['Cluster'] = random.sample(data_list_semi['Cluster'].tolist(), len(data_list_semi))
        
        elif semi_option == "random":
            data_list_semi['Cluster'] = random.randint(0, max(data_list_semi['Cluster']))
        
        elif semi_option == "normal":
            pass
    else:
        if semi_option == "permutation":
            for gene in geneSymbols:
                data_list_semi[gene] = minmax_scale(data_list_semi[gene])
                data_list_semi[gene] = random.sample(data_list_semi[gene].tolist(), len(data_list_semi))

        elif semi_option == "random":
            for gene in geneSymbols:
                data_list_semi[gene] = [random.uniform(0, 1) for i in range(len(data_list_semi))]

        elif semi_option == "normal":
            for gene in geneSymbols:
                data_list_semi[gene] = minmax_scale(data_list_semi[gene])


    print("data_list_semi"+str(i_semi+1)+": "+str(data_list_semi.shape))
    data_list_semi.head()

    data_list_semi.to_csv("../out/data_list_semi"+str(i_semi+1)+".txt", index=False, sep='\t', float_format='%.6f')


    ## add semi dataset
    data_list_student = pd.concat([data_list_teacher, data_list_semi])
    data_list_teacher = data_list_student.copy()
    
    data_list_student = data_list_student.reset_index(drop=True)
    
    ### 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    count = 0
    for train_index, test_index in kf.split(data_list_student.index, data_list_student.index):
        if count == cross_index:
            data_list_student.loc[data_list_student.index[train_index], 'phase'] = 'train'
            data_list_student.loc[data_list_student.index[test_index], 'phase'] = 'valid'

        count += 1
    
    data_list_student.to_csv("../out/data_list_student"+str(i_semi+1)+".txt", index=False, sep='\t', float_format='%.6f')

    print("data_list_student"+str(i_semi+1)+": "+str(data_list_student.shape))
    data_list_student.head()

    
    dataloaders_dict_student = makeTrainDataloader(rootDir=dataDir,
                                                   data_list_df=data_list_student,
                                                   geneSymbols=geneSymbols,
                                                   size=size,
                                                   mean=mean,
                                                   std=std,
                                                   augmentation=augmentation,
                                                   batch_size=batch_size,
                                                   ClusterPredictionMode=ClusterPredictionMode)

    print("### save dataloader ###")
    with open("../out/dataloaders_dict_student"+str(i_semi+1)+".pickle", mode='wb') as f:
        pickle.dump(dataloaders_dict_student, f)

    
    print("### make model ###")
    if ClusterPredictionMode:
        net, params_to_update = make_model(use_pretrained=True,
                                           num_features=len(data_list_teacher['Cluster'].unique()),
                                           transfer=transfer,
                                           model=model)
    else:
        net, params_to_update = make_model(use_pretrained=True,
                                           num_features=len(geneSymbols),
                                           transfer=transfer,
                                           model=model)
    
    print("### set optimizer ###")
    optimizer = optim.Adam(params=params_to_update, lr=lr, weight_decay=weight_decay)

    
    ### Training ###
    print("### run train ###")
    run_train(outDir=outDir,
              net=net,
              dataloaders_dict=dataloaders_dict_student,
              optimizer=optimizer,
              num_epochs=num_epochs,
              device=device,
              early_stop_max=early_stop_max,
              ClusterPredictionMode=ClusterPredictionMode,
              name='student'+str(i_semi+1))

    
    ### validation ###
    data_list_test_student, net_best = run_test(outDir=outDir,
                                                data_list_df=data_list_test,
                                                dataloaders_dict=dataloaders_dict_test,
                                                model=model,
                                                device=device,
                                                geneSymbols=geneSymbols,
                                                num_features=len(data_list_test['Cluster'].unique()),
                                                ClusterPredictionMode=ClusterPredictionMode,
                                                name="student"+str(i_semi+1))

    data_list_test_student.to_csv("../out/data_list_test_student"+str(i_semi+1)+".txt", index=False, sep='\t', float_format='%.6f')

    print("data_list_test_student"+str(i_semi+1)+": "+str(data_list_test_student.shape))
    data_list_test_student.head() 

















