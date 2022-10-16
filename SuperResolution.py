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

from torchvision import models, transforms

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import StratifiedKFold

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

from BasicLib import ImageTransform
from BasicLib import SpotImageDataset
from BasicLib import plot_loss
from BasicLib import plot_correlation_scatter_hist



parser = argparse.ArgumentParser(description='Super-resolution')

parser.add_argument('--dataDir', type=str, default='/home/'+os.environ['USER']+'/DeepSpaCE/data',
                   help='Data directory (default: '+'/home/'+os.environ['USER']+'/DeepSpaCE/data'+')')

parser.add_argument('--outDir', type=str, default='/home/'+os.environ['USER']+'/DeepSpaCE/out',
                   help='Output directory (default: '+'/home/'+os.environ['USER']+'/DeepSpaCE/out'+')')

parser.add_argument('--sampleNames', type=str, default='Human_Breast_Cancer_Block_A_Section_1',
                   help='Sample names (default: 0)')

parser.add_argument('--seed', type=int, default=0,
                   help='Random seed (default: 0)')

parser.add_argument('--threads', type=int, default=8,
                   help='Number of CPU threads (default: 8)')

parser.add_argument('--full', action='store_true',
                    help='Enables full training')

parser.add_argument('--model', type=str, choices=['VGG16','DenseNet121'], default='DenseNet121',
                    help='Deep learning model')

parser.add_argument('--modelName', type=str, choices=['teacher', 'student1', 'student2', 'student3', 'student4', 'student5'], default='teacher',
                   help=' (default: teacher)')

parser.add_argument('--batch_size', type=int, default=128,
                    help='Input batch size for training (default: 128)')

parser.add_argument('--extraSize', type=int, default=150,
                   help='Extra image size (default: 150)')

parser.add_argument('--quantileRGB', type=int, default=80,
                   help='Threshold of quantile RGB (default: 80)')

parser.add_argument('--geneSymbols', type=str, default='ESR1,ERBB2,MKI67',
                   help='Gene symbols (default: ESR1,ERBB2,MKI67)')

parser.add_argument('--ClusterPredictionMode', action='store_true',
                   help='Enables ClusterPredictionMode')

parser.add_argument('--cluster_num', type=int, default=7,
                   help='Number of clusters (default: 7)')

args = parser.parse_args()



print(args)



dataDir = args.dataDir
print("dataDir: "+str(dataDir))

outDir = args.outDir
print("outDir: "+str(outDir))

batch_size = args.batch_size
print("batch_size: "+str(batch_size))

quantileRGB = args.quantileRGB
print("quantileRGB: "+str(quantileRGB))

seed = args.seed
print("seed: "+str(seed))

threads = args.threads
print("threads: "+str(threads))

extraSize = args.extraSize
print("extraSize: "+str(extraSize))

model = args.model
print("model: "+str(model))

modelName = args.modelName
print("modelName: "+str(modelName))

ClusterPredictionMode = args.ClusterPredictionMode
print("ClusterPredictionMode: "+str(ClusterPredictionMode))

cluster_num = args.cluster_num
print("cluster_num: "+str(cluster_num))



sampleNames = args.sampleNames.split(',')
print("sampleName: "+str(sampleNames))



geneSymbols = args.geneSymbols.split(',')
print(geneSymbols)



size = 224
print("size: "+str(size))

mean = (0.485, 0.456, 0.406)
print("mean: "+str(mean))

std = (0.229, 0.224, 0.225)
print("std: "+str(std))



print("### Set seeds ###")
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.set_num_threads(threads)



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# # make DataSet


print("### load image list ###")
image_list = pd.DataFrame(columns=['Sample','No','pos_x1', 'pos_y1', 'radius', 'ImageFilter', 'image_path', 'mean_RGB'] )

for sample in sampleNames:
    tmp = pd.read_csv(dataDir+"/"+sample+"/CropImage/size_"+str(extraSize)+"/RGB_"+str(quantileRGB)+"/image_list_inter.txt", sep='\t')
    tmp['Sample'] = sample
    image_list = image_list.append(tmp, ignore_index=True)

print("image_list: "+str(image_list.shape))
image_list.head()



image_list = image_list.loc[image_list['ImageFilter'] == 'OK',]

print("image_list: "+str(image_list.shape))
image_list.head()



print("### make dataset ###")
valid_dataset = SpotImageDataset(file_list=image_list.loc[:,'image_path'].tolist(),
                                 label_df=image_list.loc[:,['No']],
                                 transform=ImageTransform(size, mean, std),
                                 phase='valid',
                                 param='none')

print("### check ###")
index = 1
print(valid_dataset.__getitem__(index)[0].size())
print(valid_dataset.__getitem__(index)[1])



print("### make DataLoader ###")
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

print("### make dictionary###")
dataloaders_dict = {"valid": valid_dataloader}

print("### check ###")
batch_iterator = iter(dataloaders_dict["valid"])
inputs, labels = next(batch_iterator)
print(inputs.size())
print(labels.size())






# # Validation


files = os.listdir(outDir+"/model_"+modelName)
best_model = files[0]
best_model






# # make model


#use_pretrained = False
#print("use_pretrained: "+str(use_pretrained))
#net = models.vgg16(pretrained=use_pretrained)
#
#if ClusterPredictionMode:
#    net.classifier[6] = nn.Linear(in_features=4096, out_features=cluster_num)
#else:
#    net.classifier[6] = nn.Linear(in_features=4096, out_features=len(geneSymbols))



use_pretrained = False
print("use_pretrained: "+str(use_pretrained))


if ClusterPredictionMode:
    num_features = cluster_num
else:
    num_features = len(geneSymbols)
    

if model == "VGG16":
    # load VGG16 model
    net = torchvision.models.vgg16(pretrained=use_pretrained)

    # change the last unit of VGG16
    net.classifier[6] = nn.Linear(in_features=4096, out_features=num_features)

elif model == "DenseNet121":
    # load DenseNet121 model
    net = torchvision.models.densenet121(pretrained=use_pretrained)

    # change the last unit of VGG16
    net.classifier = nn.Linear(in_features=1024, out_features=num_features)



print("### load the best model ###")
net.load_state_dict(torch.load(outDir+"/model_"+modelName+"/"+best_model, map_location={'cuda:0': 'cpu'}))



print("### Predict validation set ###")
net.eval()

valid_preds = np.array([[]])
valid_labels = np.array([[]])

phase = 'valid'
check_first = True

for inputs, labels in tqdm(dataloaders_dict[phase]):

    with torch.set_grad_enabled(phase == 'train'):
        outputs = net(inputs)

        if ClusterPredictionMode:
            _, preds = torch.max(outputs, 1)  # predict labels
            valid_preds = np.append(valid_preds, preds.clone().numpy())
            valid_labels = np.append(valid_labels, labels[:,0].data.clone().numpy())

        else:
            if check_first:
                valid_preds = outputs.clone().numpy()
                valid_labels = labels.clone().numpy()
                check_first = False
            else:
                valid_preds = np.concatenate([valid_preds, outputs.clone().numpy()])
                valid_labels = np.concatenate([valid_labels, labels.clone().numpy()])


valid_preds_df = pd.DataFrame(valid_preds)
valid_preds_df.columns = [s+"_pred" for s in geneSymbols]

print("valid_preds_df: "+str(valid_preds_df.shape))
valid_preds_df.head()



image_list_pred = image_list.copy()
image_list_pred = image_list_pred.reset_index()
image_list_pred = pd.concat([image_list_pred, valid_preds_df], axis=1)
image_list_pred = image_list_pred.drop('index', axis=1)

image_list_pred.to_csv(outDir+"/image_list_pred.txt", index=False, sep='\t')













