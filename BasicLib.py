#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import math

import torch.utils.data as data

import albumentations as albu
from albumentations.pytorch import ToTensor

#from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from mlxtend.plotting import plot_confusion_matrix
from mlxtend.evaluate import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_squared_error

import cv2

import torch







class ImageTransform():

    def __init__(self, resize, mean, std):
        self.data_transform = {
            'init': albu.Compose([
                albu.Resize(resize, resize)
            ]),
            'end': albu.Compose([
                albu.Normalize(mean,std),
                ToTensor()
            ]),
            'flip': albu.Compose([
                albu.RandomRotate90(p=0.5),
                albu.Flip(p=0.5),
                albu.Transpose(p=0.5)
            ], p=1.0),
            'noise': albu.Compose([
                albu.OneOf([
                    albu.IAAAdditiveGaussianNoise(p=1.0),
                    albu.GaussNoise(p=1.0)
                ], p=1.0),
            ], p=1.0),
            'blur': albu.Compose([
                albu.OneOf([
                    albu.MotionBlur(p=1.0),
                    albu.MedianBlur(p=1.0),
                    albu.Blur(p=1.0)
                ], p=1.0),
            ], p=1.0),
            'dist': albu.Compose([
                albu.OneOf([
                    albu.OpticalDistortion(p=1.0),
                    albu.GridDistortion(p=1.0),
                    albu.IAAPiecewiseAffine(p=1.0),
                    albu.ShiftScaleRotate(p=1.0)
                ], p=1.0),
            ], p=1.0),
            'contrast': albu.Compose([
                albu.RandomContrast(p=0.5),
                albu.RandomGamma(p=0.5),
                albu.RandomBrightness(p=0.5)
            ], p=1.0),
            'color': albu.Compose([
                albu.HueSaturationValue(p=0.5),
                albu.ChannelShuffle(p=0.5),
                albu.RGBShift(p=0.5)
            ], p=1.0),
            'crop': albu.Compose([
                albu.RandomResizedCrop(height=resize, width=resize, scale=(0.5, 1.0), p=0.5),
            ], p=1.0),
            'random': albu.Compose([
                albu.OneOf([
                    albu.OneOf([
                        albu.IAAAdditiveGaussianNoise(p=1.0),
                        albu.GaussNoise(p=1.0)
                    ], p=1.0),
                    albu.OneOf([
                        albu.MotionBlur(p=1.0),
                        albu.MedianBlur(p=1.0),
                        albu.Blur(p=1.0)
                    ], p=1.0),
                    albu.OneOf([
                        albu.OpticalDistortion(p=1.0),
                        albu.GridDistortion(p=1.0),
                        albu.IAAPiecewiseAffine(p=1.0),
                        albu.ShiftScaleRotate(p=1.0)
                    ], p=1.0),
                ], p=1.0),            
            ], p=1.0),
            'valid': albu.Compose([
                albu.Resize(resize, resize),
                albu.CenterCrop(resize, resize),
                albu.Normalize(mean, std),
                ToTensor()
            ])
        }

    def __call__(self, img, phase='train', param=''):
        if phase == 'train':
            img = self.data_transform['init'](image=img)

            if param != 'none':
                param = param.split(',')
                
                for para in param:
                    img = self.data_transform[para](image=img['image'])

            img = self.data_transform['end'](image=img['image'])
        elif phase == 'valid':
            img = self.data_transform['valid'](image=img)
        
        return img['image']



class SpotImageDataset(data.Dataset):
    def __init__(self, file_list, label_df, transform=None, phase='train', param=''):
        self.file_list = file_list
        self.label_df = label_df
        self.transform = transform
        self.phase = phase
        self.param = param

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # load image of index
        img = cv2.imread(self.file_list[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Image transform
        img_transformed = self.transform(img, self.phase, self.param)

        # label
        label = self.label_df.iloc[index,:].tolist()
        
        return img_transformed, torch.tensor(label)







# Plot Loss
def plot_loss(outDir, loss_acc_df, name):
    fig = plt.figure()

    plt.scatter(loss_acc_df.index, loss_acc_df['train_loss'], label="train")
    plt.scatter(loss_acc_df.index, loss_acc_df['valid_loss'], label="valid")
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.tick_params(labelsize=12)

    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.3, fontsize=15)

    plt.yscale('log')

    plt.xticks(np.arange(0, math.ceil((loss_acc_df.shape[0]-1)/50) * 50 + 1, math.ceil((loss_acc_df.shape[0]-1)/50) * 10))

    plt.subplots_adjust(left=0.2, right=0.9, bottom=0.16, top=0.9)

    fig.savefig(outDir+"/loss_plot_"+name+".png")

    plt.close()



# Plot Acc
def plot_acc(outDir, loss_acc_df, name):
    fig = plt.figure()

    plt.scatter(loss_acc_df.index, loss_acc_df['train_acc'], label="train")
    plt.scatter(loss_acc_df.index, loss_acc_df['valid_acc'], label="valid")
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.tick_params(labelsize=12)

    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.3, fontsize=15)

    plt.xticks(np.arange(0, math.ceil((loss_acc_df.shape[0]-1)/50) * 50 + 1, math.ceil((loss_acc_df.shape[0]-1)/50) * 10))

    plt.subplots_adjust(left=0.2, right=0.9, bottom=0.16, top=0.9)

    fig.savefig(outDir+"/acc_plot_"+name+".png")

    plt.close()



### Plot confusion matrix ###
def plot_conf_matrix(outDir, valid_labels, valid_preds, class_names, name):
    
    conf_mat = confusion_matrix(y_target=valid_labels, y_predicted=valid_preds)

    plt.figure()

    plt.rcParams["font.size"] = 15

    fig, ax = plot_confusion_matrix(conf_mat=conf_mat,
                                    colorbar=False,
                                    show_absolute=True,
                                    show_normed=True,
                                    class_names=class_names,
                                    cmap=plt.cm.Blues,
                                    figsize=(10,10))

    plt.xlabel("Predicted label", fontsize=25)
    plt.ylabel("True label", fontsize=25)
    plt.tick_params(labelsize=12)

    plt.ylim(conf_mat.shape[0]-0.5,-0.5)

    plt.subplots_adjust(left=0.2, right=0.9, bottom=0.16, top=0.9)
    plt.savefig(outDir+"/confusion_matrix_plot_"+name+".png")

    #plt.show()
    plt.close()



### make classification_report ###
def make_classification_report(outDir, valid_labels, valid_preds, class_names, name):
    
    report_df = classification_report(valid_labels, valid_preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report_df)
    report_df = report_df.T
    report_df['Cluster'] = report_df.index
    report_df = report_df.loc[:,['Cluster','precision','recall','f1-score','support']]
    report_df['support'] = report_df['support'].astype(np.int64)

    report_df.to_csv(outDir+"/classification_report_"+name+".txt", index=False, sep='\t', float_format='%.6f')

    report_df



def plot_correlation_scatter_hist(outDir, valid_labels, valid_preds, geneSymbols, scatter, name):
    corR = []
    
    corR_df = pd.DataFrame(columns=['geneSymbols','corR','RMSE','RMSE_MinMax'] )

    
    for i in range(len(geneSymbols)):
        
        idx = np.isnan(valid_labels[:,i])
        lab = valid_labels[~idx,i]
        pred = valid_preds[~idx,i]

        corR.append(np.corrcoef(lab, pred)[0,1])
        corR_tmp = np.corrcoef(lab, pred)[0,1]
        rmse = np.sqrt(mean_squared_error(lab, pred))
        rmse_minmax = np.sqrt(mean_squared_error(lab, minmax_scale(pred)))
                              
        corR_df = corR_df.append(pd.Series([geneSymbols[i],corR_tmp,rmse,rmse_minmax], index=['geneSymbols','corR','RMSE','RMSE_MinMax']), ignore_index=True)

                              
    corR_df.to_csv(outDir+"/corR_"+name+".txt", index=False, sep='\t', float_format='%.6f')


    with open(outDir+"/correlation_"+name+".txt", mode='w') as f:
        for i in range(len(geneSymbols)):
            f.write(geneSymbols[i]+"\t"+'{:.3f}'.format(corR[i])+"\n")

    ### scatter plot
    if scatter == True:
        for i in range(len(geneSymbols)):
            fig = plt.figure()
            ax = fig.add_subplot(111)

            idx = np.isnan(valid_labels[:,i])
            lab = valid_labels[~idx,i]
            pred = valid_preds[~idx,i]

            
#            plt.scatter(valid_labels[:,i], valid_preds[:,i], label="test")
            plt.scatter(lab, pred, label="test")
            plt.xlabel("log10(UMIcount+1)", fontsize=20)
            plt.ylabel("Predicted expression", fontsize=20)
            plt.tick_params(labelsize=12)

            plt.subplots_adjust(left=0.16, right=0.9, bottom=0.16, top=0.9)

            plt.text(0.025, 0.9, "r="+'{:.3f}'.format(corR[i]), size = 20, color = "black", transform=ax.transAxes)

            fig.savefig(outDir+"/plot/scatter_plot_"+geneSymbols[i]+"_"+name+".png")

            plt.close()

    ### Hist
    fig = plt.figure()

    plt.hist(corR, bins=20)
    plt.xlim([0,1])

    plt.xlabel("Pearson correlation coefficient", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)

    fig.savefig(outDir+"/correlation_"+name+".png")
    plt.close()

    ### Hist2
    fig = plt.figure()

    plt.hist(corR, bins=20)
    plt.xlim([-1,1])

    plt.xlabel("Pearson correlation coefficient", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)

    fig.savefig(outDir+"/correlation2_"+name+".png")
    plt.close()





