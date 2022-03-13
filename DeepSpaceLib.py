#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import pickle
import time
import sys
import subprocess
import glob
import random
import os
import math
import itertools

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision

from torchvision import models, transforms

from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import KFold

sys.path.append('./')

from BasicLib import ImageTransform
from BasicLib import SpotImageDataset
from BasicLib import plot_loss
from BasicLib import plot_acc
from BasicLib import plot_correlation_scatter_hist
from BasicLib import plot_conf_matrix
from BasicLib import make_classification_report


# ## makeDataList


def makeDataList(rootDir, sampleNames, clusteringMethod, extraSize, geneSymbols, quantileRGB, seed, cross_index, train_equals_test, is_test, rm_cluster):
    print("### load cluster list ###")
    cluster_list = pd.DataFrame(columns=['Sample','Barcode','Cluster'] )

    for sample in sampleNames:
        tmp = pd.read_csv(rootDir+"/"+sample+"/SpaceRanger/analysis/clustering/"+clusteringMethod+"/clusters.csv")
        tmp['Sample'] = sample
        cluster_list = cluster_list.append(tmp, ignore_index=True)

    print("cluster_list: "+str(cluster_list.shape))
    print(cluster_list.head())

    
    print("### remove cluster ###")
    cluster_list['have_cluster'] = [True if i != rm_cluster else False for i in cluster_list['Cluster']]

    cluster_list['Cluster'] = [i if i != rm_cluster else -1 for i in cluster_list['Cluster']]
    cluster_list['Cluster'] = [i if i < rm_cluster else i - 1 for i in cluster_list['Cluster']]

    print("cluster_list: "+str(cluster_list.shape))
    print(cluster_list.head())

    print(cluster_list['Cluster'].unique())


    print("### load tissue_position_list.csv ###")
    tissue_pos = pd.DataFrame(columns=['Sample','Barcode','in_tissue','array_row','array_col','pxl_row_in_fullres','pxl_col_in_fullres','imageID'] )

    for sample in sampleNames:
        tmp = pd.read_csv(rootDir+"/"+sample+"/SpaceRanger/spatial/tissue_positions_list.csv", header=None)

        tmp.columns = ['Barcode','in_tissue','array_row','array_col','pxl_row_in_fullres','pxl_col_in_fullres']
        tmp['imageID'] = tmp.index

        tmp['Sample'] = sample
        tissue_pos = tissue_pos.append(tmp, ignore_index=True)

    print("tissue_pos: "+str(tissue_pos.shape))
    print(tissue_pos.head())

    
    print("### merge cluster file and tissue position file ###")
    cluster_pos_df = pd.merge(cluster_list, tissue_pos, how='inner', on=['Sample','Barcode'])

    cluster_pos_df['image_path'] = [rootDir+"/"+sample+"/CropImage/size_"+str(extraSize)+"/spot_images/spot_image_"+str(s_id).zfill(4)+".tif" for sample,s_id in zip(cluster_pos_df['Sample'].tolist(), cluster_pos_df['imageID'].tolist())]

    cluster_pos_df = cluster_pos_df.sort_values('imageID')

    cluster_pos_df.index = cluster_pos_df['imageID'].tolist()

    print("cluster_pos_df: "+str(cluster_pos_df.shape))
    print(cluster_pos_df.head())

    
    print("### load expression metrix ###")
    exp_mat = pd.DataFrame(columns=['Sample','Barcode']+geneSymbols)

    for sample in sampleNames:
        tmp = pd.read_csv(rootDir+"/"+sample+"/NormUMI/exp_mat_fil_SCT_log10.txt", sep='\t')

        tmp = tmp.T
        tmp.columns = tmp.iloc[0,:].tolist()
        tmp = tmp.drop("symbol", axis=0)
        tmp = tmp.loc[:,geneSymbols]

        tmp['Barcode'] = tmp.index
        tmp['Barcode'] = tmp['Barcode'].str.replace('.','-')
        tmp['Sample'] = sample

        exp_mat = exp_mat.append(tmp, ignore_index=True)
       

    print("exp_mat: "+str(exp_mat.shape))
    print(exp_mat.head())

    
#    print("### Min-Max scaling ###")
#    for i in range(2,exp_mat.shape[1]):
#        exp_mat.iloc[:,i] = minmax_scale(exp_mat.iloc[:,i].tolist())

    print("### Min-Max scaling ###")
    exp_mat_np = exp_mat.iloc[:,range(2,exp_mat.shape[1])].to_numpy()
    
    for i in range(exp_mat.shape[1]-2):
        exp_mat_np[:,i] = minmax_scale(exp_mat_np[:,i].tolist())
    
    exp_mat_np = pd.DataFrame(exp_mat_np)
    exp_mat_np = exp_mat_np.astype('float64')
    exp_mat_np.columns = exp_mat.iloc[:,range(2,exp_mat.shape[1])].columns

    exp_mat = pd.concat([exp_mat.iloc[:,0:2],exp_mat_np], axis=1)

    print(exp_mat)

        
    exp_mat['have_exp'] = True
        
    print("exp_mat: "+str(exp_mat.shape))
    print(exp_mat.head())

    
    print("### merge cluster file and expression metrix ###")
    cluster_pos_df = pd.merge(cluster_pos_df, exp_mat, how='left', on=['Sample','Barcode'])

    print("cluster_pos_df: "+str(cluster_pos_df.shape))
    print(cluster_pos_df.head())

    
    print("### Filter image ###")
    cluster_pos_filter_df = pd.DataFrame(columns=['Sample','Barcode','ImageFilter'] )

    for sample in sampleNames:
        tmp = pd.read_csv(rootDir+"/"+sample+"/CropImage/size_"+str(extraSize)+"/RGB_"+str(quantileRGB)+"/cluster_position_filter.txt", sep='\t')
        tmp.index = tmp['imageID'].tolist()

        tmp = tmp.query('ImageFilter == "OK"').copy()

        tmp = tmp.loc[:,['Barcode','ImageFilter']]

        tmp['Sample'] = sample
        cluster_pos_filter_df = cluster_pos_filter_df.append(tmp, ignore_index=True)

    print("cluster_pos_filter_df: "+str(cluster_pos_filter_df.shape))
    print(cluster_pos_filter_df.head())

    
    print("### Use only OK images ###")
    data_list_df = pd.merge(cluster_pos_df, cluster_pos_filter_df, how='inner', on=['Sample','Barcode'])

    print("data_list_df: "+str(data_list_df.shape))
    print(data_list_df.head())


    
    ### 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    count = 0
    for train_index, test_index in kf.split(data_list_df.index, data_list_df.index):
        if count == cross_index:
            data_list_df.loc[data_list_df.index[train_index], 'phase'] = 'train'
            data_list_df.loc[data_list_df.index[test_index], 'phase'] = 'valid'

        count += 1
        
    
    if is_test:
        data_list_df['phase'] = 'valid'

    else:
        data_list_df_train = data_list_df.query('phase == "train"').copy()
        data_list_df_test = data_list_df.query('phase == "valid"').copy()

        data_list_df_train_index = data_list_df_train.query('have_exp != True').index
        data_list_df_train = data_list_df_train.drop(data_list_df_train_index)

        data_list_df_train_index = data_list_df_train.query('have_cluster != True').index
        data_list_df_train = data_list_df_train.drop(data_list_df_train_index)

        if train_equals_test:
            data_list_df_test['phase'] = "test"

            count = 0
            for train_index, test_index in kf.split(data_list_df_train.index, data_list_df_train.index):
                if count == cross_index:
                    data_list_df_train.loc[data_list_df_train.index[train_index], 'phase'] = 'train'
                    data_list_df_train.loc[data_list_df_train.index[test_index], 'phase'] = 'valid'

                count += 1

        else: # Just for training data
            data_list_df_test_index = data_list_df_test.query('have_exp != True').index
            data_list_df_test = data_list_df_test.drop(data_list_df_test_index)

            data_list_df_train_index = data_list_df_train.query('have_cluster != True').index
            data_list_df_train = data_list_df_train.drop(data_list_df_train_index)

                
        data_list_df = pd.concat([data_list_df_train, data_list_df_test])
    

    data_list_df = data_list_df.sort_values(['Sample','imageID'])
    data_list_df = data_list_df.reset_index(drop=True)

    print("data_list_df: "+str(data_list_df.shape))
    print(data_list_df.head())
    
    return data_list_df



### verification ###
if __name__ == '__main__':
    data_list_teacher = makeDataList(rootDir="/home/"+os.environ['USER']+"/DeepSpaCE/data",
                                     sampleNames=['Human_Breast_Cancer_Block_A_Section_1'],
                                     clusteringMethod="graphclust",
                                     extraSize=150,
                                     geneSymbols=['MKI67', 'ESR1', 'ERBB2'],
                                     quantileRGB=80,
                                     seed=0,
                                     cross_index=0,
                                     train_equals_test=True,
                                     is_test=False,
                                     rm_cluster=8)
    
    print("data_list_teacher: "+str(data_list_teacher.shape))
    data_list_teacher.head()

    data_list_teacher.to_csv("./temp.txt", index=False, sep='\t', float_format='%.6f')

    data_list_teacher_tmp = data_list_teacher.copy()
    data_list_teacher = data_list_teacher_tmp.query('phase != "test"').copy()
    data_list_test = data_list_teacher_tmp.query('phase == "test"').copy()
    data_list_test['phase'] = 'valid'






# ## makeTrainDataloader


def makeTrainDataloader(rootDir, data_list_df, geneSymbols, size, mean, std, augmentation, batch_size, ClusterPredictionMode):

    data_list_df = data_list_df.reset_index(drop=True)
    
    print("### make dataset ###")
    if ClusterPredictionMode:
        train_dataset = SpotImageDataset(file_list=data_list_df.loc[data_list_df['phase'] == 'train', 'image_path'].tolist(),
                                         label_df=data_list_df.loc[data_list_df['phase'] == 'train', ['Cluster']] - 1,
                                         transform=ImageTransform(size, mean, std),
                                         phase='train',
                                         param=augmentation)

        valid_dataset = SpotImageDataset(file_list=data_list_df.loc[data_list_df['phase'] == 'valid', 'image_path'].tolist(),
                                         label_df=data_list_df.loc[data_list_df['phase'] == 'valid', ['Cluster']] - 1,
                                         transform=ImageTransform(size, mean, std),
                                         phase='valid',
                                         param=augmentation)
    else:
        train_dataset = SpotImageDataset(file_list=data_list_df.loc[data_list_df['phase'] == 'train', 'image_path'].tolist(),
                                         label_df=data_list_df.loc[data_list_df['phase'] == 'train', geneSymbols],
                                         transform=ImageTransform(size, mean, std),
                                         phase='train',
                                         param=augmentation)

        valid_dataset = SpotImageDataset(file_list=data_list_df.loc[data_list_df['phase'] == 'valid', 'image_path'].tolist(),
                                         label_df=data_list_df.loc[data_list_df['phase'] == 'valid', geneSymbols],
                                         transform=ImageTransform(size, mean, std),
                                         phase='valid',
                                         param=augmentation)

    print("### check ###")
    index = 1
    print(train_dataset.__getitem__(index)[0].size())
    print(train_dataset.__getitem__(index)[1])

    
    print("### make DataLoader ###")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=1, pin_memory=False)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=False)

    
    print("### make dictionary###")
    dataloaders_dict = {"train": train_dataloader, "valid": valid_dataloader}

    
    print("### check2 ###")
    batch_iterator = print(dataloaders_dict)
    batch_iterator = iter(dataloaders_dict["train"])
    inputs, labels = next(batch_iterator)
    print(inputs.size())
    print(labels.size())
    
    return dataloaders_dict



### verification ###
if __name__ == '__main__':
    dataloaders_dict = makeTrainDataloader(rootDir="/home/"+os.environ['USER']+"/DeepSpaCE/out",
                                           data_list_df=data_list_teacher,
                                           geneSymbols=['MKI67', 'ESR1', 'ERBB2'],
                                           size=224,
                                           mean=(0.485, 0.456, 0.406),
                                           std=(0.229, 0.224, 0.225),
                                           augmentation="flip",
                                           batch_size=128,
                                           ClusterPredictionMode=False)






# # make network model


def make_model(use_pretrained, num_features, transfer, model):
    
    print("use_pretrained: "+str(use_pretrained))

    if model == "VGG16":
        # load VGG16 model
        net = torchvision.models.vgg16(pretrained=use_pretrained)

        # change the last unit of VGG16
        net.classifier[6] = nn.Linear(in_features=4096, out_features=num_features)

    elif model == "DenseNet121":
        # load DenseNet121 model
        net = torchvision.models.densenet121(pretrained=use_pretrained)

        # change the last unit of DenseNet121
        net.classifier = nn.Linear(in_features=1024, out_features=num_features)


    # train mode
    net.train()

    params_to_update = []

    if transfer == False:
        print("### Full learning ###")
        for name, param in net.named_parameters():
            param.requires_grad = True
            params_to_update.append(param)

    else:
        print("### Transfer learning ###")
        if model != "VGG16":
            print("Transfer leraning is available only for VGG16")
            sys.exit()
        
        
        # parameters for training
        update_param_names = ["classifier.6.weight", "classifier.6.bias"]

        for name, param in net.named_parameters():
            if name in update_param_names:
                param.requires_grad = True
                params_to_update.append(param)
                print(name)
            else:
                param.requires_grad = False

    print("-----------")
    print(params_to_update)
    print("-----------")

    return net, params_to_update



### verification ###
if __name__ == '__main__':
    print("### make model ###")
    net, params_to_update = make_model(use_pretrained=True,
                                       num_features=len(data_list_teacher['Cluster'].unique()),
                                       transfer=False,
                                       model="VGG16")






# # loss function


def loss_function(outputs, labels):
    
    criterion = nn.SmoothL1Loss()
    num_gene = outputs.shape[1]

    loss = 0
    
    for i in range(num_gene):
        loss += criterion(outputs[:,i], labels[:,i]) / num_gene

    return loss



def calc_cor(outputs, labels):

    num_gene = outputs.shape[1]

    corR = []

    for i in range(num_gene):
        corR.append(np.corrcoef(outputs[:,i].to('cpu').detach().numpy(), labels[:,i].to('cpu').detach().numpy())[0,1])

    corR = np.array(corR)

    corR[np.isnan(corR)] = 0.0
        
    print("corR: "+str(corR))

    return np.mean(corR)






# ## train_model


def run_train(outDir, net, dataloaders_dict, optimizer, num_epochs, device, early_stop_max, name, ClusterPredictionMode):

    ### set multi GPU
    if str(device) != 'cpu':
        net.to(device)
        
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)

    if ClusterPredictionMode:
        res_df = pd.DataFrame(columns=['train_loss','valid_loss','train_acc','valid_acc'])
    else:
        res_df = pd.DataFrame(columns=['train_loss','valid_loss','train_cor','valid_cor'])
    
    time_df = pd.DataFrame(columns=['time'])

    valid_loss_prev = 1e+100; valid_loss_best = 1e+100; valid_cor_best = -1e+100;
    early_stop_count = 0
    
    # epoch roop
    for epoch in range(num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-------------')
        start = time.time()

        train_loss = 0; valid_loss = 0;
        train_acc_or_cor = 0; valid_acc_or_cor = 0;

        # train and valid roop per epoch
        for phase in ['train', 'valid']:
            net.train() if phase == 'train' else net.eval() # train or eval mode

            epoch_loss = 0.0  # sum of loss
            epoch_corrects_or_cor = 0  # sum of corrects or correlation

            # skip trainging if epoch == 0
            if (epoch == 0) and (phase == 'train'): continue
                
            # extract minibatch from dataloader
            for inputs, labels in tqdm(dataloaders_dict[phase]):
                # if GPU is avalable
                if str(device) != 'cpu':
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                # initialize optimizer
                optimizer.zero_grad()

                # forward calculation
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    
                    # calculate loss
                    if ClusterPredictionMode:
                        criterion = nn.CrossEntropyLoss()
                        loss = criterion(outputs, labels[:,0])
                        _, preds = torch.max(outputs, 1)  # predict labels
                    else:
                        loss = loss_function(outputs, labels)
                    
                    # backpropagation if training phase
                    if phase == 'train': 
                        loss.backward()
                        optimizer.step()

                    # update sum of loss
                    epoch_loss += loss.item() * inputs.size(0)
                    
                    # update sum of corrects
                    if ClusterPredictionMode:
                        epoch_corrects_or_cor += torch.sum(preds == labels[:,0])
                    # update sum of correlation
                    else:
                        epoch_corrects_or_cor += calc_cor(outputs, labels)

            # print loss
            if ClusterPredictionMode:
                epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
                epoch_corrects_or_cor = epoch_corrects_or_cor.double() / len(dataloaders_dict[phase].dataset)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_corrects_or_cor))
            else:
                epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # train loss or valid loss
            if phase == 'train':
                train_acc_or_cor = float(epoch_corrects_or_cor)
                train_loss = epoch_loss
            else:
                valid_acc_or_cor = float(epoch_corrects_or_cor)
                valid_loss = epoch_loss
                
        ### append loss to DataFrame
        res_df = res_df.append([pd.Series([train_loss,valid_loss,train_acc_or_cor,valid_acc_or_cor],index=res_df.columns)], ignore_index=True)
        
        ### save training_loss.txt
        res_df.to_csv(outDir+"/training_loss_"+name+".txt", sep='\t', float_format='%.6f')

        ### save best model
        save_best = False
        
        if ClusterPredictionMode:
            if valid_loss_best > valid_loss:
                valid_loss_best = valid_loss
                save_best = True
        else:
            if valid_cor_best < valid_acc_or_cor:
                valid_cor_best = valid_acc_or_cor
                save_best = True

        if save_best:
            if str(device) != 'cpu' and torch.cuda.device_count() > 1:
                subprocess.call(['rm','-r',outDir+'/model_'+name+'/'])
                subprocess.call(['mkdir',outDir+'/model_'+name+'/'])
                torch.save(net.module.state_dict(), outDir+"/model_"+name+"/model_"+str(epoch)+".pth")
            else:
                subprocess.call(['rm','-r',outDir+'/model_'+name+'/'])
                subprocess.call(['mkdir',outDir+'/model_'+name+'/'])
                torch.save(net.state_dict(), outDir+"/model_"+name+"/model_"+str(epoch)+".pth")

        ### early stopping
        if valid_loss_prev > valid_loss:
            early_stop_count = 0
        else:
            early_stop_count += 1

        print("early_stop_count: "+str(early_stop_count))

        if early_stop_count == early_stop_max: break

        valid_loss_prev = valid_loss

        ## elapsed time
        elapsed_time = time.time() - start
        print ("elapsed_time:{:.2f}".format(elapsed_time) + "[sec]")

        ### append loss to DataFrame
        time_df = time_df.append([pd.Series([elapsed_time],index=time_df.columns)], ignore_index=True)
        ### save training_loss.txt
        time_df.to_csv(outDir+"/time_"+name+".txt", sep='\t', float_format='%.6f')



### verification ###
if __name__ == '__main__':
    optimizer = optim.Adam(params=params_to_update, lr=1e-4, weight_decay=1e-4)
    
    print("### run train ###")
    run_train(outDir="/home/"+os.environ['USER']+"/DeepSpaCE/",
              net=net,
              dataloaders_dict=dataloaders_dict,
              optimizer=optimizer,
              num_epochs=3,
              device='cpu',
              early_stop_max=10,
              name='teacher',
              ClusterPredictionMode=False)






# # Validation (Test set)


def makeTestDataloader(rootDir, data_list_df, model, geneSymbols, size, mean, std, augmentation, batch_size, ClusterPredictionMode):
    
    data_list_df = data_list_df.reset_index(drop=True)
    
    print("### make dataset ###")
    if ClusterPredictionMode:
        test_dataset = SpotImageDataset(file_list=data_list_df.loc[data_list_df['phase'] == 'valid', 'image_path'].tolist(),
                                        label_df=data_list_df.loc[data_list_df['phase'] == 'valid', ['Cluster']] - 1,
                                        transform=ImageTransform(size, mean, std),
                                        phase='valid',
                                        param=augmentation)
    else:
        test_dataset = SpotImageDataset(file_list=data_list_df.loc[data_list_df['phase'] == 'valid', 'image_path'].tolist(),
                                        label_df=data_list_df.loc[data_list_df['phase'] == 'valid', geneSymbols],
                                        transform=ImageTransform(size, mean, std),
                                        phase='valid',
                                        param=augmentation)
        
    print("### check ###")
    index = 1
    print(test_dataset.__getitem__(index)[0].size())
    print(test_dataset.__getitem__(index)[1])

    
    print("### make DataLoader ###")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=False)

    print("### make dictionary ###")
    dataloaders_dict_test = {"valid": test_dataloader}


    print("### check ###")
    batch_iterator = iter(dataloaders_dict_test["valid"])
    inputs, labels = next(batch_iterator)
    print(inputs.shape)
    print(labels.shape)
    
    return dataloaders_dict_test



### verification ###
if __name__ == '__main__':
    dataloaders_dict_test = makeTestDataloader(rootDir="/home/"+os.environ['USER']+"/DeepSpaCE/",
                                               data_list_df=data_list_test,
                                               model="VGG16",
                                               geneSymbols=['MKI67', 'ESR1', 'ERBB2'],
                                               size=224,
                                               mean=(0.485, 0.456, 0.406),
                                               std=(0.229, 0.224, 0.225),
                                               augmentation="none",
                                               batch_size=128,
                                               ClusterPredictionMode=False)



def run_test(outDir, data_list_df, dataloaders_dict, model, device, geneSymbols, num_features, ClusterPredictionMode, name):
    print("### load loss_acc_df ###")
    loss_acc_df = pd.read_csv(outDir+"/training_loss_"+name+".txt", sep='\t')

    loss_acc_df = loss_acc_df.rename(columns={'Unnamed: 0':'no'})

    print("loss_acc_df: "+str(loss_acc_df.shape))
    loss_acc_df.head()

    
    print("### Best model ###")
    best_files = glob.glob(outDir+"/model_"+name+"/model_*")

    print(best_files)
    best_model = best_files[0]
        
    print("best_model: "+str(best_model))

    with open(outDir+"/best_model_"+name+".txt", mode='w') as f:
        f.write(str(best_model))
        
        
    loss_acc_df.loc[0,'train_loss'] = np.nan

    print("### Plot Loss ###")
    plot_loss(outDir, loss_acc_df, name)

    if ClusterPredictionMode:
        loss_acc_df.loc[0,'train_acc'] = np.nan
        
        print("### Plot Acc ###")
        plot_acc(outDir, loss_acc_df, name)

    
    if ClusterPredictionMode:
        net, params_to_update = make_model(use_pretrained=False,
                                           num_features=num_features,
                                           transfer=False,
                                           model=model)
    else:
        net, params_to_update = make_model(use_pretrained=False,
                                           num_features=len(geneSymbols),
                                           transfer=False,
                                           model=model)

    print("### load the best model ###")
    if str(device) != 'cpu':
        net.load_state_dict(torch.load(best_model))
    else:
        net.load_state_dict(torch.load(best_model, map_location={'cuda:0': 'cpu'}))

        
    print("### Predict test set ###")
    net.eval()   # eval mode

    valid_preds = np.array([[]])
    valid_labels = np.array([[]])

    phase = 'valid'
    check_first = True

    # extract minibatch from dataloader
    for inputs, labels in tqdm(dataloaders_dict[phase]):

        # forward calculation
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
    
    print(valid_preds_df)
    
    if ClusterPredictionMode:
        valid_preds_df.columns = ["Cluster_pred"]
    else:
        valid_preds_df.columns = [s+"_pred" for s in geneSymbols]

    print("valid_preds_df: "+str(valid_preds_df.shape))
    valid_preds_df.head()

    data_list_df = data_list_df.reset_index(drop=True)
    data_list_df = pd.concat([data_list_df, valid_preds_df], axis=1)

    
    if ClusterPredictionMode:
        data_list_df['Cluster_pred'] = [int(i)+1 for i in valid_preds.tolist()]        
  
        print("### Plot confusion matrix ###")
        idx = [int(i+1) != -1 for i in valid_labels]
        valid_labels = list(itertools.compress(valid_labels, idx))
        valid_preds = list(itertools.compress(valid_preds, idx))
        
        print("valid_labels: "+str(valid_labels))
        print("valid_preds: "+str(valid_preds))
        print(["Cluster"+str(i+1) for i in range(num_features)])
        
        plot_conf_matrix(outDir, valid_labels, valid_preds, ["Cluster"+str(i+1) for i in range(num_features)], name)

        print("### make classification_report ###")
        make_classification_report(outDir, valid_labels, valid_preds, ["Cluster"+str(i+1) for i in range(num_features)], name)

    else:
        print("### plot_correlation_scatter_hist ###")
        plot_correlation_scatter_hist(outDir, valid_labels, valid_preds, geneSymbols, scatter=False, name=name)

    return data_list_df, net







### verification ###
if __name__ == '__main__':
    data_list_df, net = run_test(outDir="/home/"+os.environ['USER']+"/DeepSpaCE/",
                                 data_list_df=data_list_test,
                                 dataloaders_dict=dataloaders_dict_test,
                                 model="VGG16",
                                 device="cpu",
                                 geneSymbols=['MKI67', 'ESR1', 'ERBB2'],
                                 num_features=3,
                                 ClusterPredictionMode=False,
                                 name="teacher")
    
    print("data_list_df: "+str(data_list_df.shape))
    data_list_df.head()


# ## Semi-supervised


def makeDataListSemi(rootDir, sampleNames, semiType, ImageSet, semiName):
    image_list = pd.DataFrame(columns=['ImageSet', 'sample_No', 'sample_id', 'image_path_old', 'image_path'])

    if semiType == "Visium":
        for sampleName in sampleNames:
            for i_imageSet in ImageSet:
                image_l = pd.read_csv(rootDir+"/"+semiType+"/ImageSet/"+sampleName+"/ImageSet_"+str(i_imageSet)+"/image_list.txt", sep='\t')
                image_list = image_list.append(image_l, ignore_index=True)
        
        idx_rnd = random.sample(image_list.index.tolist(), 2000*len(ImageSet))
        image_list = image_list.loc[idx_rnd,:]

    else:
        for i_imageSet in ImageSet:
            image_l = pd.read_csv(rootDir+"/"+semiType+"/ImageSet/ImageSet_"+str(i_imageSet)+"/image_list.txt", sep='\t')
            image_list = image_list.append(image_l, ignore_index=True)


    image_list['Sample'] = semiType
    image_list['imageID'] = image_list['sample_No']
    image_list['Barcode'] = image_list['ImageSet']
    image_list['pxl_row_in_fullres'] = ""
    image_list['pxl_col_in_fullres'] = ""
    image_list['Cluster'] = ""
    image_list['in_tissue'] = ""
    image_list['array_row'] = ""
    image_list['array_col'] = ""
    image_list['phase'] = "valid"
    image_list['semiName'] = semiName
    
    image_list = image_list.loc[:,['Sample','Barcode','Cluster','in_tissue','array_row','array_col','pxl_row_in_fullres','pxl_col_in_fullres','imageID','image_path','phase','semiName']]
    image_list = image_list.reset_index(drop=True)

    return image_list



### verification ###
if __name__ == '__main__':
    data_list_semi = makeDataListSemi(rootDir="/home/"+os.environ['USER']+"/DeepSpaCE/",
                                      sampleNames=['Human_Breast_Cancer_Block_A_Section_1','Human_Breast_Cancer_Block_A_Section_2'],
                                      semiType="Visium",
                                      ImageSet=['0','1'],
                                      semiName="semi")
    
    #data_list_semi = data_list_semi.loc[1:10,]

    print(data_list_semi.shape)
    print(data_list_semi)



def makeSemiDataloader(rootDir, data_list_df, size, mean, std, augmentation, batch_size):
    
    data_list_df = data_list_df.reset_index(drop=True)
    
    print("### make dataset ###")
    test_dataset = SpotImageDataset(file_list=data_list_df.loc[:,'image_path'].tolist(),
                                    label_df=data_list_df.loc[:,['imageID']],
                                    transform=ImageTransform(size, mean, std),
                                    phase='valid',
                                    param=augmentation)
    
    print("### check ###")
    index = 1
    print(test_dataset.__getitem__(index)[0].size())
    print(test_dataset.__getitem__(index)[1])

    
    print("### make DataLoader ###")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=False)

    print("### make dictionary ###")
    dataloaders_dict_test = {"valid": test_dataloader}

    print("### check ###")
    batch_iterator = iter(dataloaders_dict_test["valid"])
    inputs, labels = next(batch_iterator)
    print(inputs.shape)
    print(labels.shape)
    
    return dataloaders_dict_test



### verification ###
if __name__ == '__main__':
    dataloaders_dict_semi = makeSemiDataloader(rootDir="/home/"+os.environ['USER']+"/Visium/",
                                               data_list_df=data_list_semi,
                                               size=224,
                                               mean=(0.485, 0.456, 0.406),
                                               std=(0.229, 0.224, 0.225),
                                               augmentation="none",
                                               batch_size=128)



def predict_semi_label(net, data_list_semi, dataloaders_dict_semi, geneSymbols, ClusterPredictionMode):
    print("### Predict validation set ###")
    net.eval()   # eval mode

    valid_preds = np.array([[]])
    valid_labels = np.array([[]])

    phase = 'valid'
    check_first = True

    # extract minibatch from dataloader
    for inputs, labels in tqdm(dataloaders_dict_semi[phase]):

        # forward calculation
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
    
    if ClusterPredictionMode:
        valid_preds_df.columns = ["Cluster"]
    else:
        valid_preds_df.columns = geneSymbols

    print("valid_preds_df: "+str(valid_preds_df.shape))
    valid_preds_df.head()

    ### add predicted values
    if ClusterPredictionMode:
        data_list_semi['Cluster'] = [int(i)+1 for i in valid_preds.tolist()]
    else:
        for g in geneSymbols:
            data_list_semi[g] = valid_preds_df[g]

    data_list_semi['phase'] = 'train'

    data_list_semi = data_list_semi.reset_index(drop=True)

    print("data_list_semi: "+str(data_list_semi.shape))
    data_list_semi.head()

    return data_list_semi



### verification ###
if __name__ == '__main__':
    data_list_semi = predict_semi_label(net=net,
                                        data_list_semi=data_list_semi,
                                        dataloaders_dict_semi=dataloaders_dict_semi,
                                        geneSymbols=["COL1A1"],
                                        ClusterPredictionMode=True)









