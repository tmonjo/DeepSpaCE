#!/usr/bin/env python
# coding: utf-8


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import argparse
import subprocess
import os

import cv2
from PIL import Image



parser = argparse.ArgumentParser(description='Crop Image')

parser.add_argument('--dataDir', type=str, default='/home/'+os.environ['USER']+'/DeepSpaCE/data',
                   help='Data directory (default: '+'/home/'+os.environ['USER']+'/DeepSpaCE/data'+')')

parser.add_argument('--sampleName', type=str, default='Human_Breast_Cancer_Block_A_Section_1',
                   help='Sample name (default: Human_Breast_Cancer_Block_A_Section_1)')

parser.add_argument('--transposeType', type=int, default=0,
                   help='0: No transpose, 1: Rotate90CC, 2: Rotate90CC+Flip (default: 0)')

parser.add_argument('--radiusPixel', type=int, default=40,
                   help='Radius [pixel] (default: 40)')

parser.add_argument('--extraSize', type=int, default=150,
                   help='Extra image size (default: 150)')

parser.add_argument('--quantileRGB', type=int, default=80,
                   help='Threshold of quantile RGB (default: 80)')

parser.add_argument('--threads', type=int, default=1,
                    help='Number of CPU threads (default: 1)')

args = parser.parse_args()



print(args)



dataDir = args.dataDir
print("dataDir: "+str(dataDir))

sampleName = args.sampleName
print("sampleName: "+sampleName)

transposeType = args.transposeType
print("transposeType: "+str(transposeType))

radiusPixel = args.radiusPixel
print("radiusPixel: "+str(radiusPixel))

extraSize = args.extraSize
print("extraSize: "+str(extraSize))

quantileRGB = args.quantileRGB
print("quantileRGB: "+str(quantileRGB))

threads = args.threads
print("threads: "+str(threads))



os.environ["OMP_NUM_THREADS"] = str(threads)



def pil2cv(image):
    ''' PIL -> OpenCV '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


# # Crop Image


dirName = dataDir+"/"+sampleName

subprocess.call(['mkdir','-p',dirName+"/CropImage/size_"+str(extraSize)+"/spot_images"])



### load Image
Image.MAX_IMAGE_PIXELS = 1000000000
I = Image.open(dirName+"/SpaceRanger/image.tif")

I = pil2cv(I)

print(I.shape)



## Transpose
if transposeType == 1:
    I = cv2.rotate(I, cv2.ROTATE_90_COUNTERCLOCKWISE)

elif transposeType == 2:
    I = cv2.rotate(I, cv2.ROTATE_90_COUNTERCLOCKWISE)
    I = np.fliplr(I)

cv2.imwrite(dirName+"/CropImage/size_"+str(extraSize)+"/transpose_image.tif", I)

### load Image
I = cv2.imread(dirName+"/CropImage/size_"+str(extraSize)+"/transpose_image.tif")

print(I.shape)



# keep original image
I_org = I.copy()



### load tissue_position
tissue_pos = pd.read_csv(dirName+"/SpaceRanger/spatial/tissue_positions_list.csv", header=None)
tissue_pos.columns = ['Barcode','in_tissue','array_row','array_col','pxl_row_in_fullres','pxl_col_in_fullres']

tissue_pos['imageID'] = tissue_pos.index

print("tissue_pos: "+str(tissue_pos.shape))
print(tissue_pos.head())



## calc radius
radius = math.ceil(radiusPixel * (1 + extraSize/100))
print("radius: "+str(radius))



### draw line and text
for i in range(4992):
    pos_x1 = tissue_pos.iloc[i,5]-radius
    pos_x2 = tissue_pos.iloc[i,5]+radius
    pos_y1 = tissue_pos.iloc[i,4]-radius
    pos_y2 = tissue_pos.iloc[i,4]+radius
    
    I = cv2.rectangle(I, (pos_x1,pos_y1), (pos_x2,pos_y2), (0, 0, 0), 2)
    I = cv2.putText(I, str(i), (pos_x1,pos_y1), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), lineType=cv2.LINE_AA, thickness=2)



### save Image
cv2.imwrite(dirName+"/CropImage/size_"+str(extraSize)+"/original_image_with_spots.tif", I)



### split Image
I = I_org.copy()

for i in range(4992):
    pos_x1 = tissue_pos.iloc[i,5]-radius
    pos_x2 = tissue_pos.iloc[i,5]+radius
    pos_y1 = tissue_pos.iloc[i,4]-radius
    pos_y2 = tissue_pos.iloc[i,4]+radius

    I2 = I[pos_y1:pos_y2, pos_x1:pos_x2]
    cv2.imwrite(dirName+"/CropImage/size_"+str(extraSize)+"/spot_images/spot_image_"+str(i).zfill(4)+".tif", I2)


# # Make RGB filter list


### load cluster list ###
cluster_list = pd.read_csv(dirName+"/SpaceRanger/analysis/clustering/graphclust/clusters.csv")

print("cluster_list: "+str(cluster_list.shape))
print(cluster_list.head())



### merge cluster file and tissue position file ###
cluster_pos_df = pd.merge(cluster_list, tissue_pos, how='left', on='Barcode')

cluster_pos_df['image_path'] = [dirName+"/CropImage/size_"+str(extraSize)+"/spot_images/spot_image_"+str(s).zfill(4)+".tif" for s in cluster_pos_df['imageID'].tolist()]
cluster_pos_df = cluster_pos_df.sort_values('imageID')
cluster_pos_df.index = cluster_pos_df['imageID'].tolist()

cluster_pos_df['ImageFilter'] = "null"
cluster_pos_df

print("cluster_pos_df: "+str(cluster_pos_df.shape))
print(cluster_pos_df.head())



### mkdir
subprocess.call(["mkdir","-p",dirName+"/CropImage/size_"+str(extraSize)+"/RGB_"+str(quantileRGB)+"/NG/"])
subprocess.call(["mkdir","-p",dirName+"/CropImage/size_"+str(extraSize)+"/RGB_"+str(quantileRGB)+"/OK/"])



### count mean RGB values ###
mean_value_list = list()

for i in cluster_pos_df.index:
    #print(i)
    I = cv2.imread(cluster_pos_df.loc[i,'image_path'])
    #print(I.shape)

    total_value = np.sum(I[:,:,0]) + np.sum(I[:,:,1]) + np.sum(I[:,:,2])
    
    total_value = total_value / (I.shape[0] * I.shape[1]) / 3
    
    #print(total_value)
    mean_value_list.append(total_value)



cluster_pos_df['mean_RGB'] = [round(f, 4) for f in mean_value_list]

print("cluster_pos_df: "+str(cluster_pos_df.shape))
print(cluster_pos_df.head())



## Define threshold
c_array = np.percentile(mean_value_list, q=[quantileRGB])
c_array

white_th = 255 if quantileRGB == 100 else c_array[0]

print("white_th: "+str(white_th))



with open(dirName+"/CropImage/size_"+str(extraSize)+"/RGB_"+str(quantileRGB)+"/white_th.txt", mode='w') as f:
    f.write(str(white_th)+"\n")



# Histgram
fig = plt.figure()

plt.hist(mean_value_list, bins=50)

plt.xlabel("mean RGB", fontsize=20)
plt.ylabel("Frequency", fontsize=20)

plt.axvline(x=white_th, color='r')

fig.savefig(dirName+"/CropImage/size_"+str(extraSize)+"/RGB_"+str(quantileRGB)+"/meanRGB.png")

plt.close()



### Threshold percentage ###
pixel_th_white = I.shape[0] * I.shape[1] * 0.5

print("pixel_th_white: "+str(pixel_th_white))



### load Image
for i in cluster_pos_df.index:
    #print(i)
    I = cv2.imread(cluster_pos_df.loc[i,'image_path'])
    #print(I.shape)
    
    ### color threshold (white)
    count_white = sum(np.logical_and.reduce((I[:,:,0] > white_th, I[:,:,1] > white_th, I[:,:,2] > white_th)))
    
    if sum(count_white) > pixel_th_white:
        subprocess.call(["cp","-p",dirName+"/CropImage/size_"+str(extraSize)+"/spot_images/spot_image_"+str(i).zfill(4)+".tif",dirName+"/CropImage/size_"+str(extraSize)+"/RGB_"+str(quantileRGB)+"/NG/"])
        cluster_pos_df.loc[cluster_pos_df.index == i,"ImageFilter"] = "NG"
    else:
        subprocess.call(["cp","-p",dirName+"/CropImage/size_"+str(extraSize)+"/spot_images/spot_image_"+str(i).zfill(4)+".tif",dirName+"/CropImage/size_"+str(extraSize)+"/RGB_"+str(quantileRGB)+"/OK/"])
        cluster_pos_df.loc[cluster_pos_df.index == i,"ImageFilter"] = "OK"
 



cluster_pos_df.to_csv(dirName+"/CropImage/size_"+str(extraSize)+"/RGB_"+str(quantileRGB)+"/cluster_position_filter.txt", index=False, sep='\t')


# # Crop Image (interpolation)


I = I_org.copy()

subprocess.call(['mkdir','-p',dirName+"/CropImage/size_"+str(extraSize)+"/spot_images_inter"])



### draw line and text
for i in range(4992):
    pos_x1 = tissue_pos.iloc[i,5]-radius
    pos_x2 = tissue_pos.iloc[i,5]+radius
    pos_y1 = tissue_pos.iloc[i,4]-radius
    pos_y2 = tissue_pos.iloc[i,4]+radius
    
    I = cv2.rectangle(I, (pos_x1,pos_y1), (pos_x2,pos_y2), (0, 0, 0), 2)
    I = cv2.putText(I, str(i), (pos_x1,pos_y1), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), lineType=cv2.LINE_AA, thickness=2)



### draw line and text
count_inter = 0

for i in range(4992):
    
    if tissue_pos.iloc[i,3] == 0 or tissue_pos.iloc[i,3] == 127:
        continue
       
    center_x = (tissue_pos.iloc[i-1,5] + tissue_pos.iloc[i,5] + tissue_pos.iloc[i+1,5]) / 3.0
    center_y = (tissue_pos.iloc[i-1,4] + tissue_pos.iloc[i,4] + tissue_pos.iloc[i+1,4]) / 3.0

    pos_x1 = int(center_x)-radius
    pos_x2 = int(center_x)+radius
    pos_y1 = int(center_y)-radius
    pos_y2 = int(center_y)+radius
    
    I = cv2.rectangle(I, (pos_x1,pos_y1), (pos_x2,pos_y2), (0, 255, 0), 2)
    I = cv2.putText(I, str(count_inter), (pos_x1,pos_y1), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), lineType=cv2.LINE_AA, thickness=2)

    count_inter += 1
    
    if i % 2 == 0:
        if i-127 < 0:
            continue

        center_x = (tissue_pos.iloc[i-129,5] + tissue_pos.iloc[i,5] + tissue_pos.iloc[i-127,5]) / 3.0
        center_y = (tissue_pos.iloc[i-129,4] + tissue_pos.iloc[i,4] + tissue_pos.iloc[i-127,4]) / 3.0

    else:
        if i+127 > 4991:
            continue

        center_x = (tissue_pos.iloc[i+127,5] + tissue_pos.iloc[i,5] + tissue_pos.iloc[i+129,5]) / 3.0
        center_y = (tissue_pos.iloc[i+127,4] + tissue_pos.iloc[i,4] + tissue_pos.iloc[i+129,4]) / 3.0

    pos_x1 = int(center_x)-radius
    pos_x2 = int(center_x)+radius
    pos_y1 = int(center_y)-radius
    pos_y2 = int(center_y)+radius
    
    I = cv2.rectangle(I, (pos_x1,pos_y1), (pos_x2,pos_y2), (0, 0, 255), 2)
    I = cv2.putText(I, str(count_inter), (pos_x1,pos_y1), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA, thickness=2)

    count_inter += 1



### save Image
cv2.imwrite(dirName+"/CropImage/size_"+str(extraSize)+"/original_image_with_spots_inter.tif", I)



### split Image
count_inter = 0

I = I_org.copy()

image_list = pd.DataFrame(columns=['No','pos_x1','pos_y1','radius'])

for i in range(4992):
    
    if tissue_pos.iloc[i,3] == 0 or tissue_pos.iloc[i,3] == 127:
        continue
       
    center_x = (tissue_pos.iloc[i-1,5] + tissue_pos.iloc[i,5] + tissue_pos.iloc[i+1,5]) / 3.0
    center_y = (tissue_pos.iloc[i-1,4] + tissue_pos.iloc[i,4] + tissue_pos.iloc[i+1,4]) / 3.0

    pos_x1 = int(center_x)-radius
    pos_x2 = int(center_x)+radius
    pos_y1 = int(center_y)-radius
    pos_y2 = int(center_y)+radius
    
    I2 = I[pos_y1:pos_y2, pos_x1:pos_x2]
    cv2.imwrite(dirName+"/CropImage/size_"+str(extraSize)+"/spot_images_inter/spot_image_inter_"+str(count_inter).zfill(4)+"_"+str(int(center_x))+"_"+str(int(center_y))+".tif", I2)
    
    image_list = image_list.append([pd.Series([count_inter,int(center_x),int(center_y),radius],index=image_list.columns)], ignore_index=True)
    
    count_inter += 1
    
    if i % 2 == 0:
        if i-127 < 0:
            continue

        center_x = (tissue_pos.iloc[i-129,5] + tissue_pos.iloc[i,5] + tissue_pos.iloc[i-127,5]) / 3.0
        center_y = (tissue_pos.iloc[i-129,4] + tissue_pos.iloc[i,4] + tissue_pos.iloc[i-127,4]) / 3.0

    else:
        if i+127 > 4991:
            continue

        center_x = (tissue_pos.iloc[i+127,5] + tissue_pos.iloc[i,5] + tissue_pos.iloc[i+129,5]) / 3.0
        center_y = (tissue_pos.iloc[i+127,4] + tissue_pos.iloc[i,4] + tissue_pos.iloc[i+129,4]) / 3.0

    pos_x1 = int(center_x)-radius
    pos_x2 = int(center_x)+radius
    pos_y1 = int(center_y)-radius
    pos_y2 = int(center_y)+radius
    
    I2 = I[pos_y1:pos_y2, pos_x1:pos_x2]
    cv2.imwrite(dirName+"/CropImage/size_"+str(extraSize)+"/spot_images_inter/spot_image_inter_"+str(count_inter).zfill(4)+"_"+str(int(center_x))+"_"+str(int(center_y))+".tif", I2)
    
    image_list = image_list.append([pd.Series([count_inter,int(center_x),int(center_y),radius],index=image_list.columns)], ignore_index=True)

    count_inter += 1


# # Make RGB filter list (interpolation)


### interpolated image list ###
image_list['ImageFilter'] = "null"
image_list['image_path'] = [dirName+"/CropImage/size_"+str(extraSize)+"/spot_images_inter/spot_image_inter_"+str(no).zfill(4)+"_"+str(x)+"_"+str(y)+".tif" for no,x,y in zip(image_list['No'],image_list['pos_x1'],image_list['pos_y1'])]

print("image_list: "+str(image_list.shape))
print(image_list.head())



### mkdir
subprocess.call(["mkdir","-p",dirName+"/CropImage/size_"+str(extraSize)+"/RGB_"+str(quantileRGB)+"/NG_inter/"])
subprocess.call(["mkdir","-p",dirName+"/CropImage/size_"+str(extraSize)+"/RGB_"+str(quantileRGB)+"/OK_inter/"])



### count mean RGB values ###
mean_value_list = list()

for i in image_list.index:
    #print(i)
    
    I = cv2.imread(image_list.loc[i,'image_path'])

    total_value = np.sum(I[:,:,0]) + np.sum(I[:,:,1]) + np.sum(I[:,:,2])
    
    total_value = total_value / (I.shape[0] * I.shape[1]) / 3
    
    mean_value_list.append(total_value)



image_list['mean_RGB'] = [round(f, 4) for f in mean_value_list]

print("image_list: "+str(image_list.shape))
print(image_list.head())



## read RGB threshold
with open(dirName+"/CropImage/size_"+str(extraSize)+"/RGB_"+str(quantileRGB)+"/white_th.txt", mode='r') as f:
    white_th = f.read().splitlines()[0]
    white_th = float(white_th)

white_th

print("white_th: "+str(white_th))



# Histgram
fig = plt.figure()

plt.hist(mean_value_list, bins=50)

plt.xlabel("mean RGB", fontsize=20)
plt.ylabel("Frequency", fontsize=20)

plt.axvline(x=white_th, color='r')

fig.savefig(dirName+"/CropImage/size_"+str(extraSize)+"/RGB_"+str(quantileRGB)+"/meanRGB_inter.png")

plt.close()



### Threshold percentage ###
pixel_th_white = I.shape[0] * I.shape[1] * 0.5

print("pixel_th_white: "+str(pixel_th_white))



### load Image
for i in image_list.index:
    #print(i)
    
    I = cv2.imread(image_list.loc[i,'image_path'])
    
    ### color threshold (white)
    count_white = sum(np.logical_and.reduce((I[:,:,0] > white_th, I[:,:,1] > white_th, I[:,:,2] > white_th)))
    
    if sum(count_white) > pixel_th_white:
        subprocess.call(["cp","-p",image_list.loc[image_list.index == i,"image_path"].tolist()[0],dirName+"/CropImage/size_"+str(extraSize)+"/RGB_"+str(quantileRGB)+"/NG_inter/"])
        image_list.loc[image_list.index == i,"ImageFilter"] = "NG"
    else:
        subprocess.call(["cp","-p",image_list.loc[image_list.index == i,"image_path"].tolist()[0],dirName+"/CropImage/size_"+str(extraSize)+"/RGB_"+str(quantileRGB)+"/OK_inter/"])
        image_list.loc[image_list.index == i,"ImageFilter"] = "OK"
 



image_list.to_csv(dirName+"/CropImage/size_"+str(extraSize)+"/RGB_"+str(quantileRGB)+"/image_list_inter.txt", index=False, sep='\t')













