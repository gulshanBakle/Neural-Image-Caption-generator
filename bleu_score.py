# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:20:47 2020

@author: gulsh
"""

import os
import io
import sys
import torch
import requests
from testingRnnCnn import clean_sentence, get_prediction
from data_loader import get_loader
from PIL import Image
from torchvision import transforms
sys.path.append('C:/Users/gulsh/OneDrive/Desktop/CS 535 Deep Learning/Final Project/cocoapi')
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
data_dir='C:/Users/gulsh/OneDrive/Desktop/CS 535 Deep Learning/Final Project/cocoapi'
data_type='test2014'

instances_annfile=os.path.join(data_dir,'annotations/instances_{}.json'.format(dataType))
coco=COCO(instances_annfile)
captions_annFile = os.path.join(dataDir, 'annotations/captions_{}.json'.format(dataType))
coco_caps = COCO(captions_annFile)
ids = list(coco.anns.keys())
#print(ids)

import numpy as np
import skimage.io 


def get_imgs():
    ann_id = np.random.choice(ids)
    img_id = coco.anns[ann_id]['image_id']
    img = coco.loadImgs(img_id)[0]
    print(img)
    url = img['coco_url']
    #flickrurl=img['flickr_url']
    #print(url)
    #print(flickrurl)
    orig_img = skimage.io.imread(url)
    plt.axis('off')
    plt.imshow(orig_img)
    plt.show()
    #print(orig_img.shape)
    
    
    
    response = requests.get(url)
    img_pil = Image.open(io.BytesIO(response.content)).convert('RGB')
    
    transform_test = transforms.Compose([ 
        transforms.Resize(256),                          # smaller edge of image resized to 256
        transforms.RandomCrop(224),                      # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
        transforms.ToTensor()]) 
    
    
    I_transf=transform_test(img_pil)  
    return orig_img,I_transf,img                       


# load and display captions
annIds = coco_caps.getAnnIds(imgIds=img['id']);
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)


for i in range(2):
    orig_img,I_transf,img=get_imgs()
    print('original captions')
    annIds = coco_caps.getAnnIds(imgIds=img['id']);
    anns = coco_caps.loadAnns(annIds)
    coco_caps.showAnns(anns)
    get_prediction()
    
    
    
    
    