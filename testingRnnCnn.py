# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 00:31:17 2020

@author: gulsh
"""
import sys
import os
sys.path.append('C:/Users/gulsh/OneDrive/Desktop/CS 535 Deep Learning/Final Project/cocoapi/PythonAPI')
from pycocotools.coco import COCO
from data_loader import get_loader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import nltk
import nltk.translate.bleu_score as bleu

import math

# TODO #1: Define a transform to pre-process the testing images.
transform_test = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406),
                                                          (0.229, 0.224, 0.225))
                                    ])

#-#-#-# Do NOT modify the code below this line. #-#-#-#

# Create the data loader.
data_loader = get_loader(transform=transform_test,    
                         mode='test')

orig_image, image = next(iter(data_loader))

# Visualize sample image, before pre-processing.
plt.imshow(np.squeeze(orig_image))
plt.title('example image')
plt.show()

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
import torch
from model import EncoderCNN, DecoderRNN

# TODO #2: Specify the saved models to load.
encoder_file = 'encoder-3.pkl'
decoder_file = 'decoder-3.pkl'

# TODO #3: Select appropriate values for the Python variables below.
embed_size = 256
hidden_size = 512

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the encoder and decoder, and set each to inference mode.
encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()

# Load the trained weights.
encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file)))
decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file)))

# Move models to GPU if CUDA is available.
encoder.to(device)
decoder.to(device)
image = image.to(device)

# Obtain the embedded image features.
features = encoder(image).unsqueeze(1)

# Pass the embedded image features through the model to get a predicted caption.
output = decoder.sample(features)
print('example output:', output)

assert (type(output)==list), "Output needs to be a Python list" 
assert all([type(x)==int for x in output]), "Output should be a list of integers." 
assert all([x in data_loader.dataset.vocab.idx2word for x in output]), "Each entry in the output needs to correspond to an integer that indicates a token in the vocabulary."

def clean_sentence(output):
    
    words_sequence = []
    
    for i in output:
        if (i == 1):
            continue
        words_sequence.append(data_loader.dataset.vocab.idx2word[i])
    print(words_sequence)
    words_sequence = words_sequence[1:-1] 
    sentence = ' '.join(words_sequence) 
    sentence = sentence.capitalize()
    
    return sentence


sentence = clean_sentence(output)
print('example sentence:', sentence)

assert type(sentence)==str, 'Sentence needs to be a Python string!'

import skimage.io 
instances_annfile=os.path.join(dataDir,'annotations/instances_{}.json'.format(dataType))
coco=COCO(instances_annfile)
captions_annFile = os.path.join(dataDir, 'annotations/captions_{}.json'.format(dataType))
coco_caps = COCO(captions_annFile)
ids = list(coco.anns.keys())


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
    return orig_img,I_transf.view(1,3,224,224),img      

def get_prediction():
    orig_image, image,img = get_imgs()
    plt.imshow(np.squeeze(orig_image))
    plt.title('Sample Image')
    plt.show()
    #print(caption)
    print(orig_image.shape)
    print(image.shape)
    print("original captions")
    annIds = coco_caps.getAnnIds(imgIds=img['id']);
    anns = coco_caps.loadAnns(annIds)
    print("caption 1")
    ref1=str(anns[0]['caption']).split()
    ref2=str(anns[1]['caption']).split()
    ref3=str(anns[2]['caption']).split()
    ref4=str(anns[2]['caption']).split()
    print(ref1)
    #coco_caps.showAnns(anns)
    #plt.imshow(np.squeeze(image).view(224,224,3))
    #plt.title('Sample Image')
    #plt.show()
    image = image.to(device)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)   
    print(output)
    sentence = clean_sentence(output)
    print("generated caption")
    print(sentence)
    sentence=str(sentence[:len(sentence)-1]).split()
    print(sentence)
    print("bleu score \t")
    score=bleu.sentence_bleu(sentence,ref2,(0.25,0.25,0.25,0.25))
    print(score)
    
get_prediction()

#print(data_loader)

