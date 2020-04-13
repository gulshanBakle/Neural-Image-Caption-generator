# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 18:12:08 2020

@author: gulsh
"""
import os
import sys
sys.path.append('C:/Users/gulsh/OneDrive/Desktop/CS 535 Deep Learning/Final Project/cocoapi')
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
dataDir='C:/Users/gulsh/OneDrive/Desktop/CS 535 Deep Learning/Final Project/cocoapi'
dataType='val2014'
instances_annFile = os.path.join(dataDir, 'annotations/instances_{}.json'.format(dataType))
coco = COCO(instances_annFile)  
captions_annFile = os.path.join(dataDir, 'annotations/captions_{}.json'.format(dataType))
coco_caps = COCO(captions_annFile)
ids = list(coco.anns.keys())
#print(ids)
import numpy as np
import skimage.io as io
ann_id = np.random.choice(ids)
img_id = coco.anns[ann_id]['image_id']
img = coco.loadImgs(img_id)[0]
print(img)
url = img['coco_url']
flickrurl=img['flickr_url']
print(url)
print(flickrurl)
I = io.imread(url)
print(I.shape)
plt.axis('off')
plt.imshow(I)
plt.show()

I_flickr=io.imread(flickrurl)
plt.axis('on')
plt.imshow(I_flickr)
plt.show()
import data_loader
import nltk
from data_loader import get_loader
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models

# Define a transform to pre-process the training images.
from collections import Counter
import math
from model import EncoderCNN, DecoderRNN

#counter=Counter(data_loader.dataset.caption_lengths)
#print(counter)
#lengths=sorted(counter.items(),key=lambda pair:pair[1],reverse=True)
#for value, count in lengths:
#    print('value: %2d --- count: %5d' % (value, count))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#Lets specify hyperparamters for training Encoder -Decoder
batch_size=64
vocab_threshold=4
vocab_from_file=True
embed_size=256
hidden_size=512
epochs=4
save_every=1
print_every=100
log_file='train_final.txt'
log_file2='validation.txt'

transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])


train_data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=True)
vocab_size=len(train_data_loader.dataset.vocab)
print(vocab_size)

val_data_loader=get_loader(transform=transform_train,
                         mode='val',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=True)


encoder=EncoderCNN(embed_size)
decoder=DecoderRNN(embed_size,hidden_size,vocab_size)

encoder.to(device)
decoder.to(device)

criterion = nn.CrossEntropyLoss().cuda()
params=list(decoder.parameters())+list(encoder.embed.parameters())
print(len(params))

optimizer= torch.optim.Adam(params)
total_step = math.ceil(len(train_data_loader.dataset.caption_lengths) / train_data_loader.batch_sampler.batch_size)
print(total_step)

import numpy as np
import time
import requests
f=open(log_file,'w')
#f2=open(log_file2,'w)

# old_time = time.time()
# response = requests.request("GET", 
#                             "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token", 
# 

loss_list=[]      
perplex_100=[]  
val_loss_list=[]                    
for epoch in range(1,3):
    for step in range(1,total_step+1):
        
#         if time.time() - old_time > 60:
#             old_time = time.time()
#             requests.request("POST", 
#                              "https://nebula.udacity.com/api/v1/remote/keep-alive", 
#                              headers={'Authorization': "STAR " + response.text})
        
        
        
        indices = train_data_loader.dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        train_data_loader.batch_sampler.sampler = new_sampler
        
        # Obtain the batch.
        images, captions = next(iter(train_data_loader))
        
        # Move batch of images and captions to GPU if CUDA is available.
        images = images.to(device)
        captions = captions.to(device)
        
        # Zero the gradients.
        decoder.zero_grad()
        encoder.zero_grad()
        
        
        # set the encoder decoder in training mode
        encoder.train()
        decoder.train()
        
        # Pass the inputs through the CNN-RNN model.
        features = encoder(images)
        outputs = decoder(features, captions)
        
        # Calculate the batch loss.
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        
        # Backward pass.
        loss.backward()
        
        # Update the parameters in the optimizer.
        optimizer.step()
        
        with torch.no_grad():
            
            encoder.eval()
            decoder.eval()
            val_images,val_captions=next(iter(val_data_loader))
            
            val_images=val_images.to(device)
            val_captions=val_captions.to(device)
            
            val_features=encoder(val_images)
            val_op=decoder(val_features,val_captions)
            
            val_loss=criterion(val_op.view(-1,vocab_size),val_captions.view(-1))
        val_loss_list.append(val_loss.item())
            
            
        # Get training statistics.
        stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f,Val loss: %.4f, Perplexity: %5.4f' % (epoch, epochs, step, total_step, loss.item(),val_loss.item(), np.exp(loss.item()))
        
        # Print training statistics (on same line).
        print('\r' + stats, end="")
        sys.stdout.flush()
        
        # Print training statistics to file.
        f.write(stats + '\n')
        f.flush()
        
        # Print training statistics (on different line).
        if step % print_every == 0:
            print('\r' + stats)
            loss_list.append(loss.item())
            perplex_100.append(np.exp(loss.item()))
            
            
    # Save the weights.
    if epoch % save_every == 0:
        torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-%d.pkl' % 5+epoch))
        torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-%d.pkl' % 5+epoch))

# Close the training log file.
f.close()

#print(len(loss_list))
#print(perplex_100[:-5])
