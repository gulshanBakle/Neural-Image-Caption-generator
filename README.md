# Neural-Image-Caption-generator

An image caption generator, which accepts image as input and generates captions describing them. In this project I have tried to implement paper https://arxiv.org/abs/1411.4555. Model described in this paper is similar to an encoder-decoder model used in machine translation tasks. Only difference being is a pre-trained CNN like ResNet, ImageNet or Inception is used as an encoder and Recurrent networks like LSTMs or GRUs as decoder.  
originally the paper has been implemented on flickr8k,flick30k and MSCOCO dataset, but this project shows results with only MSCOCO dataset(http://cocodataset.org/#download). Most part of implementaion is done using Python's widely used library- Pytorch. Other libraries like Numpy, Sklearn and Matplotlib are used for data pre-processing. Natural Language ToolKit (NLTK) is also used for data wrangling with captions. 

Running the model for 3-4 epochs took around 12 hours of training on 2GB Nividia graphics card, which resulted in perlexity in the range of 5-10, but this can be improved by training the model for much larger time. Average bleu score was around 46.
