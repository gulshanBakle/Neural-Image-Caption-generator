import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        
        
        #self.gru=nn.GRU(input_size = embed_size,hidden_size = hidden_size,
                            #num_layers = num_layers, batch_first = True,bidirectional=True)
        
        
        self.lstm = nn.LSTM(input_size = embed_size,hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        captions = captions[:, :-1]
        #print(captions.shape)
        embed = self.embedding_layer(captions)
        #print(embed.shape)
        embed = torch.cat((features.unsqueeze(1), embed), dim=1)
        #print(embed.shape)
        lstm_outputs, _ = self.lstm(embed)
        out = self.linear(lstm_outputs)

        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output_sentence = []
#        lstm_outputs = torch.zeros((128, self.hidden_size)).cuda()
#        states = torch.zeros((128, self.hidden_size)).cuda()
        for i in range(max_len):
            lstm_outputs, states = self.lstm(inputs, states)
            
            lstm_outputs = lstm_outputs.squeeze(1)
            out = self.linear(lstm_outputs)
            last_pick = out.max(1)[1]
            output_sentence.append(last_pick.item())
            inputs = self.embedding_layer(last_pick).unsqueeze(1)
        
        return output_sentence