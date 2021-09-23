import torch
from torch import nn
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from utils import *
from torch.utils import data
import sys

test_file = sys.argv[1]
output_file = sys.argv[2]


class TwitterDataset(data.Dataset):
    def __init__(self, X, y):
        self.data = X
        self.label = y
    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]
    def __len__(self):
        return len(self.data)


class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True, bidirectional=False):
        super(LSTM_Net, self).__init__()
        # 製作 embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # 是否將 embedding fix住，如果fix_embedding為False，在訓練過程中，embedding也會跟著被訓練
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)

        self.classifier = nn.Sequential( nn.Dropout(dropout),
                                         nn.Linear(hidden_dim *2 * (bidirectional +1), 1500),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(1500, 200),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(200, 1),
                                         nn.Sigmoid()
                                         )

    def apply_attention(self, rnn_output, final_hidden_state):
        hidden_state = final_hidden_state.unsqueeze(2)
        attention_scores = torch.bmm(rnn_output, hidden_state).squeeze(2)
        soft_attention_weights = F.softmax(attention_scores, 1).unsqueeze(2) #shape = (batch_size, seq_len, 1)
        attention_output = torch.bmm(rnn_output.permute(0,2,1), soft_attention_weights).squeeze(2)
        return attention_output
        

    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, (h_n,c_n) = self.lstm(inputs, None)
        #x, h_n = self.gru(inputs, None)
        batch_size = h_n.shape[1]
        h_n_final_layer = h_n.view(self.num_layers, self.bidirectional + 1, batch_size, self.hidden_dim)[-1,:,:,:]
        
        # Convert input to (batch_size, num_directions * hidden_size) for attention
        final_hidden_state = torch.cat([h_n_final_layer[i,:,:] for i in range(h_n_final_layer.shape[0])], dim=1)
        
        attention_out = self.apply_attention(x, final_hidden_state)
        # Attention_out.shape = (batch_size, num_directions * hidden_size)
        
        concatenated_vector = torch.cat([final_hidden_state, attention_out], dim=1)
        return self.classifier(concatenated_vector)


import os
import torch
import argparse
import numpy as np
from torch import nn
from gensim.models import word2vec
from sklearn.model_selection import train_test_split

def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs>=0.5] = 1 # 大於等於0.5為負面
            outputs[outputs<0.5] = 0 # 小於0.5為正面
            ret_output += outputs.int().tolist()
    
    return ret_output


w2v_path = os.path.join('model/w2v_all.model') 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sen_len = 30
fix_embedding = True# fix embedding during training
batch_size = 256
lr = 0.001

# 開始測試模型並做預測
print("loading testing data ...")
test_x = load_testing_data(test_file)
preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
test_x = preprocess.sentence_word2idx()
test_dataset = TwitterDataset(X=test_x, y=None)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)
print('\nload model ...')
model = torch.load(os.path.join('model/ckpt.model'))
model = model.to(device)
outputs = testing(batch_size, test_loader, model, device)

tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
print("save csv ...")
tmp.to_csv(output_file, index=False)
print("Finish Predicting")

