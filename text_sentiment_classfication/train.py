# -*- coding: utf-8 -*-

# this is for filtering the warnings
import warnings
warnings.filterwarnings('ignore')

"""### Utils"""

import torch
from torch import nn
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import sys
from utils import *
from torch.utils import data
import os

train_file = sys.argv[1]

class TwitterDataset(data.Dataset):

    def __init__(self, X, y):
        self.data = X
        self.label = y
    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]
    def __len__(self):
        return len(self.data)

"""### Model"""

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

def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    model.train()
    criterion = nn.BCELoss() # 定義損失函數，這裡我們使用binary cross entropy loss
    t_batch = len(train) 
    v_batch = len(valid) 
    optimizer = optim.Adam(model.parameters(), lr=lr) # 將模型的參數給optimizer，並給予適當的learning rate
    total_loss, total_acc, best_acc = 0, 0, 0
    
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long) 
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(inputs) 
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward() # 算loss的gradient
            optimizer.step() # 更新訓練模型的參數
            correct = evaluation(outputs, labels) # 計算此時模型的training accuracy
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
            	epoch+1, i+1, t_batch, loss.item(), correct*100/batch_size), end='\r')

        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch, total_acc/t_batch*100))

        model.eval() # 將model的模式設為eval，這樣model的參數就會固定住
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.float)
                outputs = model(inputs)
                outputs = outputs.squeeze() # 去掉最外面的dimension，好讓outputs可以餵進criterion()
                loss = criterion(outputs, labels)
                correct = evaluation(outputs, labels)
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))

            if total_acc > best_acc:
                best_acc = total_acc
                torch.save(model, "{}/ckpt.model".format(model_dir))
                print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
        print('-----------------------------------------------')
        model.train()


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

w2v_path = os.path.join('model/w2v_all.model')
model_dir = os.path.join('model/')

sen_len = 30
fix_embedding = True# fix embedding during training
batch_size = 256
epoch = 7
lr = 0.001


print("loading data ...")
train_x, y = load_training_data(train_file)


# 對input跟labels做預處理
preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
train_x = preprocess.sentence_word2idx()
y = preprocess.labels_to_tensor(y)


# 製作一個model的對象
model = LSTM_Net(embedding, embedding_dim=400, hidden_dim=1000, num_layers=1, dropout=0.6, fix_embedding=fix_embedding, bidirectional=True)
model = model.to(device)

print('\nload model ...')

X_train, X_val, y_train, y_val = train_x[:180000], train_x[180000:], y[:180000], y[180000:]

train_dataset = TwitterDataset(X=X_train, y=y_train)
val_dataset = TwitterDataset(X=X_val, y=y_val)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 8)

val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)

# 開始訓練
training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)

