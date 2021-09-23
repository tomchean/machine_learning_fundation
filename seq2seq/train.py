# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.data.sampler as sampler
import torchvision
from torchvision import datasets, transforms
import math

import numpy as np
import sys
import os
import random
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 判斷是用 CPU 還是 GPU 執行運算
import numpy as np

class LabelTransform(object):
  def __init__(self, size, pad):
    self.size = size
    self.pad = pad

  def __call__(self, label):
    label = np.pad(label, (0, (self.size - label.shape[0])), mode='constant', constant_values=self.pad)
    return label

import re
import json

class EN2CNDataset(data.Dataset):
  def __init__(self, root, max_output_len, set_name):
    self.root = root

    self.word2int_cn, self.int2word_cn = self.get_dictionary('cn')
    self.word2int_en, self.int2word_en = self.get_dictionary('en')

    # 載入資料
    self.data = []
    with open(os.path.join(self.root, f'{set_name}.txt'), "r") as f:
      for line in f:
        self.data.append(line)
    print (f'{set_name} dataset size: {len(self.data)}')

    self.cn_vocab_size = len(self.word2int_cn)
    self.en_vocab_size = len(self.word2int_en)
    self.transform = LabelTransform(max_output_len, self.word2int_en['<PAD>'])

  def get_dictionary(self, language):
    # 載入字典
    with open(os.path.join(self.root, f'word2int_{language}.json'), "r") as f:
      word2int = json.load(f)
    with open(os.path.join(self.root, f'int2word_{language}.json'), "r") as f:
      int2word = json.load(f)
    return word2int, int2word

  def __len__(self):
    return len(self.data)

  def __getitem__(self, Index):
    # 先將中英文分開
    sentences = self.data[Index]
    sentences = re.split('[\t\n]', sentences)
    sentences = list(filter(None, sentences))
    #print (sentences)
    assert len(sentences) == 2

    # 預備特殊字元
    BOS = self.word2int_en['<BOS>']
    EOS = self.word2int_en['<EOS>']
    UNK = self.word2int_en['<UNK>']

    # 在開頭添加 <BOS>，在結尾添加 <EOS> ，不在字典的 subword (詞) 用 <UNK> 取代
    en, cn = [BOS], [BOS]
    # 將句子拆解為 subword 並轉為整數
    sentence = re.split(' ', sentences[0])
    sentence = list(filter(None, sentence))
    #print (f'en: {sentence}')
    for word in sentence:
      en.append(self.word2int_en.get(word, UNK))
    en.append(EOS)

    # 將句子拆解為單詞並轉為整數
    # e.g. < BOS >, we, are, friends, < EOS > --> 1, 28, 29, 205, 2
    sentence = re.split(' ', sentences[1])
    sentence = list(filter(None, sentence))
    #print (f'cn: {sentence}')
    for word in sentence:
      cn.append(self.word2int_cn.get(word, UNK))
    cn.append(EOS)

    en, cn = np.asarray(en), np.asarray(cn)

    # 用 <PAD> 將句子補到相同長度
    en, cn = self.transform(en), self.transform(cn)
    en, cn = torch.LongTensor(en), torch.LongTensor(cn)

    return en, cn

class Encoder(nn.Module):
  def __init__(self, en_vocab_size, emb_dim, hid_dim, n_layers, dropout):
    super().__init__()
    self.embedding = nn.Embedding(en_vocab_size, emb_dim)
    self.hid_dim = hid_dim
    self.n_layers = n_layers
    self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
    self.dropout = nn.Dropout(dropout)

  def forward(self, input):
    # input = [batch size, sequence len, vocab size]
    embedding = self.embedding(input)
    outputs, hidden = self.rnn(self.dropout(embedding))
    # outputs = [batch size, sequence len, hid dim * directions]
    # hidden =  [num_layers * directions, batch size  , hid dim]
    # outputs 是最上層RNN的輸出
        
    return outputs, hidden

class Decoder(nn.Module):
  def __init__(self, cn_vocab_size, emb_dim, hid_dim, n_layers, dropout, isatt, max_len):
    super().__init__()
    self.cn_vocab_size = cn_vocab_size
    self.hid_dim = hid_dim * 2
    self.n_layers = n_layers
    self.embedding = nn.Embedding(cn_vocab_size, config.emb_dim)
    self.isatt = isatt
    self.attention = Attention(hid_dim, emb_dim, max_len)
    # 如果使用 Attention Mechanism 會使得輸入維度變化，請在這裡修改
    self.input_dim = emb_dim + hid_dim * 2 if isatt else emb_dim
    self.rnn = nn.GRU(self.input_dim, self.hid_dim, self.n_layers, dropout = dropout, batch_first=True)
    self.embedding2vocab1 = nn.Linear(self.hid_dim, self.hid_dim * 2)
    self.embedding2vocab2 = nn.Linear(self.hid_dim * 2, self.hid_dim * 4)
    self.embedding2vocab3 = nn.Linear(self.hid_dim * 4, self.cn_vocab_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, input, hidden, encoder_outputs):
    # input = [batch size, vocab size]
    # hidden = [batch size, n layers * directions, hid dim]
    # Decoder 只會是單向，所以 directions=1
    input = input.unsqueeze(1)
    embedded = self.dropout(self.embedding(input))
    # embedded = [batch size, 1, emb dim]
    if self.isatt:
      attn = self.attention(encoder_outputs, hidden, embedded)
      # concat Attention with embedded
      embedded = torch.cat([embedded, attn], dim=2)

    output, hidden = self.rnn(embedded, hidden)
    # output = [batch size, 1, hid dim]
    # hidden = [num_layers, batch size, hid dim]

    # 將 RNN 的輸出轉為每個詞出現的機率
    output = self.embedding2vocab1(output.squeeze(1))
    output = self.embedding2vocab2(output)
    prediction = self.embedding2vocab3(output)
    # prediction = [batch size, vocab size]
    return prediction, hidden

class BeamNode():
  def __init__(self, input, result, score):
    self.output_seq = input
    self.output_result = result
    self.output_scores = score # Prefix score, list of float

  def score(self):
    return sum(self.output_scores) / len(self.output_scores)

class Attention(nn.Module):
  def __init__(self, hid_dim, emb_dim, max_len):
    super(Attention, self).__init__()
    self.hid_dim = hid_dim
    self.emb_dim = emb_dim
    #self.attn = nn.Linear(hid_dim *2 + emb_dim, max_len)
    self.attn = nn.Sequential(
      nn.Linear(hid_dim *2 + emb_dim, 2 * max_len),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(2 * max_len, max_len)
    )

  
  def forward(self, encoder_outputs, encoder_hidden, embedding):
    # encoder_outputs = [batch size, sequence len, hid dim * directions]
    # encoder_hidden = [num_layers, batch size, hid dim * directions]
    # 一般來說是取 Encoder 最後一層的 hidden state 來做 attention

    final_hidden_layer = encoder_hidden[-1,:,:]
    # final_hidden_layer = [batch size, hid dim]

    final_hidden_layer = final_hidden_layer.unsqueeze(2)
    # final_hidden_layer = [batch size, hid dim, 1]

    tmp = torch.cat((embedding.squeeze(1), final_hidden_layer.squeeze(2)), 1)
    attention_scores = self.attn(tmp)
    # attention_scores = [batch size, seq_len]

    soft_attention_weights = F.softmax(attention_scores, 1).unsqueeze(2)
    # soft_attention_weights = [batch size, seq_len, 1]

    attention = torch.bmm(encoder_outputs.permute(0,2,1), soft_attention_weights).squeeze(2)
    # attention = [batch size, hid dim * directions]
    attention = attention.unsqueeze(1) 
    # attention = [batch size, 1, hid dim * directions]

    return attention
  '''
  def forward(self, encoder_outputs, encoder_hidden, embedding):
    # encoder_outputs = [batch size, sequence len, hid dim * directions]
    # encoder_hidden = [num_layers, batch size, hid dim * directions]
    # 一般來說是取 Encoder 最後一層的 hidden state 來做 attention

    final_hidden_layer = encoder_hidden[-1,:,:]
    # final_hidden_layer = [batch size, hid dim]

    final_hidden_layer = final_hidden_layer.unsqueeze(2)
    # final_hidden_layer = [batch size, hid dim, 1]

    #attention_scores = torch.bmm(encoder_outputs, final_hidden_layer).squeeze(2)
    attention_scores = torch.bmm(embedding, final_hidden_layer).squeeze(2)
    # attention_scores = [batch size, seq_len]

    soft_attention_weights = F.softmax(attention_scores, 1).unsqueeze(2)
    # soft_attention_weights = [batch size, seq_len, 1]

    attention = torch.bmm(encoder_outputs.permute(0,2,1), soft_attention_weights).squeeze(2)
    # attention = [batch size, hid dim * directions]
    attention = attention.unsqueeze(1) 
    # attention = [batch size, 1, hid dim * directions]

    return attention
  '''

class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder, device):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.device = device
    assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
            
  def forward(self, input, target, teacher_forcing_ratio):
    # input  = [batch size, input len, vocab size]
    # target = [batch size, target len, vocab size]
    # teacher_forcing_ratio 是有多少機率使用正確答案來訓練
    batch_size = target.shape[0]
    target_len = target.shape[1]
    vocab_size = self.decoder.cn_vocab_size

    # 準備一個儲存空間來儲存輸出
    outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
    # 將輸入放入 Encoder
    encoder_outputs, hidden = self.encoder(input)
    # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
    # encoder_outputs 主要是使用在 Attention
    # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
    # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
    hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
    hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
    # 取的 <BOS> token
    input = target[:, 0]
    preds = []
    for t in range(1, target_len):
      output, hidden = self.decoder(input, hidden, encoder_outputs)
      outputs[:, t] = output
      # 決定是否用正確答案來做訓練
      teacher_force = random.random() <= teacher_forcing_ratio
      # 取出機率最大的單詞
      top1 = output.argmax(1)
      # 如果是 teacher force 則用正解訓練，反之用自己預測的單詞做預測
      input = target[:, t] if teacher_force and t < target_len else top1
      preds.append(top1.unsqueeze(1))
    preds = torch.cat(preds, 1)
    return outputs, preds

  def inference(self, input, target, beamsize):
    # 在這裡實施 Beam Search
    # 此函式的 batch size = 1  
    # input  = [batch size, input len]
    # target = [batch size, target len]
    batch_size = input.shape[0]
    input_len = input.shape[1]
    vocab_size = self.decoder.cn_vocab_size

    # 將輸入放入 Encoder
    encoder_outputs, hidden = self.encoder(input)
    hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
    hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)

    # 取的 <BOS> token
    input = target[:, 0]
    # 準備一個儲存空間來儲存輸出
    beams = [BeamNode([input[0]], [], [1])]
    preds = []
    for t in range(1, input_len):
      nextbeams = []
      for beam in beams:
        output, hidden = self.decoder(beam.output_seq[-1].unsqueeze(0), hidden, encoder_outputs)

        prob = F.log_softmax(output, dim=-1)

        beam.output_result.append(output)
        # 取出機率最大的單詞
        topv, topi = prob.squeeze(0).topk(beamsize)
        for i in range(len(topv)):
          newnode = BeamNode(beam.output_seq + [topi[i]], 
            beam.output_result + [],
            beam.output_scores + [topv[i]]
          )
          nextbeams.append(newnode)

      nextbeams.sort(key=lambda b: b.score(), reverse=True)
      beams = nextbeams[:beamsize]

    outputs = torch.zeros(batch_size, input_len, vocab_size).to(self.device)
    for i in range(1, input_len):
      outputs[:,i] = beams[0].output_result[i-1]

    preds = []
    for i in range(1, input_len):
      preds.append(beams[0].output_seq[i].unsqueeze(0).unsqueeze(1))

    preds = torch.cat(preds, 1)
    return outputs, preds

def save_model(model, optimizer, store_model_path, step):
  torch.save(model.state_dict(), f'{store_model_path}/model_{step}.ckpt')
  return

def load_model(model, load_model_path):
  print(f'Load model from {load_model_path}')
  model.load_state_dict(torch.load(f'{load_model_path}.ckpt'))
  return model

def build_model(config, en_vocab_size, cn_vocab_size):
  # 建構模型
  encoder = Encoder(en_vocab_size, config.emb_dim, config.hid_dim, config.n_layers, config.dropout)
  decoder = Decoder(cn_vocab_size, config.emb_dim, config.hid_dim, config.n_layers, config.dropout, config.attention, config.max_output_len)
  model = Seq2Seq(encoder, decoder, device)
  print(model)
  # 建構 optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
  print(optimizer)
  if config.load_model:
    model = load_model(model, config.load_model_path)
  model = model.to(device)

  return model, optimizer

def tokens2sentence(outputs, int2word):
  sentences = []
  for tokens in outputs:
    sentence = []
    for token in tokens:
      word = int2word[str(int(token))]
      if word == '<EOS>':
        break
      sentence.append(word)
    sentences.append(sentence)
  
  return sentences

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

def computebleu(sentences, targets):
  score = 0 
  assert (len(sentences) == len(targets))

  def cut_token(sentence):
    tmp = []
    for token in sentence:
      if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
        tmp.append(token)
      else:
        tmp += [word for word in token]
    return tmp 

  for sentence, target in zip(sentences, targets):
    sentence = cut_token(sentence)
    target = cut_token(target)
    score += sentence_bleu([target], sentence, weights=(1, 0, 0, 0))                                                                                          
  
  return score

def infinite_iter(data_loader):
  it = iter(data_loader)
  while True:
    try:
      ret = next(it)
      yield ret
    except StopIteration:
      it = iter(data_loader)

"""## schedule_sampling"""

def schedule_sampling(decay, total_steps, num_steps):
    if decay == "teacher":
        return 1
    elif decay == "noteacher":
        return 0
    elif decay == "linear":
        return 1 - (total_steps / num_steps)
    elif decay == "tanh":
        return 1 - math.tanh(total_steps/num_steps)
    elif decay == "exp":
        return math.exp(-4 * (total_steps / num_steps))

def train(model, optimizer, train_iter, loss_function, total_steps, summary_steps, train_dataset, decay, num_steps):
  model.train()
  model.zero_grad()
  losses = []
  loss_sum = 0.0
  for step in range(summary_steps):
    sources, targets = next(train_iter)
    sources, targets = sources.to(device), targets.to(device)
    outputs, preds = model(sources, targets, schedule_sampling(decay, total_steps + step, num_steps))
    # targets 的第一個 token 是 <BOS> 所以忽略
    outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
    targets = targets[:, 1:].reshape(-1)
    loss = loss_function(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()

    loss_sum += loss.item()
    if (step + 1) % 5 == 0:
      loss_sum = loss_sum / 5
      print ("\r", "train [{}] loss: {:.3f}, Perplexity: {:.3f}      ".format(total_steps + step + 1, loss_sum, np.exp(loss_sum)), end=" ")
      losses.append(loss_sum)
      loss_sum = 0.0

  return model, optimizer, losses

def test(model, dataloader, loss_function, beamsize=1):
  model.eval()
  loss_sum, bleu_score= 0.0, 0.0
  n = 0
  result = []
  time = 0
  for sources, targets in dataloader:
    print(time)
    time = time + 1
    sources, targets = sources.to(device), targets.to(device)
    batch_size = sources.size(0)
    outputs, preds = model.inference(sources, targets, beamsize)
    # targets 的第一個 token 是 <BOS> 所以忽略
    outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
    targets = targets[:, 1:].reshape(-1)

    loss = loss_function(outputs, targets)
    loss_sum += loss.item()

    # 將預測結果轉為文字
    targets = targets.view(sources.size(0), -1)
    preds = tokens2sentence(preds, dataloader.dataset.int2word_cn)
    sources = tokens2sentence(sources, dataloader.dataset.int2word_en)
    targets = tokens2sentence(targets, dataloader.dataset.int2word_cn)
    for source, pred, target in zip(sources, preds, targets):
      result.append((source, pred, target))
    # 計算 Bleu Score
    bleu_score += computebleu(preds, targets)

    n += batch_size

  return loss_sum / len(dataloader), bleu_score / n, result

def train_process(config):
  # 準備訓練資料
  train_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'training')
  train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
  train_iter = infinite_iter(train_loader)
  # 準備檢驗資料
  val_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'validation')
  val_loader = data.DataLoader(val_dataset, batch_size=1)
  # 建構模型
  model, optimizer = build_model(config, train_dataset.en_vocab_size, train_dataset.cn_vocab_size)
  loss_function = nn.CrossEntropyLoss(ignore_index=0)

  train_losses, val_losses, bleu_scores = [], [], []
  total_steps = 0
  while (total_steps < config.num_steps):
    # 訓練模型
    model, optimizer, loss = train(model, optimizer, train_iter, loss_function, total_steps, config.summary_steps, train_dataset, config.decay, config.num_steps)
    train_losses += loss
    # 檢驗模型
    val_loss, bleu_score, result = test(model, val_loader, loss_function, config.beamsize)
    val_losses.append(val_loss)
    bleu_scores.append(bleu_score)

    total_steps += config.summary_steps
    print ("\r", "val [{}] loss: {:.3f}, Perplexity: {:.3f}, blue score: {:.3f}       ".format(total_steps, val_loss, np.exp(val_loss), bleu_score))
    
    # 儲存模型和結果
    if total_steps % config.store_steps == 0 or total_steps >= config.num_steps:
      save_model(model, optimizer, config.store_model_path, total_steps)
      with open(f'{config.store_model_path}/output_{total_steps}.txt', 'w') as f:
        for line in result:
          print (line, file=f)
    
  return train_losses, val_losses, bleu_scores

def test_process(config):
  # 準備測試資料
  test_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'testing')
  test_loader = data.DataLoader(test_dataset, batch_size=1)
  # 建構模型
  model, optimizer = build_model(config, test_dataset.en_vocab_size, test_dataset.cn_vocab_size)
  print ("Finish build model")
  loss_function = nn.CrossEntropyLoss(ignore_index=0)
  model.eval()
  # 測試模型
  test_loss, bleu_score, result = test(model, test_loader, loss_function, config.beamsize)
  # 儲存結果
  with open(f'./test_output.txt', 'w') as f:
    for line in result:
      print (line, file=f)

  return test_loss, bleu_score

import os, sys
data_path = sys.argv[1]
output_path = sys.argv[2]

class configurations(object):
  def __init__(self):
    self.batch_size = 256
    self.emb_dim = 256
    self.hid_dim = 512
    self.n_layers = 3
    self.dropout = 0.5
    self.learning_rate = 0.00005
    self.max_output_len = 50              # 最後輸出句子的最大長度
    self.num_steps = 12000                # 總訓練次數
    self.store_steps = 2000               # 訓練多少次後須儲存模型
    self.summary_steps = 300              # 訓練多少次後須檢驗是否有overfitting
    self.load_model = False               # 是否需載入模型
    self.store_model_path = "./ckpt"      # 儲存模型的位置
    self.load_model_path = None           # 載入模型的位置 e.g. "./ckpt/model_{step}" 
    self.data_path = data_path            # 資料存放的位置
    self.attention = True                 # 是否使用 Attention Mechanism
    self.decay = 'linear'
    self.beamsize = 1

if __name__ == '__main__':
  config = configurations()
  print ('config:\n', vars(config))
  train_losses, val_losses, bleu_scores = train_process(config)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(train_losses)
plt.xlabel('次數')
plt.ylabel('loss')
plt.title('train loss')
plt.show()

import matplotlib.pyplot as plt
plt.figure()
plt.plot(val_losses)
plt.xlabel('次數')
plt.ylabel('loss')
plt.title('validation loss')
plt.show()

import matplotlib.pyplot as plt
plt.figure()
plt.plot(bleu_scores)
plt.xlabel('次數')
plt.ylabel('BLEU score')
plt.title('BLEU score')
plt.show()

