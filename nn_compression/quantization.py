# -*- coding: utf-8 -*-
"""hw7_Weight_Quantization.ipynb
# Readme


HW7的任務是模型壓縮 - Neural Network Compression。

Compression有很多種門派，在這裡我們會介紹上課出現過的其中四種，分別是:
## Weight Quantization

我們這邊會示範如何實作第一條: Using less bits to represent a value。

## 好的Quantization很重要。
這邊提供一些TA的數據供各位參考。

|bit|state_dict size|accuracy|
|-|-|-|
|32|1047430 Bytes|0.81315|
|16|522958 Bytes|0.81347|
|8|268472 Bytes|0.80791|
|7|268472 Bytes|0.80791|

## Byte Cost
根據[torch的官方手冊](https://pytorch.org/docs/stable/tensors.html)，我們知道torch.FloatTensor預設是32-bit，也就是佔了4byte的空間，而FloatTensor系列最低可以容忍的是16-bit。

為了方便操作，我們之後會將state_dict轉成numpy array做事。
因此我們可以先看看numpy有甚麼樣的type可以使用。
![](https://i.imgur.com/3N7tiEc.png)
而我們發現numpy最低有float16可以使用，因此我們可以直接靠轉型將32-bit的tensor轉換成16-bit的ndarray存起來。

# Read state_dict

下載我們已經train好的小model的state_dict進行測試。
"""
import os
import torch

"""# 32-bit Tensor -> 16-bit"""

import numpy as np
import pickle

def encode16(params, fname):
    '''將params壓縮成16-bit後輸出到fname。

    Args:
      params: model的state_dict。
      fname: 壓縮後輸出的檔名。
    '''

    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        # 有些東西不屬於ndarray，只是一個數字，這個時候我們就不用壓縮。
        if type(param) == np.ndarray:
            custom_dict[name] = np.float16(param)
        else:
            custom_dict[name] = param

    pickle.dump(custom_dict, open(fname, 'wb'))


def decode16(fname):
    '''從fname讀取各個params，將其從16-bit還原回torch.tensor後存進state_dict內。

    Args:
      fname: 壓縮後的檔名。
    '''

    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        param = torch.tensor(param)
        custom_dict[name] = param

    return custom_dict

"""# 32-bit Tensor -> 8-bit (OPTIONAL)

這邊提供轉成8-bit的方法，僅供大家參考。
因為沒有8-bit的float，所以我們先對每個weight記錄最小值和最大值，進行min-max正規化後乘上$2^8-1$在四捨五入，就可以用np.uint8存取了。

$W' = round(\frac{W - \min(W)}{\max(W) - \min(W)} \times (2^8 - 1)$)

> 至於能不能轉成更低的形式，例如4-bit呢? 當然可以，待你實作。
"""

def encode8(params, fname):
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        if type(param) == np.ndarray:
            min_val = np.min(param)
            max_val = np.max(param)
            param = np.round((param - min_val) / (max_val - min_val) * 255)
            param = np.uint8(param)
            custom_dict[name] = (min_val, max_val, param)
        else:
            custom_dict[name] = param

    pickle.dump(custom_dict, open(fname, 'wb'))


def decode8(fname):
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        if type(param) == tuple:
            min_val, max_val, param = param
            param = np.float64(param)
            param = (param / 255 * (max_val - min_val)) + min_val
            param = torch.tensor(param)
        else:
            param = torch.tensor(param)

        custom_dict[name] = param

    return custom_dict

def truncate8(model):
    params = model.state_dict()
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        if type(param) == np.ndarray:
            min_val = np.min(param)
            max_val = np.max(param)
            param = np.round((param - min_val) / (max_val - min_val) * 255)
            param = np.uint8(param)
            custom_dict[name] = (min_val, max_val, param)
        else:
            custom_dict[name] = param

    trun_dict = {}
    for (name, param) in custom_dict.items():
        if type(param) == tuple:
            min_val, max_val, param = param
            param = np.float64(param)
            param = (param / 255 * (max_val - min_val)) + min_val
            param = torch.tensor(param)
        else:
            param = torch.tensor(param)

        trun_dict[name] = param

    model.load_state_dict(trun_dict)
    return 
