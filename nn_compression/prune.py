# -*- coding: utf-8 -*-
# Commented out IPython magic to ensure Python compatibility.
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from model import StudentNet

"""Network Pruning
===
在這裡我們會教Neuron Pruning。
<img src="https://i.imgur.com/Iwp90Wp.png" width="500px">

簡單上來說就是讓一個已經學完的model中的neuron進行刪減，讓整個網路變得更瘦。

## Weight & Neuron Pruning
* weight和neuron pruning差別在於prune掉一個neuron就等於是把一個matrix的整個column全部砍掉。但如此一來速度就會比較快。因為neuron pruning後matrix整體變小，但weight pruning大小不變，只是有很多空洞。

## What to Prune?
* 既然要Neuron Pruning，那就必須要先衡量Neuron的重要性。衡量完所有的Neuron後，就可以把比較不重要的Neuron刪減掉。
* 在這裡我們介紹一個很簡單可以衡量Neuron重要性的方法 - 就是看batchnorm layer的$\gamma$因子來決定neuron的重要性。 (by paper - Network Slimming)
  ![](https://i.imgur.com/JVpCm2r.png)
* 相信大家看這個pytorch提供的batchnorm公式應該就可以意識到為甚麼$\gamma$可以當作重要性來衡量了:)

* Network Slimming其實步驟沒有這麼簡單，有興趣的同學可以check以下連結。[Netowrk Slimming](https://arxiv.org/abs/1708.06519)


## 為甚麼這會 work?
* 樹多必有枯枝，有些neuron只是在躺分，所以有他沒他沒有差。
* 困難的說可以回想起老師說過的大樂透假說(The Lottery Ticket Hypothesis)就可以知道了。

## 要怎麼實作?
* 為了避免複雜的操作，我們會將StudentNet(width_mult=$\alpha$)的neuron經過篩選後移植到StudentNet(width_mult=$\beta$)。($\alpha > \beta$)
* 篩選的方法也很簡單，只需要抓出每一個block的batchnorm的$\gamma$即可。

## 一些實作細節
* 假設model中間兩層是這樣的:

|Layer|Output # of Channels|
|-|-|
|Input|in_chs|
|Depthwise(in_chs)|in_chs|
|BatchNorm(in_chs)|in_chs|
|Pointwise(in_chs, **mid_chs**)|**mid_chs**|
|**Depthwise(mid_chs)**|**mid_chs**|
|**BatchNorm(mid_chs)**|**mid_chs**|
|Pointwise(**mid_chs**, out_chs)|out_chs|

則你會發現利用第二個BatchNorm來做篩選的時候，跟他的Neuron有直接關係的是該層的Depthwise&Pointwise以及上層的Pointwise。
因此再做neuron篩選時記得要將這四個(包括自己, bn)也要同時prune掉。

* 在Design Architecure內，model的一個block，名稱所對應的Weight；

|#|name|meaning|code|weight shape|
|-|-|-|-|-|
|0|cnn.{i}.0|Depthwise Convolution Layer|nn.Conv2d(x, x, 3, 1, 1, group=x)|(x, 1, 3, 3)|
|1|cnn.{i}.1|Batch Normalization|nn.BatchNorm2d(x)|(x)|
|2||ReLU6|nn.ReLU6||
|3|cnn.{i}.3|Pointwise Convolution Layer|nn.Conv2d(x, y, 1),|(y, x, 1, 1)|
|4||MaxPooling|nn.MaxPool2d(2, 2, 0)||
"""

def network_slimming(old_model, new_model):
    params = old_model.state_dict()
    new_params = new_model.state_dict()
    print("new model para :", len(new_params))
    print("old model para :", len(params))
    
    # selected_idx: 每一層所選擇的neuron index
    selected_idx = []
    # 我們總共有7層CNN，因此逐一抓取選擇的neuron index們。
    for i in range(8):
        # 根據上表，我們要抓的gamma係數在cnn.{i}.1.weight內。
        importance = params[f'cnn.{i}.1.weight']
        # 抓取總共要篩選幾個neuron。
        old_dim = len(importance)
        new_dim = len(new_params[f'cnn.{i}.1.weight'])
        # 以Ranking做Index排序，較大的會在前面(descending=True)。
        ranking = torch.argsort(importance, descending=True)
        # 把篩選結果放入selected_idx中。
        selected_idx.append(ranking[:new_dim])

    now_processed = 1
    for (name, p1), (name2, p2) in zip(params.items(), new_params.items()):
        # 如果是cnn層，則移植參數。
        # 如果是FC層，或是該參數只有一個數字(例如batchnorm的tracenum等等資訊)，那麼就直接複製。
        if name.startswith('cnn') and p1.size() != torch.Size([]) and now_processed != len(selected_idx):
            # 當處理到Pointwise的weight時，讓now_processed+1，表示該層的移植已經完成。
            if name.startswith(f'cnn.{now_processed}.3'):
                now_processed += 1

            # 如果是pointwise，weight會被上一層的pruning和下一層的pruning所影響，因此需要特判。
            if name.endswith('3.weight'):
                # 如果是最後一層cnn，則輸出的neuron不需要prune掉。
                if len(selected_idx) == now_processed:
                    new_params[name] = p1[:,selected_idx[now_processed-1]]
                # 反之，就依照上層和下層所選擇的index進行移植。
                # 這裡需要注意的是Conv2d(x,y,1)的weight shape是(y,x,1,1)，順序是反的。
                else:
                    new_params[name] = p1[selected_idx[now_processed]][:,selected_idx[now_processed-1]]
            else:
                new_params[name] = p1[selected_idx[now_processed]]
        else:
            new_params[name] = p1

    # 讓新model load進被我們篩選過的parameters，並回傳new_model。        
    new_model.load_state_dict(new_params)
    return new_model

import re
import torch
from glob import glob
from PIL import Image
import torchvision.transforms as transforms

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, folderName, transform=None):
        self.transform = transform
        self.data = []
        self.label = []

        for img_path in glob(folderName + '/*.jpg'):
            try:
                # Get classIdx by parsing image path
                class_idx = int(re.findall(re.compile(r'\d+'), img_path)[1])
            except:
                # if inference mode (there's no answer), class_idx default 0
                class_idx = 0
 
            image = Image.open(img_path)
            # Get File Descriptor
            image_fp = image.fp
            image.load()
            # Close File Descriptor (or it'll reach OPEN_MAX)
            image_fp.close()

            self.data.append(image)
            self.label.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.label[idx]


trainTransform = transforms.Compose([
    transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
testTransform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

def get_dataloader(mode='training', batch_size=32):

    assert mode in ['training', 'testing', 'validation']

    dataset = MyDataset(
        f'./food-11/{mode}',
        transform=trainTransform if mode == 'training' else testTransform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'training'))

    return dataloader

# get dataloader
train_dataloader = get_dataloader('training', batch_size=32)
valid_dataloader = get_dataloader('validation', batch_size=32)

net = StudentNet().cuda()
net.load_state_dict(torch.load('./model/student_custom_small.bin'))

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=1e-3)

"""# Start Training
* 每次Prune rate是0.95，Prune完後會重新fine-tune 3 epochs。
* 其餘的步驟與你在做Hw3 - CNN的時候一樣。
"""

def run_epoch(dataloader, update=True, alpha=0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        # 清空 optimizer
        optimizer.zero_grad()
        # 處理 input
        inputs, labels = batch_data
        inputs = inputs.cuda()
        labels = labels.cuda()
  
        logits = net(inputs)
        loss = criterion(logits, labels)
        if update:
            loss.backward()
            optimizer.step()

        total_hit += torch.sum(torch.argmax(logits, dim=1) == labels).item()
        total_num += len(inputs)
        total_loss += loss.item() * len(inputs)

    return total_loss / total_num, total_hit / total_num

now_width_mult = 1
for i in range(5):
    now_width_mult *= 0.95
    new_net = StudentNet(width_mult=now_width_mult).cuda()
    params = net.state_dict()
    net = network_slimming(net, new_net)
    now_best_acc = 0
    for epoch in range(5):
        net.train()
        train_loss, train_acc = run_epoch(train_dataloader, update=True)
        net.eval()
        valid_loss, valid_acc = run_epoch(valid_dataloader, update=False)
        # 在每個width_mult的情況下，存下最好的model。
        if valid_acc > now_best_acc:
            now_best_acc = valid_acc
            torch.save(net.state_dict(), f'custom_small_rate_{now_width_mult}.bin')
        print('rate {:6.4f} epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(now_width_mult, 
            epoch, train_loss, train_acc, valid_loss, valid_acc))

