# -*- coding: utf-8 -*-
"""hw5_colab.ipynb

# 安裝lime套件
# 這份作業會用到的套件大部分 colab 都有安裝了，只有 lime 需要額外安裝
!pip install lime==0.1.1.37

"""## Start our python script"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.segmentation import slic
from lime import lime_image
from pdb import set_trace

"""## Argument parsing"""

workspace_dir = sys.argv[1]
output_dir = sys.argv[2]


"""## Dataset definition and creation"""

class FoodDataset(Dataset):
    def __init__(self, paths, labels, mode):
        # mode: 'train' or 'eval'
        
        self.paths = paths
        self.labels = labels
        trainTransform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
        ])
        evalTransform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),
        ])
        self.transform = trainTransform if mode == 'train' else evalTransform
        self._trans = trainTransform

    # 這個 FoodDataset 繼承了 pytorch 的 Dataset class
    # 而 __len__ 和 __getitem__ 是定義一個 pytorch dataset 時一定要 implement 的兩個 methods
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        X = Image.open(self.paths[index])
        X = self.transform(X)
        Y = self.labels[index]
        return X, Y

    def openImg(self, index):
        X = Image.open(self.paths[index])
        Y = self.labels[index]
        return X, Y

    # 這個 method 並不是 pytorch dataset 必要，只是方便未來我們想要指定「取哪幾張圖片」出來當作一個 batch 來 visualize
    def getbatch(self, indices):
        images = []
        labels = []
        for index in indices:
          image, label = self.__getitem__(index)
          images.append(image)
          labels.append(label)
        return torch.stack(images), torch.tensor(labels)

    def getbatch_withTrans(self, indices):
        images = []
        labels = []
        for index in indices:
          image, label = self.openImg(index)
          images.append(self.transform(image))
          labels.append(label)
          for i in range(3):
            images.append(self._trans(image))
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)


# 給予 data 的路徑，回傳每一張圖片的「路徑」和「class」
def get_paths_labels(path):
    imgnames = os.listdir(path)
    imgnames.sort()
    imgpaths = []
    labels = []
    for name in imgnames:
        imgpaths.append(os.path.join(path, name))
        labels.append(int(name.split('_')[0]))
    return imgpaths, labels
train_paths, train_labels = get_paths_labels(os.path.join(workspace_dir, 'training'))

# 這邊在 initialize dataset 時只丟「路徑」和「class」，之後要從 dataset 取資料時
# dataset 的 __getitem__ method 才會動態的去 load 每個路徑對應的圖片
train_set = FoodDataset(train_paths, train_labels, mode='eval')

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #torch.nn.MaxPool2d(kernel_size, stride, padding)
        #input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]
            nn.Dropout(0.3),

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            nn.Dropout(0.3),
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
            nn.Dropout(0.3),
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 11)
        )
        print(self.cnn)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

    def hook(self, grad):
        self.camgrad = grad

    def _forward(self, x):
        out = self.cnn(x)
        h = out.register_hook(self.hook)
        out = out.view(out.size()[0], -1)
        return self.fc(out)



model = Classifier().cuda()

model.load_state_dict(torch.load('best_model'))
model.eval()

def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

def compute_saliency_maps(x, y, model):
  model.eval()
  x = x.cuda()

  x.requires_grad_()
  
  y_pred = model._forward(x)
  loss_func = torch.nn.CrossEntropyLoss()
  loss = loss_func(y_pred, y.cuda())
  loss.backward()

  saliencies = x.grad.abs().detach().cpu()
  saliencies = torch.stack([normalize(item) for item in saliencies])
  return saliencies

'''
cam = None
def compute_gradcam(x, y, model, target):
  model.eval()

  def hook(model, input, output):
    global cam
    cam = output

  hook_handle = model.cnn[22].register_forward_hook(hook)

  y = model._forward(x.cuda())

  objective = y[:, target]
  objective.backward()

  pooled_gradient = torch.mean(model.camgrad, dim=[0,2,3])

  activations = cam.detach().cpu()
  for i in range(512):
      activations[:, i, :, :] *= pooled_gradient[i]
      
  heatmap = torch.mean(activations, dim=1).squeeze()
  
  heatmap = np.maximum(heatmap, 0)
  
  # normalize the heatmap
  heatmap /= torch.max(heatmap)
  print(heatmap)

  plt.matshow(heatmap.squeeze())
  plt.show()

  hook_handle.remove()

  return 

img_indices = [2000]
images, labels = train_set.getbatch(img_indices)
compute_gradcam(images, labels, model, 0)
'''

# 指定想要一起 visualize 的圖片 indices
img_indices = [0, 2000, 3000, 4000, 5000, 6000, 7000, 7400, 7700, 9700, 1000]

images, labels = train_set.getbatch(img_indices)
saliencies = compute_saliency_maps(images, labels, model)

# 使用 matplotlib 畫出來
fig, axs = plt.subplots(2, len(img_indices), figsize=(20, 10))
for row, target in enumerate([images, saliencies]):
  for column, img in enumerate(target):
    axs[row][column].imshow(img.permute(1, 2, 0).numpy())
    axs[row][column].axis('off')

plt.savefig(output_dir + '/p1.png')

layer_activations = None
def filter_explaination(x, model, cnnid, filterid, iteration=100, lr=1):
  # x: 要用來觀察哪些位置可以 activate 被指定 filter 的圖片們
  # cnnid, filterid: 想要指定第幾層 cnn 中第幾個 filter
  model.eval()

  def hook(model, input, output):
    global layer_activations
    layer_activations = output
  
  hook_handle = model.cnn[cnnid].register_forward_hook(hook)

  # Filter activation: 我們先觀察 x 經過被指定 filter 的 activation map
  model(x.cuda())
  # 這行才是正式執行 forward，因為我們只在意 activation map，所以這邊不需要把 loss 存起來
  filter_activations = layer_activations[:, filterid, :, :].detach().cpu()
  
  # 根據 function argument 指定的 filterid 把特定 filter 的 activation map 取出來
  # 因為目前這個 activation map 我們只是要把他畫出來，所以可以直接 detach from graph 並存成 cpu tensor
  
  # Filter visualization: 接著我們要找出可以最大程度 activate 該 filter 的圖片
  x = x.cuda()
  # 從一張 random noise 的圖片開始找 (也可以從一張 dataset image 開始找)
  x.requires_grad_()
  # 我們要對 input image 算偏微分
  optimizer = Adam([x], lr=lr)
  # 利用偏微分和 optimizer，逐步修改 input image 來讓 filter activation 越來越大
  for iter in range(iteration):
    optimizer.zero_grad()
    model(x)
    
    objective = -layer_activations[:, filterid, :, :].sum()
    # 與上一個作業不同的是，我們並不想知道 image 的微量變化會怎樣影響 final loss
    # 我們想知道的是，image 的微量變化會怎樣影響 activation 的程度
    # 因此 objective 是 filter activation 的加總，然後加負號代表我們想要做 maximization
    
    objective.backward()
    # 計算 filter activation 對 input image 的偏微分
    optimizer.step()
    # 修改 input image 來最大化 filter activation
  filter_visualization = x.detach().cpu().squeeze()[0]
  # 完成圖片修改，只剩下要畫出來，因此可以直接 detach 並轉成 cpu tensor

  hook_handle.remove()
  # 很重要：一旦對 model register hook，該 hook 就一直存在。如果之後繼續 register 更多 hook
  # 那 model 一次 forward 要做的事情就越來越多，甚至其行為模式會超出你預期 (因為你忘記哪邊有用不到的 hook 了)
  # 因此事情做完了之後，就把這個 hook 拿掉，下次想要再做事時再 register 就好了。

  return filter_activations, filter_visualization

img_indices = [0, 2000, 3000, 4000, 5000, 6000, 7000, 7400, 7700, 9700, 1000]
images, labels = train_set.getbatch(img_indices)
filter_activations, filter_visualization = filter_explaination(images, model, cnnid=7, filterid=0, iteration=100, lr=0.1)

# 畫出 filter visualization
fig = plt.figure(0)
plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
plt.savefig(output_dir + '/p2_1.png')
plt.close(0)

# 畫出 filter activations
fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
for i, img in enumerate(images):
  axs[0][i].imshow(img.permute(1, 2, 0))
  axs[0][i].axis('off')
for i, img in enumerate(filter_activations):
  axs[1][i].imshow(normalize(img))
  axs[1][i].axis('off')
plt.savefig(output_dir + '/p2_2.png')

"""## Lime
Lime 的部分因為有現成的套件可以使用，因此下方直接 demo 如何使用該套件。其實非常的簡單，只需要 implement 兩個 function 即可。
"""

def predict(input):
    # input: numpy array, (batches, height, width, channels)
    
    model.eval()
    input = torch.FloatTensor(input).permute(0, 3, 1, 2)
    # 需要先將 input 轉成 pytorch tensor，且符合 pytorch 習慣的 dimension 定義
    # 也就是 (batches, channels, height, width)

    output = model(input.cuda())
    return output.detach().cpu().numpy()

def segmentation(input):
    # 利用 skimage 提供的 segmentation 將圖片分成 100 塊
    return slic(input, n_segments=100, compactness=1, sigma=1)

img_indices = [0, 2000, 3000, 4000, 5000, 6000, 7000, 7400, 7700, 9700, 1000]

images, labels = train_set.getbatch(img_indices)
fig, axs = plt.subplots(1, 11, figsize=(15, 8))
np.random.seed(16)                                                                                                                                                       
# 讓實驗 reproducible
for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
    x = image.astype(np.double)
    explainer = lime_image.LimeImageExplainer() 
    explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation, top_labels=11)

    lime_img, mask = explaination.get_image_and_mask(label=label.item(),positive_only=False,hide_rest=False,num_features=11,min_weight=0.05)

    axs[idx].imshow(lime_img)
    axs[idx].axis('off')

plt.savefig(output_dir + '/p3.png')

img_indices = [0, 2000, 3000, 4000, 5000, 6000, 7000, 7400, 7700, 9700, 1000]
images, labels = train_set.getbatch(img_indices)

fig, axs = plt.subplots(6, len(img_indices), figsize=(15, 8))
for i, img in enumerate(images):
  axs[0][i].imshow(img.permute(1, 2, 0))
  axs[0][i].axis('off')

for layer in range(5):
    filter_activations, filter_visualization = filter_explaination(images, model, cnnid=layer*5, filterid=0, iteration=100, lr=0.1)
    fig = plt.figure(0)
    plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
    plt.savefig(output_dir + '/layer{}.png'.format(layer))
    plt.close(0)

    for i, img in enumerate(filter_activations):
      axs[layer+1][i].imshow(normalize(img))
      axs[layer+1][i].axis('off')

plt.savefig(output_dir + '/p4_1.png')

img_indices = [2000, 3000]
images, labels = train_set.getbatch_withTrans(img_indices)

fig, axs = plt.subplots(7, len(img_indices)*4, figsize=(15, 8))
for i, img in enumerate(images):
  axs[0][i].imshow(img.permute(1, 2, 0))
  axs[0][i].axis('off')

for layer in range(5):
    filter_activations, filter_visualization = filter_explaination(images, model, cnnid=layer*5, filterid=0, iteration=100, lr=0.1)

    for i, img in enumerate(filter_activations):
      axs[layer+1][i].imshow(normalize(img))
      axs[layer+1][i].axis('off')

for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
    x = image.astype(np.double)
    explainer = lime_image.LimeImageExplainer()
    explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation, top_labels=11)

    lime_img, mask = explaination.get_image_and_mask(label=label.item(),positive_only=False,hide_rest=False,num_features=11,min_weight=0.05)

    axs[6][idx].imshow(lime_img)
    axs[6][idx].axis('off')

plt.savefig(output_dir + '/p4_2.png')

img_indices = [7010, 7001, 7018, 7013, 7014, 7015, 7016]
images, labels = train_set.getbatch(img_indices)

fig, axs = plt.subplots(7, len(img_indices), figsize=(15, 8))
for i, img in enumerate(images):
  axs[0][i].imshow(img.permute(1, 2, 0))
  axs[0][i].axis('off')

for layer in range(5):
    filter_activations, filter_visualization = filter_explaination(images, model, cnnid=layer*5, filterid=0, iteration=100, lr=0.1)

    for i, img in enumerate(filter_activations):
      axs[layer+1][i].imshow(normalize(img))
      axs[layer+1][i].axis('off')

for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
    x = image.astype(np.double)
    explainer = lime_image.LimeImageExplainer()
    explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation, top_labels=11)

    lime_img, mask = explaination.get_image_and_mask(label=label.item(),positive_only=False,hide_rest=False,num_features=11,min_weight=0.05)

    axs[6][idx].imshow(lime_img)
    axs[6][idx].axis('off')

plt.savefig(output_dir + '/p4_3.png')


img_indices = [0, 2000, 3000, 4000, 5000, 6000, 7000, 7400, 7700, 9700, 1000]
images, labels = train_set.getbatch(img_indices)

model.load_state_dict(torch.load('model_10'))
model.eval()

fig, axs = plt.subplots(6, len(img_indices), figsize=(15, 8))
for i, img in enumerate(images):
  axs[0][i].imshow(img.permute(1, 2, 0))
  axs[0][i].axis('off')

for layer in range(5):
    filter_activations, filter_visualization = filter_explaination(images, model, cnnid=layer*5, filterid=0, iteration=100, lr=0.1)

    for i, img in enumerate(filter_activations):
      axs[layer+1][i].imshow(normalize(img))
      axs[layer+1][i].axis('off')

plt.savefig(output_dir + '/p4_1_10.png')


img_indices = [0, 2000, 3000, 4000, 5000, 6000, 7000, 7400, 7700, 9700, 1000]
images, labels = train_set.getbatch(img_indices)

model.load_state_dict(torch.load('model_20'))
model.eval()

fig, axs = plt.subplots(6, len(img_indices), figsize=(15, 8))
for i, img in enumerate(images):
  axs[0][i].imshow(img.permute(1, 2, 0))
  axs[0][i].axis('off')

for layer in range(5):
    filter_activations, filter_visualization = filter_explaination(images, model, cnnid=layer*5, filterid=0, iteration=100, lr=0.1)

    for i, img in enumerate(filter_activations):
      axs[layer+1][i].imshow(normalize(img))
      axs[layer+1][i].axis('off')

plt.savefig(output_dir + '/p4_1_20.png')

