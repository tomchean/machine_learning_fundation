# -*- coding: utf-8 -*-
import os
import pandas as pd
from PIL import Image
import numpy as np
import sys
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda")
unloader = transforms.ToPILImage()

data_dir = sys.argv[1]
output_dir = sys.argv[2]

class Adverdataset(Dataset):
    def __init__(self, root, label, transforms):
        self.root = root
        self.label = torch.from_numpy(label).long()
        self.transforms = transforms
        self.fnames = []

        for i in range(200):
            self.fnames.append("{:03d}".format(i))

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.fnames[idx] + '.png'))
        img = self.transforms(img)
        label = self.label[idx]
        return img, label
    
    def __len__(self):
        return 200

class Attacker:
    def __init__(self, img_dir, label):
        self.model = models.densenet121(pretrained = True)
        self.model.cuda()
        self.model.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # 把圖片 normalize 到 0~1 之間 mean 0 variance 1
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        transform = transforms.Compose([                
                        transforms.Resize((224, 224), interpolation=3),
                        transforms.ToTensor(),
                        self.normalize
                    ])
        self.dataset = Adverdataset(img_dir, label, transform)
        
        self.loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size = 1,
                shuffle = False)

    def output(self, epsilon):
        index = 0
        for (data, target) in self.loader:
            data, target = data.to(device), target.to(device)
            data_raw = data;
            data.requires_grad = True
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]

            if init_pred.item() != target.item():
                data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                data_raw = data_raw.squeeze().detach().cpu()
                img = unloader(data_raw)
                img.save(output_dir + '/{:03d}.png'.format(index))
                index = index + 1
                continue

            tries = 1
            _sum = 0
            while True:
                output = self.model(data)
                loss = F.nll_loss(output, target)
                self.model.zero_grad()
                loss.backward()
                data_grad = data.grad.data

                if tries == 1:
                    new_grad = data_grad / data_grad.abs().sum()
                    epsilon = 0.03 / new_grad.max()
                else :
                    new_grad = 0.95 * new_grad + data_grad / data_grad.abs().sum()

                delta = new_grad * epsilon / tries
                _sum = _sum + delta.max().item()

                perturbed_data = data + delta

                output = self.model(perturbed_data)
                final_pred = output.max(1, keepdim=True)[1]

                if final_pred.item() == target.item():
                    data = perturbed_data
                    data = data.detach().clone()
                    data.requires_grad = True
                    tries = tries + 1
                    continue

                print('success {}, tries {}, sum {}'.format(index, tries, _sum))

                adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                adv_ex = adv_ex.squeeze().detach().cpu()
                adv_ex = torch.clamp(adv_ex , 0, 1)
                break

            img = unloader(adv_ex)
            img.save(output_dir + '/{:03d}.png'.format(index))
            index = index + 1
        print(index)
        return 

if __name__ == '__main__':
    df = pd.read_csv(data_dir + "/labels.csv")
    df = df.loc[:, 'TrueLabel'].to_numpy()

    label_name = pd.read_csv(data_dir + "/categories.csv")
    label_name = label_name.loc[:, 'CategoryName'].to_numpy()

    attacker = Attacker(data_dir + '/images', df)
    attacker.output(0.01)
