# -*- coding: utf-8 -*-
import os
import pandas as pd
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda")
"""# 讀取資料庫"""

unloader = transforms.ToPILImage()

output_dir = './output'

class Adverdataset(Dataset):
    def __init__(self, root, label, transforms):
        self.root = root
        self.label = torch.from_numpy(label).long()
        self.transforms = transforms
        self.fnames = []

        for i in range(200):
            self.fnames.append("{:03d}".format(i))

    def __getitem__(self, idx):
        # 利用路徑讀取圖片
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
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        transform = transforms.Compose([                
                        transforms.Resize((224, 224), interpolation=3),
                        transforms.ToTensor(),
                        self.normalize
                    ])
        self.dataset = Adverdataset('./data/images', label, transform)
        
        self.loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size = 1,
                shuffle = False)

    def fgsm_attack(self, image, epsilon, data_grad):
        sign_data_grad = data_grad.sign()
        delta = epsilon * sign_data_grad
        perturbed_image = image + delta
        return perturbed_image , delta

    def output(self, epsilon):
        index = 0
        for (data, target) in self.loader:
            data, target = data.to(device), target.to(device)
            data_raw = data;
            data.requires_grad = True
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]

            # 如果 class 錯誤 就不進行攻擊
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
                '''
                if tries == 40:
                    data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                    data_raw = data_raw.squeeze().detach().cpu()
                    adv_ex = data_raw
                    print('give up', index)
                    break
                '''

                output = self.model(data)

                loss = F.nll_loss(output, target)
                self.model.zero_grad()
                loss.backward()
                data_grad = data.grad.data

                if tries == 1:
                    new_grad = data_grad / data_grad.abs().sum()
                    epsilon = 0.0005 / new_grad.max()

                else :
                    new_grad = 0.95 * new_grad + data_grad / data_grad.abs().sum()

                delta = new_grad *  epsilon / ( tries **0.5)
                _sum = _sum + delta.max().item()

                perturbed_data = data + delta

                output = self.model(perturbed_data)
                prob = torch.nn.functional.softmax(output[0], dim=0)
                prob = prob.sort(0, descending=True)
                tmp = prob[0][:2]
                tmp1 = prob[1][:3]
                diff = tmp[0] - tmp[1]
                final_pred = output.max(1, keepdim=True)[1]

                #if final_pred.item() == target.item() or ( tries < 40 and diff < 0.10 and target.item() in tmp1) :
                #if final_pred.item() == target.item() or ( tries < 20 and tmp[0] < 0.3) :
                if final_pred.item() == target.item():
                    data = perturbed_data #data + delta
                    data = data.detach().clone()
                    data.requires_grad = True
                    tries = tries + 1
                    continue

                print('success {}, tries {}, confidence {}, sum {}'.format(index, tries, tmp[0], _sum))

                adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                adv_ex = adv_ex.squeeze().detach().cpu()
                adv_ex = torch.clamp(adv_ex , 0, 1)

                break

            img = unloader(adv_ex)
            img.save(output_dir + '/{:03d}.png'.format(index))
            index = index + 1
        print(index)
        return 


    def sample(self):
        adv_examples = []
        for (data, target) in self.loader:
            data, target = data.to(device), target.to(device)
            data_raw = data;
            data.requires_grad = True
            # 將圖片丟入 model 進行測試 得出相對應的 class
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]
            init_prob = torch.nn.functional.softmax(output[0], dim=0)
            init_prob = init_prob.sort(0, descending=True)
            init_prob = init_prob[0][0]

            # 如果 class 錯誤 就不進行攻擊
            if init_pred.item() != target.item():
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
                    epsilon = 0.0005 / new_grad.max()

                else :
                    new_grad = 0.95 * new_grad + data_grad / data_grad.abs().sum()

                delta = new_grad *  epsilon / ( tries **0.5)
                _sum = _sum + delta.max().item()

                perturbed_data = data + delta

                output = self.model(perturbed_data)
                prob = torch.nn.functional.softmax(output[0], dim=0)
                prob = prob.sort(0, descending=True)
                tmp = prob[0][:3]
                tmp1 = prob[1][:3]
                final_pred = output.max(1, keepdim=True)[1]

                if final_pred.item() == target.item():
                    data = perturbed_data
                    data = data.detach().clone()
                    data.requires_grad = True
                    tries = tries + 1
                    continue

                adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                adv_ex = adv_ex.squeeze().detach().cpu()
                adv_ex = torch.clamp(adv_ex , 0, 1)

                data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                data_raw = data_raw.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), init_prob.item(), tmp1.detach().cpu().numpy(), tmp.detach().cpu().numpy(), data_raw , adv_ex) )        

                break

            if len(adv_examples) == 5:
                break

        return adv_examples 

if __name__ == '__main__':
    df = pd.read_csv("./data/labels.csv")
    df = df.loc[:, 'TrueLabel'].to_numpy()
    label_name = pd.read_csv("./data/categories.csv")
    label_name = label_name.loc[:, 'CategoryName'].to_numpy()

    attacker = Attacker('./data/images', df)

    examples = attacker.sample()
    for ex in examples:
        ori_label = label_name[ex[0]].split(',')[0]
        ori_prob = ex[1]
        print("Origin :", ori_label, ori_prob)

        for j in range(len(ex[2])):
            post_label = label_name[ex[2][j]].split(',')[0]
            post_prob = ex[3][j]
            print("post {}:".format(j), post_label, post_prob)

        _orimg = np.transpose(ex[4], (1, 2, 0))
        _postimg = np.transpose(ex[5], (1, 2, 0))
        plt.subplot(2,1,1)
        plt.imshow(_orimg)
        plt.subplot(2,1,2)
        plt.imshow(_postimg)
        plt.show()
