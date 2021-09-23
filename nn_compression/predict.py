import sys
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from model import StudentNet
from quantization import *

#workspace_dir = sys.argv[1]
#output_file = sys.argv[2]

def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss

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
        file_list = glob(folderName + '/*.jpg')
        file_list.sort()

        for img_path in file_list:
            try:
                class_idx = int(re.findall(re.compile(r'\d+'), img_path)[1])
            except:
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

def get_dataloader(mode='training', batch_size=32, data_path = './food-11/'):

    assert mode in ['training', 'testing', 'validation']

    dataset = MyDataset(
        data_path + f'/{mode}',
        transform=trainTransform if mode == 'training' else testTransform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'training'))

    return dataloader

import sys

data_path = sys.argv[1]
out_file = sys.argv[2]

# get dataloader
test_dataloader = get_dataloader('testing', batch_size=128, data_path=data_path)

student_net = StudentNet(base=16).cuda()

static_dict = decode8('./model.pkl')
student_net.load_state_dict(static_dict)
student_net.eval()

student_net.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        inputs, _ = data
        inputs = inputs.cuda()
        test_pred = student_net(inputs)
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

#將結果寫入 csv 檔
with open(out_file, 'w') as f:
    f.write('Id,label\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))

