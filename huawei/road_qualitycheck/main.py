#!pip install pandas numpy cv2 torch torchvision codecs PIL glob
import os
import glob
from tkinter import W
from PIL import Image
import csv, time
import numpy as np

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data as data

import pandas as pd
import codecs

class ImageSet(data.Dataset):
    def __init__(
            self,
            images,
            labels,
            transform):
        self.transform = transform
        self.images = images
        self.labels = labels

    def __getitem__(self, item):
        imagename = self.images[item]
        
        try:
            image = Image.open(imagename)
            image = image.convert('RGB')
        except:
            image = Image.fromarray(np.zeros((256, 256), dtype=np.int8))
            image = image.convert('RGB')

        image = self.transform(image)
        return image, torch.tensor(self.labels[item])

    def __len__(self):
        return len(self.images)


data_root = '/home/wangmao/code/kdxf/data/digix-2022-cv-sample-0829/'

lines = codecs.open(data_root + 'train_label.csv').readlines()
train_label = pd.DataFrame({
    'image': [data_root + 'train_image/' + x.strip().split(',')[0] for x in lines],
    'label': [x.strip().split(',')[1:] for x in lines],
})

train_label['new_label'] = train_label['label'].apply(lambda x: int('0' in x))

def check_image(path):
    try:
        if os.path.exists(path):
            return True
        else:
            return False
    except:
        return False
train_is_valid = train_label['image'].apply(lambda x: check_image(x) )
train_label = train_label[train_is_valid]

trfs = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageSet(train_label['image'].values[:1000],
                         train_label['new_label'].values[:1000],
                         trfs)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
)

test_images = glob.glob(data_root + 'test_images/*')
test_dataset = ImageSet(test_images, [0] * len(test_images), trfs)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=5,
    pin_memory=True,
)

for data in train_loader:
    break
    
for data in test_loader:
    break

model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, 2)
model = model.to('cuda')

optimizer = optim.SGD(model.parameters(), lr=0.001)

loss = nn.CrossEntropyLoss()

epochs = 3
for epoch in range(epochs):
    start_t = time.time()
    epoch_l = 0
    epoch_t = 0
    
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        image, label = batch
        image, label = image.to('cuda'), label.to('cuda')
        output = model(image)

        l = loss(output, label)
        l.backward()
        optimizer.step()

        batch_l = l.item()
        epoch_l += batch_l
        batch_t = time.time() - start_t
        epoch_t += batch_t
        start_t = time.time()
        
        if batch_idx % 10 == 0:
            print(l.item(), batch_idx, len(train_loader))

    epoch_t = epoch_t / len(train_loader)
    epoch_l = epoch_l / len(train_loader)
    print('...epoch: {:3d}/{:3d}, loss: {:.4f}, average time: {:.2f}.'.format(
        epoch + 1, epochs, epoch_l, epoch_t))

model.eval()
to_prob = nn.Softmax(dim=1)
with torch.no_grad():
    imagenames, probs = list(), list()
    for batch_idx, batch in enumerate(test_loader):
        image, _ = batch
        image = image.to('cuda')
        pred = model(image)
        prob = to_prob(pred)
        prob = list(prob.data.cpu().numpy())
        probs += prob

import csv
with open('submission.csv', 'w',newline = '', encoding='utf8') as fp:
    writer = csv.writer(fp)
    writer.writerow(['imagename', 'defect_prob'])
    for imagename, prob in zip(test_images, probs):
        imagename = os.path.basename(imagename)
        writer.writerow([imagename, str(prob[0])])
