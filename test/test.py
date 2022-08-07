import cv2
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import models

import albumentations as A
import timm

#----------------框架设置----------------
#设置torch使用gpu
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#输出cuda说明使用gpu，输出cpu说明使用cpu，最好使用gpu训练
print(device)

test_df = pd.read_csv('data/test.csv')
test_df['path'] = 'data/test/' + test_df['image'] 

print("Creating model test ----{}".format('resnet18d'))
model = timm.create_model('resnet18d', pretrained=True, num_classes=24)  # 通过修改模型名字更换不同模型  
checkpoint_path='./checkpoint/resnet18d_model_epoch_4.pkl'
loaded_dict = torch.load(checkpoint_path)
model.state_dict = loaded_dict
model = model.to(device)

# 定义数据集读取方法
class XunFeiDataset(Dataset):
    def __init__(self, img_path, label, transform=None):        
        self.img_path = img_path        
        self.label = label        
        if transform is not None:           
            self.transform = transform        
        else:            
            self.transform = None        
    def __getitem__(self, index):        
        img = cv2.imread(self.img_path[index])                    
        img = img.astype(np.float32)                
        img /= 255.0        
        img -= 1                
        
        if self.transform is not None:            
            img = self.transform(image = img)['image']        
        img = img.transpose([2,0,1])                
        
        return img,torch.from_numpy(np.array(self.label[index]))        
    
    def __len__(self):        
        return len(self.img_path)

# 模型预测

test_loader = torch.utils.data.DataLoader(    
    XunFeiDataset(train_df['path'].values[-200:], train_df['label'].values[-200:],            
    A.Compose([            
        A.RandomCrop(450, 750),        
        ])    
        ), batch_size=2, shuffle=False, num_workers=0, pin_memory=False)    


test_loader = torch.utils.data.DataLoader(    
    XunFeiDataset(test_df['path'].values, [0] * test_df.shape[0],            
        A.Compose([            
            A.RandomCrop(450, 750),        
            ])    
            ), batch_size=2, shuffle=False, num_workers=0, pin_memory=False)

model.eval()    
val_acc = 0.0        

test_pred = []    
with torch.no_grad():  
    j = 0
    # for input, _ in tqdm(test_loader):
    for i, (input, target) in enumerate(test_loader):            
        target = target.to(device)            

        # print(img[0])
        input = input.to(device)                      
        output = model(input)      
        val_acc += (output.argmax(1) == target).sum().item()                

        test_pred.append(output.data.cpu().numpy())     
        j = j+1
print(val_acc / j)

pred = np.vstack(test_pred)


#----------------结果输出----------------
pd.DataFrame(    
    {        
        'image': [x.split('/')[-1] for x in test_df['path'].values],        
        'label': pred.argmax(1)
        }).to_csv('result.csv', index=None)