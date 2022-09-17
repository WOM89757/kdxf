#安装相关依赖库 如果是windows系统，cmd命令框中输入pip安装，或在Jupyter notebook中!pip安装，参考上述环境配置
#!pip install pandas numpy cv2 torch torchvision time albumentations timm tqdm
# pip install pandas numpy  torch torchvision  albumentations timm tqdm

#---------------------------------------------------
#导入库
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


#----------------数据处理----------------
#基础数据读取
train_df = pd.read_csv('data/train.csv')
train_df['path'] = 'data/train/' + train_df['image']


test_df = pd.read_csv('data/test.csv')
test_df['path'] = 'data/test/' + test_df['image'] 

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
   
   
#使用torch批量数据读取
train_loader = torch.utils.data.DataLoader(    
    XunFeiDataset(train_df['path'].values[:-200], train_df['label'].values[:-200],           
    A.Compose([            
        A.RandomCrop(450, 750),
        ])
        ), batch_size=5, shuffle=True, num_workers=0, pin_memory=False)
    
val_loader = torch.utils.data.DataLoader(    
    XunFeiDataset(train_df['path'].values[-200:], train_df['label'].values[-200:],            
    A.Compose([            
        A.RandomCrop(450, 750),        
        ])    
        ), batch_size=2, shuffle=False, num_workers=0, pin_memory=False)    



       
#----------------模型训练----------------
       
# 模型训练一个epoch的函数
def train(train_loader, model, criterion, optimizer):   
    model.train()    
    train_loss = 0.0    
    
    for i, (input, target) in enumerate(train_loader):        
        input = input.to(device)        
        target = target.to(device)        
        output = model(input)        
        loss = criterion(output, target)        
        optimizer.zero_grad()        
        loss.backward()        
        optimizer.step()
        if i % 40 == 0:            
            print(loss.item())                    
            
        train_loss += loss.item()        
    
    return train_loss/len(train_loader)
    
# 模型验证一个epoch的函数
def validate(val_loader, model, criterion):    
    model.eval()
    val_acc = 0.0        
    
    with torch.no_grad():        
        end = time.time()        
        for i, (input, target) in enumerate(val_loader):            
            input = input.to(device)            
            target = target.to(device)            
            output = model(input)            
            # loss = criterion(output, target)
            val_acc += (output.argmax(1) == target).sum().item()                
            
        return val_acc / len(val_loader.dataset)    
        
# 模型预测函数     
def predict(test_loader, model, criterion):    
    model.eval()    
    val_acc = 0.0        
    
    test_pred = []    
    with torch.no_grad():        
        end = time.time()        
        for i, (input, target) in enumerate(test_loader):    
            input = input.to(device)            
            target = target.to(device)            
            output = model(input)            
            test_pred.append(output.data.cpu().numpy())                
            return np.vstack(test_pred)

# model_name = 'resnet50d'
epoch_num = 50
model_name = 'resnet18d'
# 定义模型，使用resnet18
print("Creating model----{}".format(model_name))
checkpoint_path='./checkpoint/{}_model_epoch_4.pkl'.format(model_name)

# model = timm.create_model(model_name, pretrained=True, num_classes=24)  # 通过修改模型名字更换不同模型  
model = timm.create_model(model_name, pretrained=True, num_classes=24, checkpoint_path=checkpoint_path)  # 通过修改模型名字更换不同模型  
model.load_state_dict(torch.load(checkpoint_path))

# if True:
#     loaded_dict = torch.load(checkpoint_path)
#     model.state_dict = loaded_dict
#     model.cuda()
# model = timm.create_model('resnet18d', pretrained=True, num_classes=24, checkpoint_path='./checkpoint/resnet18d_model_epoch_4.pkl')  # 通过修改模型名字更换不同模型  
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.0001)

# 模型训练
#resnet18大概消耗显存：3.7GB，使用Gtx1060时，训练每个epoch花费大概2min
for i  in range(epoch_num):
    print("------epoch{}----------".format(i))   
    print("-------Loss----------")      
    train_loss = train(train_loader, model, criterion, optimizer) 
    print(train_loss)

    print("-------Val acc----------") 
    val_acc = validate(val_loader, model, criterion)   
    print(val_acc)   
    if i % 10 == 0:
        save_path = "./checkpoint/{}_model_epoch_{}.pkl".format(model_name, i)
        # torch.save(model, save_path)
        torch.save(model.state_dict(), save_path)
        # torch.save(model.state_dict(),'./checkpoint/timm_model.pth')
        # model.load_state_dict(torch.load('./checkpoint/timm_model.pth'))

save_path = "./checkpoint/{}_model_epoch_{}.pkl".format(model_name, 'last')
# torch.save(model, save_path)
torch.save(model.state_dict(), save_path)

# 模型预测
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
    for input, _ in tqdm(test_loader): 

        # print(img[0])
        input = input.to(device)                      
        output = model(input)               

        test_pred.append(output.data.cpu().numpy())

pred = np.vstack(test_pred)


#----------------结果输出----------------
pd.DataFrame(    
    {        
        'image': [x.split('/')[-1] for x in test_df['path'].values],        
        'label': pred.argmax(1)
        }).to_csv('result.csv', index=None)