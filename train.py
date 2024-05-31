import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models

import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd

import time
from VGG import VGG16,VGGBlock
# from ResNet import *

# pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
# conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
print(torch.__version__) #检查torch版本
print(torch.cuda.device_count()) #Gpu数量
print(torch.version.cuda) #检查cuda版本
print(torch.cuda.is_available()) #检查cuda是否可用

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else: device = torch.device("cpu")
print(device)


def train(loaders, optimizer, criterion, epochs=10, save_param=True, dataset="mnist"):
    global device
    global model
    try:
        model = model.to(device)
        history_loss = {"train": [], "test": []}
        history_accuracy = {"train": [], "test": []}
        best_test_accuracy = 0
        
        start_time = time.time()

        for epoch in range(epochs):
            sum_loss = {"train": 0, "test": 0}
            sum_accuracy = {"train": 0, "test": 0}

            for split in ["train", "test"]:
                if split == "train":
                    model.train()
                else:
                    model.eval()
                
                for (inputs, labels) in loaders[split]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    prediction = model(inputs)
                    labels = labels.long()
                    loss = criterion(prediction, labels)
                    sum_loss[split] += loss.item()  # 更新loss

                    if split == "train":
                        loss.backward()  # 计算梯度
                        optimizer.step()
                    
                    _,pred_label = torch.max(prediction, dim = 1)
                    pred_labels = (pred_label == labels).float()

                    batch_accuracy = pred_labels.sum().item() / inputs.size(0)
                    sum_accuracy[split] += batch_accuracy  # 更新accuracy
                    
            # 计算批次的loss/accuracy
            epoch_loss = {split: sum_loss[split] / len(loaders[split]) for split in ["train", "test"]}
            epoch_accuracy = {split: sum_accuracy[split] / len(loaders[split]) for split in ["train", "test"]}

            # 以最佳测试精度存储参数
            if save_param and epoch_accuracy["test"] > best_test_accuracy:
                torch.save(model.state_dict(), f"./vgg16_{dataset}_model.pth")
                best_test_accuracy = epoch_accuracy["test"]

            # 更新历史
            for split in ["train", "test"]:
                history_loss[split].append(epoch_loss[split])
                history_accuracy[split].append(epoch_accuracy[split])
                
            print(f"Epoch: [{epoch + 1} | {epochs}]\nTrain Loss: {epoch_loss['train']:.4f}, Train Accuracy: {epoch_accuracy['train']:.2f}, \
            Test Loss: {epoch_loss['test']:.4f}, Test Accuracy: {epoch_accuracy['test']:.2f}, Time Taken: {(time.time() - start_time) / 60:.2f} mins")
    
    except KeyboardInterrupt: # 用户键盘中断异常
        print("Interrupted")
    
    finally: # 绘制图表
        plt.show()
        plt.title("Loss")
        for split in ["train", "test"]:
            plt.plot(history_loss[split], label=split)
        plt.legend()
        plt.show()

        plt.title("Accuracy")
        for split in ["train", "test"]:
            plt.plot(history_accuracy[split], label=split)
        plt.legend()
        plt.show()



# main
model = VGG16((1,32,32), batch_norm=True)
# optimizer = optim.SGD(model.parameters(), lr=0.001) # 随机梯度下降（SGD）
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=5E-5)
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数


transform = transforms.Compose([
  transforms.Resize(32),
  transforms.ToTensor(),
])
# 加载数据集
train_set = torchvision.datasets.MNIST(root='', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='', train=False, download=True, transform=transform)
# 查看数据集信息
print(f"Number of training samples: {len(train_set)}")
print(f"Number of test samples: {len(test_set)}")

# 提取数据标签
x_train, y_train = train_set.data, train_set.targets
print(x_train, y_train)
# 如果训练集的图像数据的维度是3，则添加一个维度，使其变为B*C*H*W的格式
if len(x_train.shape) == 3:
      x_train = x_train.unsqueeze(1)
print(x_train.shape)

# 制作 40 张图像的网格，每行 8 张图像
x_grid = torchvision.utils.make_grid(x_train[:40], nrow=8, padding=2)
print(x_grid.shape)
# 将 tensor 转换为 numpy 数组
npimg = x_grid.numpy()
# 转换为 H*W*C 形状
npimg_tr = np.transpose(npimg, (1, 2, 0))
plt.imshow(npimg_tr, interpolation='nearest')

image, label = train_set[200]
plt.imshow(image.squeeze(), cmap='gray')
print('Label:', label)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
loaders = {"train": train_loader,
           "test": test_loader}


train(loaders, optimizer, criterion, epochs=10)  
