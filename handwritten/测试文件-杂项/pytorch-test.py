# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms    # datasets包含常用的数据集，transform 对图像进行预处理


# training settings
batch_size = 64

# MNIST Dataset，注意这里的关键工具，torch.utils, data.Dataloader，这个可以有效的读取数据，是一个得到batch的生成器
# 引入MNIST数据集通过datasets函数包进行导入
# root是数据的位置，train=True是下载训练有关的集合，download是决定下不下载数据，一斤固有数据集就download=False

train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='./data_set',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader（Input Pipeline)是一个迭代器，torch.utils.data.DataLoader作用就是随机的在样本中选取数据组成一个小的batch。shuffle决定数据是否打乱
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 可视化数据图像
for i in range(5):
    plt.figure()
    plt.imshow(train_loader.dataset.train_data[i].numpy())

x = torch.randn(2, 2, 2)
# firstly change the data into diresed dimension, then reshape the tensor according to what I want
x.view(-1, 1, 4)

# 理解迭代器的深层含义，torch.utils.data.DataLoader的作用理解
for (data, target) in train_loader:
    for i in range(4):
        plt.figure()
        print(target[1])
        plt.imshow(data[i].numpy()[0])

    break

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision import datasets, transforms    # datasets包含常用的数据集，transform 对图像进行预处理


# training settings
batch_size = 64

# MNIST Dataset，注意这里的关键工具，torch.utils, data.Dataloader，这个可以有效的读取数据，是一个得到batch的生成器
# 引入MNIST数据集通过datasets函数包进行导入
# root是数据的位置，train=True是下载训练有关的集合，download是决定下不下载数据，一斤固有数据集就download=False

train_dataset = datasets.MNIST(root='./data_set/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='./data_set',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader（Input Pipeline)是一个迭代器，torch.utils.data.DataLoader作用就是随机的在样本中选取数据组成一个小的batch。shuffle决定数据是否打乱
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 可视化数据图像
for i in range(5):
    plt.figure()
    plt.imshow(train_loader.dataset.train_data[i].numpy())

x = torch.randn(2, 2, 2)
# firstly change the data into diresed dimension, then reshape the tensor according to what I want
x.view(-1, 1, 4)

# 理解迭代器的深层含义，torch.utils.data.DataLoader的作用理解
for (data, target) in train_loader:
    for i in range(4):
        plt.figure()
        print(target[1])
        plt.imshow(data[i].numpy()[0])

    break

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2) #pytorch文档，torch.nn.Conv2d函数参数定义
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) #全连接层就是线性层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.max_pool2d(F.tanh(self.conv1(x)), (2, 2))
            x = F.dropout(x, p = 0.3, training=self.training)
            x = F.max_pool2d(F.tanh(self.conv2(x)), (2, 2))
            x = F.dropout(x, p = 0.3, training=self.training)
            x = x.view(-1, self.num_flat_features(x))   # view函数用来改变维度，-1是占位符

            x = F.tanh(self.fc1(x))
            x = F.dropout(x, p = 0.3, training=self.training)
            x = F.tanh(self.fc2(x))
            x = F.dropout(x, p = 0.3, training=self.training)
            x = self.fc3(x)

        # 定义num_flat_features函数进行尺度的变换
        def num_flat_features(self, x):
            size = x.size()[1:]
            num_features = 1
            for s in size:
                num_features *= s
            return num_features


model = LeNet5()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.9)
criterion = nn.CrossEntropyLoss()

def train(epoch):
    model.train()   # 第一行固定，model.train是用来实现训练期间用的网络

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()   # tidings清零
        output = model(data)
        loss = criterion(output, target)
        loss.backward() # 反向传播
        optimizer.step()
        if batch_idx % 10 == 0:
            Loss.append(loss.data[0])
            print('Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
            100.*batch_idx / len(train_loader), loss.data[0]))
        return loss.data[0]

    def test():
        model.eval()    # 测试期间用的网络
        test_loss = 0
        correct = 0
        # test数据集进行测试
        for data, target in test_loader:
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).data[0]
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]  # 预测输出的结果
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{}({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    Loss = []

    for epoch in range(60):
        loss = train(epoch)
        Loss.append(loss)
        test()