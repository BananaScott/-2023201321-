import warnings
import argparse

warnings.filterwarnings('ignore')

import os
import torch
import random
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import pdb

# 设置随机种子，方便复现效果
def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
seed_torch(123)

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 定义一个卷积层（输入，输出，卷积核的大小）
        self.pool = nn.MaxPool2d(2, 2)  # 定义一个池化层（大小2*2，步长2）
        self.conv2 = nn.Conv2d(6, 6, 5)
        self.fc1 = nn.Linear(6 * 5 * 5, 120)  # 定义一个全连接层（输入大小6*5*5，输出大小120）
        self.fc2 = nn.Linear(120, 84) # （2,1）
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):  # 定义各个层之间的关系
        x = self.pool(F.relu(self.conv1(x)))  # #F.relu（）是非线性激活函数
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 6 * 5 * 5)
       # x.view（）是将数据展开在进入全连接层之前对数据进行处理，第一个参数-1表示这个参数由其他的参数决定，比如知道矩阵的总元素个数还有他的列数你就会知道他的行数，第二个参#数是全连接层的输入
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def get_dataloader(args):  # 从文件中载入数据和标签
    data = np.load('data.npy', allow_pickle=True)
    label = np.load('label.npy', allow_pickle=True)  
    data = data[:, np.newaxis, :, :]  # 将数据的维度增加一个新的维度
    
    indices = np.random.permutation(len(data))  # 划分训练集和测试集
    train_data, train_label = data[indices[:int(0.8 * len(indices))]], label[indices[:int(0.8 * len(indices))]]
    test_data, test_label = data[indices[int(0.8 * len(indices)):]], label[indices[int(0.8 * len(indices)):]]
    print(data.shape)
    print(args.imb, args.gen)
    
    if args.imb:                                   # 训练集剔除元素
        indices1 = np.where(train_label == args.cls_idx)[0]  # 获取标签为 cls_idx 的数据索引
        indices2 = np.where(train_label != args.cls_idx)[0]  # 获取标签不为 cls_idx 的数据索引
        train_data = train_data[np.concatenate([indices1[:2], indices2])]
        train_label = train_label[np.concatenate([indices1[:2], indices2])] # 选择 2 个标签为 cls_idx 的数据和标签不为 cls_idx 的数据，然后拼接起来
    
    if args.gen:  # 如果 gen 为 True，则合并生成的数据和标签
        gen_data = np.load('gen_data.npy', allow_pickle=True)
        gen_label = np.load('gen_label.npy', allow_pickle=True)  # 载入生成的数据和标签
        train_data = np.concatenate([train_data, gen_data], axis=0)
        train_label = np.concatenate([train_label, gen_label], axis=0)

    print(train_data.shape, train_label.shape)
    print(test_data.shape, test_label.shape)
    # pdb.set_trace()
    train_dataset = TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_label).long())
    test_dataset = TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_label).long())
    train_dl = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    valid_dl = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    return train_dl, valid_dl  # 返回训练和测试用的数据加载器

  


def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # 确认执行设备
    model = CNNCifar(args)
    model = model.to(device)  # 得到模型的实例
    train_dataloader, test_dataloader = get_dataloader(args)  # 得到数据的迭代器
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean').to(device)  # 交叉熵损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=2e-4)  # 模型参数优化器
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)

    best_loss = 100000  # 用于判断验证集loss最小的参数
    history = {'train loss': [], 'test loss': [], 'train acc': [], 'test acc': []}  # 保存历史loss和准确率

    for epoch in range(args.n_epochs):
        mean_train_loss = []  # 保存这个epoch中的训练集loss，用于绘图
        mean_test_loss = []  # 保存历史的验证集loss，用于绘图
        labels = []
        preds = []
        model.train()  # model切换为train模式
        for item in tqdm(train_dataloader):  # 遍历数据集，获取Sample
            optimizer.zero_grad()  # 清除优化器的梯度，这是每次更新参数必需的操作
            value, label = item  # 获取数据集sample中的value和label
            output = model(value.to(device))  # 得到输出
            loss = loss_func(output, label.to(device))  # 由 output和label计算loss
            loss.backward()  # 由loss进行BP得到梯度
            optimizer.step()  # 优化器更新参数
            preds.extend(np.argmax(output.detach().cpu().numpy(), 1))  # 保存预测值
            labels.extend(label.numpy())  # 保存label，以便计算当前轮的准确率
            mean_train_loss.append(loss.detach().cpu().numpy())  # 把loss放入历史信息

        test_preds = []
        test_labels = []
        model.eval()  # model切换为评估模式
        with torch.no_grad():
            for item in test_dataloader:  # 遍历验证集
                value, label = item  # 获取数据
                output = model(value.to(device))  # 得到输出
                loss = loss_func(output.squeeze(), label.to(device).squeeze())  # 计算loss
                test_preds.extend(np.argmax(output.detach().cpu().numpy(), 1))
                test_labels.extend(np.squeeze(label.numpy()))
                mean_test_loss.append(loss.detach().cpu().numpy())  # 保存loss
        # 上面的mean_train_loss和mean_test_loss是保存一个epoch内的loss信息，历史信息保存的是每个epoch的loss
        history['train loss'].append(np.mean(mean_train_loss))
        history['test loss'].append(np.mean(mean_test_loss))
        history['train acc'].append(accuracy_score(labels, preds))
        history['test acc'].append(accuracy_score(test_labels, test_preds))

        if epoch % 5 == 0 or epoch == args.n_epochs - 1:
            print(
                'Epoch {}/{}: train loss is {:.4f}, test loss is {:.4f}, train acc is {:.4f}, test acc is {:.4f}'.format(
                    epoch + 1, args.n_epochs, np.mean(mean_train_loss), np.mean(mean_test_loss),
                    accuracy_score(labels, preds), accuracy_score(test_labels, test_preds)))
            if args.imb:
                labels = np.array(labels).astype(np.int)
                indices = np.where(labels == 5)[0]
                # print(indices)
                print('Acc on imbalanced class: {:.2f}'.format(accuracy_score(labels[indices], np.array(preds)[indices])))
        # lr_scheduler.step() # 更新学习率
        if np.mean(mean_test_loss) < best_loss:  # 这是判断保存验证集loss最小模型的操作
            best_loss = np.mean(mean_test_loss)
            torch.save(model, 'model_{}'.format(args.gen))
        #scheduler.step()  # 更新学习率

    pd.DataFrame(history).to_csv('result_train_{}.csv'.format(args.gen), index=False)

    plt.figure(dpi=300)
    plt.plot(history['train loss'])
    plt.plot(history['test loss'])
    plt.legend(history.keys())
    plt.savefig('loss_{}.png'.format(args.gen))
    plt.figure(dpi=300)
    plt.plot(history['train acc'])
    plt.plot(history['test acc'])
    plt.legend(['train acc', 'test acc'])
    plt.savefig('acc_{}.png'.format(args.gen))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=5e-3, type=float)  # 学习率
    parser.add_argument('--n_epochs', default=150, type=int)  # epoochs
    parser.add_argument('--batch', default=64, type=int)  # batch
    parser.add_argument('--gen', default=False, type=bool)  # use generated data,加入fake数据
    parser.add_argument('--imb', default=False, type=bool)  # imbalanced，不平衡模式
    parser.add_argument('--cls_idx', default=6, type=int)   # 剔除的类别


    args = parser.parse_args()
    train(args)
    