import numpy as np
import torch
from torch.nn import init

import torch.nn as nn

class GeneratorCDCGAN(nn.Module):
    def __init__(self, latent_dim, n_classes, img_shape):
        super(GeneratorCDCGAN, self).__init__()

        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, 100)  # 以n_classes和100为参数建立了一个Embedding层，用来嵌入标签，并将其编码为100维向量
        self.l1 = nn.Linear(latent_dim + 100, img_shape[-1] ** 2)  # latent_dim是生成器中噪音输入的维度，因此这里给噪音和标签的嵌入以及全连接层的输出之间的结构做了定型
   # nn.Linear则是定义了一个全连接层，其输入为latent_dim+100，输出为img_shape[-1]**2
        self.channel1 = 32  #定义第一个卷积核的通道数
        self.channel2 = 64  #定义第二个卷积核的通道数
        self.channel3 = 64  #定义第三个卷积核的通道数

        self.conv = nn.Sequential(
            nn.BatchNorm2d(self.img_shape[0]),  # 首先对于图像的通道数进行了BatchNorm，这是一个规范化的方法

            nn.Conv2d(self.img_shape[0], self.channel1, 3, stride=1, padding=1),  # 卷积操作1，输出通道数量为self.channel1即32，卷积核为3*3，步长为1，填充为1
            nn.BatchNorm2d(self.channel1, 0.8),  # 对每个通道进行了BatchNorm，防止梯度消失，把数据分布到0附近
            nn.LeakyReLU(0.2, inplace=True),  # 进行了非线性激活函数

            nn.Conv2d(self.channel1, self.channel2, 3, stride=1, padding=1),  # 卷积操作2，输出通道数量为self.channel2即64，卷积核为3*3，步长为1，填充为1
            nn.BatchNorm2d(self.channel2, 0.8),  # 对每个通道进行了BatchNorm
            nn.LeakyReLU(0.2, inplace=True),  # 进行了非线性激活函数

            nn.Conv2d(self.channel2, self.channel3, 3, stride=1, padding=1),  # 卷积操作3，输出通道数量为self.channel3即64，卷积核为3*3，步长为1，填充为1
            nn.BatchNorm2d(self.channel3, 0.8),  # 对每个通道进行了BatchNorm
            nn.LeakyReLU(0.2, inplace=True),  # 进行了非线性激活函数
            nn.Conv2d(self.channel3, self.img_shape[0], 3, stride=1, padding=1),  # 卷积操作4，输出通道数量为img_shape[0]，即为图像的通道数，卷积核为3*3，步长为1，填充为1
        )

    def forward(self, noise, labels):
        em_labels = self.label_emb(labels)  # 输入标签到Embedding层并得到了嵌入标签
        inputs = torch.cat((em_labels, noise), -1)  # 通过torch.cat将其与noise（噪音）合并，作为一个向量输入到全连接层
        inputs = self.l1(inputs)  # 首先通过一个linear映射，将合并后的向量映射成了一个img_shape[-1]**2维的全连接层输出
        out = inputs.view(inputs.size(0), *self.img_shape)  # 将合并后的输出向量reshape成合适的输出图像张量大小（如[batch_size，1，28，28]）
        img = self.conv(out)  # 通过一个序列的卷积操作序列得到了生成器对于这个label和noise的生成结果
        return img


class DiscriminatorCDCGAN(nn.Module):  # 定义一个DCGAN鉴别器
    def __init__(self, n_classes, img_shape, latent_dim):
        super(DiscriminatorCDCGAN, self).__init__()

        self.img_shape = img_shape  # 记录下图像的形状
        self.em_label = nn.Embedding(n_classes, latent_dim)  # 定义了一个将分类进行了嵌入的层
        self.l1 = nn.Linear(latent_dim + img_shape[1] * img_shape[2], 256)  #将嵌入的标签和图像进行了连接，并输出到了一个256维的全连接层上，实现了一个引入标签信息的判别模型的结构构造
        self.l2 = nn.Linear(256, img_shape[1] * img_shape[2])  #继续接下来的全连接层，输出为img_shape[1]*img_shape[2]

        self.channel1 = 32  # 定义第一个卷积核的通道数
        self.channel2 = 64  # 定义第二个卷积核的通道数
        self.channel3 = 16  # 定义第三个卷积核的通道数
        self.conv = nn.Sequential(
            nn.Conv2d(img_shape[0], self.channel1, 3, 1, 0),  # 32 to 30 定义第一层卷积层，通道数为32，核大小为3*3，步长为1，且输入高宽都匹配
            nn.LeakyReLU(0.2, inplace=True),   # 进行了非线性激活函数
            nn.Dropout2d(0.25),  # dropout，以减少过拟合
            nn.MaxPool2d(2),  # 30 to 15 最大池化，减小高宽

            nn.Conv2d(self.channel1, self.channel2, 4, 1, 0),  # 15 to 12  定义第二层卷积层，通道数为64，核大小为4*4（比3*3大），步长为1，且输入高宽都匹配
            nn.LeakyReLU(0.2, inplace=True),  # 进行了非线性激活函数
            nn.Dropout2d(0.25),  # dropout，以减少过拟合
            nn.MaxPool2d(2),  # 12 to 6  最大池化，减小高宽

            nn.Conv2d(self.channel2, self.channel3, 3, 1, 1),  # 定义第三层卷积层，通道数为16，核大小为3*3，步长为1，且输入高宽都匹配

        )
        self.l3 = nn.Sequential(
            nn.Linear(6 * 6 * self.channel3, 1),  # 定义全连接层，线性变化将输入维度f_latents映射成输出维度的1个单元
            nn.Sigmoid()  # sigmoid激活函数进行信息压缩
        )

    def forward(self, inputs, labels):
        inputs = torch.cat((self.em_label(labels), inputs.view(inputs.size(0), -1)), -1)  # 将鉴别器输入的图像flatten后与嵌入的标签连接到一起
        inputs = self.l1(inputs)  # 输入到l1全连接层
        inputs = self.l2(inputs)  # 继续输入到l2全连接层
        inputs = inputs.view(inputs.size(0), *self.img_shape)  # 将其reshape成了batch_size, img_shape的形式
        inputs = self.conv(inputs)  # 送到卷积层进行特征的学习
        out = inputs.view(inputs.shape[0], -1)  # 将输出向量拉成一维向量
        valid = self.l3(out)  # 最后输入到了l3层，该层最终输出一个logit

        return valid  # 将输出logit返回作为预测有效样本的值
