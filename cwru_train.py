import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from MyModule import G_D_Module
from MyModule import TrainFunction


os.makedirs("images", exist_ok=True)  # 如果不存在images文件夹，则创建它
parser = argparse.ArgumentParser()  # 解析命令行参数
parser.add_argument("--n_epochs", type=int, default=10000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--data_length", type=int, default=1024, help="size of the data length")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=20, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--Gchannels", type=int, default=128, help="start_channels_for_G")
parser.add_argument("--n_classes", type=int, default=10, help="num of class of data (labels)")
opt = parser.parse_args()
print(opt)  # 打印命令行参数


cuda = True if torch.cuda.is_available() else False  # 检测是否有可以使用的GPU设备，如有，则将cuda变量设置为True;否则设置为False（使用CPU）

# cuda = False
img_shape = (opt.channels, opt.img_size, opt.img_size)  # 定义一个元组，表示生成图像的大小和通道数

plt.rcParams['figure.figsize'] = (opt.img_size, opt.img_size)  # 设置figure_size尺寸
# plt.rcParams['image.interpolation'] = 'nearest'  # 设置 interpolation style
# plt.rcParams['image.cmap'] = 'gray'  # 设置 颜色 style
plt.rcParams['figure.dpi'] = 10

def ex_cdcgan():
    data = np.load('./data.npy', allow_pickle=True)
    label = np.load('./label.npy', allow_pickle=True)
    dataset = TensorDataset(torch.tensor(data).float(), torch.tensor(label))  # # 创建张量数据集并打包成数据标记张量
    # 创建数据集迭代器（DataLoader），批次大小为256，用于训练
    data_loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
    )
    latent_dim = 20  # 生成器的隐向量的维度
    generator = G_D_Module.GeneratorCDCGAN(latent_dim, opt.n_classes, img_shape)  # latent_dim should be 20 创建生成器实例
    discriminator = G_D_Module.DiscriminatorCDCGAN(opt.n_classes, img_shape, latent_dim)  # 创建判别器实例
# 训练CGAN
    TrainFunction.train_cdcgan(generator, discriminator, data_loader, opt.n_epochs, opt.lr, opt.b1, opt.b2,
                               opt.latent_dim, opt.n_classes, cuda,
                               fist_train=False)




if __name__ == '__main__':
    ex_cdcgan()

