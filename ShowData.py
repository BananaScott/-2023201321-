import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as pltImage
import random
from MyModule import G_D_Module
import torch
import os

# from torch.autograd import Variable

# 设置图像的字体和颜色
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def show_cdcgan_data():  # 展示CDCGAN生成的数据
    latent_dim = 20

    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor  
    generator = G_D_Module.GeneratorCDCGAN(latent_dim, 10, (1, 32, 32))
    generator.load_state_dict(torch.load('GANParameters/CDCGAN/generator.pt'))  # 创建生成器实例并从预先训练的模型中加载状态信息

    noise = FloatTensor(np.random.normal(0, 1, (10 ** 2, latent_dim)))  # 生成噪声
    single_list = list(range(10))  # 创建label实例
    label = LongTensor(single_list * 10)
    gen_imags = generator(noise, label)  # 使用生成器来预测caches/gen.jpg

    # real
    # imgs = np.empty([len(data) ** 2, 1, 32, 32], dtype=float)
    # for i in range(len(data)):
    #     for j in range(len(data)):
    #         index = random.randint(0, len(data[j]) - 1)
    #         imgs[i * len(data) + j][0] = data[j][index]
    # for i in range(imgs.shape[0]):
    #     plt.subplot(len(data_list), len(data_list), i + 1)
    #     plt.axis('off')
    #     plt.contourf(imgs[i][0])
    # plt.savefig('caches/real.jpg', bbox_inches='tight')
    # plt.close()

    for i in range(gen_imags.shape[0]):  # 展示由CDCGAN生成的数据
        plt.subplot(10, 10, i + 1)
        plt.axis('off')
        plt.contourf(gen_imags[i][0].detach().numpy())
    plt.savefig('caches/gen.jpg', bbox_inches='tight')  # 将它们保存在caches/gen.jpg中
    plt.close()




if __name__ == '__main__':
    show_cdcgan_data()
