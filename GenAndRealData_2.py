import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
from MyModule import G_D_Module
import os
import pdb

plt.rcParams['figure.figsize'] = (32, 32)
plt.rcParams['figure.dpi'] = 1
plt.rcParams['image.cmap'] = 'gray'

save_path = 'Performance/'
'''
GenAndRealImgs
Performance
'''

os.makedirs(save_path + 'real', exist_ok=True)
os.makedirs(save_path + 'gen', exist_ok=True)


'''
iner function...........................................................................
'''


def read_csv_data():
    origin_list = os.listdir('data')
    data_list = []  # csv data list
    for name in origin_list:
        if name[-3:] == 'csv':
            data_list.append(name)

    data_length = 1024
    data_np_list = []
    for i in range(len(data_list)):
        source_data = np.loadtxt('data/' + data_list[i])
        length = len(source_data)
        data_count = length // data_length
        data = source_data[:data_count * data_length]
        data = np.reshape(data, (data_count, data_length))
        data_np_list.append(data)
    return data_np_list
'''
iner function...........................................................................
'''


def cdcgan_data(img_epoch_num=50,  cls_idx=6, cuda=True):
    '''
    :param img_epoch_num:
    :param cuda:
    :return: imgs(.jpg) note: change the save path
    '''
    latent_dim = 20  # details in G_D_Module
    n_class = 10  # details in G_D_Module
    img_shape = (1, 32, 32)  # details in G_D_Module
    if cuda:
        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor

    generator = G_D_Module.GeneratorCDCGAN(latent_dim, n_class, img_shape)
    generator.load_state_dict(torch.load('GANParameters/CDCGAN/generator.pt'))

    noise = FloatTensor(np.random.normal(0, 1, (img_epoch_num * n_class, latent_dim)))
    single_list = list(range(n_class))
    label_cpu = single_list * img_epoch_num
    label = LongTensor(label_cpu)
    if cuda:
        label.cuda()
        generator.cuda()
    gen_imags = generator(noise, label)
    gen_imags = gen_imags.cpu().detach().numpy()
    gen_imags = gen_imags[list(range(args.cls_idx, len(label), 10))]
    label_cpu = np.array(label_cpu)[list(range(args.cls_idx, len(label), 10))]
    print(label_cpu, label_cpu.shape)

    return gen_imags, label_cpu


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls_idx', default=6, type=int)   # 剔除掉的类别(改default)
    args = parser.parse_args()
    print(f"class index: {args.cls_idx}")
    
    gen_imgs0, gen_label0 = cdcgan_data(200, cls_idx=args.cls_idx)
    gen_imgs1, gen_label1 = cdcgan_data(200, cls_idx=args.cls_idx)
    
    gen_imgs = np.concatenate([gen_imgs0, gen_imgs1], axis=0)
    gen_labels = np.concatenate([gen_label0, gen_label1], axis=0)
    
    # pdb.set_trace()
    np.save('gen_data', gen_imgs)
    np.save('gen_label', gen_labels)




