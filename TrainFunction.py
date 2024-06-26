import matplotlib.pyplot as plt  # 引入画图库
import os  # 用于进行文件夹的操作
from torch.autograd import Variable  # 自动求导，方便后向传递梯度
import torch  # PyTorch框架
import numpy as np  # 使用numpy来生成随机数

# 定义训练CDCGAN的函数
def train_cdcgan(generator, discriminator, data_loader, n_epochs, lr, b1, b2, latent_dim, n_classes, cuda,
                 fist_train=False):
    path = "GANParameters/CDCGAN"
    os.makedirs(path, exist_ok=True)  # 用于创建文件夹的函数，如果已经有了这个文件夹，则不操作

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor  # 判断当前是否可使用CUDA


    loss = torch.nn.BCELoss()  # 定义构造判别模型的损失函数

    if not fist_train:  # 如果不是第一次train，则加载已经训练好的generator和discriminator
        generator.load_state_dict(torch.load(path + "/generator.pt"))
        discriminator.load_state_dict(torch.load(path + "/discriminator.pt"))

    if cuda:  # 使用CUDA
        generator.cuda()
        discriminator.cuda()
        loss.cuda()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))  # 定义生成器的优化器
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))  # 定义鉴别器的优化器

    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(data_loader):
            batch_size = imgs.shape[0]  # 当前batch的大小

            # Adversarial ground truths（事实）
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)  # 定义当前epoch的真实和假的图片

            # Configure input（配置输入）
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))  # 将label和真实的图片reshape，准备输入

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()  # 清空生成器的所有梯度

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)  # 使用生成器得到生成的图片

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = loss(validity, valid)  # 得到生成器的损失函数

            g_loss.backward()  # 反向传递梯度
            optimizer_G.step()  # 使用优化器来更新生成器的参数

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()  # 清空鉴别器的所有梯度

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = loss(validity_real, valid)  # 得到当前真实图片的损失函数

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = loss(validity_fake, fake)  #得到当前生成图片的损失函数

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2  # 得到总的鉴别器的损失函数

            d_loss.backward()  # 反向传递梯度
            optimizer_D.step()  # 使用优化器来更新鉴别器的参数

            if i == 1:  #控制输出的频率
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(data_loader), d_loss.item(), g_loss.item())
                )
                if epoch % 50 == 0:  # 每50个epoch保存生成的图片
                    if cuda:
                        gen_imgs = gen_imgs.cpu()
                        gen_labels = gen_labels.cpu()
                    plt.contourf(gen_imgs[0][0].detach().numpy())
                    plt.axis('off')
                    plt.savefig('caches/label' + str(gen_labels[0].detach().numpy()) + '_' + str(epoch) + '.jpg')
                    plt.close()
                    if epoch % 200 == 0:  # 每200个epoch保存当前的generator和discriminator
                        torch.save(generator.state_dict(), path + "/generator.pt")
                        torch.save(discriminator.state_dict(), path + "/discriminator.pt")
