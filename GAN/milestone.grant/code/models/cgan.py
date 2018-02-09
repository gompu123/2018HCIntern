import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as tcuda
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable

import pandas as pd

from models.gan import GAN
from models.generators import ConditionalBNGenerator, ConditionalGenerator
from models.discriminators import ConditionalBNDiscriminator, ConditionalDiscriminator

import matplotlib.pyplot as plt

from tqdm import tqdm
from tqdm import trange


class CGAN(GAN):
    def __init__(self, **kwargs):
        self.batch_size = kwargs['batch_size']
        self.train_loader = kwargs['train_loader']
        self.G = ConditionalBNGenerator(**kwargs)
        # self.G = ConditionalGenerator(**kwargs)
        self.D = ConditionalBNDiscriminator(**kwargs)
        # self.D = ConditionalDiscriminator(**kwargs)
        self.z_size = kwargs['z_size']
        self.class_num = kwargs['class_num']
        self.fixed_z = torch.rand(10, self.z_size)

        if tcuda.is_available():
            self.G, self.D = self.G.cuda(), self.D.cuda()

        # todo --> customizable
        self.BCE_loss = nn.BCELoss()

        # todo --> customizable
        # todo --> weight decay setting
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=kwargs['lrG'], betas=(0.5, 0.999))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=kwargs['lrD'], betas=(0.5, 0.999))
        # self.G_scheduler = MultiStepLR(self.G_optimizer, milestones=[30, 40], gamma=0.1)
        # self.D_scheduler = MultiStepLR(self.D_optimizer, milestones=[30, 40], gamma=0.1)


    def train(self, epoch_num=10):
        self.G.weight_init(mean=0, std=0.02)
        self.D.weight_init(mean=0, std=0.02)
        # self.gen_plot(0)
        # print('...')
        for epoch in trange(epoch_num, desc='Epoch', leave=True, position=1):
            pbar2 = tqdm(total=len(self.train_loader), leave=False, position=2)

            generator_losses = []
            discriminator_losses = []

            # self.G_scheduler.step()
            # self.D_scheduler.step()
            for x, y in self.train_loader:
                # print(x, y)
                # print(self.G, self.D)
                # exit()
                self.D.zero_grad()
                batch_size = x.size()[0]

                y_real = torch.ones(batch_size)
                y_fake = torch.zeros(batch_size)
                y_label = torch.zeros(batch_size, self.class_num)
                y_label.scatter_(1, y.view(batch_size, 1), 1)
                # print(batch_size, y_real, y_fake, y_label)
                # exit()
                if tcuda.is_available():
                    x, y_real, y_fake, y_label = x.cuda(), y_real.cuda(), y_fake.cuda(), y_label.cuda()
                x, y_real, y_fake, y_label = Variable(x), Variable(y_real), Variable(y_fake), Variable(y_label)

                y_pred = self.D(x, y_label).squeeze()
                real_loss = self.BCE_loss(y_pred, y_real)
                # print(real_loss)
                z = torch.rand((batch_size, self.z_size))
                y = (torch.rand(batch_size, 1) * self.class_num).type(torch.LongTensor)
                y_label = torch.zeros(batch_size, self.class_num)
                y_label.scatter_(1, y.view(batch_size, 1), 1)
                # print(z, y, y_label)
                # exit()
                if tcuda.is_available():
                    z, y_label = z.cuda(), y_label.cuda()
                z, y_label = Variable(z), Variable(y_label)
                y_pred = self.D(self.G(z, y_label), y_label).squeeze()
                fake_loss = self.BCE_loss(y_pred, y_fake)

                D_train_loss = real_loss + fake_loss
                D_train_loss.backward()
                discriminator_losses.append(D_train_loss.data[0])

                self.D_optimizer.step()

                self.G.zero_grad()
                z = torch.rand((batch_size, self.z_size))
                y = (torch.rand(batch_size, 1) * self.class_num).type(torch.LongTensor)
                y_label = torch.zeros(batch_size, self.class_num)
                y_label.scatter_(1, y.view(batch_size, 1), 1)

                if tcuda.is_available():
                    z, y_label = z.cuda(), y_label.cuda()
                z, y_label = Variable(z), Variable(y_label)
                y_pred = self.D(self.G(z, y_label), y_label).squeeze()
                # print(y_real)
                # print(y_label)
                # exit()
                G_train_loss = self.BCE_loss(y_pred, y_real)
                G_train_loss.backward()
                generator_losses.append(G_train_loss.data[0])

                self.G_optimizer.step()

                pbar2.update(batch_size)

            pbar2.close()
            tqdm.write('Training [{:>5}:{:>5}] D Loss {:.6f}, G Loss {:.6f}'.format(
                epoch + 1, epoch_num,
                torch.mean(torch.FloatTensor(discriminator_losses)),
                torch.mean(torch.FloatTensor(generator_losses))))
            # self.gen_plot(epoch + 1)
            self.gen_df(epoch + 1)

    # testing
    def gen_plot(self, epoch_num):
        z = torch.rand(10, 100)
        c_ = torch.zeros(1, 1)
        for i in range(1, self.class_num):
            temp = torch.zeros(1, 1) + i
            c_ = torch.cat([c_, temp], 0)
        c = torch.zeros(10, 10)
        c.scatter_(1, c_.type(torch.LongTensor), 1)
        z, c = Variable(z), Variable(c)
        self.G.eval()
        results = self.G(z, c)
        self.G.train()

        fig, ax = plt.subplots(10, 1, figsize=(10, 10))
        for i, d2 in enumerate(results.data):
            # print(d2)
            # print(type(d2))
            ax[i].imshow(d2.numpy().reshape(28, -1), cmap='gray')

        plt.savefig('../data/fig_{}.'.format(epoch_num))

    # testing
    def gen_df(self, epoch_num):
        # z = torch.rand(10, self.z_size)
        # print(self.fixed_z)
        c_ = torch.zeros(5, 1)
        for i in range(1, self.class_num):
            temp = torch.zeros(5, 1) + i
            c_ = torch.cat([c_, temp], 0)
        c = torch.zeros(10, self.class_num)
        c.scatter_(1, c_.type(torch.LongTensor), 1)
        z, c = Variable(self.fixed_z), Variable(c)
        self.G.eval()
        results = self.G(z, c)
        self.G.train()

        a = pd.DataFrame(
            torch.cat([results.data, c_], 1).numpy(),
            columns=self.train_loader.dataset.df.columns
        )
        # print(a)
        a.to_csv('../data/gen_data_{}.csv'.format(epoch_num), index=False)


    def generate(self, gen_num=10):
        z = torch.rand(gen_num, self.z_size)
        c_ = torch.zeros(gen_num // self.class_num, 1)
        for i in range(1, self.class_num):
            temp = torch.zeros(gen_num // self.class_num, 1) + i
            c_ = torch.cat([c_, temp], 0)
        c = torch.zeros(gen_num, self.class_num)
        c.scatter_(1, c_.type(torch.LongTensor), 1)
        if tcuda.is_available():
            z, c = z.cuda(), c.cuda()
        z, c = Variable(z), Variable(c)
        self.G.eval()
        results = self.G(z, c)
        resultsd = torch.cat([results.data, c_], 1)
        print(resultsd)
        self.G.train()
        # return pd.DataFrame(
        #     resultsd.numpy(),
        #     columns=None
        # )
        return pd.DataFrame(
            resultsd.numpy(),
            columns=self.train_loader.dataset.df.columns
        )


    def save(self, generator_path, discriminator_path):
        super().save(generator_path, discriminator_path)
