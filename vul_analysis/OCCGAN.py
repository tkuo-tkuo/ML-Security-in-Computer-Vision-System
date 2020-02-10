from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np

class Generator(nn.Module):
    def __init__(self, num_of_features):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(1024, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 1024)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 1024))
        z = self.reparametrize(mu, logvar)
        return (self.decode(z)).view(-1, 1, 32, 32)


# class Generator(nn.Module):
#     def __init__(self, num_of_features):
#         super().__init__()
#         self.relu = nn.ReLU()

#         # compress layer 
#         self.compress_conv1 = nn.Conv2d(1, num_of_features, 4)
#         self.compress_conv2 = nn.Conv2d(num_of_features, num_of_features, 4)
#         self.compress_conv3 = nn.Conv2d(num_of_features, num_of_features, 5)
#         self.compress_pool = nn.MaxPool2d(2, stride=2)
#         self.compress_bn = nn.BatchNorm2d(num_of_features)

#         # input layer
#         self.input_deconv = nn.ConvTranspose2d(num_of_features, num_of_features, 4)

#         # dense block
#         self.dense_conv1 = nn.Conv2d(num_of_features, num_of_features, 1)
#         self.dense_conv2 = nn.Conv2d(
#             num_of_features, num_of_features, 3, padding=1)
#         self.dense_bn = nn.BatchNorm2d(num_of_features)

#         # transition layer
#         self.trans_conv = nn.Conv2d(num_of_features, num_of_features, 1)
#         self.trans_deconv = nn.ConvTranspose2d(
#             num_of_features, num_of_features, 2, stride=2)
#         self.trans_bn = nn.BatchNorm2d(num_of_features)

#         # output layer
#         self.output_conv = nn.Conv2d(num_of_features, 1, 1)
#         self.output_bn = nn.BatchNorm2d(1)

#     # exp (return B, 32, 1, 1)
#     def compress_block(self, x):
#         x2 = self.compress_conv1(x)
#         x2 = self.relu(self.compress_bn(x2))
#         x3 = self.compress_pool(x2)
#         x4 = self.compress_conv2(x3)
#         x4 = self.relu(self.compress_bn(x4))
#         x5 = self.compress_pool(x4)
#         x6 = self.compress_conv3(x5)
#         x6 = self.relu(self.compress_bn(x6))
#         return x6

#     def dense_block(self, x):
#         x2 = self.dense_conv1(x)
#         x2 = self.relu(self.dense_bn(x2))
#         x3 = self.dense_conv2(x2)
#         x3 = self.relu(self.dense_bn(x3))
#         return x3 + x

#     def transition_layer(self, x):
#         x2 = self.trans_conv(x)
#         x2 = self.relu(self.trans_bn(x2))
#         x3 = self.trans_deconv(x2)
#         return x3

#     def output_layer(self, x):
#         x2 = self.output_conv(x)
#         x2 = self.relu(self.output_bn(x2))
#         return x2

#     def forward(self, img):
#         x = self.compress_block(img)
#         print(x.shape)

#         x = self.input_deconv(x)

#         x = self.dense_block(x)
#         x = self.transition_layer(x)

#         x = self.dense_block(x)
#         x = self.transition_layer(x)

#         x = self.dense_block(x)
#         x = self.transition_layer(x)

#         x = self.output_layer(x)

#         return x / torch.max(x)


class Discriminator(nn.Module):
    def __init__(self, num_of_features):
        super().__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        # input layer
        self.input_conv = nn.Conv2d(1, num_of_features, 1)
        self.input_bn = nn.BatchNorm2d(num_of_features)

        # dense block
        self.dense_conv1 = nn.Conv2d(num_of_features, num_of_features, 1)
        self.dense_conv2 = nn.Conv2d(
            num_of_features, num_of_features, 3, padding=1)
        self.dense_bn = nn.BatchNorm2d(num_of_features)

        # transition layer
        self.trans_conv = nn.Conv2d(num_of_features, num_of_features, 1)
        self.trans_pool = nn.MaxPool2d(2, stride=2)
        self.trans_bn = nn.BatchNorm2d(num_of_features)

        # output layer
        self.output_fc = nn.Linear(32*4*4, 2)

    def input_layer(self, x):
        x2 = self.input_conv(x)
        x2 = self.relu(self.input_bn(x2))
        return x2

    def dense_block(self, x):
        x2 = self.dense_conv1(x)
        x2 = self.relu(self.dense_bn(x2))
        x3 = self.dense_conv2(x2)
        x3 = self.relu(self.dense_bn(x3))
        return x3 + x

    def transition_layer(self, x):
        x2 = self.trans_conv(x)
        x2 = self.relu(self.trans_bn(x2))
        x3 = self.trans_pool(x2)
        return x3

    def forward(self, x):
        x = self.input_layer(x)

        x = self.dense_block(x)
        x = self.transition_layer(x)

        x = self.dense_block(x)
        x = self.transition_layer(x)

        x = self.dense_block(x)
        x = self.transition_layer(x)

        x = x.view(-1, 32*4*4)
        x = self.output_fc(x)

        return self.softmax(x)


def pretrain_G(G, G_optim, model, loader, hyperparas):
    # correct penalty -> use torch.bmm
    softmax = nn.Softmax()

    # reconstruction loss
    reconstruction_function = nn.BCELoss()
    reconstruction_function.size_average = False

    # gen dis loss (game loss) -> ignored in pretraining

    for _ in range(hyperparas['pre_train_epoches']):
        for i, (X, Y) in enumerate(loader):
            # noises = torch.randn((X.shape[0], 1, 1, 1))
            G_X = G(X)

            # compute correct penalty
            remove_padding_G_X = G_X[:, :, 2:-2, 2:-2]
            fX = softmax(model(remove_padding_G_X))
            mask = torch.zeros(fX.shape)
            mask[range(len(Y)), Y] = 1
            B, S = fX.shape[0], fX.shape[1]
            correct_penalty = hyperparas['cor_pen_para']*torch.sum(
                torch.bmm(fX.view(B, 1, S), mask.view(B, S, 1))) / B

            # compute reconstruction loss
            reconstruction_loss = hyperparas['recon_para']*reconstruction_function(G_X, X)

            # compute total loss
            G_loss = correct_penalty + reconstruction_loss
            print('Gen loss:', round(G_loss.item(), 3), 'Cor penality:', round(
                correct_penalty.item(), 3), 'Recon loss:', round(reconstruction_loss.item(), 3))

            G_optim.zero_grad()
            G_loss.backward()
            G_optim.step()

            # Visual
            if i % hyperparas['draw_interval'] == 0:
                img_counter = 1
                img_bound = 60

                for x in X:
                    x = x[:, 2:-2, 2:-2]
                    x = x.reshape(28, 28)
                    plt.subplot(6, 10, img_counter)
                    img_counter += 1
                    plt.axis('off')
                    plt.imshow(x, cmap='gray')
                    if img_counter == img_bound/2+1:
                        break

                for x in G_X:
                    x = x.detach().numpy()
                    x = x[:, 2:-2, 2:-2]
                    x = x.reshape(28, 28)
                    plt.subplot(6, 10, img_counter)
                    img_counter += 1
                    plt.axis('off')
                    plt.imshow(x, cmap='gray')
                    if img_counter == img_bound+1:
                        break

                plt.show()

    return G


def game(G, G_optim, D, D_optim, model, loader, hyperparas):
    # Game between G and D
    padding = nn.ConstantPad2d(2, 0)
    softmax = nn.Softmax()
    reconstruction_function = nn.BCELoss()
    mse_loss = nn.MSELoss()

    sub_accs = [[] for _ in range(8)]

    for _ in range(hyperparas['train_epoches']):
        for i, (X, Y) in enumerate(loader):
            # noises = torch.randn((X.shape[0], 1, 1, 1))
            for _ in range(hyperparas['g_repeat_num']):
                # G_X = G(noises, X)
                G_X = G(X)

                # compute correct penalty
                remove_padding_G_X = G_X[:, :, 2:-2, 2:-2]
                fX = softmax(model(remove_padding_G_X))
                mask = torch.zeros(fX.shape)
                mask[range(len(Y)), Y] = 1
                B, S = fX.shape[0], fX.shape[1]
                correct_penalty = hyperparas['cor_pen_para']*torch.sum(torch.bmm(fX.view(B, 1, S), mask.view(B, S, 1))) / B

                # compute reconstruction loss
                reconstruction_loss = hyperparas['recon_para']*reconstruction_function(G_X, X)

                # compute gen dis loss
                S_D_GX = D(G_X)
                labels = torch.zeros(S_D_GX.shape)
                labels[:, 1] = 1.
                gen_dis_loss = hyperparas['g_dis_para']*(1 - mse_loss(S_D_GX, labels))

                # compute total loss
                G_loss = correct_penalty + reconstruction_loss + gen_dis_loss
                G_optim.zero_grad()
                G_loss.backward()
                G_optim.step()

            S_D_X = D(X)
            labels = torch.zeros(S_D_X.shape)
            labels[:, 0] = 1.
            DX_loss = hyperparas['cor_pen_para']*mse_loss(S_D_X, labels)

            # G_X = G(noises, X)
            G_X = G(X)

            S_D_GX = D(G_X)
            labels = torch.zeros(S_D_GX.shape)
            labels[:, 1] = 1.
            DGX_loss = hyperparas['gen_sam_para']*mse_loss(S_D_GX, labels)

            D_loss = DX_loss + DGX_loss
            D_optim.zero_grad()
            D_loss.backward(retain_graph=True)
            D_optim.step()

            # display training process
            print('Mini-btach', i+1)
            print('Gen loss:', round(G_loss.item(), 3), 
            'cor pen:', round(correct_penalty.item(), 3), 
            'recon:', round(reconstruction_loss.item(), 3), 
            'g_dis:', round(gen_dis_loss.item(), 3))
            print('Dis loss:', round(D_loss.item(), 3),
            'Dis (DX) loss:', round(DX_loss.item(), 3),
            'Dis (DGX) loss:', round(DGX_loss.item(), 3))

            # visualize
            if i % hyperparas['draw_interval'] == 0:
                img_counter = 1
                img_bound = 60

                for x in X:
                    x = x[:, 2:-2, 2:-2]
                    x = x.reshape(28, 28)
                    plt.subplot(6, 10, img_counter)
                    img_counter += 1
                    plt.axis('off')
                    plt.imshow(x, cmap='gray')
                    if img_counter == img_bound/2+1:
                        break

                for x in G_X:
                    x = x.detach().numpy()
                    x = x[:, 2:-2, 2:-2]
                    x = x.reshape(28, 28)
                    plt.subplot(6, 10, img_counter)
                    img_counter += 1
                    plt.axis('off')
                    plt.imshow(x, cmap='gray')
                    if img_counter == img_bound+1:
                        break

                plt.show()

                # acc in adv_images (evaluation)
                for adv_i, adv_type in enumerate(hyperparas['adv_types']):
                    correct, total = 0, 0
                    for i in range(100):
                        if adv_type == 'None':
                            fn = 'adv_images/'+'benign'+str(i)+'.npy'
                            x = np.load(fn)
                            x = torch.from_numpy(x)
                            x = padding(x)
                            x = x.reshape(1, 1, 32, 32)
                            output = D(x)
                            if (torch.argmax(output).item() == 0):
                                correct += 1
                            total += 1

                        else:
                            fn = 'adv_images/'+adv_type+str(i)+'.npy'
                            try:
                                x = np.load(fn)
                            except:
                                continue

                            x = torch.from_numpy(x)
                            x = padding(x)
                            x = x.reshape(1, 1, 32, 32)
                            output = D(x)
                            if (torch.argmax(output).item() == 1):
                                correct += 1
                            total += 1

                    sub_accs[adv_i].append(correct/total)

                # draw training process
                num_of_adv_types = len(sub_accs)
                for i in range(num_of_adv_types):
                    accs = sub_accs[i]
                    plt.plot(accs)

                plt.legend(hyperparas['adv_types'], loc='lower left')
                plt.title('eval')
                plt.ylim([0, 1.05])
                plt.show()

    return G, D
