import random
import numpy as np
np.random.seed(1234)
import os

import torch
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD, Adam
from utils import Adamp as AdamPlus
import torch.autograd as autograd
from torch.optim.optimizer import required

from scipy.stats import gaussian_kde
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from numpy import linalg

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=2)
parser.add_argument('--modes_idx', type=int, default=2)
parser.add_argument('--update_rule_idx', type=int, default=4)
parser.add_argument('--learning_rate_idx', type=int, default=1)
parser.add_argument('--momentum_idx', type=int, default=2)
parser.add_argument('--dis_iter_idx', type=int, default=0)
args = parser.parse_args()

iterations = args.iterations

modes = ['gp', 'no_gp', 'sat', 'co']
mode = modes[args.modes_idx]

update_rules = ['sgdnm', 'adam_plus', 'adam', 'mixed_adam_plus', 'mixed_sgd']
update_rule = update_rules[args.update_rule_idx]

learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]
learning_rate = learning_rates[args.learning_rate_idx]

momentums = [-0.5, -0.1, 0, 0.5]
momentum = momentums[args.momentum_idx]

dis_iters = [1, 5]
dis_iter = dis_iters[args.dis_iter_idx]
_batch_size = 128
dim = 256
use_cuda = True
# LAMBDA = 0.1
LAMBDA = 0.05
z_dim = 8
save_directory = ('results/' + str(mode) + '_' +
                  str(update_rule) + '_' +
                  str(learning_rate) + '_' +
                  str(momentum) + '_' +
                  str(dis_iter) + '_' +
                  '/')
print('save_directory: ' + save_directory)
print('iterations: ' + str(iterations))

if not os.path.exists(save_directory):
    os.makedirs(save_directory)


class Gen(nn.Module):

    def __init__(self):
        super(Gen, self).__init__()

        main = nn.Sequential(
            nn.Linear(z_dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, 2),
        )
        self.main = main

    def forward(self, noise):
            output = self.main(noise)
            return output


class Dis(nn.Module):

    def __init__(self):
        super(Dis, self).__init__()

        main = nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, 1),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# def weights_load():


def get_8gaussians(batch_size):
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x, y in centers]
    while True:
        dataset = []
        for i in range(batch_size):
            point = np.random.randn(2) * .05
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        dataset /= 1.414
        out = Variable(torch.Tensor(dataset))
        if use_cuda:
            out = out.cuda()
        yield out


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(_batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(
            disc_interpolates.size()).cuda() if use_cuda else torch.ones(
            disc_interpolates.size()),
        create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def apply_consensus(x, y):
    grad_x += 0.5 * x
    grad_y += 0.5 * y
    x -= learning_rate * grad_x
    y -= learning_rate * grad_y


class SGDNM(SGD):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)


def get_dens_real(batch_size):
    data = get_8gaussians(batch_size).next()
    real = np.array(data.data)
    kde_real = gaussian_kde(real.T, bw_method=0.22)
    x, y = np.mgrid[-2:2:(200 * 1j), -2:2:(200 * 1j)]
    z_real = kde_real((x.ravel(), y.ravel())).reshape(*x.shape)
    return z_real

z_real = get_dens_real(1000)


def plot(fake, epoch):
    plt.figure(figsize=(20, 9))
    fake = np.array(fake.data)
    kde_fake = gaussian_kde(fake.T, bw_method=0.22)

    x, y = np.mgrid[-2:2:(200 * 1j), -2:2:(200 * 1j)]
    z_fake = kde_fake((x.ravel(), y.ravel())).reshape(*x.shape)

    ax1 = plt.subplot(1, 2, 1)
    # ax1.hold(True)
    ax1.pcolor(x, y, z_real)

    ax2 = plt.subplot(1, 2, 2)
    # ax2.hold(True)
    ax2.pcolor(x, y, z_fake)

    ax1.scatter(real[:, 0], real[:, 1])
    ax2.scatter(fake[:, 0], fake[:, 1])
    plt.savefig(save_directory + str(epoch) + '.png')
    plt.close()


def plot_eigens(points):
    fig, ax = plt.subplots()
    ax.set_xlabel("real")
    ax.set_ylabel("imaginaries")
    reals = [p.real for p in points]
    imaginaries = [p.imag for p in points]
    plt.plot(reals, imaginaries, 'o', color='blue')
    plt.grid()
    fig.set_size_inches(10, 10)
    plt.savefig('eigen_plot/eigen_init.png')
    plt.close(fig)

dis = Dis()
gen = Gen()

# dis.load_state_dict(torch.load('models/dis_sat_mixed_sgd_0.01_0_1_'))
# gen.load_state_dict(torch.load('models/gen_sat_mixed_sgd_0.01_0_1_'))

dis.apply(weights_init)
gen.apply(weights_init)

if use_cuda:
    dis = dis.cuda()
    gen = gen.cuda()

if update_rule == 'adam':
    dis_optimizer = Adam(dis.parameters(),
                         lr=learning_rate,
                         betas=(0.5, 0.9))
    gen_optimizer = Adam(gen.parameters(),
                         lr=learning_rate,
                         betas=(0.5, 0.9))
elif update_rule == 'adam_plus':
    dis_optimizer = AdamPlus(dis.parameters(),
                             lr=learning_rate,
                             betas=(momentum, 0.9))
    gen_optimizer = AdamPlus(gen.parameters(),
                             lr=learning_rate,
                             betas=(momentum, 0.9))
elif update_rule == 'sgd':
    dis_optimizer = SGD(dis.parameters(), lr=learning_rate)
    gen_optimizer = SGD(gen.parameters(), lr=learning_rate)
elif update_rule == 'sgdnm':
    dis_optimizer = SGDNM(
        dis.parameters(), lr=learning_rate, momentum=momentum)
    gen_optimizer = SGDNM(
        gen.parameters(), lr=learning_rate, momentum=momentum)
elif update_rule == 'mixed_adam_plus':
    dis_optimizer = AdamPlus(dis.parameters(),
                             lr=learning_rate,
                             betas=(momentum, 0.9))
    gen_optimizer = Adam(gen.parameters(),
                         lr=learning_rate,
                         betas=(0.5, 0.9))
elif update_rule == 'mixed_sgd':
    dis_optimizer = SGDNM(
        dis.parameters(), lr=learning_rate, momentum=momentum)
    gen_optimizer = SGD(gen.parameters(), lr=learning_rate)

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda()
    mone = mone.cuda()

dataset = get_8gaussians(_batch_size)
criterion = nn.BCEWithLogitsLoss()

ones = Variable(torch.ones(_batch_size))
zeros = Variable(torch.zeros(_batch_size))
if use_cuda:
    criterion = criterion.cuda()
    ones = ones.cuda()
    zeros = zeros.cuda()

gen_losses = []
dis_losses = []
points = []
for iteration in range(iterations):
    for iter_d in range(dis_iter):
        dis.zero_grad()

        noise = torch.randn(_batch_size, z_dim)
        if use_cuda:
            noise = noise.cuda()

        noise = autograd.Variable(noise)
        real = dataset.next()
        pred_real = criterion(dis(real), ones)

        # fake = Variable(gen(noise).data)
        fake = gen(noise)
        pred_fake = criterion(dis(fake), zeros)

        gradient_penalty = 0
        import ipdb; ipdb.set_trace()
        if mode == 'gp':
            gradient_penalty = calc_gradient_penalty(
                dis, real.data, fake.data)
            pred_tot = pred_real + pred_fake + gradient_penalty
        else:
            pred_tot = pred_real + pred_fake
        pred_tot.backward(create_graph=True, retain_graph=True)
        # pred_tot.backward()

        dis_loss = gradient_penalty + pred_real + pred_fake
        dis_optimizer.step()
        if iteration == 0:
            dis_param = dis.parameters().next()
            gen_param = gen.parameters()
            for i in range(6):
                gen_param.next()
            gen_param = gen_param.next()

            dis_param_grad = dis_param.grad.view(-1)

            dis_grad_dis = torch.stack(
                [torch.autograd.grad(dis_param_grad[p],
                 dis_param, create_graph=True, retain_graph=True)
                 [0].view(-1) for p in range(len(dis_param_grad))])
            dis_grad_gen = torch.stack(
                [torch.autograd.grad(dis_param_grad[p],
                 gen_param, create_graph=True, retain_graph=True)
                 [0].view(-1) for p in range(len(dis_param_grad))])
            joc_dis = torch.cat((dis_grad_dis, dis_grad_gen), 1)

    gen.zero_grad()
    noise = torch.randn(_batch_size, z_dim)
    ones = Variable(torch.ones(_batch_size))
    zeros = Variable(torch.zeros(_batch_size))
    if use_cuda:
        noise = noise.cuda()
        ones = ones.cuda()
        zeros = zeros.cuda()
    noise = autograd.Variable(noise)
    fake = gen(noise)
    if mode == 'sat':
        gen_loss = -criterion(dis(fake), zeros)
    else:
        gen_loss = criterion(dis(fake), ones)
    gen_loss.backward(create_graph=True, retain_graph=True)
    gen_optimizer.step()

    if iteration == 0:
        gen_param_grad = gen_param.grad.view(-1)
        gen_grad_gen = torch.stack(
            [torch.autograd.grad(gen_param_grad[p],
             gen_param, create_graph=True, retain_graph=True)
             [0].view(-1) for p in range(len(gen_param_grad))])
        gen_grad_dis = torch.stack(
            [torch.autograd.grad(gen_param_grad[p],
             dis_param, create_graph=True, retain_graph=True)
             [0] for p in range(len(gen_param_grad))])
        gen_grad_dis = gen_grad_dis.view(512, 512)

        joc_gen = torch.cat((gen_grad_dis, gen_grad_gen), 1)
        joc = torch.cat((joc_gen, joc_dis), 0)
        eigvals = linalg.eigvals(joc.cpu().data.numpy())
        points.append(eigvals)

    if iteration % 250 == 0:
        print("iteration: " + str(iteration) +
              " gen_loss: " + str(float(gen_loss)) +
              " dis_loss: " + str(float(dis_loss)))
        gen_losses += [float(gen_loss)]
        dis_losses += [float(dis_loss)]
    if iteration < 5000:
        freq = 500
    elif iteration < 25000:
        freq = 2000
    elif iteration < 70000:
        freq = 5000
    else:
        freq = 10000
    if iteration % freq == 0:
        noise = torch.randn(1000, z_dim)
        if use_cuda:
            noise = noise.cuda()
        noise = autograd.Variable(noise)
        fake_for_plot = gen(noise)
        plot(fake_for_plot, iteration)

plot_eigens(points)

# torch.save(gen.state_dict(), save_directory + 'gen')
# torch.save(dis.state_dict(), save_directory + 'dis')
# np.save(save_directory + 'dis_losses', dis_losses)
# np.save(save_directory + 'gen_losses', gen_losses)
