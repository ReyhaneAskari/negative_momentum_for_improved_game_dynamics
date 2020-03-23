import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim import SGD
from torch.optim.optimizer import required
from gradient_color import linear_gradient
import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate_idx', type=int, default=0)
parser.add_argument('--epochs', type=int, default=5000000)
args = parser.parse_args()
update_rule = 'sgd'
sgd_sim = False
epochs = args.epochs
starting_phi = 0.5
starting_theta = 0.5
start_to_nash_distance = starting_phi ** 2 + starting_theta ** 2
grid_length = 2
momentum = -0.5
grad_scale = 15
learning_rate_idx = args.learning_rate_idx

epsilon = 0.001
lr_N = 4
mom_N = 4
max_distance = 10

# lr_list = ([1 * 10**(lr) for lr in np.linspace(-4, 0.1, lr_N)])
lr_list = [0.1, 0.001, 0.0001]
lr_list.sort()
learning_rate = lr_list[learning_rate_idx]
# mom_list = np.arange(-1.2, 0.1, 1.3 / mom_N)
mom_list = [-0.9, -0.5, -0.1, 0, 0.1, 0.5, 0.9]

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


def plot_fig(dis_grad_list, gen_grad_list):
    plt.figure()
    plt.scatter(
        dis_grad_list, gen_grad_list,
        color=linear_gradient('#445ACC', '#b30000', epochs + 1), s=1)
    my_mesh = np.array(
        np.meshgrid(
            np.linspace(-grid_length, grid_length, 20),
            np.linspace(-grid_length, grid_length, 20))).reshape((2, -1)).T
    for i in range(len(my_mesh)):
        grad_x = my_mesh[i][1]
        grad_y = my_mesh[i][0]
        plt.arrow(my_mesh[i][0], my_mesh[i][1],
                  -grad_x / grad_scale, grad_y / grad_scale,
                  head_width=0.02, head_length=0.02,
                  color='gray')

    plt.xlim([-grid_length, grid_length])
    plt.ylim([-grid_length, grid_length])

    plt.savefig('figs_2/' + update_rule +
                '_' + str(sgd_sim) + '_' +
                str(learning_rate) + '_' + str(momentum) + '.png', dpi=300)


class Dis(nn.Module):
    def __init__(self):
        super(Dis, self).__init__()
        self.theta = Parameter(torch.Tensor(1))
        self.theta.data = torch.FloatTensor([starting_theta])

    def forward(self):
        return self.theta


class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.phi = Parameter(torch.Tensor(1))
        self.phi.data = torch.FloatTensor([starting_phi])

    def forward(self):
        return self.phi


def train(update_rule, learning_rate, momentum, sgd_sim):
    conv_time = -2
    dis = Dis()
    gen = Gen()
    optimizer_d = SGDNM(
        dis.parameters(), lr=learning_rate, momentum=momentum)
    optimizer_g = SGDNM(
        gen.parameters(), lr=learning_rate, momentum=momentum)
    dis_grad_list = []
    gen_grad_list = []
    for i in range(epochs):
        gen_out = gen()
        dis_out = dis()
        loss_d = gen_out.detach() * dis_out
        loss_g = -gen_out * dis_out.detach()
        optimizer_d.zero_grad()
        optimizer_g.zero_grad()
        if sgd_sim:
            loss_d.backward()
            loss_g.backward()
            optimizer_d.step()
            optimizer_g.step()
        else:
            loss_d.backward()
            optimizer_d.step()
            loss_g.backward()
            optimizer_g.step()
        dis_grad_list.append(dis.theta.grad.cpu().data.numpy()[0])
        gen_grad_list.append(gen.phi.grad.cpu().data.numpy()[0])
        distance = gen.phi.data ** 2 + dis.theta.data**2
        distance = distance.cpu().numpy()[0]
        if distance < epsilon:
            conv_time = i
            break
        if distance > max_distance:
            break
    if distance > start_to_nash_distance:
        conv_time = -1
    elif distance == start_to_nash_distance:
        conv_time = - 0.5
    elif distance > epsilon:
        conv_time = 0
    plot_fig(dis_grad_list, gen_grad_list)
    return conv_time


conv_times = np.ones((mom_N)) * -1
for idx_m, momentum in enumerate(mom_list):
    print momentum
    conv_times[idx_m] = train(update_rule, learning_rate, momentum, sgd_sim)
    print conv_times[idx_m]

# np.save(("new_np_out/" +
#          update_rule + '_' + str(sgd_sim) + '_' +
#          str(learning_rate_idx)), conv_times)
# np.load("out_np/"+ update_rule + '_' + str(sgd_sim) + '_' + str(lr) + '_' + str(momentum)))

# np.save("conv_matrix_learning_rate", conv_times)
# plt.figure()
# plt.imshow(conv_times, interpolation='nearest')
# plt.savefig("out_learning_rate.png")
