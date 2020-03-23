import os, sys
sys.path.append(os.getcwd())
import time
from utils import load, save_images, Adamp, SGDNM
import numpy as np
import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
import cifar10

np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
# DATA_DIR = '/mnt/dataset2'
# if not os.path.exists(DATA_DIR):
#     DATA_DIR = '/u/pezeshki/cifar-10-batches-py'

# else:
#     if not os.path.exists('results_2/cifar10'):
#         os.makedirs('results_2/cifar10')
DATA_DIR = '/network/data1/cifar-10-batches-py/'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

print('DATA_DIR: ' + DATA_DIR)
# MODE = 'wgan-gp' # Valid options are dcgan, wgan, or wgan-gp

mode = str(sys.argv[1])
# mode = 'mixed_adam_plus'
print('Mode: ' + mode)

DIM = 128
# LAMBDA = float(sys.argv[6])
LAMBDA = 0

BATCH_SIZE = 64
ITERS = 50000
OUTPUT_DIM = 3072

LR = float(sys.argv[2])
# LR = 0.00001
print('LR: ' + str(LR))

MOM = float(sys.argv[3])
# MOM = -0.46
print('MOM: ' + str(MOM))

bn = str(sys.argv[4])
# bn = 'no'
print('bn: ' + str(bn))

CRITIC_ITERS = int(sys.argv[5])
# CRITIC_ITERS = 1

name = str(mode) + '_lr_' + str(LR) + '_mom_' + str(MOM) + '_bn_' + bn + '_dis_iter_' + str(CRITIC_ITERS)
if not os.path.exists('results_2/' + str(name)):
    os.makedirs('results_2/' + str(name))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        if bn == 'yes':
            preprocess = nn.Sequential(
                nn.Linear(128, 4 * 4 * 4 * DIM),
                nn.BatchNorm1d(4 * 4 * 4 * DIM),
                nn.ReLU(True))
            block1 = nn.Sequential(
                nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),
                nn.BatchNorm2d(2 * DIM),
                nn.ReLU(True))
            block2 = nn.Sequential(
                nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
                nn.BatchNorm2d(DIM),
                nn.ReLU(True))
        else:
            preprocess = nn.Sequential(
                nn.Linear(128, 4 * 4 * 4 * DIM),
                nn.ReLU(True))
            block1 = nn.Sequential(
                nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),
                nn.ReLU(True))
            block2 = nn.Sequential(
                nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
                nn.ReLU(True))

        deconv_out = nn.ConvTranspose2d(DIM, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4 * 4 * 4 * DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4 * 4 * 4 * DIM)
        output = self.linear(output)
        return output

netG = Generator()
netD = Discriminator()
# print(netG)
# print(netD)

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

if mode == 'gp' or mode == 'dc' or mode == 'cp':
    optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.9))

if mode == 'nm':
    optimizerD = SGDNM(netD.parameters(), lr=LR, momentum=MOM)
    optimizerG = SGDNM(netG.parameters(), lr=LR, momentum=MOM)

if 'adamp' in mode:
    optimizerD = Adamp(netD.parameters(), lr=LR, betas=(MOM, 0.9))
    optimizerG = Adamp(netG.parameters(), lr=LR, betas=(MOM, 0.9))

if 'mixed_adam_plus' in mode:
    optimizerD = Adamp(netD.parameters(), lr=LR, betas=(MOM, 0.9))
    optimizerG = Adamp(netG.parameters(), lr=LR, betas=(0.5, 0.9))

if ('mixed_adam' in mode) and ('mixed_adam_plus' not in mode):
    optimizerD = Adamp(netD.parameters(), lr=LR, betas=(MOM, 0.9), md=0)
    optimizerG = Adamp(netG.parameters(), lr=LR, betas=(0.5, 0.9), md=0)

# if 'mixed_sgd' in mode:
#     optimizerD = SGDNM(netD.parameters(), lr=LR, momentum=MOM)
#     optimizerG = SGD(netG.parameters(), lr=LR, momentum=0.9)


def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(
        BATCH_SIZE,
        int(real_data.nelement() / BATCH_SIZE)).contiguous().view(
            BATCH_SIZE, 3, 32, 32)
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    if use_cuda:
        gradients = autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
    else:
        gradients = autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


# For generating samples
def generate_image(frame, netG):
    fixed_noise_128 = torch.randn(128, 128)
    if use_cuda:
        fixed_noise_128 = fixed_noise_128.cuda(gpu)
    with torch.no_grad():
        noisev = autograd.Variable(fixed_noise_128)
    samples = netG(noisev)
    samples = samples.view(-1, 3, 32, 32)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()
    save_images(
        samples, 'results_2/' + str(name) + '/samples_' + str(frame) + '.png')
    # save_images(samples, './samples_{}.jpg'.format(frame))

# Dataset iterator
# train_gen = load(BATCH_SIZE, data_dir=DATA_DIR)
train_gen, dev_gen = cifar10.load(BATCH_SIZE, data_dir=DATA_DIR)


def inf_train_gen():
    while True:
        for images in train_gen():
            yield images
gen = inf_train_gen()
preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

criterion = nn.BCEWithLogitsLoss()
label = torch.FloatTensor(BATCH_SIZE)
if use_cuda:
    criterion.cuda()
    label = label.cuda()

G_costs = []
D_costs = []
for iteration in range(ITERS):
    start_time = time.time()
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
    for i in range(CRITIC_ITERS):
        _data = gen.next()
        # _data = gen.next()
        netD.zero_grad()

        # train with real
        _data = _data.reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
        real_data = torch.stack([preprocess(item) for item in _data])

        if use_cuda:
            real_data = real_data.cuda(gpu)
        real_data_v = autograd.Variable(real_data)

        label.resize_(BATCH_SIZE, 1).fill_(1)
        labelv = autograd.Variable(label)
        output = netD(real_data_v)
        D_cost_real = criterion(output, labelv)
        D_cost_real.backward(retain_graph=True)

        # train with fake
        noise = torch.randn(BATCH_SIZE, 128)
        if use_cuda:
            noise = noise.cuda(gpu)
        with torch.no_grad():
            noisev = autograd.Variable(noise)  # totally freeze netG
        fake = autograd.Variable(netG(noisev).data)
        inputv = fake

        label.resize_(BATCH_SIZE, 1).fill_(0)
        labelv = autograd.Variable(label)
        output = netD(inputv)
        D_cost_fake = criterion(output, labelv)
        D_cost_fake.backward(retain_graph=True)

        if 'gp' in mode:
            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(
                netD, real_data_v.data, fake.data)
            D_cost = D_cost_real + D_cost_fake + gradient_penalty
            gradient_penalty.backward()
        else:
            D_cost = D_cost_real + D_cost_fake

        # D_cost.backward()

        optimizerD.step()
    ############################
    # (2) Update G network
    ###########################
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    netG.zero_grad()

    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise)
    fake = netG(noisev)

    label.resize_(BATCH_SIZE, 1).fill_(1)
    labelv = autograd.Variable(label)
    output = netD(fake)
    if 'sat' in mode:
        label.resize_(BATCH_SIZE, 1).fill_(0)
        labelv = autograd.Varirble(label)
        G_cost = - criterion(output, labelv)
    else:
        G_cost = criterion(output, labelv)
    G_cost.backward()

    optimizerG.step()

    # Calculate dev loss and generate samples every 100 iters
    if iteration % 10 == 0:
        print('iter: ' + str(iteration) + ', ' +
              'G_cost: ' + str(G_cost.cpu().data.numpy()) + ', ' +
              'D_cost: ' + str(D_cost.cpu().data.numpy()) + ', ')
    G_costs += [G_cost.cpu().data.numpy()]
    D_costs += [D_cost.cpu().data.numpy()]
    if iteration % 5000 == 0:
        generate_image(iteration, netG)
        # np.save('./G_costs', np.array(G_costs))
        # np.save('./D_costs', np.array(D_costs))
        np.save('results_2/' + str(name) + '/G_costs', np.array(G_costs))
        np.save('results_2/' + str(name) + '/D_costs', np.array(D_costs))
        torch.save(netG.state_dict(), 'results_2/' + str(name) + '/gen_' + str(iteration))
        torch.save(netD.state_dict(), 'results_2/' + str(name) + '/dis_' + str(iteration))
