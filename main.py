from __future__ import print_function
import os
import argparse
import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

#from skimage import io,transform

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
#from torchvision import datasets

import torchvision.datasets as dset
import torch.optim as optim

# LOAD MORE LIBS
from PIL import Image
from multiprocessing import Pool
import numpy as np

## LOAD ARGUMENTS
import settings
settings.initialize()
opt = settings.opt

import models
from models import weights_init_normal

# DEFINE LOSS FUNCTIONS
adversarial_loss = nn.BCELoss()

# INITIALIZE GENERATOR AND DESCRIMINATOR
netG = models.netG_()
netD = models.netD_()

#GPU specific
cuda = True if torch.cuda.is_available() else False

#GPU specific
if cuda:
    print('HAS CUDA')
    netG.cuda()
    netD.cuda()
    adversarial_loss.cuda()

# APPLY WEIGHT
netG.apply(weights_init_normal)
netD.apply(weights_init_normal)

#IF THERE EXIST PATHS TO G AND D, LOAD THEM
if opt.netG != '':
    if opt.latent_dim == 20:
        netG.load_state_dict(torch.load(opt.netG2))
    elif opt.latent_dim == 50:
        netG.load_state_dict(torch.load(opt.netG5))
print(netG)

if opt.netD != '':
    if opt.latent_dim == 20:
        netD.load_state_dict(torch.load(opt.netD2))
    elif opt.latent_dim == 50:
        netD.load_state_dict(torch.load(opt.netD5))
print(netD)

n_cpu = opt.n_cpu

dataset = dset.ImageFolder(
    root=opt.data_root,
    transform=transforms.Compose([
            transforms.Resize(opt.image_size),# resize 96x96 images to 64x64
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),# bring images to (-1,1)
        ])
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

# OPTIMIZERS
optimizer_G = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

#GPU specific
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#Tensor = torch.FloatTensor

# TRACK PROGRESS FOR PLOTTING
D_LOSSES = []
G_LOSSES = []
IMAGE_LIST = []

# TO TRAIN
for epoch in range(opt.n_epochs):
    for i, (images,_) in enumerate(dataloader):

        # ADVERSARIAL GROUND TRUTH
        valid = Tensor(images.shape[0], 1).fill_(1.0)
        valid.requires_grad = False
        fake = Tensor(images.shape[0], 1).fill_(0.0)
        fake.require_grad = False

        # CONFIGURE INPUT
        real_images = images.type(Tensor)

        # TRAIN GENERATOR
        optimizer_G.zero_grad()

        # SAMPLE NOISE AS GENERATOR'S INPUT
        noise = Tensor(np.random.normal(0, 1, (images.shape[0], opt.latent_dim)))

        # generate a batch of images
        gen_images = netG(noise)

        # CALCULATE LOSS (GENERATOR'S ABILITY TO FOOL THE DISCRIMINATOR)
        G_loss = adversarial_loss(netD(gen_images), valid)

        G_loss.backward()
        optimizer_G.step()

        # TRAIN DESCRIMINATOR
        optimizer_D.zero_grad()

        # CALCULATE LOSS (DISCRIMINATOR'S ABILITY TO CLASSIFY REAL AND FAKE SAMPLES)
        real_loss = adversarial_loss(netD(real_images), valid)
        fake_loss = adversarial_loss(netD(gen_images.detach()), fake)
        D_loss = (real_loss + fake_loss)/2

        D_loss.backward()
        optimizer_D.step()

        # CALCULATE DISCRIMINATOR ACCURACY
        # _, argmax = torch.max(outputs, 1)

        print("[Epoch %d %d] [Batch %d %d] [D loss:%f] [G loss %f]" % (epoch, opt.n_epochs, i, len(dataloader), D_loss.item(), G_loss.item()))
        # SAVE LOSSES FOR PLOTTING LATER
        D_LOSSES.append(D_loss.item())
        G_LOSSES.append(G_loss.item())

        # CHECKPOINT: SAVE G AND D
        if epoch % 1 == 0:
            if opt.latent_dim == 20:
                torch.save(netG.state_dict(), '%s/netG2.pth' % (opt.outDir))
                torch.save(netD.state_dict(), '%s/netD2.pth' % (opt.outDir))
            elif opt.latent_dim == 50:
                torch.save(netG.state_dict(), '%s/netG5.pth' % (opt.outDir))
                torch.save(netD.state_dict(), '%s/netD5.pth' % (opt.outDir))
        batches_done = epoch * len(dataloader) + 1

        # SAVE IMAGES EVERY FEW ITERATIONS
        if i == 0:
            print("save->")
            save_image(gen_images.data[9:16], '%s/%d.png' % (opt.outDir, batches_done), nrow=8, normalize=True)

        # PLOTTING
        if i == 0:
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(G_LOSSES, label="G")
            plt.plot(D_LOSSES, label="D")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
        # TENSOR BOARD LOGGING
        # 1. LOG SCALAR SUMMARY
        # info = {'loss':loss.item(),'accuracy': acc}