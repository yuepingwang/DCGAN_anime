from __future__ import print_function
import os
import argparse
import numpy as np
import math
import random
import time

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
# criterion = nn.BCELoss()
# criterion_MSE = nn.MSELoss()

# INITIALIZE GENERATOR AND DESCRIMINATOR
netG = models.netG_()
netD = models.netD_()

# APPLY WEIGHT
netG.apply(weights_init_normal)
netD.apply(weights_init_normal)

# IF THERE EXIST PATHS TO G AND D, LOAD THEM
if opt.netG != '':
    if opt.latent_dim == 20:
        netG.load_state_dict(torch.load(opt.netG2))
    elif opt.latent_dim == 10:
        netG.load_state_dict(torch.load(opt.netG))
print(netG)

if opt.netD != '':
    if opt.latent_dim == 20:
        netD.load_state_dict(torch.load(opt.netD2))
    elif opt.latent_dim == 10:
        netD.load_state_dict(torch.load(opt.netD))
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

Tensor = torch.FloatTensor

real_label = 1
fake_label = 0

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
        print("[Epoch %d %d] [Batch %d %d] [D loss:%f] [G loss %f]" % (epoch, opt.n_epochs, i, len(dataloader), D_loss.item(), G_loss.item()))

        # CHECKPOINT: SAVE G AND D
        if epoch % 1 == 0:
            if opt.latent_dim == 20:
                torch.save(netG.state_dict(), '%s/netG2.pth' % (opt.outDir))
                torch.save(netD.state_dict(), '%s/netD2.pth' % (opt.outDir))
            elif opt.latent_dim == 10:
                torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outDir))
                torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outDir))
        batches_done = epoch * len(dataloader) + 1

        # SAVE IMAGES EVERY FEW ITERATIONS
        if epoch % 1 == 0 and i == 98:
            print("save->")
            save_image(gen_images.data[9:16], '%s/%d.png' % (opt.outDir, batches_done), nrow=8, normalize=True)