import torch
import torch.nn as nn
import settings
settings.initialize()
opt = settings.opt

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# DCGAN GENERATOR
class netG_(nn.Module):
    def __init__(self):
        super().__init__()

        self.init_size = opt.image_size//4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128,0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64,opt.channels,3,stride=1,padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        out = self.l1(input)
        out = out.view(out.shape[0],128,self.init_size,self.init_size)
        img = self.conv_blocks(out)
        return img

# DCGAN DISCRIMINATOR
class netD_(nn.Module):
    def __init__(self):
        super().__init__()

        def netD_block(in_filters, out_filters, bn=True):
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *netD_block(opt.channels, 16, bn=False),
            *netD_block(16, 32),
            *netD_block(32, 64),
            *netD_block(64, 128),
        )

        # HEIGHT AND WIDTH OF THE DOWNSAMPLED IMAGE
        ds_size = opt.image_size//2**4
        self.adv_layer = nn.Sequential(nn.Linear(128*ds_size**2, 1), nn.Sigmoid())


    def forward(self, input):
        out = self.model(input)
        out = out.view(out.shape[0], -1)## make the output same shape as input shape
        validity = self.adv_layer(out)

        return validity
