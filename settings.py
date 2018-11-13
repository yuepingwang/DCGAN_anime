import argparse

def initialize():
    global opt
    parser = argparse.ArgumentParser()
    parser.add_argument('--netG10', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD10', default='', help="path to netD (to continue training)")
    parser.add_argument('--netG2', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD2', default='', help="path to netD (to continue training)")
    parser.add_argument('--netG5', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD5', default='', help="path to netD (to continue training)")
    parser.add_argument('--data_root', required=True, help='path to the dateset (real images)')
    parser.add_argument('--outDir', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--n_epochs', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='for adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='for adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='for adam: decay of second order momentum of gradient')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of CPU threads to use during batch generation')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--channels', type=int, default=3, help='the number of image channels(RGB)')
    parser.add_argument('--sample_interval', type=int, default=8, help='interval between image sampling')

    # SPECIFY PARAMS
    arg_list = [
        '--data_root', '/Users/ywang/Desktop/small_set',
        '--batch_size', '64',
        '--image_size', '64',## rescale actual dataset to this size
        '--latent_dim', '100',## start w/ small number
        '--n_epochs', '40',## start w/ small number of iterations
        '--lr', '0.0002',
        '--b1', '0.5',
        '--n_cpu', '6',
        '--netG10', '/Users/ywang/Desktop/GANs/DCGAN/results/netG10.pth',
        '--netD10', '/Users/ywang/Desktop/GANs/DCGAN/results/netD10.pth',
        '--netG2', '/Users/ywang/Desktop/GANs/DCGAN/results/netG2.pth',
        '--netD2', '/Users/ywang/Desktop/GANs/DCGAN/results/netD2.pth',
        '--netG5', '/Users/ywang/Desktop/GANs/DCGAN/results/netG5.pth',
        '--netD5', '/Users/ywang/Desktop/GANs/DCGAN/results/netD5.pth',
        '--outDir', '/Users/ywang/Desktop/GANs/DCGAN/results',
        ]
    opt = parser.parse_args(arg_list)