"""Model training

author: Masahiro hayashi

This script defines the training procedure for the CUT-GAN framework
"""
import os
import argparse
import time

import torch
from torch import nn, optim, cuda
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms, datasets
import numpy as np
from matplotlib import pyplot as plt

from cut import ResnetGenerator, PatchGAN, ProjectionHead
from loss import PatchNCELoss
from afhq import AFHQ

def parse_args():
    """Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Train CUTGAN')
    parser.add_argument(
        '--batch-size', type=int, default=1, metavar='N',
        help='input batch size for training (default: 1)'
    )
    parser.add_argument(
        '--val-batch-size', type=int, default=1, metavar='N',
        help='input batch size for validation (default: 1)'
    )
    parser.add_argument(
        '--epochs', type=int, default=10, metavar='N',
        help='number of epochs to train (default: 10)'
    )
    parser.add_argument(
        '--lr', type=float, default=2e-4, metavar='LR',
        help='learning rate (default: 0.0002)'
    )
    parser.add_argument(
        '--momentum', type=float, default=0.5, metavar='M',
        help='SGD momentum (default: 0.5)'
    )
    parser.add_argument(
        '--nce_T', type=float, default=0.07, help='temperature for NCE loss'
    )
    parser.add_argument(
        '--num_workers', type=int, default=6,
        help='number of workers to load data'
    )
    parser.add_argument(
        '--no-cuda', action='store_true', default=False,
        help='disables CUDA training'
    )
    parser.add_argument(
        '--amp', action='store_true', default=False,
        help='automatic mixed precision training'
    )
    parser.add_argument(
        '--opt-level', type=str
    )
    parser.add_argument(
        '--keep_batchnorm_fp32', type=str, default=None,
        help='keep batch norm layers with 32-bit precision'
    )
    parser.add_argument(
        '--loss-scale', type=str, default=None
    )
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)'
    )
    parser.add_argument(
        '--log-interval', type=int, default=1, metavar='N',
        help='how many batches to wait before logging training status'
    )
    parser.add_argument(
        '--save', action='store_true', default=False,
        help='save the current model'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='model to retrain'
    )
    parser.add_argument(
        '--tensorboard', action='store_true', default=False,
        help='record training log to Tensorboard'
    )
    args = parser.parse_args()
    return args

def epoch_time(start_time, end_time):
    """Convert time to minutes and seconds
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_train_loader(mean=None, std=None, batch_size=1):
    image_transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])
    train_data = AFHQ(
        root='afhq/train/',
        transforms=image_transform
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )
    return train_loader

def get_val_loader(mean=None, std=None, batch_size=1):
    image_transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])
    test_data = AFHQ(
        root='afhq/val/',
        transforms=image_transform
    )
    val_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )
    return val_loader

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations

    Args:
        nets (list(nn.Module)): a list of networks
        requires_grad (bool): whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def train(
    G, H, D, optimizer_G, optimizer_H, optimizer_D,
    criterion_GAN, criterion_NCE,
    layers_nce,
    train_loader, device, epoch, log_interval
):
    """Train network for one epoch
    """
    G.train()
    H.train()
    D.train()
    # print(D)
    loss = 0.
    for step, (X, Y) in enumerate(train_loader):
        B, C, h, w = X.size()
        patch_size = 16
        n_patch = B * (h // patch_size) * (w // patch_size)
        value_real = 1.
        value_fake = 0.
        label_real = torch.full(
            (n_patch, 1, patch_size, patch_size),
            value_real, dtype=torch.float, device=device
        )
        label_fake = torch.full(
            (n_patch, 1, patch_size, patch_size),
            value_fake, dtype=torch.float, device=device
        )

        #######################################################################
        # Update Discriminator D
        #######################################################################
        set_requires_grad(D, True)
        real_X = X.to(device)
        fake_Y, encoded_real_X = G(real_X, layers_nce)
        real_Y = Y.to(device)
        score_real = D(real_Y)
        score_fake = D(fake_Y.detach())

        # Least square GAN loss
        # V(D) = 0.5 * E_x[(D(x) - 1)^2] + 0.5 * E_z[(D(G(z)))^2]
        loss_GAN_D = 0.5 * criterion_GAN(score_real, label_real) ** 2
        loss_GAN_D += 0.5 * criterion_GAN(score_fake, label_fake) ** 2
        loss_D = loss_GAN_D
        # update parameters in D
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        #######################################################################
        # Update Generator G
        #######################################################################
        set_requires_grad(D, False)

        encoded_fake_Y = G(fake_Y, layers_nce, encode_only=True)
        fake_X, encoded_real_Y = G(real_Y, layers_nce)
        encoded_fake_X = G(fake_X, layers_nce, encode_only=True)
        # Projections for NCE calculation
        projections_real_X, i_X = H(encoded_real_X)
        projections_fake_Y, _ = H(encoded_fake_Y, idx_patch=i_X)
        projections_real_Y, i_Y  = H(encoded_real_Y)
        projections_fake_X, _ = H(encoded_fake_X, idx_patch=i_Y)

        # Least square GAN loss
        # V(G) = 0.5 * E_z[(D(G(x)) - 1)^2]
        score_fake = D(fake_Y)
        loss_GAN_G = 0.5 * criterion_GAN(score_fake, label_real)
        # Weighting for NCE loss
        lambda_X = 1
        lambda_Y = 1
        # NCE Loss: L_NCE(G, H, X)
        NCE_X = [
            criterion(proj_X, proj_Y, B).mean()
            for proj_X, proj_Y, criterion in zip(
                projections_real_X, projections_fake_Y, criterion_NCE
            )
        ]
        # Identity Loss: L_NCE(G, H, Y)
        NCE_Y = [
            criterion(proj_Y, proj_X, B).mean()
            for proj_Y, proj_X, criterion in zip(
                projections_real_Y, projections_fake_X, criterion_NCE
            )
        ]
        # Total loss for G: L_GAN + L_NCE
        n_layer = len(layers_nce)
        NCE = (lambda_X * sum(NCE_X) + lambda_Y * sum(NCE_Y)) / n_layer
        loss_G = loss_GAN_G + NCE
        # update parameters in G and H
        optimizer_G.zero_grad()
        optimizer_H.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        optimizer_H.step()

        loss = loss_D + loss_G
        # log progress
        if (step+1) % log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (step+1) * B, len(train_loader.dataset),
                    100. * (step+1) / len(train_loader), loss.item()
                )
            )
        # break
        # im_fake_B = visualize(G)
        # writer.add_image('Fake Dog', im_fake_B)

    return loss.item(), loss_G.item(), loss_D.item()

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Args:
        input_image (tensor): the input image tensor array
        imtype (type): the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        # get the data from a variable
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        # convert it into a numpy array
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # post-processing: tranpose and scaling
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def init_visualize(real_X, fake_Y, real_Y):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    im1 = plt.imshow(real_X)
    ax1.set_title('Real Cat')
    ax2 = fig.add_subplot(1, 3, 2)
    plt.imshow(fake_Y)
    ax2.set_title('Fake Dog')
    ax3 = fig.add_subplot(1, 3, 3)
    plt.imshow(real_Y)
    ax3.set_title('Real Dog')
    fig.tight_layout()
    plt.draw()


def visualize(G):
    G.eval()
    fake_B = G(real_A)
    im_fake_B = tensor2im(fake_B)
    im_fake_B = np.moveaxis(im_fake_B, -1, 0)
    # im_fake_B = np.moveaxis(im_fake_B, (-1, 256, 256, 3))
    # print(im_fake_B.shape)
    return im_fake_B


def initialize_model_dict(args):
    model_dict = {
        'total_epoch': args.epochs,
        'model_state_dict': None,
        'optimizer_state_dict': None,
        'train_loss': {
            'G': list(),
            'D': list(),
            'total': list()
        },
        'metrics': {
            'last': {
                'loss': None,
                'epoch': 0
            },
            'best': {
                'loss': None,
                'epoch': 0
            }
        }
    }
    return model_dict

def get_model(args, layers_nce, device):
    if args.model:
        # Load model checkpoint
        model_path = os.path.join(os.getcwd(), f'models/{args.model}')
        model_dict = torch.load(model_path)
    else:
        model_dict = initialize_model_dict(args)
    G = ResnetGenerator().to(device)
    X = torch.rand((1, 3, 256, 256))
    encoded_X = G(X, layers_nce, encode_only=True)
    H = ProjectionHead().to(device)
    _, _ = H(encoded_X)
    D = PatchGAN().to(device)
    optimizer_G = optim.Adam(G.parameters(), args.lr)
    optimizer_H = optim.Adam(H.parameters(), args.lr)
    optimizer_D = optim.Adam(D.parameters(), args.lr)
    if args.model:
        G.load_state_dict(model_dict['model_state_dict'])
        optimizer_G.load_state_dict(model_dict['optimizer_state_dict'])
    return G, H, D, optimizer_G, optimizer_H, optimizer_D, model_dict

if __name__ == '__main__':
    # get command line arguments
    args = parse_args()
    # initialize training settings
    if args.tensorboard:
        writer = SummaryWriter()
    device = (
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )
    train_loader = get_train_loader(batch_size=args.batch_size)
    val_loader = get_val_loader(batch_size=args.val_batch_size)
    layers_nce = [0, 2, 3, 4, 8]
    # initialize model
    print('Initialize model...', end='')
    G, H, D, optimizer_G, optimizer_H, optimizer_D, model_dict = get_model(
        args, layers_nce, device
    )
    print('Done')
    # Initialize visualization
    real_A, real_B = next(iter(val_loader))
    if args.tensorboard:
        writer.add_image(
            'Image/Real Cat',
            np.moveaxis(tensor2im(real_A), -1, 0)
        )
        writer.add_image(
            'Image/Real Dog',
            np.moveaxis(tensor2im(real_B), -1, 0)
        )
    # initialize loss functions
    criterion_GAN = nn.MSELoss()
    criterion_NCE = [PatchNCELoss().to(device) for layer in layers_nce]
    # Training Loop
    start_epoch = 1 if not args.model else model_dict['total_epoch'] + 1
    n_epoch = start_epoch + args.epochs - 1
    model_dict['total_epoch'] = n_epoch
    model_name = f'models/CUT{n_epoch}.pt'
    for epoch in range(start_epoch, n_epoch+1):
        start_time = time.time()
        train_loss, train_loss_G, train_loss_D = train(
            G, H, D, optimizer_G, optimizer_H, optimizer_D,
            criterion_GAN, criterion_NCE, layers_nce,
            train_loader, device, epoch, args.log_interval
        )
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # print statistics and update training progress
        print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
        im_fake_B = visualize(G)
        if args.tensorboard:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalars(
                'Loss/G_D',
                {'G': train_loss_G, 'D': train_loss_D},
                epoch
            )
            writer.add_image('Image/Fake Dog', im_fake_B, epoch)
        # log results to model dictionary
        model_dict['train_loss']['G'].append(train_loss)
        model_dict['train_loss']['D'].append(train_loss_G)
        model_dict['train_loss']['total'].append(train_loss_D)
        model_dict['metrics']['last']['loss'] = train_loss
        model_dict['metrics']['last']['epoch'] = epoch
        if epoch == 1 or train_loss < model_dict['metrics']['best']['loss']:
            model_dict['model_state_dict'] = G.state_dict()
            model_dict['optimizer_state_dict'] = optimizer_G.state_dict()
            model_dict['metrics']['best']['epoch'] = epoch
            model_dict['metrics']['best']['loss'] = train_loss
        if args.save:
            torch.save(model_dict, model_name)
