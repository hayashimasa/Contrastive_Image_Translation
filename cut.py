"""Model architectures for Contrastive Unpaired Translation

author: Masahiro Hayashi

This script defines the network architectures for the Contrastive Unpaired
Translation framework, which consists of 3 components: a generator,
a discriminator, and a projection head.

The file is organized into 4 sections:
    - the building blocks for defining the network
    - one section for each component of the CUT-GAN framework

For testing, please run this script as the main program. It will make one
forward pass for each network defined in this script.
"""
import torch
from torch import nn

###############################################################################
# Building Blocks
###############################################################################
class ConvBnReLU(nn.Module):
    """Convoutional-Batch Normalization-ReLU block
    """
    def __init__(self, in_dim, out_dim, filter_size=3, stride=1, padding=1):
        self.name = 'Conv-BN-ReLU'
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, filter_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvBnLeakyReLU(ConvBnReLU):
    """Convoutional-Batch Normalization-LeakyReLU block
    """
    def __init__(self, in_dim, out_dim, filter_size=3, stride=1, padding=1):
        super(ConvBnLeakyReLU, self).__init__(in_dim, out_dim)
        self.name = 'Conv-BN-LeakyReLU'
        self.relu = nn.LeakyReLU(2e-1, inplace=True)

class ResidualBlock(nn.Module):
    """Basic Residiual Block
    """
    def __init__(
        self, in_dim, out_dim, padding=1, padding_type='reflect',
        downsample=None, stride=1, use_dropout=False
    ):
        self.name = 'Basic Residual Block'
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        _pad = {
            'reflect': nn.ReflectionPad2d(padding),
            'replicate': nn.ReflectionPad2d(padding)
        }
        self.block = []
        if padding_type == 'zero':
            self.padding = 1
        else:
            self.padding = 0
            self.pad = _pad.get(padding_type, None)
            if self.pad is None:
                NotImplementedError(
                    f'padding [{padding_type}] is not implemented'
                )
        # conv-bn-relu 1
        if self.padding == 0:
            self.block.append(self.pad)
        dropout = nn.Dropout2d(0.5) if use_dropout else None
        conv1 = nn.Conv2d(
            in_dim, out_dim, 3, stride=stride, padding=self.padding
        )
        bn1 = nn.BatchNorm2d(out_dim)
        relu = nn.ReLU(inplace=True)
        self.block += [conv1, bn1, relu]
        if use_dropout:
            self.block.append(dropout)
        # conv-bn-relu 2
        if self.padding == 0:
            self.block.append(self.pad)
        conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=self.padding)
        bn2 = nn.BatchNorm2d(out_dim)
        self.block += [conv2, bn2]
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        identity = x
        # if not self.downsample is None:
        #     identity = self.downsample(identity)
        residual =self.block(x)
        out = identity + residual
        # out = self.relu(out)
        return out

class DeconvBnReLU(nn.Module):
    """Decovolution-Batch Normalization-ReLU block
    """

    def __init__(
        self, in_dim, out_dim, filter_size=3, stride=2,
        padding=1, output_padding=1
    ):
        self.name = 'Deconv-BN-ReLU'
        super(DeconvBnReLU, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_dim, out_dim, filter_size, stride, padding, output_padding
        )
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Normalize(nn.Module):
    """Normalization layer
    """
    def __init__(self, power=2, eps=1e-7):
        super(Normalize, self).__init__()
        self.power = power
        self.eps = eps

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + self.eps)
        return out

###############################################################################
# Generator (G)
###############################################################################

class ResnetGenerator(nn.Module):
    """ResNet-based Generator model architecture
    """
    def __init__(self, im_dim=3, n_blocks=9, dropout=False):
        self.name = 'Generator'
        super(ResnetGenerator, self).__init__()
        # params
        self.im_dim = im_dim
        self.filter_size = [64, 128, 256]
        self.n_blocks = n_blocks
        # initialize model
        self.model = []
        # input
        rpad = nn.ReflectionPad2d(3)
        conv1 = ConvBnReLU(self.im_dim, self.filter_size[0], 7, 1, 0)
        self.model += [rpad, conv1]
        # Downsampling blocks
        down1 = ConvBnReLU(self.filter_size[0], self.filter_size[1], 3, 2)
        down2 = ConvBnReLU(self.filter_size[1], self.filter_size[2], 3, 2)
        self.model += [down1, down2]
        # Residual blocks
        resblock = [
            ResidualBlock(self.filter_size[2], self.filter_size[2])
            for _ in range(self.n_blocks)
        ]
        self.model += resblock
        # Upsampling blocks
        up1 = DeconvBnReLU(self.filter_size[2], self.filter_size[1])
        up2 = DeconvBnReLU(self.filter_size[1], self.filter_size[0])
        self.model += [up1, up2]
        # output
        output = nn.Conv2d(self.filter_size[0], self.im_dim, 7)
        self.model += [rpad, output, nn.Tanh()]

        self.model = nn.Sequential(*self.model)

    def forward(self, x, layers=[], encode_only=False):
        """Forward pass of model

        x (Tensor): input
        layers (list(int)): indicies of desired intermediate features
        encode_only (bool): only return intermediate features
        """
        if not layers:
            fake = self.model(x)
            return fake
        else:
            features = [] # intermediate features
            for i, layer in enumerate(self.model):
                x = layer(x)
                if i in layers:
                    features.append(x)
                if i == layers[-1] and encode_only:
                    return features
            return x, features

###############################################################################
# Discriminator (D)
###############################################################################

class PatchGAN(nn.Module):
    """PatchGAN architecture
    """
    def __init__(self):
        self.name = 'PatchGAN'
        super(PatchGAN, self).__init__()
        # self.patch_size = patch_size
        self.filter_size = [64, 128, 256, 512]
        self.block = self._make_block()
        self.output = nn.Conv2d(self.filter_size[-1], 1, 1)

    def _make_block(self):
        blocks = []
        in_dim = 3
        for i, n_filter in enumerate(self.filter_size):
            out_dim = n_filter
            block = ConvBnLeakyReLU(in_dim, out_dim, 4, stride=2)
            blocks.append(block)
            self.add_module(f'ConvBnLeakyReLU{i}', block)
            in_dim = out_dim
        return blocks

    def forward(self, x):
        B, C, H, W = x.size()
        size = 16
        Y = H // size
        X = W // size
        x = x.view(B, C, Y, size, X, size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * Y * X, C, size, size)
        for block in self.block:
            x = block(x)
        out = self.output(x)
        return out

###############################################################################
# Projection Head
###############################################################################


class ProjectionHead(nn.Module):
    """MLP Projection Head for feature maps
    """

    def __init__(self, h_dim=256, init_mlp=False, mlp=[]):
        self.name = 'MLP Projection Head'
        super(ProjectionHead, self).__init__()
        self.h_dim = h_dim
        self.init_mlp = init_mlp
        self.mlp = mlp
        self.l2norm = Normalize(2)

    def _make_MLP(self, features):
        for i, feature in enumerate(features):
            B, C, H, W = feature.size()
            mlp = nn.Sequential(
                nn.Linear(C, self.h_dim), # one node per channel
                nn.ReLU(True),
                nn.Linear(self.h_dim, self.h_dim)
            )
            self.mlp.append(mlp)
            self.add_module(f'mlp{i}', mlp)
        self.init_mlp = True

    def forward(self, features, n_patches=64, idx_patch=None):
        """Make one forward pass

        Args:
            features (Tensor): intermediate feautres for projection
            n_patches: number of patches to select
            idx_patch (list(list(int))): indices of patches for each feature
        """
        i_select = [] # indicies of selected patches for each feature
        projections = []
        # initialize mlp
        if not self.init_mlp:
            self._make_MLP(features)
        device = features[0].device
        for i, feature in enumerate(features):
            B, C, H, W = feature.size()
            feature = feature.permute(0, 2, 3, 1).flatten(1, 2)
            # get indices of patches to select
            if idx_patch is None:
                i_patch = torch.randperm(H*W, device=device)
                i_patch = i_patch[:int(min(n_patches, i_patch.shape[0]))]
            else:
                i_patch = idx_patch[i]
            feature = feature[:, i_patch, :].flatten(0, 1)
            # forward pass through MLP projection head
            feature_projection = self.mlp[i](feature)
            feature_projection = self.l2norm(feature_projection)
            i_select.append(i_patch)
            projections.append(feature_projection)
        return projections, i_select

###############################################################################
# For Testing
###############################################################################
if __name__ == '__main__':
    layers = [0, 4, 8, 12, 16]
    X = torch.rand((1, 3, 256, 256))
    G = ResnetGenerator()
    # print(G.shape)

    fake, encoded_X = G(X, layers)
    print([enc_x.shape for enc_x in encoded_X])
    print(fake.shape)

    H = ProjectionHead()
    proj_X, _ = H(encoded_X)
    # print(H)
    print([proj.shape for proj in proj_X])


    Y = torch.rand((1, 3, 256, 256))
    D = PatchGAN()
    # print(D)
    t = D(Y)
    print(t.shape)
    del G
    del H
    del D
