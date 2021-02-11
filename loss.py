"""Loss functions

This script defines the custom loss function for the CUT-GAN framework;
namely, the PatchNCELoss
"""
import torch
from torch import nn

class PatchNCELoss(nn.Module):
    """PatchNCELoss

    modified from:
        https://github.com/taesungp/contrastive-unpaired-translation/
            models/patchnce.py
    """
    def __init__(self, tau=7e-2, all_negatives_from_minibatch=False):
        super().__init__()
        self.all_negatives_from_minibatch = all_negatives_from_minibatch
        self.tau = tau
        self.CE = nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool

    def forward(self, proj_A, proj_B, batch_size):
        bp, dim = proj_A.shape
        proj_A = proj_A.detach()
        #######################################################################
        # pos logit
        #######################################################################
        l_pos = torch.bmm(proj_A.view(bp, 1, -1), proj_B.view(bp, -1, 1))
        l_pos = l_pos.view(bp, 1)

        #######################################################################
        # neg logit
        #######################################################################

        # for single-image translation, include the negatives from
        # the entire minibatch.
        if self.all_negatives_from_minibatch:
            # reshape features as if they are all negatives of
            # minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        # reshape features to batch size
        proj_A = proj_A.view(batch_dim_for_bmm, -1, dim)
        proj_B = proj_B.view(batch_dim_for_bmm, -1, dim)
        n_patch = proj_A.size(1)
        l_neg_curbatch = torch.bmm(proj_A, proj_B.transpose(2, 1))
        # diagonal entries are similarity between same features, and hence
        # just fill the diagonal with very small number, which is exp(-10)
        diagonal = torch.eye(
            n_patch, device=proj_A.device, dtype=self.mask_dtype
        )[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, n_patch)
        out = torch.cat((l_pos, l_neg), dim=1) / self.tau
        # target = 0, since we want to maximize l_pos, which has index 0
        target = torch.zeros(
            out.size(0), dtype=torch.long, device=proj_A.device
        )
        loss = self.CE(out, target)

        return loss
