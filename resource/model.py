
"""Model class for sorting numbers."""

import torch
import torch.nn as nn
import my_sinkhorn_ops

class Sinkhorn_Net(nn.Module):

    def __init__(self, latent_dim, output_dim, dropout_prob):
        super().__init__()
        self.output_dim = output_dim

        # net: output of the first neural network that connects numbers to a
        # 'latent' representation.
        # activation_fn: ReLU is default hence it is specified here
        # dropout p – probability of an element to be zeroed
        self.linear1 = nn.Linear(1, latent_dim)
        self.relu1 = nn.ReLU()
        self.d1 = nn.Dropout(p = dropout_prob)
        # now those latent representation are connected to rows of the matrix
        # log_alpha.
        self.linear2 = nn.Linear(latent_dim, output_dim)
        self.d2 = nn.Dropout(p=dropout_prob)
        self.temp = 1.0
        self.samples_per_num = 5
        self.noise_factor= 1.0
        self.n_iter_sinkhorn = 10

    def forward(self, x):
        n_numbers = x.size(1)
        # each number is processed with the same network, so data is reshaped
        # so that numbers occupy the 'batch' position.
        x = x.view(-1, 1)
        # activation_fn: ReLU
        x = self.d1(self.relu1(self.linear1(x)))
        # no activation function is enabled
        x = self.d2(self.linear2(x))
        #reshape to cubic for sinkhorn operation
        x = x.reshape(-1, self.output_dim, self.output_dim)

        #apply the gumbel sinkhorn on log alpha
        soft_perms_inf, log_alpha_w_noise = \
                my_sinkhorn_ops.my_gumbel_sinkhorn(x, self.temp,
                self.samples_per_num, self.noise_factor, self.n_iter_sinkhorn,
                squeeze=False)

        x = inv_soft_pers_flattened(soft_perms_inf, n_numbers)
        return x

def inv_soft_pers_flattened(soft_perms_inf, n_numbers):
    inv_soft_perms = torch.transpose(soft_perms_inf, 2, 3)
    inv_soft_perms = torch.transpose(inv_soft_perms, 0, 1)

    inv_soft_perms_flat = inv_soft_perms.view(-1, n_numbers, n_numbers)
    return inv_soft_perms_flat
