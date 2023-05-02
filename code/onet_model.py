import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class OnetDataset(torch.utils.data.Dataset):
    def __init__(self,
                 filename='../data/onet_train.pt'):
        self.k, self.u = torch.load(filename)
        self.length = self.k.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.k[idx], self.u[idx])


class OnetWaveEqnDataset(torch.utils.data.Dataset):
    def __init__(self,
                 filename='../data/onet_wave_train.pt'):
        self.u = torch.load(filename)
        self.u_len = self.u.shape[1]
        self.length = self.u.shape[0] * (self.u_len - 2)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_idx = idx // (self.u_len - 2)
        sample_idx = idx - (batch_idx * (self.u_len)) + 2

        return (self.u[batch_idx, sample_idx - 2, :, :],
                self.u[batch_idx, sample_idx - 1, :, :],
                self.u[batch_idx, sample_idx, :, :])


def interp_image(img, pts):
    '''
    Parameters
    ----------
    img : torch.Tensor
      B x W x H
    pts : torch.Tensor
      B x Npts x 2

    Returns
    -------
    interp : torch.Tensor
      B x Npts
    '''

    B = pts.shape[0]
    Npts = pts.shape[1]

    pts = pts * 2. - 1. # transform into the range [-1, 1]
    pts = pts.unsqueeze(2) # B x Npts x 1 x 2
    img = img.unsqueeze(1) # B x 1 x W x H

    interp = F.grid_sample(img, pts, align_corners=True)
    return interp.reshape((B, 1, Npts))


class BranchNet(nn.Module):
    def __init__(self, d_in, H, N_v):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(d_in, H, 1), nn.Tanh(),
            nn.Conv1d(H, H, 1), nn.Tanh(),
            nn.Conv1d(H, H, 1), nn.Tanh(),
            nn.Conv1d(H, 1, 1)
        )

    def forward(self, kappa_x):
        return self.model(kappa_x)


class TrunkNet(nn.Module):
    def __init__(self, H, N_v):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2 * N_v, N_v), nn.Tanh(),
            nn.Linear(N_v, N_v), nn.Tanh(),
            nn.Linear(N_v, N_v)
        )

    def forward(self, y):
        return self.model(y.flatten(1)).unsqueeze(1)


class Onet(nn.Module):
    def __init__(self, d_in, H, N_v):
        super().__init__()

        self.branch = BranchNet(d_in, H, N_v)
        self.trunk = TrunkNet(H, N_v)

    def forward(self, kappa_x, y):
        b = self.branch(kappa_x)
        t = self.trunk(y.transpose(1, 2))

        return (b*t).squeeze()
