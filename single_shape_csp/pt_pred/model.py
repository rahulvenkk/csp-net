import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def init_weights(m):
    """

    Parameters
    ----------
    m : nn.Module

    Returns
    -------

    """
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class Net(nn.Module):
    def __init__(self, omega=30):
        super(Net, self).__init__()
        
        np.random.seed(0)

        self.in_size = 3
        self.omega = omega
        
        fc_layers = []
        out_sizes = [self.in_size, 120, 512, 1024, 2048, \
                                  2048, 1024, 512,  256, 128, 3]
        for i in range(len(out_sizes) - 1):
            fc_layers.extend([nn.Linear(out_sizes[i], out_sizes[i+1]), nn.ReLU()])
        self.net = nn.Sequential(*fc_layers[:-1]) # drop the last layer's ReLU

    def init_weights(self):
        with torch.no_grad():
            for name, W in self.named_parameters():
                if type(W) == nn.Linear:
                    if self.is_first:
                        W.uniform_(-1 / self.mapping_size,
                                                     1 / self.mapping_size)
                    else:
                        W.uniform_(-np.sqrt(6 / self.mapping_size) / self.omega,
                                                     np.sqrt(6 / self.mapping_size) / self.omega)

    def loss(self, pred, gt):
        return torch.mean(((pred - gt) ** 2).sum(-1))
    
    def cosine_loss(self, pred, gt):
        pred = pred / torch.norm(pred, dim=-1, keepdim=True)
        gt = gt / torch.norm(gt, dim=-1, keepdim=True)

        return (1. - (pred * gt).sum(-1)).mean()

    def get_dist(self, query_points, cs_points):
        dist = torch.sqrt(torch.square(cs_points - query_points).sum(-1))
        return dist

    def get_normal(self, query_points, cs_points):
        # points are query points B x 3
        # pred_nn is predicted nearest point on the surface
        normal = query_points - cs_points
        dist = self.get_dist(query_points, cs_points).unsqueeze(-1)
        normal = normal / dist
        return normal

    def forward(self, x_in):
        return self.net(x_in)