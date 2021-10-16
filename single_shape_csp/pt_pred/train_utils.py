import torch
import viz_utils
import numpy as np
import time

def get_preds(net, query_pts):
    # Get predicted DF value from the network
    pred_csp = net(query_pts)
    pred_dist = net.get_dist(query_pts, pred_csp)
    pred_normal = net.get_normal(query_pts, pred_csp)

    return pred_csp, pred_dist, pred_normal

def get_losses(net, pred_csp, cspts):
    loss_pts = net.loss(pred_csp, cspts)
    return loss_pts

def train_val_step(net, query_pts, cspts, optim=None):
    """
    do one forward pass through the model

    Parameters
    ----------
    net : pytorch nn.Model
        pytorch model object
    query_pts : ndarray [N, 3]
        sampled query points
    cspts : ndarray [N, 3]
        closest surface points to the query points

    Returns
    -------
    loss_pts : float
    Loss on closest surface points

    """

    val = optim is None

    # Convert to torch tensors
    query_pts = torch.from_numpy(query_pts).cuda().float()
    cspts = torch.from_numpy(cspts).cuda().float()
    
    if not val:
        optim.zero_grad()
        
    pred_csp = net(query_pts)
    loss_pts = net.loss(pred_csp, cspts)

    if not val:
        optim.zero_grad()
        loss_pts.backward()
        optim.step()

    return loss_pts.item()