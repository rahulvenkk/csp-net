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

def get_losses(net, pred_csp, pred_dist, pred_normal, cspts, dist, normal):
    loss_pts = net.loss(pred_csp, cspts)
    loss_dist = net.loss(pred_dist, dist.abs()).item()
    loss_normal = net.cosine_loss(pred_normal, normal).item()

    return loss_pts, loss_dist, loss_normal

def train_val_step(net, query_pts, cspts, dist, normal, optim=None):
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
    pred_distance: torch tensor [batch_size, ]
        predicted distances (uDF/sDF) by the model
    gt_torch: torch tensor [batch_size, ]
        Ground truth distances
    pred_normal: torch tensor [batch_size, 3]
        predicted normals
    normal_torch: torch_tensor [batch_size, 3]
        Ground truth normals
    input_batch_torch: torch tensor [batch_size, 3]
        set of points input to the model

    """

    val = optim is None

    # Convert to torch tensors
    query_pts = torch.from_numpy(query_pts).cuda().float()
    cspts = torch.from_numpy(cspts).cuda().float()
    dist = torch.from_numpy(dist).cuda().float()
    normal = torch.from_numpy(normal).cuda().float()
    
    pred_csp, pred_dist, pred_normal = get_preds(net, query_pts)
    loss_pts, loss_dist, loss_normal = get_losses(net, pred_csp, pred_dist, pred_normal, \
                                                       cspts, dist, normal)

    if not val:
        optim.zero_grad()
        loss_pts.backward()
        optim.step()

    return loss_pts.item(), loss_dist, loss_normal