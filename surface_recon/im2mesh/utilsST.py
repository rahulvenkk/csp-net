
import numpy as np
from numpy.lib.polynomial import polydiv
#import imageio

# import faiss
import torch
from torch.nn import functional as F
import transforms3d
import time

def parse_intrinsics(intrinsics):
    intrinsics = intrinsics.cuda()

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    return fx, fy, cx, cy

def expand_as(x, y):
    if len(x.shape) == len(y.shape):
        return x

    for _ in range(len(y.shape) - len(x.shape)):
        x = x.unsqueeze(-1)

    return x

def lift(x, y, z, intrinsics, homogeneous=False):
    '''

    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    '''
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_lift = (x - expand_as(cx, x)) / expand_as(fx, x) * z
    y_lift = (y - expand_as(cy, y)) / expand_as(fy, y) * z

    if homogeneous:
        return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)
    else:
        return torch.stack((x_lift, y_lift, z), dim=-1)

def world_from_xy_depth(xy, depth, cam2world, intrinsics):
    '''Translates meshgrid of xy pixel coordinates plus depth to  world coordinates.
    '''
    batch_size, _, _ = cam2world.shape

    x_cam = xy[:, :, 0].view(batch_size, -1)
    y_cam = xy[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics, homogeneous=True)  # (batch_size, -1, 4)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = torch.bmm(cam2world, pixel_points_cam).permute(0, 2, 1)[:, :, :3]  # (batch_size, -1, 3)

    return world_coords

def get_ray_directions(xy, cam2world, intrinsics):
    '''Translates meshgrid of xy pixel coordinates to normalized directions of rays through these pixels.
    '''
    batch_size, num_samples, _ = xy.shape

    z_cam = torch.ones((batch_size, num_samples)).cuda()
    pixel_points = world_from_xy_depth(xy, z_cam, intrinsics=intrinsics, cam2world=cam2world)  # (batch, num_samples, 3)

    cam_pos = cam2world[:, :3, 3]
    ray_dirs = pixel_points - cam_pos[:, None, :]  # (batch, num_samples, 3)
    ray_dirs = F.normalize(ray_dirs, dim=2)
    return ray_dirs, pixel_points


# **Get cam rot**

def get_pixel_points(azimuth, ele, res=256, gt=False):
    """
    sample a set of pixel coordinates in world space, and shoot rays from the focal point

    Parameters
    ----------
    azimuth : float
        azimuth of the camera
    ele : float
        elevation of camera
    res : int
        pixel space resolution

    Returns
    -------
    ray_dirs: [res*res, 3]
        directions of rays shot from the focal point towards the scene
    pixel_points_world: [res*res, 3]
        set of pixel sampled, and it's coordinates returned in world space

    """
    translation_default = np.array([[-1.7, 0, 0]]).T
    rotation_default = transforms3d.euler.euler2mat(0, np.deg2rad(90), 0)
    rotation = transforms3d.euler.euler2mat(0, ele, azimuth)
    translation = np.matmul(rotation, translation_default)
    rotation = np.matmul(rotation, rotation_default)
    cam2worldMat = np.concatenate([rotation, translation], 1)
    cam2worldMat = np.expand_dims(cam2worldMat, 0)
    cam2worldMat = torch.from_numpy(cam2worldMat).float().cuda()

    # **Get intrinsics**
    res = res
    ww = res
    fx = ww
    fy = ww
    cx = ww / 2 
    cy = ww / 2
    intrinsicsMat = np.zeros([1, 3, 4])
    intrinsicsMat[:, 0, 0] = fx
    intrinsicsMat[:, 1, 1] = fy
    intrinsicsMat[:, 0, 2] = cx
    intrinsicsMat[:, 1, 2] = cy
    intrinsicsMat[:, 2, 2] = 1

    intrinsicsMat = torch.from_numpy(intrinsicsMat).float().cuda()

    # **Sample pixels**
    xy_pts = np.mgrid[0:res, 0:res]
    xy_pts = np.flip(xy_pts, axis=0).copy()
    xy_pts = xy_pts.reshape(2, -1).transpose(1, 0)
    xy_pts = np.expand_dims(xy_pts, 0)
    xy_pts = torch.from_numpy(xy_pts).float().cuda()

    ray_dirs, pixel_points_world = get_ray_directions(xy_pts, cam2worldMat, intrinsicsMat)

    return ray_dirs, pixel_points_world


def get_distance(model, xyz_world, encoding_udf):
    """
    given xyz in world coordinates get estimated shortest to the surface

    Parameters
    ----------
    net : nn.Module
        model (sDF + nF)
    xyz_world : torch tensor [N, 3]
        input points in world coordinates (pixel points to start with)

    Returns
    -------
    dists: torch tensor [N,]
        distance of each point in xyz_world to the point of intersection with the object

    """
    dists = torch.zeros([xyz_world.shape[0], 1]).cuda()

    #bounding box method: similar to BVH creation
    outside = torch.sum(((xyz_world > 0.5) | (xyz_world < -0.5)).float(), 1) >= 1
    dists[outside, 0] = 0.01
    inside_points = xyz_world[~outside]
    
    if inside_points.shape[0] !=0:
        inside_points = inside_points.unsqueeze(0)
        with torch.no_grad():
            temp = model.decode_udf(inside_points.cuda(), encoding_udf)[0]#, **kwargs)(, inside_points.cuda(), inputs)[0][0]
        dists[~outside, 0] = torch.abs(temp)
    return dists

def get_jac_normals(input_points, encoding_udf, net, eps=1e-7):
    """
    get numerically obtained normals for the given input points in world coordinates
    Parameters
    ----------
    input_points : pytorch tensor [N, 3]
         points for which normals are to be estimated
    net : torch nn.Module
        model (sDF + nF)
    eps : float
        some small number
    Returns
    -------
    norm_xyz: pytorch tensor [N, 3]
        estimated normals
    """
    is_training = net.training
    net.eval()
    with torch.enable_grad():
        input_points = input_points.detach().cpu().numpy()
        X = torch.from_numpy(input_points).cuda().float().requires_grad_(True)
        
        # grad_f1, grad_f2, grad_f3
        grad = get_batch_jacobian(net, X, encoding_udf, 3)
        Vh = np.stack([np.linalg.svd(J.cpu().numpy())[-1] for J in grad[0]])
        normals_pred = np.cross(Vh[:, 0], Vh[:, 1])
        normals_pred = normals_pred / np.sqrt((normals_pred ** 2).sum(1, keepdims=True))
    normals_pred = torch.Tensor(normals_pred).unsqueeze(0)

    net.train(is_training)
    
    return normals_pred.cuda()

def get_batch_jacobian(net, x, encoding_udf, noutputs):
    x = x.unsqueeze(-2)
    b, np, _, in_dim = x.shape
    x = x.repeat(1, 1, noutputs, 1).view(b, -1, in_dim) # b, np*out_dim, in_dim
    x.requires_grad_(True)
    
    y = net.decode_csp(x, encoding_udf)
    y = y.view(b, np, noutputs, noutputs)
    
    #x = x.view(b, np, noutputs, in_dim)
    input_val = torch.eye(noutputs).reshape(1, 1, noutputs, noutputs).repeat(b, np, 1, 1).cuda()
    grad = torch.autograd.grad((y*input_val).sum(), [x])[0].view(b, np, noutputs, in_dim)
    return grad

def normalize(norm_xyz, eps=1e-7):
    return ((norm_xyz.t()) / torch.sqrt(torch.sum(norm_xyz ** 2, 1) + eps * 0.1)).t()
