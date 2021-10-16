import numpy as np
import imageio
import faiss
import torch
from torch.nn import functional as F
import transforms3d
import math
import cv2
from viz_utils import viz_depths
import time
import trimesh
import PIL.Image
from matplotlib import pyplot as plt


def sq_err(x, y):
    return (x - y) ** 2

def arr2img(arr):
    """
    convert numpy array of range 0-1 to UInt8 Image of range 0-255

    Parameters
    ----------
    arr : ndarray [res, res, -1]
        range 0-1
    
    Returns
    -------
    img : ndarray [res, res, -1]
        range 0-255, dtype=np.uint8
    """
    return (arr * 255).round().astype(np.uint8)

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

    for i in range(len(y.shape) - len(x.shape)):
        x = x.unsqueeze(-1)

    return x

def lift(x, y, z, intrinsics, homogeneous=False):
    '''
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


def project(x, y, z, intrinsics):
    '''

    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    '''
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_proj = expand_as(fx, x) * x / z + expand_as(cx, x)
    y_proj = expand_as(fy, y) * y / z + expand_as(cy, y)

    return torch.stack((x_proj, y_proj, z), dim=-1)


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

def get_pixel_points(azimuth, ele, res=256):
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
    rotation_default = transforms3d.euler.euler2mat(0, math.pi / 2, 0)

    rotation = transforms3d.euler.euler2mat(0, ele, azimuth)
    translation = np.matmul(rotation, translation_default)

    rotation = np.matmul(rotation, rotation_default)

    cam2worldMat = np.concatenate([rotation, translation], 1)
    cam2worldMat = np.expand_dims(cam2worldMat, 0)
    cam2worldMat = torch.from_numpy(cam2worldMat).float().cuda()

    # **Get intrinsics**
    fx = res
    fy = res
    cx = res / 2
    cy = res / 2
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


def get_distance(net, xyz_world, min_dist=0.01):
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
    is_xyz_outside = ((xyz_world > 0.5) | (xyz_world < -0.5)).float()
    outside = torch.sum(is_xyz_outside, 1) >= 1
    dists[outside, 0] = min_dist
    inside_points = xyz_world[~outside].cuda()
    nn_points = net(inside_points)
    d = net.get_dist(inside_points, nn_points)
    dists[~outside, 0] = d
    return dists

def get_distance_normals(xyz_world, index):
    """
    given xyz in world coordinates get nn based shortest distance to the surface

    Parameters
    ----------
    xyz_world : torch tensor [N, 3]
        input points in world coordinates (pixel points to start with)
    index : faiss search object
        function to do fast nn serach on gpu

    Returns
    -------
    dists: ndarray [N,]
         distance of each point in xyz_world to the point of intersection with the object
    indices : ndarray [N,]
        index of the nearest point (later used to get the normals)
    """

    # Get NN for each point in data
    D, I = index.search(xyz_world, 1)

    # squeeze D, I
    D = D[:, 0]
    I = I[:, 0]

    dists = np.zeros([xyz_world.shape[0], 1])

    # bounding box method: similar to BVH creation
    outside = np.sum(((xyz_world > 0.4) | (xyz_world < -0.4)), 1) >= 1
    dists[outside, 0] = 0.01
    dists[~outside, 0] = D[~outside]

    return dists, I

def ray_trace_mesh(trimesh_mesh, xyz_world, rays, batch_size=20004):
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(trimesh_mesh)
    all_loc_intersect = []
    all_ray_index = []
    all_face_index = []

    for x in range(0, rays.shape[0], batch_size):
        loc_intersect, ray_index, face_index = \
            intersector.intersects_location(xyz_world[x:x+batch_size], rays[x:x+batch_size])

        if len(ray_index) != 0:
            ray_index = batch_size * int(x / batch_size) + ray_index
            all_loc_intersect.append(loc_intersect)
            all_ray_index.append(ray_index)
            all_face_index.append(face_index)

    if rays.shape[0] % batch_size:
        x = batch_size * int(rays.shape[0] / batch_size)
        loc_intersect, ray_index, face_index = \
            intersector.intersects_location(xyz_world[x:], rays[x:])
        if len(ray_index) != 0:
            ray_index = x + ray_index
            all_loc_intersect.append(loc_intersect)
            all_ray_index.append(ray_index)
            all_face_index.append(face_index)
    
    dists = np.ones(rays.shape[0]) * 100
    normals = np.ones(rays.shape) * 0.1
    
    try:
        all_loc_intersect = np.concatenate(all_loc_intersect, 0)
        all_face_index = np.concatenate(all_face_index, 0)
        all_ray_index = np.concatenate(all_ray_index, 0)

        for ct, xx in enumerate(all_ray_index):
            dist = np.sqrt(np.sum((xyz_world[xx] - all_loc_intersect[ct])**2, 0))

            if dist < dists[xx]:
                dists[xx] = dist
                yy = all_face_index[ct]
                normals_ = trimesh_mesh.face_normals[yy]
                reverse = np.sum(normals_ * rays[xx]) > 0
                normals__ = normals_
                if reverse:
                    normals__ = -normals_ 
                normals[xx] = normals__
    except:
        dists[dists==100] = -1
        return normals, dists, xyz_world

    
    return normals, dists, all_loc_intersect

def march_along_rays(model, xyz_world, rays, max_iters=10000, thresh=0.01):
    """
    march along the set of rays till be hit the surface

    Parameters
    ----------
    model : nn.Module
        model (sDF +nF)
    xyz_world : torch tensor [N, 3] maybe [res*res, 3]
        input points in world coordinates (pixel points to start with)
    rays : torch tensor [N, 3] -> normalized directions
         directions of rays shot from the focal point towards the scene
    max_iters : int
        max iterations after which to stop the sphere tracing
    thresh : float
        how close do we want to approach the surface
    numerical : bool
        if true estimate normals numerically

    Returns
    -------
        xyz_points: ndarray [N, 3]
            Final xyz points on surface
        dist_acc: ndarray [N,]
            shortest distance of each point to the surface
        normals_pred: ndarray [N, 3]
            estimated normals (either numerically or from net)
        normals_pred_numerical: ndarray [N, 3]
            estimated normals
    """

    step_back_dist = 0.01

    net = model
    xyz_world_cpy = xyz_world.clone()
    xyz_world_cpy_back = xyz_world_cpy.clone()
    dist_acc = torch.zeros([xyz_world.shape[0], 1]).cuda() # accumulated distance 
    valid = torch.ones_like(dist_acc)[:, 0].cuda() > 0

    it = 0
    while True:
        dists = get_distance(net, xyz_world[valid])

        dist_acc[valid] += dists
        xyz_world[valid] += rays[valid] * dists
        
        xyz_world_cpy[valid] += rays[valid] * dists
        xyz_world_cpy_back[valid] += rays[valid] * dists
        
        old_valid = valid.clone()
        
        # Update valid pixels
        valid[valid] = dists[:, 0] > thresh

        if torch.max(dists) < thresh:
            break
        
        changed = valid ^ old_valid        
        if changed.any():
            normals_pred_jac = get_jac_normals_torch(xyz_world_cpy[changed], net)[0]
            dists_ = get_distance(net, xyz_world_cpy[changed])
        
            cos_thetas =  torch.abs(torch.sum(normals_pred_jac*rays[changed], -1, keepdim=True))

            dists_ /= cos_thetas[:]
            xyz_world_cpy[changed] += rays[changed] * dists_
            dist_acc[changed] += dists_
            
            xyz_world_cpy_back[changed] = xyz_world_cpy[changed] - rays[changed] * step_back_dist
            
            xyz_world_cpy[changed] -= rays[changed] * step_back_dist
        
        it += 1

        if it >= max_iters:
            break
    
    #For sharper intersection computation
    all_dists = torch.zeros([xyz_world.shape[0], 2*max_iters]).cuda()
    xyz_world_cpy_ = xyz_world_cpy.detach().clone()
    for x in range(2*max_iters):
        dists = get_distance(net, xyz_world)
        xyz_world_cpy_ += rays * dists
        all_dists[:, x] = dists[:, 0]
        
            
    n_samp = 200
    
    perturb = torch.linspace(-0.01, 0.02, n_samp).cuda().float()
    #For sharper intersection computation
    all_dists_at_intersect = torch.zeros([xyz_world.shape[0], n_samp]).cuda()
    ct = 0

    for x in perturb:
        temp = xyz_world_cpy[~valid] + rays[~valid]*x
        dists = get_distance(net, temp)
        all_dists_at_intersect[~valid, ct] = dists[:, 0]
        ct += 1

    normals_pred = net.get_normal(xyz_world_cpy_back, net(xyz_world_cpy_back))
    to_reverse = torch.sum(normals_pred * rays, 1) > 0
    normals_pred[to_reverse] *= -1
    normals_pred[valid] = 0.1

    normals_pred_jac = get_jac_normals_torch(xyz_world_cpy, net, eps=1e-7)
    to_reverse_jac = torch.sum(normals_pred_jac * rays, 1) > 0
    normals_pred_jac[to_reverse_jac] *= -1
    normals_pred_jac[valid] = 0.1
    
    dist_acc[valid] = -1
    dist_acc_naive = dist_acc.clone()
    dist_acc_naive[valid] = -1

    xyz_world = xyz_world.detach().cpu().numpy()
    normals_pred = normals_pred.detach().cpu().numpy()
    normals_pred_jac = normals_pred_jac.detach().cpu().numpy()
    dist_acc = dist_acc.detach().cpu().numpy()
    dist_acc_naive = dist_acc_naive.detach().cpu().numpy()
    
    return xyz_world, dist_acc, dist_acc_naive, normals_pred, normals_pred_jac

def get_jac_normals_torch(input_points, net, eps=1e-7):
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
        grad = get_batch_jacobian(net, X, 3)
        Vh = np.stack([np.linalg.svd(J.cpu().numpy())[-1] for J in grad])
        normals_pred = np.cross(Vh[:, 0], Vh[:, 1])
        normals_pred = normals_pred / np.sqrt((normals_pred ** 2).sum(1, keepdims=True))
        
    net.train(is_training)
        
    return torch.Tensor(normals_pred).cuda()

def get_batch_jacobian(net, x, noutputs):
    x = x.unsqueeze(1) # b, 1 ,in_dim
    n = x.size()[0]
    x = x.repeat(1, noutputs, 1) # b, out_dim, in_dim
    x.requires_grad_(True)
    input_val = torch.eye(noutputs).reshape(1,noutputs, noutputs).repeat(n, 1, 1).cuda()
    y = net(x)
    grad = torch.autograd.grad((y*input_val).sum(), [x])[0]

    return grad

def get_ray_traced_images(trimesh_mesh, ray_dirs, pixel_points_world, num_views, \
                          batch_size_eval=100002, res=256, direct=False):
    """
    render the learnt implicit function by sphere tracing

    Parameters
    ----------
    model : Either
        net : nn.Module
            model(sDF + nF)

        Or, a Tuple of The following

        point_cloud : ndarray [N, 3]
            GT point cloud
        normals : ndarray [N, 3]
            GT normals associated with point_cloud

    ray_dirs : torch tensor [N, 3]
        direction of rays shot from focal point
    pixel_points_world : torch tensor [N, 3] maybe [res*res, 3]
        input points in world coordinates (pixel points to start with)
    direct : bool
        whether to use directional light
    numerical : bool
        if true normals are estimated numerically instead of from the net (nF)
    res : int
        pixel space resolution
    batch_size_eval: int
        maximum size of batch

    Returns
    -------
    normals_pred_: ndarray [res*res, 3]
        estimated normals at the point of intersection
    normals_pred_numerical: ndarray [res*res, 3]
        estimated normals numerically at the point of intersection
    depths: ndarray [res*res, ]
        distance of each pixel point to the point of intersection of ray
        Note: it's not exactly the depth in conventional sense (as it's the distance along the ray -
        and not the distance projected onto the camera's z axis.)
    """


    xyz_world = []
    depths = []
    normals_pred = []

    # t = time.time()
    for i in range(0, pixel_points_world[0].shape[0], batch_size_eval):
        normals_pred_, depths_, xyz_world_ = ray_trace_mesh(trimesh_mesh, \
                                                            pixel_points_world[0][i:i+batch_size_eval], \
                                                            ray_dirs[0][i:i+batch_size_eval])
        xyz_world.append(xyz_world_)
        depths.append(depths_)
        normals_pred.append(normals_pred_)

    xyz_world = np.concatenate(xyz_world, 0)
    depths = np.concatenate(depths, 0)
    normals_pred = np.concatenate(normals_pred, 0)

    normals_pred_ = normals_pred.reshape(num_views, res, res, 3).transpose([0, 2, 1, 3])[:, :, ::-1, :]

    depths = depths.reshape(num_views, res, res).transpose([0, 2, 1])[:, :, ::-1]


    imgs_normals = []
    imgs_depth = []

    for it in range(num_views):
        normals_img = np.abs(normals_pred_[it] * 0.5 + 0.5)
        #depths[it] == -1
        depths[it][depths[it] == depths[it][0,0]] = -1
        depth_img = viz_depths(depths[it])
        imgs_depth.append(depth_img)

        if direct:
            direction_light = np.array([[[0, 0, 1]]])
            direction_light = direction_light / np.linalg.norm(direction_light)

            normals_img = np.maximum(np.zeros(normals_pred_[it].shape[0:2]), np.sum(normals_pred_[it] * direction_light, 2))

        imgs_normals.append(normals_img)

    return imgs_normals, imgs_depth, depths


def get_sphere_traced_images(model, ray_dirs, pixel_points_world, num_views, \
                             direct=False, res=256, thresh=0.01, max_iters=1000, \
                             batch_size_eval=100000):
    """
    render the learnt implicit function by sphere tracing

    Parameters
    ----------
    model : Either
        net : nn.Module
            model(sDF + nF)

        Or, a Tuple of The following

        point_cloud : ndarray [N, 3]
            GT point cloud
        normals : ndarray [N, 3]
            GT normals associated with point_cloud

    ray_dirs : torch tensor [N, 3]
        direction of rays shot from focal point
    pixel_points_world : torch tensor [N, 3] maybe [res*res, 3]
        input points in world coordinates (pixel points to start with)
    direct : bool
        whether to use directional light
    numerical : bool
        if true normals are estimated numerically instead of from the net (nF)
    res : int
        pixel space resolution
    batch_size_eval: int
        maximum size of batch

    Returns
    -------
    normals_pred_: ndarray [res*res, 3]
        estimated normals at the point of intersection
    normals_pred_numerical: ndarray [res*res, 3]
        estimated normals numerically at the point of intersection
    depths: ndarray [res*res, ]
        distance of each pixel point to the point of intersection of ray
        Note: it's not exactly the depth in conventional sense (as it's the distance along the ray -
        and not the distance projected onto the camera's z axis.)
    """
    with torch.no_grad():
        xyz_world = []

        depths = []
        depths_naive = []

        normals_pred = []
        normals_pred_jac = []

        for iteration in range(0, pixel_points_world[0].shape[0], batch_size_eval):
            xyz_world_, depths_, depths_naive_, normals_pred_, normals_pred_jac_ = \
                march_along_rays(model, pixel_points_world[0][iteration:iteration+batch_size_eval], \
                                 ray_dirs[0][iteration:iteration+batch_size_eval], \
                                 max_iters=max_iters, thresh=thresh)

            xyz_world.append(xyz_world_)
            depths.append(depths_)
            depths_naive.append(depths_naive_)
            normals_pred.append(normals_pred_)
            normals_pred_jac.append(normals_pred_jac_)

            # bwd_times.append(bwd_time)
            # fwd_times.append(fwd_time)

        xyz_world = np.concatenate(xyz_world, 0)
        depths = np.concatenate(depths, 0)
        depths_naive = np.concatenate(depths_naive, 0)

        normals_pred = np.concatenate(normals_pred, 0)
        normals_pred_jac = np.concatenate(normals_pred_jac, 0)

    xyz_world = xyz_world.reshape(num_views, res, res, 3).transpose([0, 2, 1, 3])[:, :, ::-1, :]

    normals_pred_ = normals_pred.reshape(num_views, res, res, 3).transpose([0, 2, 1, 3])[:, :, ::-1, :]
    normals_pred_jac_ = normals_pred_jac.reshape(num_views, res, res, 3).transpose([0, 2, 1, 3])[:, :, ::-1, :]

    # print(depths.shape)
    depths = depths.reshape(num_views, res, res).transpose([0, 2, 1])[:, :, ::-1]
    depths_naive = depths_naive.reshape(num_views, res, res).transpose([0, 2, 1])[:, :, ::-1]

    imgs_normals = []
    imgs_normals_jac= []
    imgs_depth = []
    imgs_depth_naive = []


    for it in range(num_views):
        normals_img = np.abs(normals_pred_[it] * 0.5 + 0.5)
        normals_img_jac = np.abs(normals_pred_jac_[it] * 0.5 + 0.5)

        depth_img = viz_depths(depths[it])
        depth_naive_img = viz_depths(depths_naive[it])
        imgs_depth.append(depth_img)
        imgs_depth_naive.append(depth_naive_img)

        if direct:
            direction_light = np.array([[[0, 0, 1]]])
            direction_light = direction_light / np.linalg.norm(direction_light)

            normals_img = np.maximum(np.zeros(normals_pred_[it].shape[0:2]), np.sum(normals_pred_[it] * direction_light, 2))
            normals_img_jac = np.maximum(np.zeros(normals_pred_jac_[it].shape[0:2]),
                                     np.sum(normals_pred_jac_[it] * direction_light, 2))
        
        imgs_normals.append(normals_img)
        imgs_normals_jac.append(normals_img_jac)

    return imgs_normals, imgs_normals_jac, imgs_depth, imgs_depth_naive, depths, depths_naive

def get_multi_view_ray_tracing(trimesh_mesh, num_views=10, batch_size_eval=100000, res=256):
    """
    render multiview normal an depth map by sphere tracing the learnt function

    Parameters
    ----------
    trimesh_mesh : Trimesh object
        mesh to be rendered
    num_views : int
        number of views (360 deg. azimuth)
    batch_size_eval : int
    res : resolution of the output image

    Returns
    -------

    imgs_normal: list of images ([res, res, 3], )
        normal maps
    imgs_depth: list of images ([res, res, 3])
        depth maps
    depths: list of actual (unnormalized) depth maps ([res, res, 3], )

    """
    azims = np.deg2rad(np.linspace(0, 360, num_views))
    all_ray_dirs = []
    all_pixel_points_world = []
    for azimuth in azims:
        ray_dirs, pixel_points_world = get_pixel_points(azimuth, np.deg2rad(30), res=res)

        all_ray_dirs.append(ray_dirs)
        all_pixel_points_world.append(pixel_points_world)

    all_ray_dirs = torch.cat(all_ray_dirs, 1).cpu().numpy()
    all_pixel_points_world = torch.cat(all_pixel_points_world, 1).cpu().numpy()

    imgs_normals, imgs_depth, depths = get_ray_traced_images(trimesh_mesh, all_ray_dirs, all_pixel_points_world, num_views, batch_size_eval=batch_size_eval, res=res)
    return imgs_normals, imgs_depth, depths

def get_multi_view_sphere_tracing(model, num_views=10, thresh=0.01, max_iters=1000, res=256):
    """
    render multiview normal an depth map by sphere tracing the learnt function

    Parameters
    ----------
    model : nn.Module
    num_views : int
        number of views (360 deg. azimuth)
    numerical : bool
        if true normals are obtained numerically
    thresh: float
        how close to the surface do we want sphere tracing to stop

    Returns
    -------

    imgs: list of images ([res, res, 3], )
        normal maps
    imgs_depth: list of images ([res, res, 3])
        depth maps
    imgs_numerical: list of images ([res, res, 3], )
        normal maps numerical

    """
    azims = np.deg2rad(np.linspace(0, 360, num_views))
    all_ray_dirs = []
    all_pixel_points_world = []
    for azimuth in azims:
        ray_dirs, pixel_points_world = get_pixel_points(azimuth, np.deg2rad(30), res=res)

        all_ray_dirs.append(ray_dirs)
        all_pixel_points_world.append(pixel_points_world)

    all_ray_dirs = torch.cat(all_ray_dirs, 1)
    all_pixel_points_world = torch.cat(all_pixel_points_world, 1)

    imgs_normals, imgs_normals_jac, imgs_depth, imgs_depth_naive, depths, depths_naive = \
        get_sphere_traced_images(model, all_ray_dirs, all_pixel_points_world, num_views, \
                                 thresh=thresh, max_iters=max_iters, res=res)

    return imgs_normals, imgs_normals_jac, imgs_depth, imgs_depth_naive, depths, depths_naive

def create_gif(imgs, file_path):
    imgs_uint8 = list(map(arr2img, imgs))
    imageio.mimsave(file_path, imgs_uint8, duration=0.7)

def get_light(direction, normal_map):
    normal_map = normal_map.transpose([2, 0, 1])/(np.sqrt(np.sum(normal_map**2, -1)))
    normal_map = normal_map.transpose([1, 2, 0])
    
    direction /= np.sqrt(np.sum(direction**2) + 1e-8)
    direction = np.expand_dims(np.expand_dims(direction, 0), 0)

    lighted = np.abs(np.sum(normal_map*direction, -1))#*4
    lighted = np.stack([lighted] * 3, 2)
       
    return lighted

def get_valid_matrix(depths_pred, threshold=0.008):
    arr = depths_pred < threshold
    arr = np.argmin(~arr, -1)
    return ~(arr != 0)


def normalize(img):
    '''
    Normalize a given image to -1 to 1.

    Parameters
    ----------
    img : ndarray
        Image to be normalized, having values between 0 to 1.
    
    Returns
    -------
    ndarray
        Normalized image to -1 to 1.
    '''
    return 2 * img - 1

def min_max_normalize(img):
    '''
    Min max normalize a given image

    Parameters
    ----------
    img : ndarray
        Image to be normalized
    
    Returns
    -------
    ndarray
        Min max normalized image, range in 0 to 1.
    '''
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def evaluate_normals_depths(net, trimesh_mesh, num_views, thresh, videos_path, max_iters, res):
    """
    given a model, perform sphere tracing and save a video of multi view depth map and normal map
    for both gt and pred

    Parameters
    ----------
    net : nn.Module
        model(sDF + nF)
    trimesh_mesh : trimesh.Trimesh
    num_views : int
        number of views (360 deg. azimuth)
    thresh : float
         when to stop the sphere tracing process for pred model
    numerical : bool
        if true normals are obtained numerically
    max_iters : int
        max iters for pred sphere tracing
    videos_path : str
        path to save the videos
    res: int
        pixel space resolution

    Returns
    -------

    """
    imgs_normal_gt, imgs_depth_gt, all_depths_gt = get_multi_view_ray_tracing(trimesh_mesh, num_views, batch_size_eval=100000, res=res)
    print("done rendering gt maps")

    imgs_normal_pred, imgs_normal_pred_jac, \
        imgs_depth_pred, imgs_depth_pred_naive, \
            depths, depths_naive = get_multi_view_sphere_tracing(net, num_views=num_views, \
                                                                 thresh=thresh, max_iters=max_iters, \
                                                                 res=res)
    print("done rendering pred maps")

    final_normal_imgs = []
    final_depth_imgs = []
    normal_errs = []
    normal_jac_errs = []
    gt = []

    for it in range(num_views):
        img = np.concatenate([imgs_depth_pred_naive[it], imgs_depth_pred[it], imgs_depth_gt[it]], 1)
        
        #Lighting
        normal_map_jac = normalize(imgs_normal_pred_jac[it])
        normal_map = normalize(imgs_normal_pred[it])
        
        direction_1 = np.array([0.1, 0.3, 0.2])
        direction_2 = np.array([0.4, 0.3, 0.3])
        direction_3 = np.array([0.3, 0.8, 0.2])
        
        lighted_jac = get_light(direction_2, normal_map_jac)
        lighted_jac = np.clip(lighted_jac, 0, 1)

        lighted_pred = get_light(direction_2, normal_map)
        lighted_pred = np.clip(lighted_pred, 0, 1)
        
        normal_map_gt = normalize(imgs_normal_gt[it])
        lighted_gt = get_light(direction_2, normal_map_gt)
        lighted_gt = np.clip(lighted_gt, 0, 1)
        
        # Find common pixels to compute the error on
        bg_mask_gt = all_depths_gt[it] == all_depths_gt[it][0, 0]
        bg_mask = depths[it] == depths[it][0, 0]
        bg_mask_naive = depths_naive[it] == depths_naive[it][0, 0]

        valid_gt = np.logical_not(bg_mask_gt)
        
        valid_pred = np.logical_not(bg_mask)
        valid_common = np.logical_and(valid_pred, valid_gt) 
        
        valid_naive = np.logical_not(bg_mask_naive)
        valid_common_gt_naive = np.logical_and(valid_gt, valid_naive)
           
        lighted_jac[bg_mask] = 1
        lighted_pred[bg_mask] = 1
        lighted_gt[bg_mask_gt] = 1
    
        gt.append(lighted_gt)

        ## Compute the error
        # Normal error (fwd)
        img_normal_error = np.sqrt(sq_err(normalize(imgs_normal_pred[it]), normalize(imgs_normal_gt[it])).sum(2))
        normal_errs.append(np.sum(img_normal_error[valid_common]) / valid_common.sum())
        img_normal_error = np.stack([img_normal_error] * 3, 2)

        # Normal error (jac)
        img_normal_jac_error = np.sqrt(sq_err(normalize(imgs_normal_pred_jac[it]), normalize(imgs_normal_gt[it])).sum(2))
        normal_jac_errs.append(np.sum(img_normal_jac_error[valid_common]) / valid_common.sum())
        img_normal_jac_error = np.stack([img_normal_jac_error] * 3, 2)
        
        # Depth error
        depth_err = np.abs(all_depths_gt[it] - depths[it]) * valid_common
        depth_err_naive = np.abs(all_depths_gt[it] - depths_naive[it]) * valid_common_gt_naive

        final_depth_err = np.concatenate([depth_err_naive, depth_err, all_depths_gt[it]], 1)
        plt.imsave(videos_path + '/' + 'final_depth_err.png', arr2img(final_depth_err))
        final_depth_imgs.append(final_depth_err)

        ### Visualization of normals
        # Visualization of normal error map
        img_normal_error= min_max_normalize(img_normal_error)
        img_normal_jac_error = min_max_normalize(img_normal_jac_error)
        img_normal_error[:, :, 1:] = 0
        img_normal_jac_error[:, :, 1:] = 0

        #Find background in normal images and assign its pixel value as 1
        # In normal (jac) prediction
        bg_mask_jac = (imgs_normal_pred_jac[it] == imgs_normal_pred_jac[it][0, 0]).all(axis=-1)
        imgs_normal_pred_jac[it][bg_mask_jac] = 1

        # In normal prediction
        bg_mask = (imgs_normal_pred[it] == imgs_normal_pred[it][0, 0]).all(axis=-1)
        imgs_normal_pred[it][bg_mask] = 1

        # In GT normals
        bg_mask_gt = (imgs_normal_gt[it] == imgs_normal_gt[it][0, 0]).all(axis=-1)
        imgs_normal_gt[it][bg_mask_gt] = 1

        # Final image to visualize
        final_img_lighted = np.concatenate([lighted_pred, lighted_jac, lighted_gt], 1)
        final_img_normals = np.concatenate([imgs_normal_pred[it], imgs_normal_pred_jac[it],\
                                     imgs_normal_gt[it]], 1)
        final_img_normal_errs = np.concatenate([img_normal_error, img_normal_jac_error, \
                                     np.zeros_like(img_normal_error)], 1)
        final_img  = np.concatenate([final_img_lighted, final_img_normals, final_img_normal_errs], 0)
        plt.imsave(videos_path + '/' + 'final_normals.png', arr2img(final_img))

        final_normal_imgs.append(final_img_normals)

    create_gif(final_normal_imgs, videos_path + '/' + 'normal_viz.gif')
    create_gif(final_depth_imgs, videos_path + '/' + 'depth_viz.gif')
    create_gif(gt, videos_path + '/' + 'gt.gif')
    
    print('avg normal err: ', np.mean(normal_errs))
    print('avg normal jac err: ', np.mean(normal_jac_errs))