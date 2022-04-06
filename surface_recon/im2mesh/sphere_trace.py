
import numpy as np
import torch
import time
import trimesh

from .utilsST import *

################################################################################
############################## CSP Sphere Tracing ##############################
################################################################################
# get_multi_view_sphere_tracing
# get_sphere_traced_images_parallel
# march_along_rays
################################################################################
    
def march_along_rays(model, xyz_world, encoding_udf, rays, iters, thresh, batch_size_eval=1000):
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
    iters : int
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
    
    net = model
    xyz_world_cpy = xyz_world.clone()
    xyz_world_cpy_back = xyz_world_cpy.clone()
    dist_acc = torch.zeros([xyz_world.shape[0], 1]).cuda()
    valid = torch.ones_like(dist_acc)[:, 0].cuda() > 0
    it = 0

    while True:
        dists = get_distance(net, xyz_world[valid], encoding_udf)

        dist_acc[valid] += dists
        xyz_world[valid] += rays[valid] * dists
        
        xyz_world_cpy[valid] += rays[valid] * dists
        xyz_world_cpy_back[valid] += rays[valid] * dists
        
        old_valid = valid.clone()
        
        valid[valid.clone()] = (dists[:, 0] > thresh)
        if torch.max(dists) < thresh:
            break
        
        changed = valid ^ old_valid
                
        if changed.any():            
            xyz_world_cpy_back[changed] = xyz_world_cpy[changed] - rays[changed] * 0.005
            xyz_world_cpy[changed] -= rays[changed] * 0.005
        
        it += 1

        if it >= iters:
            break
    
    normals_pred = []

    t = time.time()
    for iteration in range(0, xyz_world.shape[0], batch_size_eval):       
        normals_pred_ = net.decode_nvf(xyz_world_cpy_back[iteration:iteration+batch_size_eval].unsqueeze(0),encoding_udf)[0]
        normals_pred.append(normals_pred_)
    
    normals_pred = torch.cat(normals_pred)
    reverse = torch.sum(normals_pred*rays, 1) > 0
    normals_pred[reverse] *= -1
    normals_pred[valid] = .1
    normals_pred_numerical = get_jac_normals(xyz_world_cpy.unsqueeze(0), encoding_udf, net)[0]
 
    dist_acc[valid] = -1
    
    reverse_numerical = torch.sum(normals_pred_numerical * rays, 1) > 0
    normals_pred_numerical[reverse_numerical] *= -1
    normals_pred_numerical[valid] = .1

    normals_pred = normals_pred.detach().cpu().numpy()    
    dist_acc = dist_acc.detach().cpu().numpy()
    xyz_world = xyz_world.detach().cpu().numpy()
    normals_pred_numerical = normals_pred_numerical.detach().cpu().numpy()

    return xyz_world, dist_acc, normals_pred, normals_pred_numerical


def get_sphere_traced_images_parallel(model, encoding_udf, ray_dirs, pixel_points_world, num_views, res, thresh, iters, batch_size_eval, batch_size_eval_nvf):
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
        normals_pred = []
        normals_pred_numerical = []

        # t = time.time()
        for iteration in range(0, pixel_points_world[0].shape[0], batch_size_eval):
            xyz_world_, depths_, normals_pred_, normals_pred_numerical_ = march_along_rays(model, pixel_points_world[0][iteration:iteration+batch_size_eval], encoding_udf, ray_dirs[0][iteration:iteration+batch_size_eval], iters, thresh, batch_size_eval=batch_size_eval_nvf)
            xyz_world.append(xyz_world_)
            depths.append(depths_)
            normals_pred.append(normals_pred_)
            normals_pred_numerical.append(normals_pred_numerical_)
            
        xyz_world = np.concatenate(xyz_world, 0)
        depths = np.concatenate(depths, 0)
        normals_pred = np.concatenate(normals_pred, 0)
        normals_pred_numerical = np.concatenate(normals_pred_numerical, 0)

        # print("time taken::", time.time() - t)
   
    xyz_world = xyz_world.reshape(num_views, res, res, 3).transpose([0, 2, 1, 3])[:, :, ::-1, :]

    normals_pred_ = normals_pred.reshape(num_views, res, res, 3).transpose([0, 2, 1, 3])[:, :, ::-1, :]
    normals_pred_numerical_ = normals_pred_numerical.reshape(num_views, res, res, 3).transpose([0, 2, 1, 3])[:, :, ::-1, :]

    depths = depths.reshape(num_views, res, res).transpose([0, 2, 1])[:, :, ::-1]
    
    return xyz_world, normals_pred_, normals_pred_numerical_, depths


def get_multi_view_sphere_tracing(model, inputs, num_views, thresh, iters, res, batch_size_eval=100000, batch_size_eval_nvf=1000):
    """
    render multiview normal an depth map by sphere tracing the learnt function

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
    encoding_udf = model.get_encoding(inputs)
#     return
    azims = np.linspace(np.deg2rad(60), np.deg2rad(300), num_views)
#     azims = np.linspace(180*math.pi/180, 300 * math.pi / 180, num_views)
    all_ray_dirs = []
    all_pixel_points_world = []
    for azimuth in azims:
        ray_dirs, pixel_points_world = get_pixel_points(azimuth, np.deg2rad(30), res=res)
        all_ray_dirs.append(ray_dirs)
        all_pixel_points_world.append(pixel_points_world)

    all_ray_dirs = torch.cat(all_ray_dirs, 1)
    all_pixel_points_world = torch.cat(all_pixel_points_world, 1)

    xyz_world, normals_pred, normals_pred_numerical, depths = get_sphere_traced_images_parallel(model, encoding_udf, all_ray_dirs, all_pixel_points_world, num_views, res, thresh, iters, batch_size_eval, batch_size_eval_nvf)

    return  xyz_world, normals_pred, normals_pred_numerical, depths


################################################################################
########################### Ground truth Ray Tracing ###########################
################################################################################
# get_multi_view_ray_tracing
# get_ray_traced_images_parallel
# ray_trace_mesh
################################################################################

def ray_trace_mesh(trimesh_mesh, xyz_world, rays, batch_size=20004):
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(trimesh_mesh)
    all_loc_intersect = []
    all_ray_index = []
    all_face_index = []
    
    for x in range(0, rays.shape[0], batch_size):
        loc_intersect, ray_index, face_index = \
            intersector.intersects_location(xyz_world[x:x + batch_size], rays[x:x + batch_size])
        if len(ray_index) != 0:
            ray_index = batch_size * int(x / batch_size) + ray_index
            # face_index = batch_size * int(x / batch_size) + face_index
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
        return all_loc_intersect, dists, normals
    except:
        dists[dists==100] = -1
        return xyz_world, dists, normals


def get_ray_traced_images_parallel(trimesh_mesh, ray_dirs, pixel_points_world, \
                                   num_views, res, batch_size_eval=100002):
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

    for iteration in range(0, pixel_points_world[0].shape[0], batch_size_eval):
        xyz_world_, depths_, normals_pred_ = ray_trace_mesh(trimesh_mesh, pixel_points_world[0][iteration:iteration+batch_size_eval], ray_dirs[0][iteration:iteration+batch_size_eval])
        xyz_world.append(xyz_world_)
        depths.append(depths_)
        normals_pred.append(normals_pred_)

    xyz_world = np.concatenate(xyz_world, 0)
    depths = np.concatenate(depths, 0)
    normals_pred = np.concatenate(normals_pred, 0)

    normals_pred_ = normals_pred.reshape(num_views, res, res, 3).transpose([0, 2, 1, 3])[:, :, ::-1, :]
    depths = depths.reshape(num_views, res, res).transpose([0, 2, 1])[:, :, ::-1]

    return normals_pred_, depths


def get_multi_view_ray_tracing(trimesh_mesh, num_views, res, batch_size_eval=100000):
    """
    render multiview normal an depth map by sphere tracing the learnt function

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
    azims = np.linspace(np.deg2rad(60), np.deg2rad(300), num_views)

    all_ray_dirs = []
    all_pixel_points_world = []
    for azimuth in azims:
        ray_dirs, pixel_points_world = get_pixel_points(azimuth, np.deg2rad(30), \
                                                        res, gt=True)

        all_ray_dirs.append(ray_dirs)
        all_pixel_points_world.append(pixel_points_world)

    all_ray_dirs = torch.cat(all_ray_dirs, 1).cpu().numpy()
    all_pixel_points_world = torch.cat(all_pixel_points_world, 1).cpu().numpy()

    normals_pred, imgs_depth = get_ray_traced_images_parallel(trimesh_mesh, \
                                                              all_ray_dirs, \
                                                              all_pixel_points_world, \
                                                              num_views, res, \
                                                              batch_size_eval=batch_size_eval)

    return normals_pred, imgs_depth
