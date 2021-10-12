import open3d as o3d
import faiss
import numpy as np
import mesh_to_sdf
import trimesh

def get_data(mesh, point_cloud, normals, num_points=250000, is_flip=False):
    """
    get densely sampled points from mesh, its normal field, distance from surface, \
    and closest surface-points \

    Parameters
    ----------
    mesh: open3D mesh object
        mesh for which we want to create data points (sDF, nF)
    point_cloud: ndarray [N, 3]
        uniformly sampled points from mesh
    normals: ndarray [N, 3]
        normals corresponding to the sampled points
    num_points: int
        how many more points to sample
    is_flip: bool
        has yz been flipped in the point_cloud

    Returns
    -------
    data: ndarray [N, 3]
        final set of sampled points
    min_distances: ndarray [Ns,]
    normals_grid: ndarray [Ns, 3]
        normals at closest surface points
    csp: ndarray [N, 3]
        closest surface points
    """

    n_points_uniform = int(num_points * 0.1)
    n_points_surface = num_points

    points_uniform = np.random.rand(n_points_uniform, 3)
    points_uniform = (points_uniform - 0.5)
    points_surface = mesh.sample_points_uniformly(n_points_surface)
    points_surface_ = np.asarray(points_surface.points)

    if is_flip:
        points_surface_ = points_surface_[:, [0, 2, 1]]

    var = 0.00025

    points_surface = points_surface_.copy()
    points_surface_small = points_surface + np.random.normal(0, np.sqrt(var), [n_points_surface, 3])
    points_surface_smaller = points_surface + np.random.normal(0, np.sqrt(var/10), [n_points_surface, 3])
    data = np.concatenate([points_uniform, points_surface_small, points_surface_smaller], axis=0).astype('float32')

    d = point_cloud.shape[1]

    # init index
    cpu_index = faiss.IndexFlatL2(d)
    index = faiss.index_cpu_to_all_gpus(cpu_index)

    # Add point cloud to index
    index.add(point_cloud.astype('float32'))

    # Get NN for each point in data
    D, I = index.search(data, 1)

    # squeeze D, I
    D = D[:, 0]
    I = I[:, 0]

    # min_distances now constitutes UDF
    csp = point_cloud[I, :]

    min_distances = np.sqrt(np.sum((data - csp) ** 2, 1))

    normals_grid = normals[I, :]

    return data, min_distances, normals_grid, csp

def get_sample_points(points, dists, normals, cspoints, batch_size, prob_dists):
    """
    sample the unsigned distance from the surface, CSP, and the normal field at few points.

    Parameters
    ----------
    points: ndarray [N, 3]
        set of points
    cspoints: ndarray [N, 3]
        set of points

    Returns
    -------
    points: ndarray [batch_size, 3]
        points sampled
    nn: ndarray [batch_size, 3]
        associated closest point on the surface
    """
    ch = np.random.choice(np.arange(points.shape[0]), batch_size, p=prob_dists)
    return points[ch], dists[ch], normals[ch], cspoints[ch]

def get_sampled_points_normals(mesh_filepath, flip_axis=False):
    """

    Parameters
    ----------
    mesh_filepath : str
        path to mesh file (.off)
    flip_axis : bool
        if true the yz axis is flipped in the mesh - mostly happens when mesh is not processed by libmesh

    Returns
    -------
    point_cloud: ndarray [250000, 3]
        sampled points
    normals: ndarray [250000, 3]
        associated normals
    mesh: open3d mesh object
    """

    mesh = o3d.io.read_triangle_mesh(mesh_filepath)

    trimesh_mesh = trimesh.load(mesh_filepath, process=False)
    point_cloud, indices = trimesh_mesh.sample(250000, True)
    normals = trimesh_mesh.face_normals[indices]

    if flip_axis:
        point_cloud = point_cloud[:, [0, 2, 1]]

        normals = normals[:, [0, 2, 1]]

    point_cloud = np.ascontiguousarray(point_cloud)

    normals = np.ascontiguousarray(normals)

    return mesh, point_cloud, normals