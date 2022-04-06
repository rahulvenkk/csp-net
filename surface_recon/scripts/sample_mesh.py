import faiss
import open3d as o3d
import argparse
import trimesh
import numpy as np
import os
import glob
import sys
from multiprocessing import Pool
from functools import partial
# TODO: do this better
sys.path.append('..')
from im2mesh.utils import binvox_rw, voxels
from im2mesh.utils.libmesh import check_mesh_contains

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser('Sample a watertight mesh.')
parser.add_argument('in_folder', type=str,
                    help='Path to input watertight meshes.')
parser.add_argument('--n_proc', type=int, default=0,
                    help='Number of processes to use.')

parser.add_argument('--resize', action='store_true',
                    help='When active, resizes the mesh to bounding box.')

parser.add_argument('--rotate_xz', type=float, default=0.,
                    help='Angle to rotate around y axis.')

parser.add_argument('--bbox_padding', type=float, default=0.,
                    help='Padding for bounding box')
parser.add_argument('--bbox_in_folder', type=str,
                    help='Path to other input folder to extract'
                         'bounding boxes.')

parser.add_argument('--pointcloud_folder', type=str,
                    help='Output path for point cloud.')
parser.add_argument('--pointcloud_size', type=int, default=250000,
                    help='Size of point cloud.')

parser.add_argument('--voxels_folder', type=str,
                    help='Output path for voxelization.')
parser.add_argument('--voxels_res', type=int, default=32,
                    help='Resolution for voxelization.')

parser.add_argument('--points_folder', type=str,
                    help='Output path for points.')
parser.add_argument('--points_size', type=int, default=500000,
                    help='Size of points.')
parser.add_argument('--points_uniform_ratio', type=float, default=0.1,
                    help='Ratio of points to sample uniformly'
                         'in bounding box.')
parser.add_argument('--points_sigma', type=float, default=0.005,
                    help='Standard deviation of gaussian noise added to points'
                         'samples on the surfaces.')
parser.add_argument('--points_padding', type=float, default=0.1,
                    help='Additional padding applied to the uniformly'
                         'sampled points on both sides (in total).')

parser.add_argument('--mesh_folder', type=str,
                    help='Output path for mesh.')

parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite output.')

# parser.add_argument('--sdf', action='store_true',
#                     help='Whether to overwrite output.')

parser.add_argument('--float16', action='store_true',
                    help='Whether to use half precision.')
parser.add_argument('--packbits', action='store_true',
                help='Whether to save truth values as bit array.')
    
def main(args):
    input_files = glob.glob(os.path.join(args.in_folder, '*.off'))
    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(process_path, args=args), input_files)
    else:
        for p in input_files:
            print("process", p)
            process_path(p, args)
#             break


def process_path(in_path, args):
    in_file = os.path.basename(in_path)
    modelname = os.path.splitext(in_file)[0]
    mesh = trimesh.load(in_path, process=False)
    
  
    mesh_o3d = o3d.io.read_triangle_mesh(in_path)
#     mesh_trimesh = trimesh.load(in_path, process=False)
    
    # Determine bounding box
    if not args.resize:
        # Standard bounding boux
        loc = np.zeros(3)
        scale = 1.
    else:
        if args.bbox_in_folder is not None:
            in_path_tmp = os.path.join(args.bbox_in_folder, modelname + '.off')
            mesh_tmp = trimesh.load(in_path_tmp, process=False)
            bbox = mesh_tmp.bounding_box.bounds
        else:
            bbox = mesh.bounding_box.bounds

        # Compute location and scale
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max() / (1 - args.bbox_padding)

        # Transform input mesh
        mesh.apply_translation(-loc)
        mesh.apply_scale(1 / scale)

        if args.rotate_xz != 0:
            angle = args.rotate_xz / 180 * np.pi
            R = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
            mesh.apply_transform(R)
        
        in_path_tmp = os.path.join(args.bbox_in_folder, modelname + '_tt.off')
        mesh.export(in_path_tmp)
        mesh_o3d = o3d.io.read_triangle_mesh(in_path_tmp)
    # Expert various modalities
    if args.pointcloud_folder is not None:
        export_pointcloud(mesh, modelname, loc, scale, args)

    if args.voxels_folder is not None:
        export_voxels(mesh, modelname, loc, scale, args)

    if args.points_folder is not None:
        export_points(mesh_o3d, mesh, modelname, loc, scale, args)

    if args.mesh_folder is not None:
        export_mesh(mesh, modelname, loc, scale, args)


def export_pointcloud(mesh, modelname, loc, scale, args):
    filename = os.path.join(args.pointcloud_folder,
                            modelname + '.npz')
    if not args.overwrite and os.path.exists(filename):
        print('Pointcloud already exist: %s' % filename)
        return

    points, face_idx = mesh.sample(args.pointcloud_size, return_index=True)
    normals = mesh.face_normals[face_idx]

    # Compress
    if args.float16:
        dtype = np.float16
    else:
        dtype = np.float32

    points = points.astype(dtype)
    normals = normals.astype(dtype)

    print('Writing pointcloud: %s' % filename)
    np.savez(filename, points=points, normals=normals, loc=loc, scale=scale)


def export_voxels(mesh, modelname, loc, scale, args):
    if not mesh.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot create voxelization.' % modelname)
        return

    filename = os.path.join(args.voxels_folder, modelname + '.binvox')

    if not args.overwrite and os.path.exists(filename):
        print('Voxels already exist: %s' % filename)
        return

    res = args.voxels_res
    voxels_occ = voxels.voxelize(mesh, res)

    voxels_out = binvox_rw.Voxels(voxels_occ, (res,) * 3,
                                  translate=loc, scale=scale,
                                  axis_order='xyz')
    print('Writing voxels: %s' % filename)
    with open(filename, 'bw') as f:
        voxels_out.write(f)



def export_points(mesh, trimesh_mesh, modelname, loc, scale, args):
#     print("in export")
#     print(modelname)
    
    filename = os.path.join(args.points_folder, modelname + '.npz')

    if not args.overwrite and os.path.exists(filename):
        print('Points already exist: %s' % filename)
        return
    
    num_points = args.pointcloud_size

    point_cloud, indices = trimesh_mesh.sample(num_points, True)
    normals = trimesh_mesh.face_normals[indices]

    point_cloud = np.ascontiguousarray(point_cloud)

    normals = np.ascontiguousarray(normals)

    n_points_uniform = int(num_points * 0.1)
    n_points_surface = num_points#num_points - n_points_uniform

    boxsize = 1 + args.points_padding
    points_uniform = np.random.rand(n_points_uniform, 3)
    points_uniform = boxsize*(points_uniform - 0.5)
    points_surface = mesh.sample_points_uniformly(n_points_surface)
    points_surface_ = np.asarray(points_surface.points)


    points_surface = points_surface_.copy()
    points_surface_small = points_surface + np.random.normal(0, np.sqrt(0.00025), [n_points_surface, 3])
    points_surface_smaller = points_surface + np.random.normal(0, np.sqrt(0.000025), [n_points_surface, 3])
    data = np.concatenate([points_uniform, points_surface_small, points_surface_smaller], axis=0).astype('float32')

    d = point_cloud.shape[1]

    # init index
    cpu_index = faiss.IndexFlatL2(d)

#     print("to_gpu")
    index = faiss.index_cpu_to_all_gpus(cpu_index)
#     print("to_gpu done")

    # Add point cloud to index
    index.add(point_cloud.astype('float32'))

    # Get NN for each point in data
    D, I = index.search(data, 1)

    # squeeze D, I
    D = D[:, 0]
    I = I[:, 0]

    # min_distances now constitutes uDF, for sDF see the following for loop
    
    nn = point_cloud[I, :]
    
    vectors = data - nn
    min_distances = np.sqrt(np.sum((data - nn) ** 2, 1))

    normals_grid = normals[I, :]
    
    try:
        inside = trimesh_mesh.contains(data)
        min_distances[inside] *= -1
    except: 
        print("could not save sdf")
        pass
    
#     return data, min_distances, normals_grid
    point_cloud = point_cloud[:, [0, 2, 1]]
    normals = normals[:, [0, 2, 1]]
    data = data[:, [0, 2, 1]]
    normals_grid = normals_grid[:, [0, 2, 1]]
    
    np.savez(filename, point_cloud=point_cloud, normals=normals, point_sampled=data, sDF=min_distances,
             nF=normals_grid, nP=nn)


def export_mesh(mesh, modelname, loc, scale, args):
    filename = os.path.join(args.mesh_folder, modelname + '.off')    
    if not args.overwrite and os.path.exists(filename):
        print('Mesh already exist: %s' % filename)
        return
    print('Writing mesh: %s' % filename)
    mesh.export(filename)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


# def export_points(mesh, modelname, loc, scale, args):
#     if not mesh.is_watertight:
#         print('Warning: mesh %s is not watertight!'
#               'Cannot sample points.' % modelname)
#         return

#     filename = os.path.join(args.points_folder, modelname + '.npz')

#     if not args.overwrite and os.path.exists(filename):
#         print('Points already exist: %s' % filename)
#         return

#     n_points_uniform = int(args.points_size * args.points_uniform_ratio)
#     n_points_surface = args.points_size - n_points_uniform

#     boxsize = 1 + args.points_padding
#     points_uniform = np.random.rand(n_points_uniform, 3)
#     points_uniform = boxsize * (points_uniform - 0.5)
#     points_surface = mesh.sample(n_points_surface)
#     points_surface += args.points_sigma * np.random.randn(n_points_surface, 3)
#     points = np.concatenate([points_uniform, points_surface], axis=0)

#     occupancies = check_mesh_contains(mesh, points)

#     # Compress
#     if args.float16:
#         dtype = np.float16
#     else:
#         dtype = np.float32

#     points = points.astype(dtype)

#     if args.packbits:
#         occupancies = np.packbits(occupancies)

#     print('Writing points: %s' % filename)
#     np.savez(filename, points=points, occupancies=occupancies,
#              loc=loc, scale=scale)

# def export_points(mesh, mesh_trimesh, modelname, loc, scale, args):
# #     print("in export")
    
#     filename = os.path.join(args.points_folder, modelname + '.npz')

#     if not args.overwrite and os.path.exists(filename):
#         print('Points already exist: %s' % filename)
#         return
    
#     mesh.compute_vertex_normals()
#     pt = mesh.sample_points_uniformly(args.points_size)
#     normals = np.asarray(pt.normals)
#     point_cloud = np.asarray(pt.points)

#     n_points_uniform = int(args.points_size * args.points_uniform_ratio)
#     n_points_surface = int(args.points_size * (1-args.points_uniform_ratio))

#     boxsize = 1 + args.points_padding
#     points_uniform = np.random.rand(n_points_uniform, 3)
#     points_uniform = boxsize * (points_uniform - 0.5)
#     points_surface = mesh.sample_points_uniformly(n_points_surface)
#     points_surface = np.asarray(points_surface.points)
#     points_surface += args.points_sigma * np.random.randn(n_points_surface, 3)
#     data = np.concatenate([points_uniform, points_surface], axis=0).astype('float32')
    
#     d = point_cloud.shape[1]
    
#     #init index
# #     index = faiss.IndexFlatL2(d)
    
#     #init index
#     cpu_index = faiss.IndexFlatL2(d)
# #     print("to gpu", d)
#     index = faiss.index_cpu_to_all_gpus(cpu_index)
    
#     #Add point cloud to index
# #     print("to gpu done")
#     index.add(point_cloud.astype('float32'))
# #     print("search doing")
#     #Get NN for each point in data
#     D, I = index.search(data, 1) 
# #     print("search done")
#     #squeeze D, I
#     D = D[:, 0]
#     I = I[:, 0]
    
#     #min_distances now constitutes uDF, for sDF see the following for loop
#     min_distances = np.sqrt(np.sum((data - point_cloud[I, :])**2, 1))

#     normals_grid = normals[I, :]
    
#     inside = mesh_trimesh.contains(data)
#     min_distances[inside] = -min_distances[inside]
    
# #     return data, min_distances, normals_grid
#     point_cloud = point_cloud[:, [0, 2, 1]]
#     normals = normals[:, [0, 2, 1]]
#     data = data[:, [0, 2, 1]]
#     normals_grid = normals_grid[:, [0, 2, 1]]
#     np.savez(filename, point_cloud=point_cloud, normals=normals, point_sampled=data, sDF=min_distances,
#              nF=normals_grid)

