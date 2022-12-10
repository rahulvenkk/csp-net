from scipy.spatial import cKDTree as KDTree
import numpy as np
import trimesh
import os
import traceback
import time

kdtree, grid_points, cfg = None, None, None
def get_occupancy_from_pc(point_cloud, res):
        
    global kdtree, grid_points, cfg
    
    if grid_points is None:
    
        grid_points = create_grid_points_from_bounds(-0.5, 0.5, res)

        kdtree = KDTree(grid_points)


    occupancies = np.zeros(len(grid_points), dtype=np.int8)
    
#     t = time.time()
    _, idx = kdtree.query(point_cloud)
    occupancies[idx] = 1
#     print("time.time", time.time() - t)

#     compressed_occupancies = np.packbits(occupancies)
    compressed_occupancies = np.reshape(occupancies, (res,)*3).astype(np.float32)
    
    return compressed_occupancies
    
def create_grid_points_from_bounds(minimun, maximum, res):
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list