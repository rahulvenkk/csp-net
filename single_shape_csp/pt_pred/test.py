
import argparse
import mesh
import model
import torch
import sphere_trace as sphere_trace
import config as config
import os
import trimesh
import PIL.Image
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

config = config.Config()

parser = argparse.ArgumentParser(description='Get multi view depth and normal maps')
parser.add_argument('-ename', '--exp_name', type=str)
parser.add_argument('-model_iter', '--model_iter', type=int)
parser.add_argument('-num_views', '--num_views', type=int)
parser.add_argument('-reverse', '--reverse', action='store_true')
parser.add_argument('-point_pred', '--point_pred', action='store_true')
parser.add_argument('-fourier', '--fourier', action='store_true')
parser.add_argument('-sine', '--sine', action='store_true')
parser.add_argument('-omega', '--omega', type=float)

args = parser.parse_args()

config.exp_name = args.exp_name
config.omega = args.omega
config.make_paths()

weights_path = config.weights_path + '/' + str(args.model_iter) + '.pth'

if not os.path.exists(config.videos_path):
    os.system('mkdir ' + config.videos_path)

print('Loading model from ',  weights_path)
net = model.Net().cuda()
net.load_state_dict(torch.load(weights_path)['net'])
net.eval()

#GT maps
file_path = config.mesh_file_path

trimesh_mesh = trimesh.load(file_path, process=False)

if config.flip_axis:
    trimesh_mesh.vertices = trimesh_mesh.vertices[:, [0, 2, 1]]
    trimesh_mesh.face_normals = trimesh_mesh.face_normals[:, [0, 2, 1]]

o3d_mesh, point_cloud, normals = mesh.get_sampled_points_normals(file_path, flip_axis=False)
print("sampling done")

sphere_trace.evaluate_normals_depths(net, trimesh_mesh, args.num_views, config.sphere_tracing_thresh, config.videos_path, config.sphere_tracing_max_iters, config.res)
