import os

os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

# os.environ["PYOPENGL_PLATFORM"] = "mesa"

os.environ['DISPLAY'] = ':0.0'

os.environ['NVIDIA_DRIVER_CAPABILITIES'] = 'graphics'

import config as config
from tensorboardX import SummaryWriter
import mesh
import torch
import trimesh
from tqdm import tqdm
import numpy as np
from model import Net, init_weights
from train_utils import train_val_step
import argparse


config = config.Config()

parser = argparse.ArgumentParser(description='Get multi view depth and normal maps')
parser.add_argument('-ename', '--exp_name', type=str, help='Expt name')
parser.add_argument('-infile', '--infile', type=str, help='Input 3D shape as .off file')

args = parser.parse_args()
config.exp_name = args.exp_name
config.mesh_file_path = args.infile

######################################################################################################################
'''
Prepare Data
'''
file_path = config.mesh_file_path

trimesh_mesh = trimesh.load_mesh(file_path, process=False)

o3d_mesh, point_cloud, normals = mesh.get_sampled_points_normals(file_path, flip_axis=config.flip_axis)
print("point sampling done")

input_points, sDF, nF, nn = mesh.get_data(o3d_mesh, point_cloud, normals, is_flip=config.flip_axis)
print("Nearest neighbour creation done")

######################################################################################################################

'''
Create path for saving weights
'''
if not os.path.exists('./weights/'):
    os.system('mkdir ' + './weights/')

if not os.path.exists('./videos/'):
    os.system('mkdir ' + './videos/')

if not os.path.exists(config.weights_path):
    os.system('mkdir ' + config.weights_path)

if not os.path.exists(config.videos_path):
    os.system('mkdir ' + config.videos_path)

######################################################################################################################

'''
initialize net
'''

net = Net().cuda()
net.apply(init_weights)

net.train()

# Load weights if needed
if config.load_weights:
    print("Loading weights from ", config.load_weights_path)
    net.load_state_dict(torch.load(config.load_weights_path))

######################################################################################################################
'''
Create optimizers
    One for normal field network 
    One for Signed/Unsigned Field Network 
'''

optim = torch.optim.Adam(net.parameters(), lr=0.0001)

######################################################################################################################

# Delete Logs if needed
if config.delete_logs:
    os.system('rm -rf ' + config.train_writer_path)
    os.system('rm -rf ' + config.val_writer_path)


#save configs
os.system('cp pt_pred/config.py ' + config.exp_path + '/')
######################################################################################################################
'''
Initialize the train and val writers
'''
train_writer = SummaryWriter(config.train_writer_path)
val_writer = SummaryWriter(config.val_writer_path)

######################################################################################################################
'''
Sample points based on the distance from the surface
Sample more points closer to the surface
'''
inv_dists = 1 / np.abs(sDF)
prob_dists = inv_dists / np.sum(inv_dists)

######################################################################################################################
# ReSampling GT and associated visualization
if config.resample:
    pts_sample, dSample, nSample, nnSample = mesh.get_sample_points(input_points, nn, config.res ** 2, prob_dists)
else:
    pts_sample, dSample, nSample, nnSample = input_points, sDF, nF, nn

######################################################################################################################

# Split train and test set
temp = np.arange(0, pts_sample.shape[0] - config.batch_size)
np.random.shuffle(temp)

train_indices = temp[:int(pts_sample.shape[0] * config.train_perc)]
val_indices = temp[int(pts_sample.shape[0] * config.train_perc):]

######################################################################################################################
# Create train-val split

p_inds_train = None
n_inds_train = None
p_inds_val = None
n_inds_val = None

pts_sample_tr = pts_sample[train_indices]
dSample_tr = dSample[train_indices]
nSample_tr = nSample[train_indices]
nnSample_tr = nnSample[train_indices]
pts_sample_vl = pts_sample[val_indices]
dSample_vl = dSample[val_indices]
nSample_vl = nSample[val_indices]
nnSample_vl = nnSample[val_indices]

train_size = pts_sample_tr.shape[0]
val_size = pts_sample_vl.shape[0]
num_epochs = int((config.num_iters / train_size) * config.batch_size)
# train_val loop
total_iters = 0
idxs = list(range(train_size))

######################################################################################################################
# Start training

for ep in tqdm(range(num_epochs)):
    np.random.shuffle(idxs)
    for it in range(0, train_size, config.batch_size):
        if (total_iters % config.save_after == 0):
            torch.save(net.state_dict(), config.weights_path + '/' + str(it) + '.pth')

        pts = pts_sample_tr[idxs[it:it+config.batch_size]]
        nn = nnSample_tr[idxs[it:it+config.batch_size]]
        dist = dSample_tr[idxs[it:it+config.batch_size]]
        normal = nSample_tr[idxs[it:it+config.batch_size]]
        loss_pts, loss_normal, loss_dist = train_val_step(net, pts, nn, dist, normal, optim)

        # Log losses to tensorboard
        train_writer.add_scalar('loss_point', loss_pts, total_iters)

        if (total_iters % config.val_after == 0):
            # validate
            print("train loss point:", loss_pts)

            net.eval()

            with torch.no_grad():
                loss_pts, loss_normal, loss_dist = train_val_step(net, pts_sample_vl, nnSample_vl, dSample_vl, nSample_vl)

                # Log losses to tensorboard
                val_writer.add_scalar('loss_point', loss_pts, total_iters)                            
                print("val loss point:", loss_pts)
                
            net.train()
        
        total_iters += 1
