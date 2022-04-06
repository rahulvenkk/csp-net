import os
from tqdm import trange
import torch
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import time
import cv2
from ..common import (
     make_3d_grid
)
from ..utils import visualize as vis
from ..training import BaseTrainer

from ..evalMeshST import EvalMeshST
import matplotlib.pyplot as plt


class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.008, eval_sample=False, max_val=1):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.max_val = max_val
        
        if isinstance(model, torch.nn.DataParallel):
            self.model_attr_accessor = self.model.module
        else:
            self.model_attr_accessor = self.model

        self.flag_udf = False

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, val_loader):
        ''' Performs an evaluation.
        
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        eval_list = defaultdict(list)
        eval_dict = {}
        ct = 0
        for data in tqdm(val_loader):
            eval_step_dict, viz_image = self.eval_step(data)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)
                
            img_path = os.path.join(self.vis_dir, '%03d_viz.png' % ct)
            
            plt.imsave(img_path, viz_image)
            
            viz_image = cv2.resize(viz_image, (int(viz_image.shape[1]/2), int(viz_image.shape[0]/2)))
            eval_dict['viz_image_' + str(ct)] = viz_image 
            ct+=1

            if ct >= self.max_val:
                break
                
        for k, v in eval_list.items():   
            eval_dict[k] = np.mean(v)
        
        return eval_dict
    
    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        #MISE (Meshing)
        resolution0 = 16 # initial resolution of Voxel grid
        upsampling_steps = 4 # scale of increasing the resolution at next step
        batch_size_eval_mesh = 1000

        n_points_chamfer = 30000

        #Sphere tracing
        num_views = 2
        res = 256
        thresh = 0.008
        max_iter_st = 1000
        batch_size_eval_udf = 500000
        batch_size_eval_nvf = 1000
        direction = np.array([0.4, 0.3, 0.3])
        
        self.model.eval()

        device = self.device
        eval_dict = {}

        model_path = data['model_path'][0]
        evaluator = EvalMeshST(resolution0, upsampling_steps,\
                     batch_size_eval_mesh, n_points_chamfer,\
                     max_iter_st, batch_size_eval_udf, batch_size_eval_nvf,\
                    model_path, num_views, res, thresh, direction, device)

        inputs = data.get('points.points').to(device)

        dict_iso, dict_metrics = evaluator.get_isosurfaces_and_metrics(\
                                        self.model_attr_accessor, inputs)

        eval_dict['chamfer'] = dict_metrics['chamfer']
        eval_dict['normal'] = dict_metrics['cosine_normals']
        eval_dict['normal_jac'] = dict_metrics['cosine_normals_jac']
        eval_dict['sill_iou'] = dict_metrics['sill_iou']
        eval_dict['depth'] = dict_metrics['abs_depth_error']

        return eval_dict, dict_iso['viz_image']

    def visualize(self, data):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device

        batch_size = data['points.points_sampled'].size(0)
        inputs = data.get('inputs', torch.empty(batch_size, 0)).to(device)

        shape = (32, 32, 32)
        p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        p = p.expand(batch_size, *p.size())

        kwargs = {}
        with torch.no_grad():
            udf, nvf = self.model(p, p, inputs, **kwargs)

        udf_hat = udf.view(batch_size, *shape)
        voxels_out = (udf_hat <= self.threshold).cpu().numpy()

        for i in trange(batch_size):
            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            vis.visualize_data(
                inputs[i].cpu(), self.input_type, input_img_path)
            vis.visualize_voxels(
                voxels_out[i], os.path.join(self.vis_dir, '%03d.png' % i))

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        
        #point in full 3d space
        p = data.get('points.points_sampled').to(device)
        
        #points on surface for getting encoding (5k)
        p_surface = data.get('points.points').to(device)
        
        #get for points_samples
        points_gt = data.get('points.nn').to(device)

        #udf loss for 3d space pts
        enc = self.model_attr_accessor.get_encoding(p_surface.clone())  
        points_pred = self.model_attr_accessor.decoder(p, enc, return_feat=False)       
        loss = self.model_attr_accessor.loss(points_pred, points_gt)
        
        return loss