import sys
sys.path.append(".")

from .utils import libmcubes
from .utils.libmise import MISE
import scipy.misc
from skimage.transform import resize
#from .generate_mesh_ndf import Generator
import numpy as np

from . import sphere_trace
import torch
from tensorboardX import SummaryWriter
import numpy as np

import time
import matplotlib; matplotlib.use('Agg')
import trimesh

import numpy as np
from scipy.spatial import cKDTree as KDTree

import matplotlib.pyplot as plt
import transforms3d
from .utilsST import get_jac_normals
# %matplotlib agg

    
def normalize(norm_xyz, eps=1e-7):
    return norm_xyz.t() / ((norm_xyz ** 2).sum(1) + eps * 0.1).sqrt().t()

class EvalMeshST():
    def __init__(self, resolution0, upsampling_steps, batch_size_eval_mesh, \
                 n_points_chamfer, max_iter_st, batch_size_eval_udf, \
                 batch_size_eval_nvf, model_path, num_views, res, thresh, \
                 direction, device, meshing=True, num=True):
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.batch_size_eval_mesh = batch_size_eval_mesh
        self.batch_size_eval_udf = batch_size_eval_udf
        self.batch_size_eval_nvf = batch_size_eval_nvf
        self.max_iter_st = max_iter_st
        self.n_points_chamfer = n_points_chamfer
        self.model_path = model_path
        self.num_views = num_views
        self.res = res
        self.thresh_st = thresh    
        self.direction = direction
        self.direction /= np.sqrt(np.sum(self.direction**2) + 1e-8)
        self.device = device
        self.meshing=meshing
        self.num=num
                
        self.mesh_gt = self.as_mesh((self.get_scaled_mesh(model_path)))
        self.normals_pred_gt, self.depth_gt = sphere_trace.get_multi_view_ray_tracing(self.mesh_gt,\
                                                                                 self.num_views, self.res)
        self.depth_gt[self.depth_gt==100] = -1
        self.valid_pixels_gt = ~(self.depth_gt == self.depth_gt[0, 0, 0])
        self.lighted_normals_gt = self.get_lighted_image(self.normals_pred_gt, \
                                                         self.depth_gt)
                                        
    def as_mesh(self, scene_or_mesh):
        """
        Convert a possible scene to a mesh.

        If conversion occurs, the returned mesh has only vertex and face data.
        """
        if isinstance(scene_or_mesh, trimesh.Scene):
            if len(scene_or_mesh.geometry) == 0:
                mesh = None  # empty scene
            else:
                # we lose texture information here
                mesh = trimesh.util.concatenate(
                    tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                        for g in scene_or_mesh.geometry.values()))
        else:
            assert(isinstance(scene_or_mesh, trimesh.Trimesh))
            mesh = scene_or_mesh

        # NOTE: it was observed that loaded mesh obj from shapenet v1 is rotated
        # Therefore, it is rotated back to the original form.
        rotation = np.eye(4)
        rotation[:3, :3] = transforms3d.euler.euler2mat(np.deg2rad(90), 0, 0)
        mesh.apply_transform(rotation)
            
        return mesh

    def get_scaled_mesh(self, model_path):
        path = self.get_obj_path(model_path)
        yourList = trimesh.load_mesh(path, process=False)

        if isinstance(yourList, list):
            vertice_list = [mesh.vertices for mesh in yourList]
            faces_list = [mesh.faces for mesh in yourList]

            faces_offset = np.cumsum([v.shape[0] for v in vertice_list])
            faces_offset = np.insert(faces_offset, 0, 0)[:-1]

            vertices = np.vstack(vertice_list)
            faces = np.vstack([face + offset for face, offset in zip(faces_list, faces_offset)])

            vertices = vertices[:, [0, 2, 1]]

            mesh = trimesh.Trimesh(vertices, faces)
        else:
            mesh = yourList
            mesh.vertices = self.as_mesh(mesh).vertices[:, [0, 2, 1]]
        
        bbox = mesh.bounding_box.bounds

        # Compute location and scale
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max() / (1 - 0.1)

        # Transform input mesh
        mesh.apply_translation(-loc)
        mesh.apply_scale(1 / scale)

        trimesh_mesh = mesh 

        return trimesh_mesh

    def get_lighted_image(self, normals, depths):
        normals = self.get_light(normals)
        normals = np.clip(normals, 0, 1)  

        valid = depths == depths[0, 0]

        normals[valid] = 1

        return normals

    def get_light(self, normal_map):
        normal_map = normal_map.transpose([3, 0, 1, 2])
        normal_map = normal_map / np.sqrt((normal_map ** 2).sum(0))
        normal_map = normal_map.transpose([1, 2,  3, 0])

        direction = self.direction[None, None, None, :]

        lighted = np.abs((normal_map * direction).sum(-1))
        lighted = np.stack([lighted] * 3, -1)

        return lighted

    def get_obj_path(self, path):
        model_path = path.split('/')[-1]
        cls = path.split('/')[-2]
        model_path = 'data/external/ShapeNetCore.v1/' + cls + '/' + model_path + '/model.obj'
        return model_path

    def get_predicted_mesh(self, model, inputs, num=True):        
        edge_length = (2 ** self.upsampling_steps) * self.resolution0
        thresh = 1/edge_length + 1/edge_length/2

        mesh_extractor = MISE(self.resolution0, self.upsampling_steps)
        points, voxels = mesh_extractor.query()
        
        c_udf = model.get_encoding(inputs)
        
        normals = torch.zeros([edge_length+1, edge_length+1, \
                               edge_length+1, 3]).to(self.device)
        total_feval = 0
        total_time = 0

        while points.shape[0] != 0:
            pointsf = points[None]
            pointsf = torch.FloatTensor(pointsf).to(self.device)

            all_udfs = []
            total_feval += pointsf.shape[1]
            for x in range(0, len(pointsf[0]), self.batch_size_eval_mesh):
                dat = pointsf[:, x:x+self.batch_size_eval_mesh]
                t = time.time()
                udf = model.decode_udf(dat/edge_length - 0.5, c_udf)
                total_time += time.time() - t

                if not num:
                    nms = model.decode_nvf(dat/edge_length - 0.5, c_udf, c_udf)
                else:
                    nms = get_jac_normals(dat/edge_length - 0.5, c_udf, model)

                normals[points[x:x+self.batch_size_eval_mesh, 0], \
                        points[x:x+self.batch_size_eval_mesh, 1], \
                        points[x:x+self.batch_size_eval_mesh, 2]] = nms
                
                values = udf.cpu().detach().numpy().astype(np.float64)[0]

                all_udfs.append(values)

            values = np.concatenate(all_udfs)

            t = time.time()
            mesh_extractor.update_udf(points, values*edge_length)
            points, voxels = mesh_extractor.query()
            total_time += time.time() - t

        value_grid = mesh_extractor.to_dense() / edge_length
        t = time.time()
        vertices, triangles = libmcubes.marching_cubes(value_grid, thresh)

        value_grid_voxel = value_grid <= thresh
        value_grid_voxel = np.where(value_grid_voxel)
        value_grid_voxel = np.stack(value_grid_voxel, 1)

        mesh = trimesh.Trimesh(vertices, triangles)

        mesh.apply_scale(1 / edge_length)
        mesh.apply_scale(1 - thresh)

        mesh.apply_translation(-np.ones(3) / 2)
                
        mesh_dmc = mesh
        
        return mesh, mesh_dmc
        
    def get_isosurfaces_and_metrics(self, model, inputs):
        dict_iso = {}
        dict_metrics = {}
        
        #Mesh
        if self.meshing:
            pred_mesh, pred_mesh_dmc = self.get_predicted_mesh(model, inputs, num=self.num)
        
            #Metrics
            chamfer, chamfer_l1 = self.compute_chamfer(pred_mesh, self.mesh_gt)
            chamfer_dmc, chamfer_dmc_l1 = self.compute_chamfer(pred_mesh_dmc, \
                                                               self.mesh_gt)
            dict_iso.update({'mesh': pred_mesh, 'mesh_dmc':pred_mesh_dmc})
            dict_metrics.update({'chamfer':chamfer, 'chamfer_l1':chamfer_l1, \
                                 'chamfer_dmc':chamfer_dmc, \
                                 'chamfer_dmc_l1':chamfer_dmc_l1,})
            
        #Sphere Tracing
        with torch.no_grad():
            xyz_world, normals_pred, normals_pred_jac, depths =\
            sphere_trace.get_multi_view_sphere_tracing(model, inputs, \
                                                       self.num_views, \
                                                       self.thresh_st, \
                                                       self.max_iter_st, \
                                                       res=self.res,\
                                                       batch_size_eval=self.batch_size_eval_udf,\
                                                       batch_size_eval_nvf=self.batch_size_eval_nvf)

        lighted_normals = self.get_lighted_image(normals_pred, depths)
        lighted_normals_gt = self.get_lighted_image(self.normals_pred_gt, self.depth_gt)
        lighted_normals_jac = self.get_lighted_image(normals_pred_jac, depths)
        
        valid_pred = ~(depths == depths[0, 0, 0])
        
        valid_comb = valid_pred & self.valid_pixels_gt
        
        temp = np.abs(np.sum(self.norm(normals_pred[valid_comb]) *\
                            self.norm(self.normals_pred_gt[valid_comb]), -1))
        normal_error = np.mean(temp)
        
        var_normal_error = np.std(np.sort(temp)[::-1][:int(0.6*len(temp))])
        
        temp = np.abs(np.sum(self.norm(normals_pred_jac[valid_comb])*self.norm(self.normals_pred_gt[valid_comb]), -1))
        normal_error_num = np.mean(temp)

        var_normal_error_num = np.std(np.sort(temp)[::-1][:int(0.6*len(temp))])
        depth_error_map = np.abs(depths - self.depth_gt)
        depth_error = np.mean(depth_error_map[valid_comb])
        depth_error_map[~valid_comb] = 0
        depth_error_map[depth_error_map>1] = 0        
        
        num_valid = np.sum(np.sum((valid_pred == self.valid_pixels_gt).astype(float), axis=1), 1)
        num_invalid = np.sum(np.sum((valid_pred != self.valid_pixels_gt).astype(float), axis=1), 1)
        sill_iou = np.mean(num_valid/(num_valid + num_invalid))    

        dict_iso.update({ 'mesh_gt': self.mesh_gt, 'normals': normals_pred, 'normals_gt':self.normals_pred_gt ,'depths_gt': self.depth_gt,\
                    'normals_jac': normals_pred_jac, 'depths': depths, 'depth_error': depth_error_map, \
                    'lighted_normals': lighted_normals, 'lighted_normals_jac': lighted_normals_jac, 'lighted_normals_gt': lighted_normals_gt})
        
        viz_pred_gt = self.get_pred_gt_img(dict_iso, inputs)
        viz = self.save_visualizations(dict_iso)
        dict_iso['viz_image'] = viz
        dict_iso['viz_pred_gt'] = viz_pred_gt

        dict_metrics.update({'cosine_normals':normal_error, 'var_cosine_normals':var_normal_error,\
                        'cosine_normals_jac': normal_error_num, 'var_cosine_normals_jac': var_normal_error_num,\
                        'abs_depth_error': depth_error, 'sill_iou': sill_iou})

        return dict_iso, dict_metrics

    def compute_chamfer(self, mesh1, mesh2):
        """
        This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.
        gt_points: trimesh.points.PointCloud of just points, sampled from the surface (see
                   compute_metrics.ply for more documentation)
        gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
                  method (see compute_metrics.py for more)
        """

        try:
            gen_points_sampled = mesh1.sample(self.n_points_chamfer)
        except: 
            arr = np.array(mesh1.vertices)
            np.random.shuffle(arr)
            gen_points_sampled = arr[:self.n_points_chamfer]
        
        try:
            gt_points = mesh2.sample(self.n_points_chamfer)
        except:
            arr = np.array(mesh2.vertices)
            np.random.shuffle(arr)
            gt_points = arr[:self.n_points_chamfer]

        gen_points_kd_tree = KDTree(gen_points_sampled)
        one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
        gt_to_gen_chamfer = np.mean(np.square(one_distances))
        
        gt_to_gen_chamfer_l1 = np.mean(one_distances)
        
        # other direction
        gt_points_kd_tree = KDTree(gt_points)
        two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
        gen_to_gt_chamfer = np.mean(np.square(two_distances))
        gen_to_gt_chamfer_l1 = np.mean(two_distances)

        return (gt_to_gen_chamfer + gen_to_gt_chamfer)*10000/2, (gt_to_gen_chamfer_l1 + gen_to_gt_chamfer_l1)*10000/2

    def norm(self, vec):
        vec = (vec.T/(np.sqrt(np.sum(vec**2, -1)))).T
        return vec
    
    def visualize_pc(self, pc):       
        fig = plt.figure(figsize=(30, 30))
        ax = fig.add_subplot(111, projection='3d')
        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        #ax.set_facecolor(None)

        # Get rid of colored axes planes
        # First remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Bonus: To get rid of the grid as well:
        ax.grid(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_axis_off()
        ax.scatter(pc[:, 0] , pc[:, 2], pc[:, 1], s=60, color='b', marker='o', alpha=0.25)
        ax.axis('auto')
        ax.dist = 8
        
        img = plt.imread("img.png")
        img = resize(img, (self.res, self.res))
        
        plt.close(fig)
        
        return img
    
    def whiten_normal(self, norm_map):
        norm_map = norm_map*0.5 + 0.5
        norm_map[norm_map == norm_map[0, 0]] = 1
        return norm_map
    
    def whiten_depth(self, dep):
        fig = plt.figure()
        plt.imsave("test.png", dep)
        img = plt.imread("test.png")[:, :, :3]
        plt.close(fig)
        return img
        
    def get_pred_gt_img(self, dict_iso, inputs):
        img_direct = dict_iso['lighted_normals'][0]
        img = dict_iso['lighted_normals_jac'][0]
        img_gt = dict_iso['lighted_normals_gt'][0]
        inputs = inputs.detach().cpu().numpy()
        # img_pc = self.visualize_pc(inputs[0])[:, :, :3]
        
        normal_map_gt = self.whiten_normal(self.normals_pred_gt[0])
        normal_map_pred_num = self.whiten_normal(dict_iso['normals_jac'][0])
        normal_map_pred = self.whiten_normal(dict_iso['normals'][0])
        
        d_gt = self.depth_gt[0].copy()
        d_gt[d_gt==-1] = np.max(d_gt) + 0.15

        d_pred = dict_iso['depths'][0].copy()
        d_pred[d_pred==-1] = np.max(d_pred) + 0.15
        
        depth_error = self.whiten_depth(dict_iso['depth_error'][0])
        depth_gt = self.whiten_depth(d_gt)
        depths = self.whiten_depth(d_pred)

        img_light = np.concatenate([ img_direct, img, img_gt], 1)
        img_normals = np.concatenate([normal_map_pred, normal_map_pred_num, normal_map_gt], 1)
        img_depth = np.concatenate([ depths, depth_error, depth_gt], 1)
                
        img = np.concatenate([img_light, img_normals, img_depth], 0)
        
        return img
    
    def save_visualizations(self, dict_iso):
        n_rows = self.normals_pred_gt.shape[0]
        n_cols = 6
        fig = plt.figure(figsize=[n_cols*6, n_rows*6])
        ct = 1

        for x in range(1, n_rows+1):
            ax = fig.add_subplot(n_rows, n_cols, ct)
            ax.imshow(self.normals_pred_gt[x-1]*0.5+0.5)
            ax.set_title("GT Normals")
            ax.set_xticks([])
            ax.set_yticks([])
            ct += 1

            ax = fig.add_subplot(n_rows, n_cols, ct)
            ax.imshow(dict_iso['normals'][x-1]*0.5+0.5)
            ax.set_title("Pred Normals")
            ax.set_xticks([])
            ax.set_yticks([])
            ct += 1

            ax = fig.add_subplot(n_rows, n_cols, ct)
            ax.imshow(dict_iso['normals_jac'][x-1]*0.5 + 0.5)
            ax.set_title("Jac. Normals")
            ax.set_xticks([])
            ax.set_yticks([])
            ct += 1

            ax = fig.add_subplot(n_rows, n_cols, ct)
            ax.imshow(dict_iso['lighted_normals_gt'][x-1])
            ax.set_title("Lighted Normals")
            ax.set_xticks([])
            ax.set_yticks([])
            ct += 1

            ax = fig.add_subplot(n_rows, n_cols, ct)
            ax.imshow(dict_iso['lighted_normals'][x-1])
            ax.set_title("Lighted Normals")
            ax.set_xticks([])
            ax.set_yticks([])
            ct += 1

            ax = fig.add_subplot(n_rows, n_cols, ct)
            ax.imshow(dict_iso['lighted_normals_jac'][x-1])
            ax.set_title("Lighted Jac. Normals")
            ax.set_xticks([])
            ax.set_yticks([])
            ct += 1
            
        fig.savefig('./temp.png')#, transparent=True)
        img = plt.imread('./temp.png')
        plt.close(fig)

        return img

