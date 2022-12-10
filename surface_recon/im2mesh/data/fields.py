import os
import glob
import random
from PIL import Image
import numpy as np
import trimesh
from ..data.core import Field
from ..utils import binvox_rw
import time
from tempfile import mkdtemp
import os.path as path
from . import ndf_utils
# import time
saved_npz  = {}

class IndexField(Field):
    ''' Basic index field.'''
    def load(self, model_path, idx, category):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return idx

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True


class CategoryField(Field):
    ''' Basic category field.'''
    def load(self, model_path, idx, category):
        ''' Loads the category field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return category

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True


class ImagesField(Field):
    ''' Image Field.

    It is the field used for loading images.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
        random_view (bool): whether a random view should be used
        with_camera (bool): whether camera data should be provided
    '''
    def __init__(self, folder_name, transform=None,
                 extension='jpg', random_view=True, with_camera=False):
        self.folder_name = folder_name
        self.transform = transform
        self.extension = extension
        self.random_view = random_view
        self.with_camera = with_camera

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        folder = os.path.join(model_path, self.folder_name)
        files = glob.glob(os.path.join(folder, '*.%s' % self.extension))
        if self.random_view:
            idx_img = random.randint(0, len(files)-1)
        else:
            idx_img = 0
        filename = files[idx_img]

        image = Image.open(filename).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        data = {
            None: image
        }

        if self.with_camera:
            camera_file = os.path.join(folder, 'cameras.npz')
            camera_dict = np.load(camera_file)
            Rt = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
            K = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)
            data['world_mat'] = Rt
            data['camera_mat'] = K

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.folder_name in files)
        # TODO: check camera
        return complete


# 3D Fields
class PointsField(Field):
    ''' Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the
            points tensor
        with_transforms (bool): whether scaling and rotation data should be
            provided

    '''
    def __init__(self, file_name, transform=None, with_transforms=False, unpackbits=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms
        self.unpackbits = unpackbits
        self.dict_ram = {}
    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        global saved_npz
        file_path = os.path.join(model_path, self.file_name)
        if file_path not in self.dict_ram.keys():

            points_dict = np.load(file_path)
            

            points = points_dict['points']
            # Break symmetry if given in float16:
            if points.dtype == np.float16:
                points = points.astype(np.float32)
                points += 1e-4 * np.random.randn(*points.shape)
            else:
                points = points.astype(np.float32)

            occupancies = points_dict['occupancies']
            if self.unpackbits:
                occupancies = np.unpackbits(occupancies)[:points.shape[0]]
            occupancies = occupancies.astype(np.float32)

            data = {
                None: points,
                'occ': occupancies,
            }

            self.dict_ram[file_path] = data

        data = self.dict_ram[file_path]
        

        if self.with_transforms:
            data['loc'] = points_dict['loc'].astype(np.float32)
            data['scale'] = points_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)


        return data


class uDFField(Field):
    ''' Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform: transformation which will be applied to the
            points tensor
        with_transforms (bool): whether scaling and rotation data should be
            provided

    '''
    def __init__(self, file_name, transform=None, with_transforms=False, ndf=False, res=256):
        self.file_name = file_name
        self.transform = transform
#         self.pointcloud_transform = pointcloud_transform
        self.with_transforms = with_transforms
        self.mmap = True
        self.dict_ram = {}
        self.model_to_indx = {}
        self.ct = 0
        self.index_sample_model = {}
        self.all_data = None
        self.max_prefetch = 4000
        self.epoch = 0
        self.ndf = ndf
        self.res = res
        # self._x = "a"
        # print("inti")

    def reset_index(self, index_model):
        for key in index_model.keys():
            index_model[key] = 0
        return index_model

    def create_mmap(self, model_path, key, arr):
        filename = path.join(mkdtemp(), model_path + '_' + key + '.dat')
        fp = np.memmap(filename, dtype='float32', mode='w+', shape=arr.shape)
        fp[:] = arr[:]
        del fp
        newfp = np.memmap(filename, dtype='float32', mode='r', shape=arr.shape)
        return newfp

    def create_mmap_large(self, key, arr):
        filename = path.join(mkdtemp(), key + '.dat')
        shp = list(arr)
        shp.insert(0, 6000)
        shp = tuple(shp)
        fp = np.memmap(filename, dtype='float32', mode='w+', shape=shp)
        # fp[:] = arr[:]
        # del fp
        # newfp = np.memmap(filename, dtype='float32', mode='r', shape=arr.shape)
        return fp

    def cleanup(self,):
        os.system('rm -rf /tmp/tmp*')
        del self.all_data
    
    def load(self, model_path, idx, category, no_return=False):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''

        file_path = os.path.join(model_path, self.file_name)
        model_name = model_path.split('/')[-1]

        #import pdb; pdb.set_trace()
        if file_path not in self.model_to_indx.keys():
            points_dict = np.load(file_path)

            if self.all_data == None:
                self.all_data = {
                'points_sampled': self.create_mmap_large('points_sampled', points_dict['point_sampled'].shape),
                'points': self.create_mmap_large('points', points_dict['point_cloud'].shape),
                'points_surf': self.create_mmap_large('points', points_dict['point_cloud'].shape),
                'nn': self.create_mmap_large('nn', points_dict['nP'].shape),
                'udf': self.create_mmap_large('udf', points_dict['sDF'].shape),
                'nvf': self.create_mmap_large('nvf', points_dict['nF'].shape),
                'normals': self.create_mmap_large('normals', points_dict['normals'].shape),
                'index_points': np.zeros(6000, dtype=np.int),
                'index_points_surface': np.zeros(6000, dtype=np.int),
                'idx_surf': np.zeros(6000, dtype=np.int),
                'max_points': points_dict['point_sampled'].shape[0],
                'max_points_surface': points_dict['point_cloud'].shape[0]
            }

                # for key in self.all_data.keys():
                #     self.index_sample_model[key] = 0 
            self.model_to_indx[file_path] = self.ct 
            self.all_data['points_sampled'][self.ct, :] = points_dict['point_sampled'].astype(float)[:]
            self.all_data['points'][self.ct, :] = points_dict['point_cloud'].astype(float)[:]
            self.all_data['points_surf'][self.ct, :] = points_dict['point_cloud'].astype(float)[:]
            self.all_data['nn'][self.ct, :] = points_dict['nP'].astype(float)[:, [0, 2, 1]]
            self.all_data['udf'][self.ct, :] = np.abs(points_dict['sDF'].astype(float)[:])
            self.all_data['nvf'][self.ct, :] = points_dict['nF'].astype(float)[:]
            self.all_data['normals'][self.ct, :] = points_dict['normals'].astype(float)[:]
            
            ch = np.arange(self.all_data['max_points'])
            np.random.shuffle(ch)
            self.all_data['points_sampled'][self.ct] = self.all_data['points_sampled'][self.ct][ch]
            self.all_data['udf'][self.ct] = self.all_data['udf'][self.ct][ch]
            self.all_data['nvf'][self.ct] = self.all_data['nvf'][self.ct][ch]
            self.all_data['nn'][self.ct] = self.all_data['nn'][self.ct][ch]
            self.all_data['index_points'][self.ct] = 0

            ch = np.arange(self.all_data['max_points_surface'])
            np.random.shuffle(ch)
            self.all_data['points'][self.ct] = self.all_data['points'][self.ct][ch]
            self.all_data['points_surf'][self.ct] = self.all_data['points_surf'][self.ct][ch]
            self.all_data['normals'][self.ct] = self.all_data['normals'][self.ct][ch]
            self.all_data['index_points_surface'][self.ct] = 0
            self.all_data['idx_surf'][self.ct] = 0
            
            self.ct += 1
        
        if no_return:
            print("no return")
            return []
        
        data = {}
        batch_size = self.transform.N
        batch_size_pc = self.transform.N_pc
        ct_idx = self.model_to_indx[file_path]
        data['points_sampled'] = self.all_data['points_sampled'][ct_idx]
        data['points'] = self.all_data['points'][ct_idx]
        data['points_surf'] = self.all_data['points_surf'][ct_idx]
        data['udf'] = self.all_data['udf'][ct_idx]
        data['nvf'] = self.all_data['nvf'][ct_idx]
        data['nn'] = self.all_data['nn'][ct_idx]
        data['normals'] = self.all_data['normals'][ct_idx]
        index_to_sample_points = self.all_data['index_points'][ct_idx]
        index_to_sample_point_cloud = self.all_data['index_points_surface'][ct_idx]
        idx_surf = self.all_data['idx_surf'][ct_idx]
        # print("type conversions", time.time() - t)

        if self.with_transforms:
            data['loc'] = points_dict['loc'].astype(np.float32)
            data['scale'] = points_dict['scale'].astype(np.float32)

        if self.transform is not None:
            # t = time.time()
            data = self.transform(data, index_to_sample_points, index_to_sample_point_cloud, idx_surf)
            # print("sampling time", time.time() - t)
        if self.ndf:
            data['points'] = ndf_utils.get_occupancy_from_pc(data['points'], res=self.res)
#             print("shp", data['points'].shape)
        
        self.all_data['index_points'][ct_idx] += batch_size
        self.all_data['index_points_surface'][ct_idx] += batch_size_pc
        self.all_data['idx_surf'][ct_idx] += batch_size

        if (self.all_data['index_points'][ct_idx]  > self.all_data['max_points'] - batch_size) or \
            (self.all_data['idx_surf'][ct_idx]  > self.all_data['max_points_surface'] - batch_size):
#             self.cleanup()
#             print("epoch incr +++", self.all_data['max_points'], len(self.all_data['index_points']))
            self.epoch += 1
            
            ch = np.arange(self.all_data['max_points'])
            np.random.shuffle(ch)
            self.all_data['points_sampled'][ct_idx] = self.all_data['points_sampled'][ct_idx][ch]
            self.all_data['udf'][ct_idx] = self.all_data['udf'][ct_idx][ch]
            self.all_data['nn'][ct_idx] = self.all_data['nn'][ct_idx][ch]
            self.all_data['nvf'][ct_idx] = self.all_data['nvf'][ct_idx][ch]
            self.all_data['index_points'][ct_idx] = 0
            
            ch = np.arange(self.all_data['max_points_surface'])
            np.random.shuffle(ch)
            self.all_data['normals'][ct_idx] = self.all_data['normals'][ct_idx][ch]
            self.all_data['points_surf'][ct_idx] = self.all_data['points_surf'][ct_idx][ch]
            self.all_data['idx_surf'][ct_idx] = 0

        if self.all_data['index_points_surface'][ct_idx] > self.all_data['max_points_surface'] - batch_size_pc:
            ch = np.arange(self.all_data['max_points_surface'])
            np.random.shuffle(ch)
            self.all_data['points'][ct_idx] = self.all_data['points'][ct_idx][ch]
            self.all_data['index_points_surface'][ct_idx] = 0

        # print("**", data['udf'].shape, data['nvf'].shape)

        return data


class VoxelsField(Field):
    ''' Voxel field class.

    It provides the class used for voxel-based data.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
    '''
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        with open(file_path, 'rb') as f:
            voxels = binvox_rw.read_as_3d_array(f)
        voxels = voxels.data.astype(np.float32)

        if self.transform is not None:
            voxels = self.transform(voxels)

        return voxels

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


class PointCloudField(Field):
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        with_transforms (bool): whether scaling and rotation dat should be
            provided
    '''
    def __init__(self, file_name, transform=None, with_transforms=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms
        self.dict_ram = {}
    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        global saved_npz
        file_path = os.path.join(model_path, self.file_name)

        if file_path not in self.dict_ram.keys():
            pointcloud_dict = np.load(file_path)
            
            points = pointcloud_dict['points'].astype(np.float32)
            normals = pointcloud_dict['normals'].astype(np.float32)

            data = {
                None: points,
                'normals': normals,
            }

            self.dict_ram[file_path] = data

        data = self.dict_ram[file_path]

        # pointcloud_dict = np.load(file_path)

        

        if self.with_transforms:
            data['loc'] = pointcloud_dict['loc'].astype(np.float32)
            data['scale'] = pointcloud_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        ''' Check if field is complete. 
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


# NOTE: this will produce variable length output.
# You need to specify collate_fn to make it work with a data laoder
class MeshField(Field):
    ''' Mesh field.

    It provides the field used for mesh data. Note that, depending on the
    dataset, it produces variable length output, so that you need to specify
    collate_fn to make it work with a data loader.

    Args:
        file_name (str): file name
        transform (list): list of transforms applied to data points
    '''
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        mesh = trimesh.load(file_path, process=False)
        if self.transform is not None:
            mesh = self.transform(mesh)

        data = {
            'verts': mesh.vertices,
            'faces': mesh.faces,
        }

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete
