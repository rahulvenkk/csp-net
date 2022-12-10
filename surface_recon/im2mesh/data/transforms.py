import numpy as np
import time

# Transforms
class PointcloudNoise(object):
    ''' Point cloud noise transformation class.

    It adds noise to point cloud data.

    Args:
        stddev (int): standard deviation
    '''

    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        noise = self.stddev * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        data_out[None] = points + noise
        return data_out


class SubsamplePointcloud(object):
    ''' Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        normals = data['normals']

        indices = np.random.randint(points.shape[0], size=self.N)
        data_out[None] = points[indices, :]
        data_out['normals'] = normals[indices, :]

        return data_out


class SubsamplePoints(object):
    ''' Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        points = data[None]
        occ = data['occ']

        data_out = data.copy()
        if isinstance(self.N, int):
            idx = np.random.randint(points.shape[0], size=self.N)
            data_out.update({
                None: points[idx, :],
                'occ':  occ[idx],
            })
        else:
            Nt_out, Nt_in = self.N
            occ_binary = (occ >= 0.5)
            points0 = points[~occ_binary]
            points1 = points[occ_binary]

            idx0 = np.random.randint(points0.shape[0], size=Nt_out)
            idx1 = np.random.randint(points1.shape[0], size=Nt_in)

            points0 = points0[idx0, :]
            points1 = points1[idx1, :]
            points = np.concatenate([points0, points1], axis=0)

            occ0 = np.zeros(Nt_out, dtype=np.float32)
            occ1 = np.ones(Nt_in, dtype=np.float32)
            occ = np.concatenate([occ0, occ1], axis=0)

            volume = occ_binary.sum() / len(occ_binary)
            volume = volume.astype(np.float32)

            data_out.update({
                None: points,
                'occ': occ,
                'volume': volume,
            })
        return data_out



class SubsamplePointsUDF(object):
    ''' Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N, N_pc):
        self.N = N
        self.N_pc = N_pc

    def __call__(self, data, index_to_sample, index_to_sample_surface, idx_surf):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        points = data['points_sampled']
        normals_on_surface = data['normals']
        points_on_surface = data['points']
        points_surf = data['points_surf']
        udf = data['udf']
        nvf = data['nvf']
        nn = data['nn']

        data_out = {}#data#.copy()

        #Randomly flip normals
        idx = np.random.randint(points.shape[0], size=self.N)

        #Normals
        data_out.update({
        'points_sampled': np.array(points[index_to_sample:index_to_sample+self.N, :]),
        'nn':  np.array(nn[index_to_sample:index_to_sample+self.N, :]),
        'udf':  np.array(udf[index_to_sample:index_to_sample+self.N]),
        'nvf': np.array(nvf[index_to_sample:index_to_sample+self.N, :]),
        'points': np.array(points_on_surface[index_to_sample_surface:index_to_sample_surface+self.N_pc, :]),
        'points_surf': np.array(points_surf[idx_surf:idx_surf+self.N, :]),    
        'normals': np.array(normals_on_surface[idx_surf:idx_surf+self.N, :])
        })

        return data_out
