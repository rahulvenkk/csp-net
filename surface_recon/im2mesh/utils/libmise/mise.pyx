# distutils: language = c++
cimport cython
from cython.operator cimport dereference as dref
from libcpp.vector cimport vector
from libcpp.map cimport map
from libc.math cimport isnan, NAN
import numpy as np


cdef struct Vector3D:
    int x, y, z


cdef struct Voxel:
    Vector3D loc
    unsigned int level
    bint is_leaf
    bint is_active
    unsigned long children[2][2][2]


cdef struct GridPoint:
    Vector3D loc
    double value
    bint known


cdef inline unsigned long vec_to_idx(Vector3D coord, long resolution):
    cdef unsigned long idx
    idx = resolution * resolution * coord.x + resolution * coord.y + coord.z
    return idx


cdef class MISE:
    cdef vector[Voxel] voxels
    cdef vector[GridPoint] grid_points
    cdef map[long, long] grid_point_hash
    cdef readonly int resolution_0
    cdef readonly int depth
    cdef readonly int voxel_size_0
    cdef readonly int resolution

    def __cinit__(self, int resolution_0, int depth):
        self.resolution_0 = resolution_0
        self.depth = depth
        self.voxel_size_0 = (1 << depth)
        self.resolution = resolution_0 * self.voxel_size_0

        # Create initial voxels
        self.voxels.reserve(resolution_0 * resolution_0 * resolution_0)
        
        cdef Voxel voxel
        cdef GridPoint point
        cdef Vector3D loc
        cdef int i, j, k
        for i in range(resolution_0):
            for j in range(resolution_0): 
                for  k in range (resolution_0):
                    loc = Vector3D(
                        i * self.voxel_size_0, 
                        j * self.voxel_size_0,
                        k * self.voxel_size_0,
                    )
                    voxel = Voxel(
                        loc=loc,
                        level=0,
                        is_leaf=True,
                        is_active=False
                    )
                    
                    assert(self.voxels.size() == vec_to_idx(Vector3D(i, j, k), resolution_0))
                    self.voxels.push_back(voxel)

        # Create initial grid points
        self.grid_points.reserve((resolution_0 + 1) * (resolution_0 + 1) * (resolution_0 + 1))
        for i in range(resolution_0 + 1):
            for j in range(resolution_0 + 1):
                for k in range(resolution_0 + 1):
                    loc = Vector3D(
                        i * self.voxel_size_0, 
                        j * self.voxel_size_0,
                        k * self.voxel_size_0,
                    )
                    assert(self.grid_points.size() == vec_to_idx(Vector3D(i, j, k), resolution_0 + 1))
                    self.add_grid_point(loc)

    

    def query(self):
        """Query points to evaluate."""
        # Find all points with unknown value
        cdef vector[Vector3D] points
        cdef int n_unknown = 0
        for p in self.grid_points:
            if not p.known:
                n_unknown += 1 

        points.reserve(n_unknown)
        for p in self.grid_points:
            if not p.known:
                points.push_back(p.loc)

        # Convert to numpy
        points_np = np.zeros((points.size(), 3), dtype=np.int64)
        cdef long[:, :] points_view = points_np
        for i in range(points.size()):
            points_view[i, 0] = points[i].x
            points_view[i, 1] = points[i].y
            points_view[i, 2] = points[i].z

        # return voxels points
        voxels_np = np.zeros((self.voxels.size(),3))
        for i in range(self.voxels.size()):
            voxels_np[i,0] = self.voxels[i].loc.x
            voxels_np[i,1] = self.voxels[i].loc.y
            voxels_np[i,2] = self.voxels[i].loc.z
        
        return points_np, voxels_np



    def update_udf(self, long[:, :] points, double[:] values):
        """Update points and set their values. Also determine all active voxels and subdivide them."""
        assert(points.shape[0] == values.shape[0])
        assert(points.shape[1] == 3)
        cdef Vector3D loc
        cdef long idx
        cdef int i

        # Find all indices of point and set value
        for i in range(points.shape[0]):
            loc = Vector3D(points[i, 0], points[i, 1], points[i, 2])
            idx = self.get_grid_point_idx(loc)
            if idx == -1:
                raise ValueError('Point not in grid!')
            # save UDF value of each grid point
            self.grid_points[idx].value = values[i]
            self.grid_points[idx].known = True
        
        # Subdivide activate voxels and add new points
        self.subdivide_voxels()
    
    cdef void subdivide_voxels(self) except +:
        '''Function to go over all the voxels in the grid and check which 
            of the voxels need to be sudivided.
        '''
        cdef int n_subdivide = 0
        cdef int curr_voxel_size
        cdef Vector3D curr_voxel_location 
        cdef Vector3D adj_loc
        cdef int point_idx
        cdef double curr_point_value
        cdef bint flag=False
        
        # first iteration is over all the voxels to count the #subdivions and #new grid point
        for idx in range(self.voxels.size()):
            if not self.voxels[idx].is_leaf or self.voxels[idx].level == self.depth:
                continue

            curr_voxel_size = 1 << (self.depth - self.voxels[idx].level)
            curr_voxel_location = self.voxels[idx].loc
            flag=False
            for i in range(0,2):
                for j in range(0,2):
                    for k in range(0,2):
                        adj_loc = Vector3D(
                            curr_voxel_location.x + i*curr_voxel_size, 
                            curr_voxel_location.y + j*curr_voxel_size,
                            curr_voxel_location.z + k*curr_voxel_size,
                        )
                        point_idx = self.get_grid_point_idx(adj_loc)
                        if point_idx == -1:
                            continue
                        if not self.grid_points[point_idx].known:
                            continue 
                        
                        curr_point_value = self.grid_points[point_idx].value
                        if curr_point_value <= curr_voxel_size:
                            n_subdivide+=1
                            flag =True
                            self.voxels[idx].is_active=True
                            break
                    if flag is True:
                        break          
                if flag is True:
                    break

        # reserve space for new voxels and grid points
        self.voxels.reserve(self.voxels.size() + 8 * n_subdivide)
        self.grid_points.reserve(self.voxels.size() + 19 * n_subdivide)
        flag = False
        
        # iterate over the voxels again to call subdivision
        for idx in range(self.voxels.size()):
            if not self.voxels[idx].is_leaf or self.voxels[idx].level == self.depth:
                continue
            # subdivide the voxel
            if self.voxels[idx].is_active is True:
                self.subdivide_voxel(idx)



    cdef void subdivide_voxel(self, long idx):
        '''Function to subdivide a voxel.
        Args:
            idx : index of the voxel.
        
        '''
        cdef Voxel voxel
        cdef GridPoint point
        cdef Vector3D loc0 = self.voxels[idx].loc
        cdef Vector3D loc
        cdef int new_level = self.voxels[idx].level + 1
        cdef int new_size = 1 << (self.depth - new_level)
        assert(new_level <= self.depth)
        assert(1 <= new_size <= self.voxel_size_0)

        # Current voxel is not leaf anymore
        self.voxels[idx].is_leaf = False
        # Add new voxels        
        cdef int i, j, k
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    loc = Vector3D(
                        x=loc0.x + i * new_size,
                        y=loc0.y + j * new_size,
                        z=loc0.z + k * new_size,
                    )
                    voxel = Voxel(
                        loc=loc, 
                        level=new_level,
                        is_leaf=True
                    )

                    self.voxels[idx].children[i][j][k] = self.voxels.size()
                    self.voxels.push_back(voxel)

        # Add new grid points
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    loc = Vector3D(
                        loc0.x + i * new_size,
                        loc0.y + j * new_size,
                        loc0.z + k * new_size,
                    )

                    # Only add new grid points
                    if self.get_grid_point_idx(loc) == -1:
                        self.add_grid_point(loc)
    


    def to_dense(self):
        """Output dense matrix at highest resolution."""
        out_array = np.full((self.resolution + 1,) * 3, np.nan)
        cdef double[:, :, :] out_view = out_array
        cdef GridPoint point
        cdef int i, j, k
        
        for point in self.grid_points:
            # Take voxel for which points is upper left corner
            # assert(point.known)
            out_view[point.loc.x, point.loc.y, point.loc.z] = point.value

        # Complete along x axis
        for i in range(1, self.resolution + 1):
            for j in range(self.resolution + 1):
                for k in range(self.resolution + 1):
                    if isnan(out_view[i, j, k]):
                        out_view[i, j, k] = out_view[i-1, j, k]

        # Complete along y axis
        for i in range(self.resolution + 1):
            for j in range(1, self.resolution + 1):
                for k in range(self.resolution + 1):
                    if isnan(out_view[i, j, k]):
                        out_view[i, j, k] = out_view[i, j-1, k]


        # Complete along z axis
        for i in range(self.resolution + 1):
            for j in range(self.resolution + 1):
                for k in range(1, self.resolution + 1):
                    if isnan(out_view[i, j, k]):
                        out_view[i, j, k] = out_view[i, j, k-1]
                    assert(not isnan(out_view[i, j, k]))
        return out_array



    def get_points(self):
        ''' Return the grid points and their values.
        
        '''
        points_np = np.zeros((self.grid_points.size(), 3), dtype=np.int64)
        values_np = np.zeros((self.grid_points.size()), dtype=np.float64)

        cdef long[:, :] points_view = points_np
        cdef double[:] values_view = values_np
        cdef Vector3D loc
        cdef int i

        for i in range(self.grid_points.size()):
            loc = self.grid_points[i].loc
            points_view[i, 0] = loc.x
            points_view[i, 1] = loc.y
            points_view[i, 2] = loc.z
            values_view[i] = self.grid_points[i].value

        return points_np, values_np



    @cython.cdivision(True) 
    cdef long get_voxel_idx(self, Vector3D loc) except +:
        """Utility function for getting voxel index corresponding to 3D coordinates."""
        # Shorthands
        cdef long resolution = self.resolution
        cdef long resolution_0 = self.resolution_0
        cdef long depth = self.depth
        cdef long voxel_size_0 = self.voxel_size_0

        # Return -1 if point lies outside bounds
        if not (0 <= loc.x < resolution and 0<= loc.y < resolution and 0 <= loc.z < resolution):
            return -1
        
        # Coordinates in coarse voxel grid
        cdef Vector3D loc0 = Vector3D(
            x=loc.x >> depth,
            y=loc.y >> depth,
            z=loc.z >> depth,
        )       

        # Initial voxels
        cdef int idx = vec_to_idx(loc0, resolution_0)
        cdef Voxel voxel = self.voxels[idx]
        assert(voxel.loc.x == loc0.x * voxel_size_0)
        assert(voxel.loc.y == loc0.y * voxel_size_0)
        assert(voxel.loc.z == loc0.z * voxel_size_0)

        # Relative coordinates
        cdef Vector3D loc_rel = Vector3D(
            x=loc.x - (loc0.x << depth),
            y=loc.y - (loc0.y << depth),
            z=loc.z - (loc0.z << depth),
        ) 

        cdef Vector3D loc_offset
        cdef long voxel_size = voxel_size_0

        while not voxel.is_leaf:
            voxel_size = voxel_size >> 1
            assert(voxel_size >= 1)

            # Determine child
            loc_offset = Vector3D(
                x=1 if (loc_rel.x >= voxel_size) else 0,
                y=1 if (loc_rel.y >= voxel_size) else 0,
                z=1 if (loc_rel.z >= voxel_size) else 0,
            )
            # New voxel
            idx = voxel.children[loc_offset.x][loc_offset.y][loc_offset.z]
            voxel = self.voxels[idx]

            # New relative coordinates
            loc_rel = Vector3D(
                x=loc_rel.x - loc_offset.x * voxel_size,
                y=loc_rel.y - loc_offset.y * voxel_size,
                z=loc_rel.z - loc_offset.z * voxel_size,
            ) 

            assert(0<= loc_rel.x < voxel_size)
            assert(0<= loc_rel.y < voxel_size)
            assert(0<= loc_rel.z < voxel_size)


        # Return idx
        return idx



    cdef inline void add_grid_point(self, Vector3D loc):
        cdef GridPoint point = GridPoint(
            loc=loc,
            value=0.,
            known=False,
        )
        self.grid_point_hash[vec_to_idx(loc, self.resolution + 1)] = self.grid_points.size()
        self.grid_points.push_back(point)



    cdef inline int get_grid_point_idx(self, Vector3D loc):
        p_idx = self.grid_point_hash.find(vec_to_idx(loc, self.resolution + 1))
        if p_idx == self.grid_point_hash.end():
            return -1

        cdef int idx = dref(p_idx).second
        assert(self.grid_points[idx].loc.x == loc.x)
        assert(self.grid_points[idx].loc.y == loc.y)
        assert(self.grid_points[idx].loc.z == loc.z)

        return idx