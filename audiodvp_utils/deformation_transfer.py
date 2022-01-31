import scipy.io as sio
import torch
import numpy as np
import scipy.sparse as sparse
from tqdm import tqdm

class transformation():
    def __init__(self, opt, src_alpha, tgt_alpha):
        self.opt = opt
        self.mat_data = sio.loadmat(opt.matlab_data_path)
        self.geo_mean = self.mat_data['geo_mean'].astype(np.float64)
        self.id_base = self.mat_data['id_base'].astype(np.float64)
        self.exp_base = self.mat_data['exp_base'].astype(np.float64)
        self.triangles64 = self.mat_data['triangles']
        self.src_alpha = src_alpha.clone().detach().numpy().astype(np.float64)
        self.tgt_alpha = tgt_alpha.clone().detach().numpy().astype(np.float64)
        
        self.src_geometry = (self.geo_mean + self.id_base @ self.src_alpha).reshape(-1, 3)
        self.tgt_geometry = (self.geo_mean + self.id_base @ self.tgt_alpha).reshape(-1, 3)
    
        self.src_span = self.get_span(self.src_geometry)
        self.tgt_span = self.get_span(self.tgt_geometry)

        self.src_inv_span = np.linalg.inv(self.src_span)
        self.tgt_inv_span = np.linalg.inv(self.tgt_span)
        
        self.V_hat = self.get_transform_matrix()
        self.A = self.get_A()
        self.LU = sparse.linalg.splu((self.A.T @ self.A).tocsc())
    
    
    def span_components(self, geometry):
        triangles64 = self.triangles64

        v1 = geometry[triangles64[:, 0], :]
        v2 = geometry[triangles64[:, 1], :]
        v3 = geometry[triangles64[:, 2], :]

        a = v2 - v1
        b = v3 - v1
        tmp = np.cross(a, b)
        c = (tmp.T / np.sqrt(np.linalg.norm(tmp, axis=1))).T
        
        return a, b, c
    
    
    def v1(self):
        return self.tgt_geometry[self.triangles64[:, 0], :]
    
    
    def get_span(self, geometry):
        a, b, c = self.span_components(geometry)
        return np.transpose((a, b, c), (1, 2, 0))
    
    
    def get_transform_matrix(self):
        row = np.array([0, 1] * 3)
        data = np.array([-1, -1, 1, 0, 0, 1])
        vertices = self.tgt_geometry
        triangles = self.triangles64
        sparse_matrix = []
        
        for f in tqdm(triangles, total=len(triangles)):
            i0, i1, i2 = f
            col = np.array([i0, i0, i1, i1, i2, i2])
            sparse_matrix.append(sparse.coo_matrix((data, (row, col)), shape=(2, len(vertices)), dtype=np.float64))       
        return sparse.vstack(sparse_matrix, dtype=np.float64).tocsc()
    
    def get_A(self):
        V_hat = self.V_hat
        exp_base = self.exp_base.reshape(35709, 3*64)
        A = V_hat.dot(exp_base).reshape(-1, 64)
        return sparse.csr_matrix(A)
        
    
    def deformation_transfer(self, src_delta):
        src_pose = self.src_geometry + (self.exp_base @ src_delta.numpy().astype(np.float64)).reshape(-1, 3)
        self.src_pose = src_pose
        src_pose_span = self.get_span(src_pose)
        s = ((src_pose_span @ self.src_inv_span) @ self.tgt_span[:, :, :2]).transpose(0, 2, 1)
        b = (np.concatenate(s) - self.V_hat.dot(self.tgt_geometry)).reshape(-1, 1)
        x = self.LU.solve(self.A.T @ b)
        return torch.from_numpy(x.astype(np.float32))
    
    def to_fourth_dimension(self):
        a, b, c = self.span_components(self.tgt_geometry)
        v4 = self.v1() + c
        new_vertices = np.concatenate((self.tgt_geometry, v4), axis=0)
        v4_indices = np.arange(len(self.tgt_geometry), len(self.tgt_geometry) + c.shape[0])
        new_triangles = np.concatenate((self.triangles64, v4_indices.reshape(-1, 1)), axis=1)
        
        return new_vertices, new_triangles