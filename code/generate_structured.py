import matplotlib.pyplot as plt
import mesh
from mesh import Mesh
import numpy as np
import pickle
from tqdm import tqdm
import torch
import opensimplex

simplex_noise = np.vectorize(opensimplex.noise2)

# Parameters
Nx = 7
Ny = 7
num_mesh = 10_000

# Get reference parameters
structured = Mesh.create_structured(Nx, Ny)
num_verts = structured.verts.shape[0]
num_nonzeros = structured.nnz

N_d = num_verts * 2 + num_nonzeros
data = torch.empty(num_mesh, N_d)

# Random anisotropic meshes
for i in tqdm(range(num_mesh)):
    # generate a random linear transform to the input coordinates of opensimplex
    # this consists of a rotation, scaling, and another rotation
    theta = np.random.uniform(0, 2 * np.pi)
    Q = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    S = np.diag(np.random.randn(2) * 3.)

    def kappa(x, y):
        X = np.column_stack((x, y))
        X = ((Q.T @ S @ Q) @ X.T).T
        return np.abs(simplex_noise(X[:,0], X[:,1]))*2. + 1e-1 # we want this strictly positive

    M = Mesh.create_structured(Nx, Ny, kappa=kappa)
    # print(M.nonzeros, torch.sum(M.nonzeros != 0), torch.sum(M.A != 0))
    #print('verts', M.verts.shape[0], 'nnz', M.nnz)
    data[i] = M.to_tensor()

torch.save((num_verts, num_nonzeros, N_d), '../data/structured_parameters.pt')
torch.save(data, '../data/structured_data.pt')
