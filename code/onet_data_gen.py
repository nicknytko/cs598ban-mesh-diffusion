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
Nx = 64
Ny = 64
num_training_mesh = 1_000
num_testing_mesh  = 300
num_mesh = num_training_mesh + num_testing_mesh

# Get reference parameters
structured = Mesh.create_structured(Nx, Ny)
Nv = structured.verts.shape[0]
sensor_pts = structured.verts

data_k = torch.empty((num_mesh, Nx, Ny))
data_utrue = torch.empty((num_mesh, Nx, Ny))

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
        return np.abs(simplex_noise(X[:,0], X[:,1]))*5. + 1e-1 # we want this strictly positive

    M = Mesh.create_structured(Nx, Ny, kappa=kappa, f=lambda x, y: x**0)

    k_sensor = torch.Tensor(kappa(M.verts[:,0], M.verts[:,1])).reshape((Nx, Ny))
    eval_u = M.calc_num_soln().reshape((Nx, Ny))

    data_k[i] = k_sensor
    data_utrue[i] = eval_u

    if i == 0:
        plt.figure()
        plt.imshow(eval_u)

        plt.figure()
        plt.imshow(k_sensor)

        plt.show(block=True)


torch.save((data_k[:num_training_mesh], data_utrue[:num_training_mesh]), '../data/onet_train.pt')
torch.save((data_k[num_training_mesh:], data_utrue[num_training_mesh:]), '../data/onet_test.pt')
