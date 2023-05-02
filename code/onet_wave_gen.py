import matplotlib.pyplot as plt
import mesh
from mesh import Mesh
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from tqdm import tqdm
import torch
import pyamg

Nx, Ny = 64, 64
N = Nx * Ny

x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
xx, yy = np.meshgrid(x, y)
xy = np.column_stack((xx.flatten(), yy.flatten()))

boundary_nodes = np.logical_or(
    np.logical_or(xy[:,0] == 0, xy[:,0] == 1), # x==0 or x==1
    np.logical_or(xy[:,1] == 0, xy[:,1] == 1)  # y==0 or y==1
)
interior_nodes = np.logical_not(boundary_nodes)
R = (sp.eye(N).tocsr())[interior_nodes] # restriction to interior nodes

dx = x[1] - x[0]
dy = y[1] - y[0]

Ax = (sp.eye(Nx) * -2 + sp.eye(Nx, k=-1) + sp.eye(Nx, k=1)) / (dx**2)
Ay = (sp.eye(Ny) * -2 + sp.eye(Ny, k=-1) + sp.eye(Ny, k=1)) / (dy**2)
A = sp.kron(Ax, sp.eye(Ny)) + sp.kron(sp.eye(Nx), Ay)

# Derivation of timestepper:
# d^2u / dt^2 = c^2 \nabla^2 u                               (wave equation PDE formulation)
# d^2u / dt^2 = c^2 Au                                       (central fd in space w/ matrix operator A)
# => (1/dt^2) (u^{i-1} - 2^{i} + u^{i+1}) = c^2 A u^{i+1}    (central fd in time)
# => (u^{i+1}) = c^2 dt^2 A u^{i+1} - u^{i-1} + 2^{i}        (rearrange)
# => u^{i+1} - c^2 dt^2 A u^{i+1} = - u^{i-1} + 2^{i}        (rearrange)
# => (I - c^2 dt^2 A) u^{i+1} = - u^{i-1} + 2u^{i}           (gather left terms)

dt = 0.01
c = 0.2

LHS = sp.eye(N) - (c**2)*(dt**2) * A
ml = pyamg.smoothed_aggregation_solver(R @ LHS @ R.T)

def wave_run():
    u0 = np.zeros(A.shape[0])
    us = [u0, u0]

    i = 0
    plt_step = 20

    f_x = np.random.randint(1, Nx - 1)
    f_y = np.random.randint(1, Ny - 1)

    cutoff_time = np.random.uniform(0.0, 10.0)
    start_idx = -1

    while True:
        t = i * dt
        u = R.T @ ml.solve(R @ (-us[-2] - 2 * -us[-1]), x0=R@us[-1], tol=1e-3)

        if t < cutoff_time:
            u[f_y * Nx + f_x] = np.sin(t * np.pi) * np.exp(-t * 0.001 - 1)
        elif start_idx == -1:
            start_idx = i

        if t > cutoff_time + 15:
            break

        us.append(u)
        i += 1

    return np.expand_dims(np.array(us[start_idx::10]).reshape((-1, Nx, Ny)), 0)

N_training_runs = 25
N_testing_runs = 10

training = []
testing = []

for i in tqdm(range(N_training_runs)):
    wr = wave_run()
    training.append(wr)
training = np.concatenate(training, axis=0)

for i in tqdm(range(N_testing_runs)):
    wr = wave_run()
    testing.append(wr)
testing = np.concatenate(testing, axis=0)

print(training.shape)
print(testing.shape)

torch.save(torch.Tensor(training).float(), '../data/onet_wave_train.pt')
torch.save(torch.Tensor(testing).float(), '../data/onet_wave_test.pt')
