import matplotlib.pyplot as plt
import mesh
from mesh import Mesh
import numpy as np
import torch

# Parameters
Nx = 7
Ny = 7
params = torch.load('../data/structured_parameters.pt')
num_verts, num_nonzeros, N_d = params
data = torch.load('../data/structured_data.pt')

print('Loaded example structured data')
print(data.shape)

# Grab the coordinates and matrix entries from the data
# here are two getters that grab the coordinate parts and matrix entries separately
coordinates = mesh.tensor_extract_coordinates(params, data)
nonzeros = mesh.tensor_extract_nonzeros(params, data)

print()
print('coordinates', coordinates.shape)
print('nonzeros', nonzeros.shape)

# Perturb the coordinates and matrix entries separately, putting them into new_data
# here are two setters to update the full data points with the new coordinates and matrix entries
new_coordinates = coordinates + torch.randn_like(coordinates)
new_nonzeros = nonzeros + torch.randn_like(nonzeros)
new_data = data.clone()
mesh.tensor_set_coordinates(params, new_data, new_coordinates)
mesh.tensor_set_nonzeros(params, new_data, new_nonzeros)

print()
print('New perturbed datapoints')
print(new_data.shape)
