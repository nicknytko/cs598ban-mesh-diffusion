import matplotlib.pyplot as plt
import mesh
from mesh import Mesh
import numpy as np
import torch

class StructuredMeshDataset(torch.utils.data.Dataset):
    def __init__(self,
                 parameters_file='../data/structured_parameters.pt',
                 data_file='../data/structured_data.pt'):
        self.params = torch.load(parameters_file)
        self.data = torch.load(data_file)
        self.vertices = mesh.tensor_extract_coordinates(self.params, self.data)
        self.nonzeros = mesh.tensor_extract_nonzeros(self.params, self.data)

    def __len__(self):
        return len(self.data.shape[0])

    def __getitem__(self, idx):
        return (self.vertices[idx], self.nonzeros[idx])


data = StructuredMeshDataset()

# Grab some data points...
coordinates, matrix = data[0:2]
print(coordinates.shape, matrix.shape)

# Add noise
coordinates = coordinates + torch.randn_like(coordinates)
matrix = matrix + torch.randn_like(matrix)
