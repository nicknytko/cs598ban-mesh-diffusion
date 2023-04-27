import matplotlib.pyplot as plt
import mesh
from mesh import Mesh
import numpy as np
import torch
from torch_geometric.nn.unpool import knn_interpolate
import torch.linalg as tla


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


def interp_loss_penalty(dataset,
                        unstruct_coord, unstruct_mat,
                        struct_coord, struct_mat):
    '''
    Introduces a loss penalty based on closeness of
    interpolating PDE solutions between the "unstructured" and "structured" meshes.

    Parameters
    ----------
    unstruct_coord : torch.Tensor
      B x N x 2 tensor containing the coordinates of the unstructured mesh
    unstruct_mat : torch.Tensor
      B x (L nnz) tensor containing the nonzeros of the lower half of the unstructured linear operators
    struct_coord : torch.Tensor
      B x N x 2 tensor containing the coordinates of the structured mesh
    struct_mat : torch.Tensor
      B x (L nnz) tensor containing the nonzeros of the lower half of the structured linear operators

    Returns
    -------
    penalty : torch.Tensor
      Scalar torch value containing the mean loss penalty over the batch
    '''

    N = unstruct_coord.shape[1] # Number of vertices
    n = int(np.round(N ** 0.5)) # Number of vertices along one dimension (assuming we are on a square-like domain)
    B = unstruct_coord.shape[0] # Batch size

    dummy_mesh = Mesh.create_structured_dummy(n, n) # we just need the mask and boundary nodes
    batch_idx = torch.arange(B).repeat_interleave(N) # batch indices for the interpolation
    bdy = dummy_mesh.boundary_verts # boundary nodes that we will add back into the matrices

    mask_r, mask_c = dummy_mesh.mask

    # Reconstruct matrix operators
    M_unstruct = torch.zeros(B, N, N)
    M_unstruct[:, mask_r, mask_c] = unstruct_mat # lower triangular part
    M_unstruct = M_unstruct + torch.tril(M_unstruct, diagonal=-1).transpose(1, 2) # copy lower to upper
    M_unstruct[:, bdy, bdy] = 1. # reintroduce boundary

    M_struct = torch.zeros(B, N, N)
    M_struct[:, mask_r, mask_c] = struct_mat # lower triangular part
    M_struct = M_struct + torch.tril(M_struct, diagonal=-1).transpose(1, 2) # copy lower to upper
    M_struct[:, bdy, bdy] = 1. # reintroduce boundary

    # We'll use random samples for the PDE forcing term (right-hand-side)
    f_U = torch.randn(B, N)
    f_S = knn_interpolate(f_U.reshape(B * N, 1),
                          unstruct_coord.reshape(B * N, 2),
                          struct_coord.reshape(B * N, 2),
                          batch_idx, batch_idx).reshape((B, N))

    # Matrix solve
    u_U = tla.solve(M_unstruct, f_U)
    u_S = tla.solve(M_struct,   f_S)

    # Interpolate structured solution to unstructured domain
    u_SonU = knn_interpolate(u_S.reshape(B * N, 1),
                             struct_coord.reshape(B * N, 2),
                             unstruct_coord.reshape(B * N, 2),
                             batch_idx, batch_idx).reshape((B, N))

    # Compute mean squared relative error between solutions
    err = u_SonU - u_U
    return (torch.einsum('ij,ij->i', err, err) / (tla.norm(u_U, dim=1) ** 2)).mean()


if __name__ == '__main__':
    data = StructuredMeshDataset()

    # Grab some data points...
    coordinates, matrix = data[0:20]

    # Pretending we are obtaining denoised data
    alpha = torch.tensor(1.0, requires_grad=True)
    beta = torch.tensor(1.0, requires_grad=True)

    optim = torch.optim.Adam([alpha, beta], lr=0.1)
    i = 0
    while (abs(beta) > 0.1):
        optim.zero_grad()
        new_coordinates = coordinates + torch.randn_like(coordinates) * alpha
        new_matrix = matrix + torch.randn_like(matrix) * beta

        loss = interp_loss_penalty(data, coordinates, matrix, new_coordinates, new_matrix)
        loss.backward()
        optim.step()

        print(f'{i}  alpha: {alpha.item():.3f}   beta: {beta.item():.3f}   loss: {loss.item():.3f}')
        i += 1
