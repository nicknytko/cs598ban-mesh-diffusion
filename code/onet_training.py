import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from onet_model import *

train = OnetDataset('../data/onet_train.pt')
test = OnetDataset('../data/onet_test.pt')

Nx = 30
Ny = 30
N_v = Nx * Ny

# Reference lattice gridpoints
# we will evaluate our model at a small perturbation to these points
x, y = torch.meshgrid(torch.linspace(0, 1, Nx), torch.linspace(0, 1, Ny), indexing='ij')
x = x.flatten()
y = y.flatten()
ref_lattice = torch.column_stack((x, y)).unsqueeze(0)

model = Onet(16, N_v)
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
batch_size = 128

# Training loop
for i in range(100):
    batch = torch.randperm(len(train))[:batch_size]

    opt.zero_grad()

    # Grab data from the dataset
    evalpts = ref_lattice + torch.rand((batch_size, N_v, 2)) * 1e-2
    k_img, u_img = train[batch]
    k = interp_image(k_img, evalpts) # interpolate k at our eval points
    u = interp_image(u_img, evalpts).squeeze() # interpolate true soln at eval pts

    # Evaluate model
    u_hat = model(k, evalpts)

    # Evaluate loss
    ell = ((u_hat - u) ** 2.).mean(dim=1).mean(dim=0)
    ell.backward()

    opt.step()

    print(i, ell.item())

torch.save(model.state_dict(), '../data/trained_onet.pt')


evalpts = ref_lattice
k_img, u_img = test[0]
k = interp_image(k_img.unsqueeze(0), evalpts)
u = interp_image(u_img.unsqueeze(0), evalpts).squeeze()

plt.figure()
plt.imshow(k_img, interpolation='bilinear')
plt.colorbar()
plt.title('kappa')

plt.figure()
plt.imshow(u_img, interpolation='bilinear')
plt.title('true')

u_hat = model(k, ref_lattice)

plt.figure()
plt.imshow(u_hat.detach().numpy().reshape((Nx, Ny)).T, interpolation='bilinear')
plt.title('predicted')

plt.show(block=True)
