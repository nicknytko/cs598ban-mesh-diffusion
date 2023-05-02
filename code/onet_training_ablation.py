import torch
import torch.nn as nn
import torch.nn.functional as F
from onet_model import *

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

train = OnetDataset('../data/onet_train.pt')
test = OnetDataset('../data/onet_test.pt')

num_neumann_steps = 3
alpha = 0.1
num_inner_epochs = 50
num_outer_epochs = 200

Nx = 30
Ny = 30
N_v = Nx * Ny

# Reference lattice gridpoints
# we will evaluate our model at a small perturbation to these points
x, y = torch.meshgrid(torch.linspace(0, 1, Nx), torch.linspace(0, 1, Ny), indexing='ij')
x = x.flatten().to(device)
y = y.flatten().to(device)
ref_lattice = torch.column_stack((x, y)).unsqueeze(0)
mod_lattice = nn.Parameter(torch.column_stack((x, y)).unsqueeze(0))
print(ref_lattice.device)

model = Onet(1, 16, N_v).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
meta_opt = torch.optim.RMSprop([mod_lattice], lr=0.001)
batch_size = 128

train_lh = []
test_lh = []

for i in range(num_outer_epochs):

    opt.zero_grad()

    # --------------------------------------
    # Inner Loop (Training network at sensor points)
    # --------------------------------------
    for _ in torch.arange(num_inner_epochs):

        train_batch = torch.randperm(len(train))[:batch_size]
        k_img_tr, u_img_tr = train[train_batch]
        k_img_tr = k_img_tr.to(device)
        u_img_tr = u_img_tr.to(device)
        # Grab data from the dataset
        evalpts = ref_lattice + torch.rand((batch_size, N_v, 2), device=device) * 1e-2
        batched_lattice = mod_lattice.expand(batch_size, -1, -1)
        k_tr = interp_image(k_img_tr, batched_lattice) # interpolate k at our eval points
        u_tr = interp_image(u_img_tr, evalpts).squeeze() # interpolate true soln at eval pts

        # Evaluate model
        u_hat = model(k_tr, evalpts)

        # Evaluate loss
        ell = ((u_hat - u_tr) ** 2.).mean(dim=1).mean(dim=0)

        opt.zero_grad()
        ell.backward()
        mod_lattice.grad.zero_()
        opt.step()

    # --------------------------------------
    # Approximating Gradient via IFT
    # --------------------------------------
    #train loss
    with torch.no_grad():
        train_batch = torch.randperm(len(train))
        k_img_tr, u_img_tr = train[train_batch]
        k_img_tr = k_img_tr.to(device)
        u_img_tr = u_img_tr.to(device)

        evalpts = ref_lattice + torch.rand((len(train), N_v, 2), device=device) * 1e-2
        batched_lattice = mod_lattice.expand(len(train), -1, -1)

        k_tr = interp_image(k_img_tr, batched_lattice) # interpolate k at our eval points
        u_tr = interp_image(u_img_tr, evalpts).squeeze() # interpolate true soln at eval pts
        u_hat = model(k_tr, evalpts)
        training_loss = ((u_hat - u_tr) ** 2.).mean(dim=1).mean(dim=0)
        print(f"[{i}] Training loss: {training_loss.item()}")

        # test loss
        test_batch = torch.randperm(len(test))
        k_img_val, u_img_val = test[test_batch]
        k_img_val = k_img_val.to(device)
        u_img_val = u_img_val.to(device)

        evalpts = ref_lattice + torch.rand((len(test), N_v, 2), device=device) * 1e-2
        batched_lattice = mod_lattice.expand(len(test), -1, -1)

        k_val = interp_image(k_img_val, batched_lattice) # interpolate k at our eval points
        u_val = interp_image(u_img_val, evalpts).squeeze() # interpolate true soln at eval pts
        u_hat = model(k_val, evalpts)
        validation_loss = ((u_hat - u_val) ** 2.).mean(dim=1).mean(dim=0)
        print(f"[{i}] Validation loss: {validation_loss.item()}")

        train_lh.append(training_loss.item())
        test_lh.append(validation_loss.item())

torch.save(torch.Tensor(train_lh), '../data/reg_train_lh.pt')
torch.save(torch.Tensor(test_lh), '../data/reg_test_lh.pt')
