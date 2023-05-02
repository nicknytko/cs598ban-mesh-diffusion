import torch
import torch.nn as nn
import torch.nn.functional as F
from onet_model import *
import matplotlib.pyplot as plt

def hypergradient(validation_loss: torch.Tensor, training_loss: torch.Tensor, lambda_: torch.tensor,
                  w: torch.Generator):
    # List[torch.Tensor]. v1[i].shape = w[i].shape
    v1 = torch.autograd.grad(validation_loss, w(), retain_graph=True)

    d_train_d_w = torch.autograd.grad(training_loss, w(), create_graph=True)
    # List[torch.Tensor]. v2[i].shape = w[i].shape
    v2 = approxInverseHVP(v1, d_train_d_w, w)

    # List[torch.Tensor]. v3[i].shape = lambda_[i].shape
    v3 = torch.autograd.grad(d_train_d_w, lambda_, grad_outputs=v2, retain_graph=True, )

    d_val_d_lambda = torch.autograd.grad(validation_loss, lambda_)
    return [d - v for d, v in zip(d_val_d_lambda, v3)]

def approxInverseHVP(v, f, w, i=3, alpha=.1):
    p = v

    for j in range(i):
        grad = torch.autograd.grad(f, w(), grad_outputs=v, retain_graph=True)
        v = [v_ - alpha * g for v_, g in zip(v, grad)]
        p = [p_ + v_ for p_, v_ in zip(p, v)]  # p += v (Typo in the arxiv version of the paper)

    return p

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

train = OnetWaveEqnDataset('../data/onet_wave_train.pt')
test = OnetWaveEqnDataset('../data/onet_wave_test.pt')

num_neumann_steps = 3
alpha = 0.001
num_inner_epochs = 100
num_outer_epochs = 200

Nx = 30
Ny = 30
N_v = Nx * Ny

ablation = False
show_plot = False

# Reference lattice gridpoints
# we will evaluate our model at a small perturbation to these points
x, y = torch.meshgrid(torch.linspace(0, 1, Nx), torch.linspace(0, 1, Ny), indexing='ij')
x = x.flatten().to(device)
y = y.flatten().to(device)
ref_lattice = torch.column_stack((x, y)).unsqueeze(0)
mod_lattice = nn.Parameter(torch.column_stack((x, y)).unsqueeze(0))

model = Onet(2, 16, N_v).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
meta_opt = torch.optim.RMSprop([mod_lattice], lr=alpha)
batch_size = 128

train_lh = []
test_lh = []

if show_plot:
    plt.figure(figsize=(6,6))
    plt.ion()
    plt.show()

for i in range(num_outer_epochs):

    opt.zero_grad()

    # --------------------------------------
    # Inner Loop (Training network at sensor points)
    # --------------------------------------
    for _ in torch.arange(num_inner_epochs):

        train_batch = torch.randperm(len(train))[:batch_size]
        u2_img_tr, u1_img_tr, u_img_tr = train[train_batch] # past two timesteps and true solution
        u2_img_tr = u2_img_tr.to(device); u1_img_tr = u2_img_tr.to(device); u_img_tr = u2_img_tr.to(device);

        # Grab data from the dataset
        evalpts = ref_lattice + torch.rand((batch_size, N_v, 2), device=device) * 1e-2
        batched_lattice = mod_lattice.expand(batch_size, -1, -1)
        u2_tr = interp_image(u2_img_tr, batched_lattice) # evaluate u^{i-2} at sensor points
        u1_tr = interp_image(u1_img_tr, batched_lattice) # evaluate u^{i-1} at sensor points
        u_tr = interp_image(u_img_tr, evalpts).squeeze() # interpolate true soln (u^{i}) at eval pts

        # Evaluate model
        u_hat = model(torch.cat((u2_tr, u1_tr), dim=1), evalpts)

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
    train_batch = torch.randperm(len(train))
    u2_img_tr, u1_img_tr, u_img_tr = train[train_batch]
    u2_img_tr = u2_img_tr.to(device); u1_img_tr = u2_img_tr.to(device); u_img_tr = u2_img_tr.to(device);

    evalpts = ref_lattice + torch.rand((len(train), N_v, 2), device=device) * 1e-2
    batched_lattice = mod_lattice.expand(len(train), -1, -1)

    u2_tr = interp_image(u2_img_tr, batched_lattice) # evaluate u^{i-2} at sensor points
    u1_tr = interp_image(u1_img_tr, batched_lattice) # evaluate u^{i-1} at sensor points
    u_tr = interp_image(u_img_tr, evalpts).squeeze() # interpolate true soln (u^{i}) at eval pts
    u_hat = model(torch.cat((u2_tr, u1_tr), dim=1), evalpts)

    training_loss = ((u_hat - u_tr) ** 2.).mean(dim=1).mean(dim=0)
    print(f"[{i}] Training loss: {training_loss.item()}")

    # test loss
    test_batch = torch.randperm(len(test))
    u2_img_val, u1_img_val, u_img_val = test[test_batch]
    u2_img_val = u2_img_val.to(device); u1_img_val = u2_img_val.to(device); u_img_val = u2_img_val.to(device);

    evalpts = ref_lattice + torch.rand((len(test), N_v, 2), device=device) * 1e-2
    batched_lattice = mod_lattice.expand(len(test), -1, -1)

    u2_val = interp_image(u2_img_val, batched_lattice) # evaluate u^{i-2} at sensor points
    u1_val = interp_image(u1_img_val, batched_lattice) # evaluate u^{i-1} at sensor points
    u_val = interp_image(u_img_val, evalpts).squeeze() # interpolate true soln (u^{i}) at eval pts
    u_hat = model(torch.cat((u2_val, u1_val), dim=1), evalpts)
    validation_loss = ((u_hat - u_val) ** 2.).mean(dim=1).mean(dim=0)
    print(f"[{i}] Validation loss: {validation_loss.item()}")

    train_lh.append(training_loss.item())
    test_lh.append(validation_loss.item())

    if not ablation:
        hyper_grads = hypergradient(validation_loss, training_loss, mod_lattice, model.parameters)

        # --------------------------------------
        # Take Meta Step to Optimize Sensors
        # --------------------------------------
        meta_opt.zero_grad()
        mod_lattice.grad = torch.clip(hyper_grads[0], -0.001, 0.001)
        #mod_lattice.grad = torch.zeros_like(mod_lattice)
        meta_opt.step()

    with torch.no_grad():
        # clamp sensor points to domain
        mod_lattice[mod_lattice < 0.] = 0.
        mod_lattice[mod_lattice > 1.] = 1.

    if show_plot:
        plt.clf()
        plt.scatter(ref_lattice.detach().cpu().numpy()[0, :, 0],
                    ref_lattice.detach().cpu().numpy()[0, :, 1])
        plt.scatter(mod_lattice.detach().cpu().numpy()[0, :, 0],
                    mod_lattice.detach().cpu().numpy()[0, :, 1])
        plt.pause(0.01)

#torch.save(model.state_dict(), '../data/trained_onet.pt')
if ablation:
    torch.save(torch.Tensor(train_lh), '../data/reg_wave_train_lh.pt')
    torch.save(torch.Tensor(test_lh), '../data/reg_wave_test_lh.pt')
else:
    torch.save(torch.Tensor(train_lh), '../data/opt_wave_train_lh.pt')
    torch.save(torch.Tensor(test_lh), '../data/opt_wave_test_lh.pt')
    torch.save(ref_lattice, '../data/ref_wave_lattice.pt')
    torch.save(mod_lattice, '../data/mod_wave_lattice.pt')
