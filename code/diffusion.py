import torch
import mesh
from mesh import Mesh
from model import Unet
import utils

# structured = Mesh.create_structured(7, 7)
# diffusion_output = structured.diffuse_grid(N_T=10, sigma=0.03)

# plt.figure()
# diffusion_output[-1].plot_grid()
# plt.show(block=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

timesteps = 300

# define beta schedule
betas = utils.linear_beta_schedule(timesteps=timesteps)

alphas, alphas_cumprod, alphas_cumprod_prev, sqrt_recip_alphas, sqrt_alphas_cumprod, \
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance = utils.generate_variables(betas)

model = Unet(
    dim=(7, 7),
    channels=1
)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


for _ in torch.arange(100):

    # TODO: masage inputs into dataloader
    for batch in dataloader:

        batch = batch.to(device)

        # TODO: x0 needs to come from the batch
        x0 = Mesh.create_structured(7, 7)

        t = torch.randint(timesteps, (batch.shape[0],), device=device).long()

        noise = torch.randn_like(x0, device=device)

        # TODO: we need to massage this to work like structured.diffuse_grid(N_T, sigma)
        xt = utils.q_sample(x0, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

        # The actual loss function works to have the network learn to predict noise from a noised input
        # (This is from the paper where they found this works better than learning the params of the 
        # reverse Gaussian)

        predicted_noise = model(xt, t)

        loss = torch.nn.functional.mse_loss(noise, predicted_noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO: add model saving

# Sampling process
# start from pure noise (for each example in the batch)
x_curr = torch.randn_like(x0)
meshes = []

for t in reversed(range(0, timesteps)):

    if t == 0:
        z = torch.zeros_like(x_curr)
    else:
        z = torch.randn_like(x_curr)

    batch_t = torch.full(x_curr.shape[0], t, device=device, dtype=torch.long)

    sqrt_recipt_alpha = utils.extract(sqrt_recip_alphas, batch_t, x_curr.shape)
    one_minus_alpha = utils.extract(betas, batch_t, x_curr.shape)
    sqrt_one_minus_alpha_cumprod = utils.extract(sqrt_one_minus_alphas_cumprod, batch_t, x_curr.shape)
    posterior_variance_t = utils.extract(posterior_variance, batch_t, x_curr.shape)

    mean = sqrt_recipt_alpha * (x_curr - one_minus_alpha / sqrt_one_minus_alpha_cumprod * model(x_curr, batch_t))
    
    x_curr = mean + torch.sqrt(posterior_variance_t) * z
    # Meshes will save the path of the reverse process
    meshes.append(x_curr.cpu().numpy())

sample = meshes[-1]