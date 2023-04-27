import torch
import mesh
from mesh import Mesh
from model import Unet
import utils
from torch.utils.data import DataLoader
from example_load_structured import StructuredMeshDataset

image_size = (7, 7)
channels = 1

dataset = StructuredMeshDataset()
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

timesteps = 300

# Variance schedule for diffusion process
# We assume we are noising the coordinates/vertices
betas = utils.linear_beta_schedule(timesteps=timesteps)

alphas, alphas_cumprod, alphas_cumprod_prev, sqrt_recip_alphas, sqrt_alphas_cumprod, \
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance = utils.generate_variables(betas)

# TODO: a second beta schedule that is dependent on the above one?
# TODO: variables dependent on second variance schedule?

model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for _ in torch.arange(100):

    for batch in dataloader:

        # TODO: x0 needs to come from the batch
        verices, non_zeros = batch
        vertices = verices.to(device)
        non_zeros = non_zeros.to(device)

        # Following Algorithm 1 of DDPM paper
        t = torch.randint(0, timesteps, (batch.shape[0],), device=device).long()
        noise = torch.randn_like(vertices, device=device)
        xt = utils.q_sample(vertices, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

        # The actual loss function works to have the network learn to predict noise from a noised input
        # (This is from the paper where they found this works better than learning the params of the 
        # reverse Gaussian)

        predicted_noise = model(xt, t)

        loss = torch.nn.functional.mse_loss(noise, predicted_noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())

        # TODO: add model saving

# Sampling process
# start from pure noise (for each example in the batch)
# x_curr = torch.randn_like(x0)
# meshes = []

# for t in reversed(range(0, timesteps)):

#     if t == 0:
#         z = torch.zeros_like(x_curr)
#     else:
#         z = torch.randn_like(x_curr)

#     batch_t = torch.full(x_curr.shape[0], t, device=device, dtype=torch.long)

#     sqrt_recipt_alpha = utils.extract(sqrt_recip_alphas, batch_t, x_curr.shape)
#     one_minus_alpha = utils.extract(betas, batch_t, x_curr.shape)
#     sqrt_one_minus_alpha_cumprod = utils.extract(sqrt_one_minus_alphas_cumprod, batch_t, x_curr.shape)
#     posterior_variance_t = utils.extract(posterior_variance, batch_t, x_curr.shape)

#     mean = sqrt_recipt_alpha * (x_curr - one_minus_alpha / sqrt_one_minus_alpha_cumprod * model(x_curr, batch_t))
    
#     x_curr = mean + torch.sqrt(posterior_variance_t) * z
#     # Meshes will save the path of the reverse process
#     meshes.append(x_curr.cpu().numpy())

# sample = meshes[-1]