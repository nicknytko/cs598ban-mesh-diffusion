# Credit: https://github.com/lucidrains/denoising-diffusion-pytorch
import torch
import torch.nn.functional as F

########################
## VARIANCE SCHEDULES ##
########################
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

##########################
## HELPERS FOR LEARNING ##
##########################
def generate_variables(betas):

    # All different values based on betas that beautifully work to 
    # make everything closed form because of math

    # define alphas 
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    return alphas, alphas_cumprod, alphas_cumprod_prev, sqrt_recip_alphas, sqrt_alphas_cumprod, \
            sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance

def extract(a, t, x_shape):
    # This allows us to extract variables from tensors while keeping batch things clean
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def q_sample(x0, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    # This is a closed form way to get the output of forward process after t steps purely based off of initial x0

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x0.shape
    )

    return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

# @torch.no_grad
# def p_sample(model, x, t, t_index):
#     # This allows for sampling in the reverse process from noise to samples

#     betas_t = extract(betas, t, x.shape)
#     sqrt_one_minus_alphas_cumprod_t = extract(
#         sqrt_one_minus_alphas_cumprod, t, x.shape
#     )
#     sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
#     # Equation 11 in the paper
#     # Use our model (noise predictor) to predict the mean
#     model_mean = sqrt_recip_alphas_t * (
#         x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
#     )

#     if t_index == 0:
#         return model_mean
#     else:
#         posterior_variance_t = extract(posterior_variance, t, x.shape)
#         noise = torch.randn_like(x)
#         # Algorithm 2 line 4:
#         return model_mean + torch.sqrt(posterior_variance_t) * noise 
    
# # Algorithm 2 (including returning all images)
# @torch.no_grad
# def p_sample_loop(model, shape):
#     device = next(model.parameters()).device

#     b = shape[0]
#     # start from pure noise (for each example in the batch)
#     img = torch.randn(shape, device=device)
#     imgs = []

#     for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
#         img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
#         imgs.append(img.cpu().numpy())
#     return imgs

# @torch.no_grad
# def sample(model, image_size, batch_size=16, channels=3):
#     return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))