import torch
import mesh
from mesh import Mesh
import matplotlib.pyplot as plt

structured = Mesh.create_structured(7, 7)
diffusion_output = structured.diffuse_grid(N_T=10, sigma=0.03)

plt.figure()
diffusion_output[-1].plot_grid()
plt.show(block=True)
