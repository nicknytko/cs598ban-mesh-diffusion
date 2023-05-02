import torch
import torch.linalg as tla
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

ref = torch.load('../data/ref_lattice.pt').squeeze().cpu().detach().numpy()
mod = torch.load('../data/mod_lattice.pt').squeeze().cpu().detach().numpy()

print(la.norm(ref.flatten() - mod.flatten()) ** 2.)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
ax0.scatter(ref[:,0], ref[:,1])
ax0.set_title('Reference sensor points')

ax1.scatter(mod[:,0], mod[:,1])
ax1.set_title('Optimized sensor points')

plt.savefig('opt_points.pdf')
plt.show(block=True)
