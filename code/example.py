import matplotlib.pyplot as plt
import mesh
from mesh import Mesh
import numpy as np

# Make up a PDE solution that we can derive a right-hand-side for
# This allows us to have an analytic solution to compare to our numerical one

def u(x, y):
    return np.sin(3*np.pi*x)*np.sin(3*np.pi*y)

def f(x, y):
    u_xx = -9*np.pi**2*np.sin(3*np.pi*x)*np.sin(3*np.pi*y) + 3*np.pi*np.sin(3*np.pi*y)*np.cos(3*np.pi*x)
    u_yy = -9*np.pi**2*np.sin(3*np.pi*x)*np.sin(3*np.pi*y) + 3*np.pi*np.sin(3*np.pi*x)*np.cos(3*np.pi*y)
    return -u_xx - u_yy

def kappa(x, y):
    return np.ones_like(x)


# Structured mesh
structured = Mesh.create_structured(15, 15, kappa=kappa, u=u, f=f)
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))

structured.plot_num_soln(ax=ax0)
structured.plot_grid(ax=ax0, alpha=0.3)
ax0.set_title('Numerical Solution on Structured Grid')

structured.plot_true_soln(ax=ax1)
structured.plot_grid(ax=ax1, alpha=0.3)
ax1.set_title('True Solution on Structured Grid')


# Unstructured mesh
unstructured = structured.perturb_points(sigma=0.01)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
unstructured.plot_num_soln(ax=ax0)
unstructured.plot_grid(ax=ax0, alpha=0.3)
ax0.set_title('Numerical Solution on Unstructured Grid')

unstructured.plot_true_soln(ax=ax1)
unstructured.plot_grid(ax=ax1, alpha=0.3)
ax1.set_title('True Solution on Unstructured Grid')


# Random Kappa

structured_aniso = Mesh.create_structured(14, 14, kappa=kappa)

fig, axs = plt.subplots(4, 4, figsize=(12, 12))
axs = axs.flatten()

for ax in axs:
    # generate a random linear transform to the input coordinates of opensimplex
    # this consists of a rotation, scaling, and another rotation
    theta = np.random.uniform(0, 2 * np.pi)
    Q = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    S = np.diag(np.random.randn(2) * 3.)

    def kappa(x, y):
        X = np.column_stack((x, y))
        X = ((Q.T @ S @ Q) @ X.T).T
        return np.abs(mesh.simplex_noise(X[:,0], X[:,1]))*2. + 1e-1 # we want this strictly positive

    structured_aniso.plot_fn(kappa, ax=ax)
    structured_aniso.plot_grid(ax=ax, alpha=0.3)

plt.show(block=True)
