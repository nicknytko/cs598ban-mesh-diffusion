import torch
import torch.linalg as tla
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.spatial as spat
import matplotlib.pyplot as plt
import matplotlib.tri
import firedrake as fd
import firedrake.cython.dmcommon as dmcommon
import pyop2.mpi


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    else:
        return torch.tensor(x)


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.numpy()


def coord_lex_argsort(xy):
    # sort first by X, then stable on Y

    p = torch.argsort(xy[:, 0])
    p2 = torch.argsort(xy[p, 1], stable=True)

    return p[p2]


def perm_inv(p):
    pinv = torch.ones_like(p)
    pinv[p] = torch.arange(len(pinv))
    return pinv


class Mesh:
    # Helper class that contains the set of points for a specific mesh, as well
    # as the matrix representing the underlying PDE operator.

    def _firedrake_gen_mesh(self, verts, edges):
        comm = pyop2.mpi.dup_comm(fd.COMM_WORLD)
        plex = fd.mesh.plex_from_cell_list(2, np.array(edges), np.array(verts), comm)
        return fd.Mesh(plex, reorder=False)

    def _firedrake_assemble_mat(self, mesh, kappa, f, elements='CG', order=1):
        V = fd.FunctionSpace(mesh, elements, order)
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)

        xy = fd.VectorFunctionSpace(mesh, elements, order)
        coords = fd.interpolate(fd.SpatialCoordinate(mesh), xy).dat.data_ro.copy()

        # kappa
        kappa_fn = fd.Function(V)
        if kappa is None:
            kappa_fn.dat.data[:] = 1.
        else:
            kappa_fn.dat.data[:] = kappa(coords[:,0], coords[:,1])

        # assemble bilinear form
        a = kappa_fn * fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
        A = fd.assemble(a, mat_type='aij')
        A = sp.csr_matrix((A.petscmat.getValuesCSR())[::-1])

        # assemble linear rhs
        f_fn = fd.Function(V)
        f_fn.dat.data[:] = f(coords[:,0], coords[:,1])
        b = fd.assemble(f_fn * v * fd.dx)
        b = np.array(b._data.dat._data)

        return A, b

    def __init__(self, verts, edges=None, kappa=None, u=None, f=None):
        self.kappa = kappa
        self.verts = to_tensor(verts)

        if u is None:
            u = lambda x, y: x ** 0
        self.u = u

        if f is None:
            f = lambda x, y: x * 0
        self.f = f

        # Create delaunay triangulation if edge connectivity is not given
        if edges is None:
            triangulation = spat.Delaunay(verts)
            self.edges = to_tensor(triangulation.simplices)
        else:
            self.edges = to_tensor(edges)

        # Find boundary vertices (we won't touch these during diffusion)
        self.boundary_verts = torch.logical_or(
            torch.logical_or(self.verts[:,0] == 0, self.verts[:,0] == 1), # x==0 or x==1
            torch.logical_or(self.verts[:,1] == 0, self.verts[:,1] == 1)  # y==0 or y==1
        )
        self.interior_verts = torch.logical_not(self.boundary_verts)

        # Generate PDE operator
        fdrake_mesh = self._firedrake_gen_mesh(self.verts.numpy(), self.edges.numpy())
        A, b = self._firedrake_assemble_mat(fdrake_mesh, self.kappa, self.f)

        self.A = to_tensor(A.todense()).float()
        self.b = to_tensor(b).float()

        # Reorder DOF on the linear system because Firedrake mangles the ordering even though we tell it not to
        fdrake_coords = torch.tensor(fdrake_mesh.coordinates.dat.data_ro)
        reg_coord_argsort = coord_lex_argsort(self.verts)
        fdrake_coord_argsort = coord_lex_argsort(fdrake_coords)
        fperm = fdrake_coord_argsort[perm_inv(reg_coord_argsort)]

        self.A = self.A[fperm, :][:, fperm]
        self.b = self.b[fperm]

        # Replace boundary rows/columns with 1 on diagonal, 0 on off-diagonal (identity)
        # We force the boundary to be 0, so we don't need to solve for it
        self.A[:, self.boundary_verts] = 0.
        self.A[self.boundary_verts, :] = 0.
        self.A[self.boundary_verts, self.boundary_verts] = 1.

    # Compute numerical PDE solution
    # This gives numerical values for each vertex.

    def calc_num_soln(self):
        return tla.solve(self.A, self.b)

    ## Random plotting functions

    @property
    def mpl_tri(self):
        v_np = to_numpy(self.verts)
        e_np = to_numpy(self.edges)
        return matplotlib.tri.Triangulation(v_np[:,0], v_np[:,1], e_np)

    def plot_grid(self, ax=None, *args, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.triplot(self.mpl_tri, color='k', *args, **kwargs)

    def plot_vals(self, x, ax=None, *args, **kwargs):
        if ax is None:
            ax = plt.gca()

        tri = self.mpl_tri
        m = ax.tricontourf(tri, x, *args, **kwargs)
        plt.colorbar(m, ax=ax)

    def eval_fn(self, f):
        return f(self.verts[:, 0], self.verts[:, 1])

    def plot_fn(self, f, *args, **kwargs):
        fv = f(self.verts[:,0], self.verts[:,1])
        self.plot_vals(fv, *args, **kwargs)

    def plot_true_soln(self, ax=None):
        self.plot_fn(self.u, ax)

    def plot_num_soln(self, ax=None):
        x = to_numpy(self.calc_num_soln())
        self.plot_vals(x, ax)

    def plot_error(self, ax=None):
        if ax is None:
            ax = plt.gca()

        xstar = self.u(self.verts[:,0], self.verts[:,1])
        err = np.abs(xstar - to_numpy(self.calc_num_soln()))
        self.plot_vals(err, ax, vmin=0, vmax=1)

    ## Create a new structured mesh

    def create_structured(Nx, Ny, kappa=None, u=None, f=None):
        x, y = np.meshgrid(np.linspace(0, 1, Nx),
                           np.linspace(0, 1, Ny))
        x = x.flatten(); y = y.flatten()
        verts = np.column_stack((x, y))

        edges = []
        for i in range(Nx - 1):
            for j in range(Ny - 1):
                edges.append([j*Nx + i, j*Nx + i + 1, (j+1)*Nx + i])
                edges.append([j*Nx + i + 1, (j+1)*Nx + i + 1, (j+1)*Nx + i])
        edges = np.array(edges)

        return Mesh(verts, edges, kappa, u, f)

    ## Perturb the interior points of this mesh to get a new one

    def perturb_points(self, sigma=1e-2):
        N = torch.sum(self.interior_verts) # number of interior vertices

        new_verts = self.verts.clone()
        new_verts[self.interior_verts, :] += torch.randn(N, 2) * sigma

        return Mesh(new_verts, None, self.kappa, self.u, self.f)

    ## Interpolate points from another mesh onto this mesh

    def interp_vals(self, other_mesh, other_vals):
        interpolator = matplotlib.tri.LinearTriInterpolator(other_mesh.mpl_tri, to_numpy(other_vals))
        here_vals = to_tensor(
            interpolator(to_numpy(self.verts[:,0]), to_numpy(self.verts[:,1]))
        ).float()
        return here_vals
