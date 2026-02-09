"""
Code to solve the collisionless problem for a razorthin disk with isotropic particle
velocities in the xy plane
"""
import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
from matplotlib import pyplot as plt
import numpy as np
import scipy as scp
from scipy.sparse import lil_matrix, csr_matrix

### 1st method: iterative sparse matrix inversion

def laplacian_2d_robin_spatial(
    Nx, Ny, Lx, Ly,
    robin_bc
):
  """
  2D Laplacian with spatially varying Robin BCs.

  robin_bc:
    'left'   : (alpha[Ny], beta[Ny], gamma[Ny])
    'right'  : (alpha[Ny], beta[Ny], gamma[Ny])
    'bottom' : (alpha[Nx], beta[Nx], gamma[Nx])
    'top'    : (alpha[Nx], beta[Nx], gamma[Nx])
  """

  hx = Lx / (Nx + 1)
  hy = Ly / (Ny + 1)

  N = Nx * Ny
  A = lil_matrix((N, N))
  b = np.zeros(N)

  def idx(i, j):
      return i + Nx * j

  for j in range(Ny):
      for i in range(Nx):
          k = idx(i, j)

          diag = -2/hx**2 - 2/hy**2

          # ----- x direction -----

          # Left boundary
          if i > 0:
              A[k, idx(i-1, j)] = 1/hx**2
          else:
              alpha, beta, gamma = (
                  robin_bc['left'][0][j],
                  robin_bc['left'][1][j],
                  robin_bc['left'][2][j],
              )
              denom = beta + alpha * hx
              diag += 2/(hx**2) * (beta / denom)
              b[k] += 2 * gamma / (hx * denom)

          # Right boundary
          if i < Nx - 1:
              A[k, idx(i+1, j)] = 1/hx**2
          else:
              alpha, beta, gamma = (
                  robin_bc['right'][0][j],
                  robin_bc['right'][1][j],
                  robin_bc['right'][2][j],
              )
              denom = beta + alpha * hx
              diag += 2/(hx**2) * (beta / denom)
              b[k] += 2 * gamma / (hx * denom)

          # ----- y direction -----

          # Bottom boundary
          if j > 0:
              A[k, idx(i, j-1)] = 1/hy**2
          else:
              alpha, beta, gamma = (
                  robin_bc['bottom'][0][i],
                  robin_bc['bottom'][1][i],
                  robin_bc['bottom'][2][i],
              )
              denom = beta + alpha * hy
              diag += 2/(hy**2) * (beta / denom)
              b[k] += 2 * gamma / (hy * denom)

          # Top boundary
          if j < Ny - 1:
              A[k, idx(i, j+1)] = 1/hy**2
          else:
              alpha, beta, gamma = (
                  robin_bc['top'][0][i],
                  robin_bc['top'][1][i],
                  robin_bc['top'][2][i],
              )
              denom = beta + alpha * hy
              diag += 2/(hy**2) * (beta / denom)
              b[k] += 2 * gamma / (hy * denom)

          A[k, k] = diag

  return csr_matrix(A), b

"""
Equations to solve:

D^2 psi = 0
with the appropriate B.Cs

In cylindrical coordinates:

[ d_z psi ]+ = -pi (psi_old^5 + 5 psi psi_old^4 sigma) +, z = 0+
psi - 1 --> const/r as r--> infinity

D^2 (alpha psi) = 0
[ d_z (alpha psi) ]+ = pi alpha psi^5 (sigma + 2 Sigma), z = 0+

Let K = K_0/(m^2 M)

sigma = 2*pi*K*E_0^2*alpha^{-3} = 2*pi*K_0*(E_0/m)^2*alpha^{-3}/M
Sigma = sigma*(1 - (alpha/(E_0/m))^2) 

"""

E_0_over_m = 0.8
M_guess = 1.0 # initial anzatz for M_ADM
"""
initial estimate for r_max, defined by alpha(r=r_max) = E_0/m
using isotropic Schwarzschild metric
"""
r_max_guess = M_guess*0.5*(1 + E_0_over_m)/(1 - E_0_over_m)
print("r_max_guess/M_guess = ",r_max_guess/M_guess)

def alpha_anzatz(R):
  return (E_0_over_m + (R - r_max_guess)/(2.0*r_max_guess))/(1 + (R - r_max_guess)/(2.0*r_max_guess))

K_0 = 1.0

def sigma(r,alpha):
  result = (r <= r_max_guess)*2*np.pi*K_0*E_0_over_m*alpha**(-3.0)/M_guess
  return result

### Method 1: the py-pde package. 
import pde 

#import CartesianGrid, solve_laplace_equation

rmin = 0.0
rmax = 10.0*M_guess
zmin = 0.0
zmax = 10.0*M_guess
N = 50
Nx = N
Ny = N
dr = (rmax - rmin)/Nx
dz = (zmax - zmin)/Ny 

r_arr = np.linspace(rmin,rmax,num=Nx,endpoint=False)
z_arr = np.linspace(zmin,zmax,num=Ny,endpoint=False)
R_top = np.sqrt(z_arr[-1]**2 + r_arr**2)
R_right = np.sqrt(r_arr[-1]**2 + z_arr**2)

#grid = pde.CartesianGrid([[rmin,rmax],[zmin,zmax]],N)
#r_arr,z_arr = grid._axes_coords
#r_arr_full = np.linspace(rmin-0.5*dr,rmax+0.5*dr,num=(N+2),endpoint=True)
#z_arr_full = np.linspace(zmin-0.5*dz,zmax+0.5*dz,num=(N+2),endpoint=True)
#r_2D, z_2D = np.meshgrid(r_arr_full, z_arr_full)
#R_2D = np.sqrt(r_2D**2 + z_2D**2)
alpha_bottom_old = alpha_anzatz(r_arr)
psi_old = np.ones(r_arr.shape)
sigma_old = sigma(r_arr,alpha_bottom_old)

robin_bc = {'left' : (np.zeros(Ny), np.ones(Ny), np.zeros(Ny)),
            'top'  : (r_arr/R_top**2, np.ones(Ny), r_arr/R_top),
            'right'  : (z_arr/R_right**2, np.ones(Ny), z_arr/R_right),
            'bottom' : (-np.pi*psi_old**4*sigma_old*5.0, np.ones(Ny), np.pi*psi_old**5*sigma_old)}

Matrix, RHS = laplacian_2d_robin_spatial(Nx, Ny, (rmax-rmin), (zmax-zmin),robin_bc)

## Now we try the iterative solver with preconditioning 

"""from scipy.sparse.linalg import cg, spilu, LinearOperator

Matrix_csc = Matrix.tocsc()
ilu = spilu(Matrix_csc, drop_tol=1e-4)
M_operator = LinearOperator(A.shape, ilu.solve)

u, info = cg(Matrix_csc, RHS, M=M_operator, tol=1e-10)"""

from scipy.sparse.linalg import gmres

u, info = gmres(Matrix, RHS, tol=1e-10)

print("info = ", info)

















"""def setter(data, args=None):
  # enforce zero derivative at the r = 0 boundary
  data[0, :] = data[1, :]  # Vanishing derivative at left side
  # enforce more tricky bc at the lower edge
  psi_at_z0 = 0.5*(data[:, 0] + data[:, 1])
  d_psi_dz = bc_zlower_func(psi_at_z0,r_arr_full)
  data[:, 0] = data[:, 1] - dr*d_psi_dz
  # now enforce fall-off condition at the outer edges
  R_right_edge_minus_1 = np.sqrt((rmax-0.5*dr)**2 + z_arr_full**2)
  R_right_edge = np.sqrt((rmax+0.5*dr)**2 + z_arr_full**2)
  dR = (R_right_edge - R_right_edge_minus_1)
  data[-1, :] = (dR + R_right_edge_minus_1*data[-2, :])/R_right_edge
  #
  R_top_edge_minus_1 = np.sqrt((zmax-0.5*dz)**2 + r_arr_full**2)
  R_top_edge = np.sqrt((zmax+0.5*dz)**2 + r_arr_full**2)
  dR = (R_right_edge - R_right_edge_minus_1)
  data[:, -1] = (dR + R_right_edge_minus_1*data[:, -2])/R_right_edge"""

#my_bc_setter = pde.grids.boundaries.axes.BoundariesSetter(setter)