# Euler equations for Sod's shocktube problem

import math
from methods_stream import out_file
from methods_shocktube import initialize, boundary_conditions, c_max, Lax_Wendroff_step, Roe_step, Lapidus_viscosity
import numpy as np
import cupy as cp

L = 1.0                     # length of shock tube
gamma = 1.4                 # ratio of specific heats
N = 200                     # number of grid points

CFL = 0.9                   # Courant-Friedrichs-Lewy number
nu = 0.0                    # artificial viscosity coefficient


def solve(step_algorithm, t_max, file_name, plots=5):
    """solves sod's shock tube problem
    :rtype :    none
    :param step_algorithm: name of the function used as step
    :param t_max: maximum time limit
    :param file_name: core name of output files
    :param plots: number of plots, default is 5
    """

    '''
    # initialize
    rho_left = 1.0
    rho_right = 0.125
    rhov = 0.0 # v = 0 -> rho * v = 0
    p = 1.0
    e = p / (gamma - 1) # v = 0 -> rho * v**2 / 2 = 0
    U_gpu = cp.zeros((N, 3), dtype=np.float64)
    U_gpu[:N/2+1, 0] = rho_left
    U_gpu[N/2+1:, 0] = rho_right
    U_gpu[:, 1] = rhov
    U_gpu[:, 2] = e
    U = cp.asnumpy(U_gpu)
    U = U.tolist()
    h = L / float(N - 1)
    print(type(U))
    print(type(U[100]))
    print(type(U[100][1]))
    print(U[98])
    print(U[99])
    print(U[100])
    print(U[101])
    '''

    U, h = initialize(L, N, gamma)
    print(type(U))
    print(type(U[100]))
    print(type(U[100][1]))
    print(U[98])
    print(U[99])
    print(U[100])
    print(U[101])

    tau = 0
    t = 0.0
    step = 0
    plot = 0
    print(" Time t\t\trho_avg\t\tu_avg\t\te_avg\t\tP_avg")
    while True:
        out_file(U, plot, file_name, t, gamma)
        plot += 1
        if plot > plots:
            print(" Solutions in files 0-..", plots, "-" + file_name)
            break
        while t < t_max * plot / float(plots):
            U = boundary_conditions(U)
            tau = CFL * h / c_max(U, gamma)
            U = step_algorithm(h, tau, U, gamma)
            U = Lapidus_viscosity(h, tau, U, nu)
            t += tau
            step += 1


print(" Sod's Shocktube Problem using various algorithms:")
print(" 1. A two-step Lax-Wendroff algorithm")
print(" 2. Mellema's Roe solver")
print(" N = ", N, ", CFL Number = ", CFL, ", nu = ", nu)
print()
print(" Lax-Wendroff Algorithm")
solve(Lax_Wendroff_step, 1.0, "lax.data")
print()
print(" Roe Solver Algorithm")
solve(Roe_step, 1.0, "roe.data")
