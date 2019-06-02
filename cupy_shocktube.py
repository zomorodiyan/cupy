# issues:
# step_algorithm argument is obsolete and should be removed

# Euler equations for Sod's shocktube problem

import math
from methods_stream import out_file
from methods_shocktube import c_max, Lax_Wendroff_step, Roe_step, Lapidus_viscosity
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
    # initial values
    p_left = 1.0
    p_right = 0.1
    rho_left = 1.0
    rho_right = 0.125
    v_left = 0.0
    v_right = 0.0

    rhov_left = rho_left * v_left
    rhov_right = rho_right * v_right
    e_left = p_left / (gamma - 1) + rho_left * v_left**2 / 2
    e_right = p_right / (gamma - 1) + rho_right * v_right**2 / 2

    U_gpu = cp.zeros((N, 3), dtype=np.float64)
    U_gpu[:N/2+1, 0] = rho_left
    U_gpu[N/2+1:, 0] = rho_right
    U_gpu[:N/2+1, 1] = rhov_left
    U_gpu[N/2+1:, 1] = rhov_right
    U_gpu[:N/2+1, 2] = e_left
    U_gpu[N/2+1:, 2] = e_right
    U = cp.asnumpy(U_gpu) # move the array to the host


    h = L / float(N - 1)
    # end of initial values

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
            U_gpu = cp.asarray(U) # move the array to the current device
            # boundary condition
            U_gpu[0, 0] = U_gpu[1, 0]
            U_gpu[0, 1] = -U_gpu[1, 1]
            U_gpu[0, 2] = U_gpu[1, 2]
            U_gpu[-1, 0] = U_gpu[-2, 0]
            U_gpu[-1, 1] = -U_gpu[-2, 1]
            U_gpu[-1, 2] = U_gpu[-2, 2]
            U = cp.asnumpy(U_gpu) # move the array to the host
            # time step
            tau = CFL * h / c_max(U, gamma)
            if(tau<0.0001):
                break

            # Lax Wendroff step and Roe step
            is_Lax_Wendroff = True
            if(is_Lax_Wendroff):
                #'''{{{ Lax_Wendroff step algorithm
                U_new_gpu = cp.zeros((N, 3), dtype=np.float64)
                F_gpu = cp.zeros((N, 3), dtype=np.float64)

                # compute flux F from U
                rho_gpu = U_gpu[:, 0]
                m_gpu = U_gpu[:, 1]
                e_gpu = U_gpu[:, 2]
                P_gpu = (gamma - 1) * (e_gpu - m_gpu**2 / rho_gpu / 2)
                F_gpu[:, 0] = m_gpu
                F_gpu[:, 1] = m_gpu**2 / rho_gpu + P_gpu
                F_gpu[:, 2] = m_gpu / rho_gpu * (e_gpu + P_gpu)

                for j in range(1, N - 1):
                    for i in range(3):
                        U_new_gpu[j, i] = ((U_gpu[j + 1, i] + U_gpu[j, i]) / 2 - tau / 2 / h * (F_gpu[j + 1, i] - F_gpu[j, i]))

                # boundary condition
                U_new_gpu[0, 0] = U_new_gpu[1, 0]
                U_new_gpu[0, 1] = -U_new_gpu[1, 1]
                U_new_gpu[0, 2] = U_new_gpu[1, 2]
                U_new_gpu[-1, 0] = U_new_gpu[-2, 0]
                U_new_gpu[-1, 1] = -U_new_gpu[-2, 1]
                U_new_gpu[-1, 2] = U_new_gpu[-2, 2]


                # compute flux at half steps
                rho_gpu = U_new_gpu[:, 0]
                m_gpu = U_new_gpu[:, 1]
                e_gpu = U_new_gpu[:, 2]
                P_gpu = (gamma - 1) * (e_gpu - m_gpu**2 / rho_gpu / 2)
                F_gpu[:, 0] = m_gpu
                F_gpu[:, 1] = m_gpu**2 / rho_gpu + P_gpu
                F_gpu[:, 2] = m_gpu / rho_gpu * (e_gpu + P_gpu)

                # step using half step flux
                for j in range(1, N - 1):
                    for i in range(3):
                        U_new_gpu[j][i] = U_gpu[j][i] - tau / h * (F_gpu[j][i] - F_gpu[j - 1][i])

                # update U from U_new
                U_gpu = U_new_gpu
                U = cp.asnumpy(U_gpu) # move the array to the host

                #'''#}}}
                pass
            else:
                #{{{ Roe step algorithm
                pass
                #}}}
            #U = step_algorithm(h, tau, U, gamma=1.4)
            U = Lapidus_viscosity(h, tau, U, nu)
            t += tau
            step += 1
        else:
            continue
        print('Diverged! tau < E-4, time step, "tau" is getting smaller and smaller.')
        break


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
