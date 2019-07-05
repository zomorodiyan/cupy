# Euler equations for Sod's shocktube problem

import math
from methods_stream import out_file, check
import numpy as np
import cupy as cp
from cupy import prof
from methods import initial_U, Lax_Wendroff_1st_half_step, Lax_Wendroff_2nd_half_step, cmax, boundary_condition_U, boundary_condition_rho_m_e, Roe_step
# profiling
#import contextlib
#import time
#import argparse

#@contextlib.contextmanager
#def timer(message):
        #cp.cuda.Stream.null.synchronize()
        #start = time.time()
        #yield
        #cp.cuda.Stream.null.synchronize()
        #end = time.time()
        #print('%s:\t%f sec' % (message, end - start))

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
    U_gpu = initial_U(N, gamma)
    h = L / float(N - 1)

    tau = 0
    t = 0.0
    step = 0
    plot = 0

    print(" Time t\t\trho_avg\t\tu_avg\t\te_avg\t\tP_avg")
    while True:
        U_gpu = boundary_condition_U(U_gpu)  # not required for solution, required for out_file()
        out_file(U_gpu, plot, file_name, t, gamma)
        plot += 1
        if plot > plots:
            print(" Solutions in files 0-..", plots, "-" + file_name)
            break
        while t < t_max * plot / float(plots):
            U_gpu = boundary_condition_U(U_gpu)
            c_max = cmax(U_gpu, N, gamma)
            tau = CFL * h / c_max # time step
            if(tau < 1e-4):
                break
            if(step_algorithm is "Lax_Wendroff_step"):
                #{{{ Lax_Wendroff_step algorithm
                # temporary variables
                tau_array = cp.full((N), tau, dtype=np.float64)   # in order to pass tau to cp.ElementwiseKernel methods
                gamma_array = cp.full((N), gamma, dtype=np.float64)   # in order to pass gamma to cp.ElementwiseKernel methods
                h_array = cp.full((N), h, dtype=np.float64)   # in order to pass h to cp.ElementwiseKernel methods
                rho = U_gpu[:, 0]
                m = U_gpu[:, 1]
                e = U_gpu[:, 2]
                rho2 = cp.empty_like(m)
                m2 = cp.empty_like(m)
                e2 = cp.empty_like(m)
                rho3 = cp.empty_like(m)
                m3 = cp.empty_like(m)
                e3 = cp.empty_like(m)

                #(rho, m, e) to (rho2, m2, e2) first half-step of Lax.
                Lax_Wendroff_1st_half_step(rho, m, e, tau_array, h_array, gamma_array, rho2, m2, e2)

                #update rho2, m2, e2
                rho2, m2, e2 = boundary_condition_rho_m_e(rho2, m2, e2)

                # (rho, m, e) & (rho2, m2, e2) to (rho3, m3, e3) second half-step of Lax.
                Lax_Wendroff_2nd_half_step(rho, m, e, rho2, m2, e2, tau_array, h_array, gamma_array, rho3, m3, e3)

                # (rho3, m3, e3) to U_gpu[N, 3]
                U_gpu[:, 0] = rho3
                U_gpu[:, 1] = m3
                U_gpu[:, 2] = e3
                #}}}
            elif(step_algorithm is "Roe_step"):
                #with timer('timer works! hooray'):
                    #{{{ Roe step algorithm
                    # temporary variables
                    tiny = 1e-30
                    sbpar1 = 2.0
                    sbpar2 = 2.0
                    rho = U_gpu[:, 0] + 0
                    m = U_gpu[:, 1] + 0
                    e = U_gpu[:, 2] + 0
                    rho1 = rho + 0 #debug
                    m1 = m + 0 #debug
                    e1 = e + 0 #debug
                    tau_array = cp.full((N), tau, dtype=np.float64)   # in order to pass tau to cp.ElementwiseKernel
                    gamma_array = cp.full((N), gamma, dtype=np.float64)   # in order to pass gamma to cp.ElementwiseKernel
                    h_array = cp.full((N), h, dtype=np.float64)   # in order to pass h to cp.ElementwiseKernel
                    rho2 = cp.empty_like(m)
                    m2 = cp.empty_like(m)
                    e2 = cp.empty_like(m)
                    #cp.cuda.profiler.start() # Enable profiling
                    Roe_step(rho1, m1, e1, tau_array, h_array, gamma_array, rho2, m2, e2)
                    #cp.cuda.profiler.stop() # Disable profiling
                    U_gpu[:, 0] = rho2
                    U_gpu[:, 1] = m2
                    U_gpu[:, 2] = e2
                    #}}}
            else:
                print("Error! Invalid step_algorithm.")
                return(1)

            # Lapidus_viscosity
            U_temp_gpu = cp.zeros((N, 3), dtype=np.float64)
            for j in range(1, N):
                U_temp_gpu[j, :] = U_gpu[j, :] - U_gpu[j - 1, :]

            # multiply Delta_U by |Delta_U|
            for j in range(1, N):
                for i in range(3):
                    U_temp_gpu[j][i] *= abs(U_temp_gpu[j][i])

            # add artificial viscosity
            for j in range(2, N):
                U_gpu[j, :] += nu * tau / h * (U_temp_gpu[j, :] - U_temp_gpu[j - 1, :])


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
#print(" Lax-Wendroff Algorithm")
#solve("Lax_Wendroff_step", 1.0, "lax.dat")
print(" Roe Solver Algorithm")
solve("Roe_step", 1.0, "roe.dat")
