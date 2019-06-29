# Euler equations for Sod's shocktube problem

import math
from methods_stream import out_file, check
import numpy as np
import cupy as cp
from methods import initial_U, Lax_Wendroff_1st_half_step, Lax_Wendroff_2nd_half_step, cmax, boundary_condition_U, boundary_condition_rho_m_e

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
            # time step
            tau = CFL * h / c_max
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
                #{{{ Roe step algorithm
                # temporary variables
                rho = U_gpu[:, 0]
                m = U_gpu[:, 1]
                e = U_gpu[:, 2]
                tiny = 1e-30
                sbpar1 = 2.0
                sbpar2 = 2.0
                w0 = cp.ones((N), dtype=np.float64)
                w1 = cp.ones((N), dtype=np.float64)
                w2 = cp.ones((N), dtype=np.float64)
                w3 = cp.ones((N), dtype=np.float64)
                vol = cp.ones((N), dtype=np.float64)
                F0 = cp.zeros((N), dtype=np.float64)
                F1 = cp.zeros((N), dtype=np.float64)
                F2 = cp.zeros((N), dtype=np.float64)
                Fc0 = cp.zeros((N), dtype=np.float64)
                Fc1 = cp.zeros((N), dtype=np.float64)
                Fc2 = cp.zeros((N), dtype=np.float64)
                Fl0 = cp.zeros((N), dtype=np.float64)
                Fl1 = cp.zeros((N), dtype=np.float64)
                Fl2 = cp.zeros((N), dtype=np.float64)
                Fr0 = cp.zeros((N), dtype=np.float64)
                Fr1 = cp.zeros((N), dtype=np.float64)
                Fr2 = cp.zeros((N), dtype=np.float64)
                fludif0 = cp.zeros((N), dtype=np.float64)
                fludif1 = cp.zeros((N), dtype=np.float64)
                fludif2 = cp.zeros((N), dtype=np.float64)
                rsumr = cp.zeros((N), dtype=np.float64)
                utilde = cp.zeros((N), dtype=np.float64)
                htilde = cp.zeros((N), dtype=np.float64)
                absvt = cp.zeros((N), dtype=np.float64)
                uvdif = cp.zeros((N), dtype=np.float64)
                ssc = cp.zeros((N), dtype=np.float64)
                vsc = cp.zeros((N), dtype=np.float64)
                a0 = cp.zeros((N), dtype=np.float64)
                a1 = cp.zeros((N), dtype=np.float64)
                a2 = cp.zeros((N), dtype=np.float64)
                ac10 = cp.zeros((N), dtype=np.float64)
                ac11 = cp.zeros((N), dtype=np.float64)
                ac12 = cp.zeros((N), dtype=np.float64)
                ac20 = cp.zeros((N), dtype=np.float64)
                ac21 = cp.zeros((N), dtype=np.float64)
                ac22 = cp.zeros((N), dtype=np.float64)
                eiglam0 = cp.zeros((N), dtype=np.float64)
                eiglam1 = cp.zeros((N), dtype=np.float64)
                eiglam2 = cp.zeros((N), dtype=np.float64)
                sgn0 = cp.zeros((N), dtype=np.float64)
                sgn1 = cp.zeros((N), dtype=np.float64)
                sgn2 = cp.zeros((N), dtype=np.float64)
                ptest = cp.zeros((N), dtype=np.float64)
                isb0 = cp.zeros((N), dtype=np.uint64)
                isb1 = cp.zeros((N), dtype=np.uint64)
                isb2 = cp.zeros((N), dtype=np.uint64)

                # initialize control variable to 0
                icntl = 0

                # find parameter vector w
                w0 = cp.sqrt(vol * rho)
                w1 = w0 * m / rho
                w3 = (gamma - 1) * (e - 0.5 * m**2 / rho)
                w2 = w0 * (e + w3) / rho

                # calculate the fluxes at the cell center
                Fc0 = w0 * w1
                Fc1 = w1 * w1 + vol * w3 #maybe I should use ** instead of repeatation
                Fc2 = w1 * w2

                # calculate the fluxes at the cell walls
                # assuming constant primitive variables
                for i in range(1, N):
                    Fl0[i] = Fc0[i - 1]
                    Fl1[i] = Fc1[i - 1]
                    Fl2[i] = Fc2[i - 1]
                    Fr0[i] = Fc0[i]
                    Fr1[i] = Fc1[i]
                    Fr2[i] = Fc2[i]

                # calculate the flux differences at the cell walls
                for i in range(1, N):
                    fludif0[i] = Fr0[i] - Fl0[i]
                    fludif1[i] = Fr1[i] - Fl1[i]
                    fludif2[i] = Fr2[i] - Fl2[i]

                # calculate the tilded U variables = mean values at the interfaces
                #--------------------------------------------------------------------
                for i in range(1, N):
                    rsumr[i] = 1 / (w0[i - 1] + w0[i])

                    utilde[i] = (w1[i - 1] + w1[i]) * rsumr[i]
                    htilde[i] = (w2[i - 1] + w2[i]) * rsumr[i]

                    absvt[i] = 0.5 * utilde[i] * utilde[i]
                    uvdif[i] = utilde[i] * fludif1[i]

                    ssc[i] = (gamma - 1) * (htilde[i] - absvt[i])
                    if ssc[i] > 0.0:
                        vsc[i] = cp.sqrt(ssc[i])
                    else:
                        vsc[i] = cp.sqrt(abs(ssc[i]))
                        icntl += 1
                #--------------------------------------------------------------------

                # calculate the eigenvalues and projection coefficients for each eigenvector
                #for i in range(1, N):
                eiglam0 = utilde - vsc
                eiglam1 = utilde
                eiglam2 = utilde + vsc
                sgn0 = -2 * (eiglam0 < 0.0) + 1.0 # -1 if <0.0, +1 if >=0 ?
                sgn1 = -2 * (eiglam1 < 0.0) + 1.0 # -1 if <0.0, +1 if >=0 ?
                sgn2 = -2 * (eiglam2 < 0.0) + 1.0 # -1 if <0.0, +1 if >=0 ?

                a0 = 0.5 * ((gamma - 1) * (absvt * fludif0 + fludif2 - uvdif) - vsc * (fludif1 - utilde * fludif0)) / ssc
                a1 = (gamma - 1) * ((htilde - 2 * absvt) * fludif0 + uvdif - fludif2) / ssc
                a2 = 0.5 * ((gamma - 1) * (absvt * fludif0 + fludif2 - uvdif) + vsc * (fludif1 - utilde * fludif0)) / ssc

                # divide the projection coefficients by the wave speeds
                # to evade expansion correction
                for i in range(1, N):
                    a0[i] /= eiglam0[i] + tiny
                    a1[i] /= eiglam1[i] + tiny
                    a2[i] /= eiglam2[i] + tiny

                # calculate the first order projection coefficients ac1
                for i in range(1, N):
                    ac10[i] = -sgn0[i] * a0[i] * eiglam0[i]
                    ac11[i] = -sgn1[i] * a1[i] * eiglam1[i]
                    ac12[i] = -sgn2[i] * a2[i] * eiglam2[i]

                # apply the 'superbee' flux correction to made 2nd order projection
                # coefficients ac2
                # implement an if statement for these six lines
                ac20[1] = ac10[1]
                ac21[1] = ac11[1]
                ac22[1] = ac12[1]
                ac20[N - 1] = ac10[N - 1]
                ac21[N - 1] = ac11[N - 1]
                ac22[N - 1] = ac12[N - 1]
                dtdx = tau / h
                #-------------------------------------------------------------
                for i in range(2, N - 1):
                    isb0[i] = i - int(sgn0[i])
                    ac20[i] = (ac10[i] + eiglam0[i] *
                        ((max(0.0, min(sbpar1 * a0[isb0[i]], max(a0[i], min(a0[isb0[i]], sbpar2 * a0[i])))) +
                        min(0.0, max(sbpar1 * a0[isb0[i]], min(a0[i], max(a0[isb0[i]], sbpar2 * a0[i]))))) *
                        (sgn0[i] - dtdx * eiglam0[i])))
                for i in range(2, N - 1):
                    isb1[i] = i - int(sgn1[i])
                    ac21[i] = (ac11[i] + eiglam1[i] *
                        ((max(0.0, min(sbpar1 * a1[isb1[i]], max(a1[i], min(a1[isb1[i]], sbpar2 * a1[i])))) +
                        min(0.0, max(sbpar1 * a1[isb1[i]], min(a1[i], max(a1[isb1[i]], sbpar2 * a1[i]))))) *
                        (sgn1[i] - dtdx * eiglam1[i])))
                for i in range(2, N - 1):
                    isb2[i] = i - int(sgn2[i])
                    ac22[i] = (ac12[i] + eiglam2[i] *
                        ((max(0.0, min(sbpar1 * a2[isb2[i]], max(a2[i], min(a2[isb2[i]], sbpar2 * a2[i])))) +
                        min(0.0, max(sbpar1 * a2[isb2[i]], min(a2[i], max(a2[isb2[i]], sbpar2 * a2[i]))))) *
                        (sgn2[i] - dtdx * eiglam2[i])))
                #-------------------------------------------------------------
                # calculate the final fluxes
                for i in range(1, N):
                    F0[i] = 0.5 * (Fl0[i] + Fr0[i] + ac20[i] + ac21[i] + ac22[i])
                    F1[i] = 0.5 * (Fl1[i] + Fr1[i] + eiglam0[i] * ac20[i] + eiglam1[i] *
                                   ac21[i] + eiglam2[i] * ac22[i])
                    F2[i] = 0.5 * (Fl2[i] + Fr2[i] + (htilde[i] - utilde[i] * vsc[i]) * ac20[i] +
                                   absvt[i] * ac21[i] + (htilde[i] + utilde[i] * vsc[i]) * ac22[i])

                # calculate test variable for negative pressure check
                for i in range(1, N - 1):
                    ptest[i] = (h * vol[i] * m[i] + tau * (F1[i] - F1[i + 1]))
                    ptest[i] = (- ptest[i] * ptest[i] + 2 * (h * vol[i] * rho[i] +
                               tau * (F0[i] - F0[i + 1])) * (h * vol[i] * e[i] + tau * (F2[i] - F2[i + 1])))

                # check for negative pressure/internal energy and set fluxes
                # left and right to first order if detected
                for i in range(1, N - 1):
                    if (ptest[i] <= 0.0 or (h * vol[i] * rho[i] + tau * (F0[i] - F0[i + 1])) <= 0.0):
                        F0[i] = 0.5 * (Fl0[i] + Fr0[i] + ac10[i] + ac11[i] + ac12[i])
                        F1[i] = 0.5 * (Fl1[i] + Fr1[i] + eiglam0[i] * ac10[i] + eiglam1[i] * ac11[i] +
                            eiglam2[i] * ac12[i])
                        F2[i] = 0.5 * (Fl2[i] + Fr2[i] + (htilde[i]-utilde[i] * vsc[i]) * ac10[i] +
                            absvt[i] * ac11[i] + (htilde[i] + utilde[i] * vsc[i]) * ac12[i])
                        F0[i + 1] = 0.5 * (Fl0[i + 1] + Fr0[i + 1] + ac10[i + 1] + ac11[i + 1] + ac12[i + 1])
                        F1[i + 1] = 0.5 * (Fl1[i + 1] + Fr1[i + 1] + eiglam0[i + 1] * ac10[i + 1] + eiglam1[i + 1] *
                             ac11[i + 1] + eiglam2[i + 1] * ac12[i + 1])
                        F2[i + 1] = 0.5 * (Fl2[i + 1] + Fr2[i + 1] + (htilde[i + 1] - utilde[i + 1] * vsc[i + 1]) * ac10[i + 1]
                             + absvt[i + 1] * ac11[i + 1] + (htilde[i + 1] + utilde[i + 1] * vsc[i + 1]) * ac12[i + 1])

                        # Check if it helped, set control variable if not

                        ptest[i] = (h * vol[i] * m[i] + tau * (F1[i] - F1[i + 1]))
                        ptest[i] = (2.0 * (h * vol[i] * rho[i] + tau * (F0[i] - F0[i + 1])) *
                                (h * vol[i] * e[i] + tau * (F2[i]- F2[i + 1])) - ptest[i] * ptest[i])
                        if (ptest[i] <= 0.0 or (h * vol[i] * rho[i] + tau * (F0[i] - F0[i + 1])) <= 0.0):
                            icntl += 1


                # update U
                for j in range(1, N - 1):   # 15N
                    rho[j] -= tau / h * (F0[j + 1] - F0[j])
                    U_gpu[j, 0] = rho[j]
                    m[j] -= tau / h * (F1[j + 1] - F1[j])
                    U_gpu[j, 1] = m[j]
                    e[j] -= tau / h * (F2[j + 1] - F2[j])
                    U_gpu[j, 2] = e[j]

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
#solve("Lax_Wendroff_step", 1.0, "lax.data")
print(" Roe Solver Algorithm")
solve("Roe_step", 1.0, "roe.data")
