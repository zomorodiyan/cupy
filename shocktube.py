# issues:
# step_algorithm argument is obsolete and should be removed

# Euler equations for Sod's shocktube problem

import math
from methods_stream import out_file
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


    h = L / float(N - 1)
    # end of initial values

    tau = 0
    t = 0.0
    step = 0
    plot = 0
    print(" Time t\t\trho_avg\t\tu_avg\t\te_avg\t\tP_avg")
    while True:
        U = cp.asnumpy(U_gpu) # move the array to the host
        out_file(U, plot, file_name, t, gamma)
        plot += 1
        if plot > plots:
            print(" Solutions in files 0-..", plots, "-" + file_name)
            break
        while t < t_max * plot / float(plots):
            #U_gpu = cp.asarray(U) # move the array to the current device
            # boundary condition
            U_gpu[0, 0] = U_gpu[1, 0]
            U_gpu[0, 1] = -U_gpu[1, 1]
            U_gpu[0, 2] = U_gpu[1, 2]
            U_gpu[-1, 0] = U_gpu[-2, 0]
            U_gpu[-1, 1] = -U_gpu[-2, 1]
            U_gpu[-1, 2] = U_gpu[-2, 2]
            U = cp.asnumpy(U_gpu) # move the array to the host

            c_max = 0.0
            for j in range(N):
                if U_gpu[j][0] != 0.0:
                    rho_gpu = U_gpu[j][0]
                    v_gpu = U_gpu[j][1] / rho_gpu
                    P_gpu = (U_gpu[j][2] - rho_gpu * v_gpu**2 / 2) * (gamma - 1)
                    c_gpu = cp.sqrt(gamma * abs(P_gpu) / rho_gpu)
                    if c_max < c_gpu + abs(v_gpu):
                        c_max = c_gpu + abs(v_gpu)
            # time step
            tau = CFL * h / c_max
            #print("tau: ", tau) # test
            if(tau < 1e-4):
                break

            # Lax Wendroff step and Roe step
            is_Lax_Wendroff = False
            if(step_algorithm is "Lax_Wendroff_step"):
                #{{{ Lax_Wendroff step algorithm
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
                #}}}
            elif(step_algorithm is "Roe_step"):
                #{{{ Roe step algorithm
                # allocate temporary arrays
                tiny = 1e-30
                sbpar1 = 2.0
                sbpar2 = 2.0
                vol_gpu = cp.ones((N), dtype=np.float64)
                vol = np.ones((N))
                F_gpu = cp.zeros((N, 3), dtype=np.float64)
                F = [[0.0] * 3 for j in range(N)]  # 3N
                fludif_gpu = cp.zeros((N, 3), dtype=np.float64)
                fludif = [ [0.0] * 3 for i in range(N) ]
                rsumr_gpu = cp.zeros((N), dtype=np.float64)
                rsumr = [ 0.0 for i in range(N) ]
                utilde_gpu = cp.zeros((N), dtype=np.float64)
                utilde = [ 0.0 for i in range(N) ]
                htilde_gpu = cp.zeros((N), dtype=np.float64)
                htilde = [ 0.0 for i in range(N) ]
                absvt_gpu = cp.zeros((N), dtype=np.float64)
                absvt = [ 0.0 for i in range(N) ]
                uvdif_gpu = cp.zeros((N), dtype=np.float64)
                uvdif = [ 0.0 for i in range(N) ]
                ssc_gpu = cp.zeros((N), dtype=np.float64)
                ssc = [ 0.0 for i in range(N) ]
                vsc_gpu = cp.zeros((N), dtype=np.float64)
                vsc = [ 0.0 for i in range(N) ]
                a_gpu = cp.zeros((N, 3), dtype=np.float64)
                a = [ [0.0] * 3 for i in range(N) ]
                ac1_gpu = cp.zeros((N, 3), dtype=np.float64)
                ac1 = [ [0.0] * 3 for i in range(N) ]
                ac2_gpu = cp.zeros((N, 3), dtype=np.float64)
                ac2 = [ [0.0] * 3 for i in range(N) ]
                w_gpu = cp.zeros((N, 4), dtype=np.float64)
                w = [ [0.0] * 4 for i in range(N) ]
                eiglam_gpu = cp.zeros((N, 3), dtype=np.float64)
                eiglam = [ [0.0] * 3 for i in range(N) ]
                sgn_gpu = cp.zeros((N, 3), dtype=np.float64)
                sgn = [ [0.0] * 3 for i in range(N) ]
                Fc_gpu = cp.zeros((N, 3), dtype=np.float64)
                Fc = [ [0.0] * 3 for i in range(N) ]
                Fl_gpu = cp.zeros((N, 3), dtype=np.float64)
                Fl = [ [0.0] * 3 for i in range(N) ]
                Fr_gpu = cp.zeros((N, 3), dtype=np.float64)
                Fr = [ [0.0] * 3 for i in range(N) ]
                ptest_gpu = cp.zeros((N), dtype=np.float64)
                ptest = [ 0.0 for i in range(N) ]
                isb_gpu = cp.zeros((N, 3), dtype=np.uint64)
                isb = [ [0] * 3 for i in range(N) ]

                # initialize control variable to 0
                icntl = 0


                # find parameter vector w
                w_gpu[:, 0] = cp.sqrt(vol_gpu * U_gpu[:, 0])
                w_gpu[:, 1] = w_gpu[:, 0] * U_gpu[:, 1] / U_gpu[:, 0]
                w_gpu[:, 3] = (gamma - 1) * (U_gpu[:, 2] - 0.5 * U_gpu[:, 1]**2 / U_gpu[:, 0])
                w_gpu[:, 2] = w_gpu[:, 0] * (U_gpu[:, 2] + w_gpu[:, 3]) / U_gpu[:, 0]

                # calculate the fluxes at the cell center
                Fc_gpu[:, 0] = w_gpu[:, 0] * w_gpu[:, 1]
                Fc_gpu[:, 1] = w_gpu[:, 1] * w_gpu[:, 1] + vol_gpu * w_gpu[:, 3]
                Fc_gpu[:, 2] = w_gpu[:, 1] * w_gpu[:, 2]

                # calculate the fluxes at the cell walls
                # assuming constant primitive variables
                for i in range(1, N):
                    Fl_gpu[i, :] = Fc_gpu[i - 1, :]
                    Fr_gpu = Fc_gpu

                # calculate the flux differences at the cell walls
                for i in range(1, N):
                    fludif_gpu[i, :] = Fr_gpu[i, :] - Fl_gpu[i, :]

                # calculate the tilded U variables = mean values at the interfaces
                #--------------------------------------------------------------------
                for i in range(1, N):
                    rsumr_gpu[i] = 1 / (w_gpu[i - 1, 0] + w_gpu[i, 0])

                    utilde_gpu[i] = (w_gpu[i - 1, 1] + w_gpu[i, 1]) * rsumr_gpu[i]
                    htilde_gpu[i] = (w_gpu[i - 1, 2] + w_gpu[i, 2]) * rsumr_gpu[i]

                    absvt_gpu[i] = 0.5 * utilde_gpu[i] * utilde_gpu[i]
                    uvdif_gpu[i] = utilde_gpu[i] * fludif_gpu[i, 1]

                    ssc_gpu[i] = (gamma - 1) * (htilde_gpu[i] - absvt_gpu[i])
                    if ssc_gpu[i] > 0.0:
                        vsc_gpu[i] = cp.sqrt(ssc_gpu[i])
                    else:
                        vsc_gpu[i] = cp.sqrt(abs(ssc_gpu[i]))
                        icntl += 1
                #--------------------------------------------------------------------

                # calculate the eigenvalues and projection coefficients for each eigenvector
                for i in range(1, N):
                    eiglam_gpu[:, 0] = utilde_gpu - vsc_gpu
                    eiglam_gpu[:, 1] = utilde_gpu
                    eiglam_gpu[:, 2] = utilde_gpu + vsc_gpu
                    sgn_gpu = -2 * (eiglam_gpu < 0.0) + 1.0 # -1 if <0.0, +1 if >=0 ?

                    a_gpu[:, 0] = 0.5 * ((gamma - 1) * (absvt_gpu * fludif_gpu[:, 0] + fludif_gpu[:, 2]
                        - uvdif_gpu) - vsc_gpu * (fludif_gpu[:, 1] - utilde_gpu
                            * fludif_gpu[:, 0])) / ssc_gpu
                    a_gpu[:, 1] = (gamma - 1) * ((htilde_gpu - 2 * absvt_gpu) * fludif_gpu[:, 0]
                            + uvdif_gpu - fludif_gpu[:, 2]) / ssc_gpu
                    a_gpu[:, 2] = 0.5 * ((gamma - 1) * (absvt_gpu * fludif_gpu[:, 0] + fludif_gpu[:, 2]
                            - uvdif_gpu) + vsc_gpu * (fludif_gpu[:, 1] - utilde_gpu
                            * fludif_gpu[:, 0])) / ssc_gpu

                # divide the projection coefficients by the wave speeds
                # to evade expansion correction
                for i in range(1, N):
                    a_gpu[i, :] /= eiglam_gpu[i, :] + tiny

                # calculate the first order projection coefficients ac1
                for i in range(1, N):
                    ac1_gpu[i, :] = -sgn_gpu[i, :] * a_gpu[i, :] * eiglam_gpu[i, :]

                # apply the 'superbee' flux correction to made 2nd order projection
                # coefficients ac2
                ac2_gpu[1, :] = ac1_gpu[1, :]
                ac2_gpu[N - 1, :] = ac1_gpu[N - 1, :]

                dtdx = tau / h
                #-------------------------------------------------------------
                for n in range(3):
                    for i in range(2, N -1):
                        isb_gpu[i][n] = i - int(sgn_gpu[i][n])
                        ac2_gpu[i][n] = (ac1_gpu[i][n] + eiglam_gpu[i][n]
                                     * ((max(0.0, min(sbpar1 * a_gpu[isb_gpu[i][n]][n],
                                    max(a_gpu[i][n], min(a_gpu[isb_gpu[i][n]][n], sbpar2
                                        * a_gpu[i][n])))) + min(0.0, max(sbpar1
                                            * a_gpu[isb_gpu[i][n]][n], min(a_gpu[i][n],
                                                max(a_gpu[isb_gpu[i][n]][n], sbpar2 * a_gpu[i][n])))) )
                                            * (sgn_gpu[i][n] - dtdx * eiglam_gpu[i][n])))
                #-------------------------------------------------------------
                # calculate the final fluxes
                for i in range(1, N):
                    F_gpu[i, 0] = 0.5 * (Fl_gpu[i, 0] + Fr_gpu[i, 0] + ac2_gpu[i, 0]
                            + ac2_gpu[i, 1] + ac2_gpu[i, 2])
                    F_gpu[i, 1] = 0.5 * (Fl_gpu[i, 1] + Fr_gpu[i, 1] +
                            eiglam_gpu[i, 0] * ac2_gpu[i, 0] + eiglam_gpu[i, 1] * ac2_gpu[i, 1] +
                            eiglam_gpu[i, 2] * ac2_gpu[i, 2])
                    F_gpu[i, 2] = 0.5 * (Fl_gpu[i, 2] + Fr_gpu[i, 2] +
                            (htilde_gpu[i] - utilde_gpu[i] * vsc_gpu[i]) * ac2_gpu[i, 0] +
                            absvt_gpu[i] * ac2_gpu[i, 1] +
                            (htilde_gpu[i] + utilde_gpu[i] * vsc_gpu[i]) * ac2_gpu[i, 2])

                # calculate test variable for negative pressure check
                for i in range(1, N - 1):
                    ptest_gpu[i] = (h * vol_gpu[i] * U_gpu[i, 1] +
                               tau * (F_gpu[i, 1] - F_gpu[i + 1, 1]))
                    ptest[i] = (- ptest_gpu[i] * ptest_gpu[i] + 2 * (h * vol_gpu[i] * U_gpu[i, 0] +
                               tau * (F_gpu[i, 0] - F_gpu[i + 1, 0])) * (h * vol_gpu[i] *
                               U_gpu[i, 2] + tau * (F_gpu[i, 2] - F_gpu[i + 1, 2])))


                # check for negative pressure/internal energy and set fluxes
                # left and right to first order if detected
                for i in range(1, N - 1):
                    if (ptest_gpu[i] <= 0.0 or (h * vol_gpu[i] * U_gpu[i, 0] + tau * (F_gpu[i, 0]
                                            - F_gpu[i + 1][0])) <= 0.0):

                        F_gpu[i, 0] = 0.5 * (Fl_gpu[i, 0] + Fr_gpu[i, 0] +
                            ac1_gpu[i, 0] + ac1_gpu[i, 1] + ac1_gpu[i, 2])
                        F_gpu[i, 1] = 0.5 * (Fl_gpu[i, 1] + Fr_gpu[i, 1] +
                            eiglam_gpu[i, 0] * ac1_gpu[i, 0] + eiglam_gpu[i, 1] * ac1_gpu[i, 1] +
                            eiglam_gpu[i, 2] * ac1_gpu[i, 2])
                        F_gpu[i, 2] = 0.5 * (Fl_gpu[i, 2] + Fr_gpu[i, 2] +
                            (htilde_gpu[i]-utilde_gpu[i] * vsc_gpu[i]) * ac1_gpu[i, 0] +
                            absvt_gpu[i] * ac1_gpu[i, 1] +
                            (htilde_gpu[i] + utilde_gpu[i] * vsc_gpu[i]) * ac1_gpu[i, 2])
                        F_gpu[i + 1, 0] = 0.5 * (Fl_gpu[i + 1, 0] + Fr_gpu[i + 1, 0] +
                             ac1_gpu[i + 1, 0] + ac1_gpu[i + 1, 1] + ac1_gpu[i + 1, 2])
                        F_gpu[i + 1, 1] = 0.5 * (Fl_gpu[i + 1, 1] + Fr_gpu[i + 1, 1] +
                             eiglam_gpu[i + 1, 0] * ac1_gpu[i + 1, 0] + eiglam_gpu[i + 1, 1] *
                             ac1_gpu[i + 1, 1] + eiglam_gpu[i + 1, 2] * ac1_gpu[i + 1, 2])
                        F_gpu[i + 1, 2] = 0.5 * (Fl_gpu[i + 1, 2] + Fr_gpu[i + 1, 2] +
                             (htilde_gpu[i + 1] - utilde_gpu[i + 1] * vsc_gpu[i + 1]) * ac1_gpu[i + 1, 0]
                             + absvt_gpu[i + 1] * ac1_gpu[i + 1, 1] +
                             (htilde_gpu[i + 1] + utilde_gpu[i + 1] * vsc_gpu[i + 1]) * ac1_gpu[i + 1, 2])

                        # Check if it helped, set control variable if not

                        ptest_gpu[i] = (h * vol_gpu[i] * U_gpu[i][1] +
                                   tau * (F_gpu[i][1] - F_gpu[i + 1][1]))
                        ptest_gpu[i] = (2.0 * (h * vol_gpu[i] * U_gpu[i][0]
                            + tau * (F_gpu[i][0]-F_gpu[i + 1][0])) * (h * vol_gpu[i] *
                            U_gpu[i][2] + tau * (F_gpu[i][2] - F_gpu[i + 1][2]))
                            - ptest_gpu[i] * ptest_gpu[i])
                        if (ptest_gpu[i] <= 0.0 or (h * vol_gpu[i] * U_gpu[i][0] + tau * (F_gpu[i][0] - F_gpu[i + 1][0])) <= 0.0):
                            icntl += 1


                # update U
                for j in range(1, N - 1):   # 15N
                    U_gpu[j, :] -= tau / h * (F_gpu[j + 1, :] - F_gpu[j, :])
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
print(" Lax-Wendroff Algorithm")
solve("Lax_Wendroff_step", 1.0, "lax.data")
print()
print(" Roe Solver Algorithm")
solve("Roe_step", 1.0, "roe.data")
