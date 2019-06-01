# Translation of G. Mellema's Roe Solver

import math
import numpy as np


def c_max(U, gamma):   # 22N
    """returns the highest sum of c, sound speed and abs(v), velocity size
    :param U: list of states values at the N points
    :rtype : float
    :return: max(c + abs(v)) in U
    """
    v_max = 0.0
    N = len(U)
    for j in range(N):
        if U[j][0] != 0.0:
            rho = U[j][0]
            v = U[j][1] / rho
            P = (U[j][2] - rho * v**2 / 2) * (gamma - 1)
            c = math.sqrt(gamma * abs(P) / rho)
            if v_max < c + abs(v):
                v_max = c + abs(v)
    return v_max


def Lapidus_viscosity(h, tau, U, nu):    # 33N
    """gets U and returns modified U ...
    :param U: list of states values at the N points
    :rtype : float[][]
    :return: U
    """
    # store Delta_U values in newU
    N = len(U)
    U_temp = [[0.0] * 3 for j in range(N)]
    for j in range(1, N):
        for i in range(3):
            U_temp[j][i] = U[j][i] - U[j - 1][i]

    # multiply Delta_U by |Delta_U|
    for j in range(1, N):
        for i in range(3):
            U_temp[j][i] *= abs(U_temp[j][i])

    # add artificial viscosity
    for j in range(2, N):
        for i in range(3):
            U[j][i] += nu * tau / h * (U_temp[j][i] - U_temp[j - 1][i])
    return U


def Lax_Wendroff_step(h, tau, U, gamma=1.4):  # 92N
    """Lax Wendroff algorithm step for Sod's shocktube problem
    :param U: list of states values at the N points
    :rtype : float[][]
    :return: U at the next step
    """


    N = len(U)
    U_new = [[0.0] * 3 for j in range(N)]
    F = [[0.0] * 3 for j in range(N)]

    # compute flux F from U
    for j in range(N):
        rho = U[j][0]
        m = U[j][1]
        e = U[j][2]
        P = (gamma - 1) * (e - m**2 / rho / 2)
        F[j][0] = m
        F[j][1] = m**2 / rho + P
        F[j][2] = m / rho * (e + P)

    # half step
    for j in range(1, N - 1):
        for i in range(3):
            U_new[j][i] = ((U[j + 1][i] + U[j][i]) / 2 -
                           tau / 2 / h * (F[j + 1][i] - F[j][i]))
    U_new = boundary_conditions(U_new)

    # compute flux at half steps
    for j in range(N):
        rho = U_new[j][0]
        m = U_new[j][1]
        e = U_new[j][2]
        P = (gamma - 1) * (e - m**2 / rho / 2)
        F[j][0] = m
        F[j][1] = m**2 / rho + P
        F[j][2] = m / rho * (e + P)

    # step using half step flux
    for j in range(1, N - 1):
        for i in range(3):
            U_new[j][i] = U[j][i] - tau / h * (F[j][i] - F[j - 1][i])

    # update U from U_new
    for j in range(1, N - 1):
        for i in range(3):
            U[j][i] = U_new[j][i]
    return U


def Roe_step(h, tau, U, gamma=1.4):  # ?N
    """Roe algorithm step for Sod's shocktube problem
    h      spatial step
    tau      time step
    gamma   adiabatic index
    vol     volume factor for 3-D problem
    U   (rho, rho*u, e) -- input
    F    flux at cell boundaries -- output
    N   number of points
    icntl   diagnostic -- bad if != 0
    """
    # allocate temporary arrays
    tiny = 1e-30
    sbpar1 = 2.0
    sbpar2 = 2.0
    N = len(U)
    vol = np.ones((N))
    F = [[0.0] * 3 for j in range(N)]  # 3N
    fludif = [ [0.0] * 3 for i in range(N) ]
    rsumr = [ 0.0 for i in range(N) ]
    utilde = [ 0.0 for i in range(N) ]
    htilde = [ 0.0 for i in range(N) ]
    absvt = [ 0.0 for i in range(N) ]
    uvdif = [ 0.0 for i in range(N) ]
    ssc = [ 0.0 for i in range(N) ]
    vsc = [ 0.0 for i in range(N) ]
    a = [ [0.0] * 3 for i in range(N) ]
    ac1 = [ [0.0] * 3 for i in range(N) ]
    ac2 = [ [0.0] * 3 for i in range(N) ]
    w = [ [0.0] * 4 for i in range(N) ]
    eiglam = [ [0.0] * 3 for i in range(N) ]
    sgn = [ [0.0] * 3 for i in range(N) ]
    Fc = [ [0.0] * 3 for i in range(N) ]
    Fl = [ [0.0] * 3 for i in range(N) ]
    Fr = [ [0.0] * 3 for i in range(N) ]
    ptest = [ 0.0 for i in range(N) ]
    isb = [ [0] * 3 for i in range(N) ]

    # initialize control variable to 0
    icntl = 0

    # find parameter vector w
    for i in range(N):
        w[i][0] = math.sqrt(vol[i] * U[i][0])
        w[i][1] = w[i][0] * U[i][1] / U[i][0]
        w[i][3] = (gamma - 1) * (U[i][2] - 0.5 * U[i][1]
                  * U[i][1] / U[i][0])
        w[i][2] = w[i][0] * (U[i][2] + w[i][3]) / U[i][0]

    # calculate the fluxes at the cell center
    for i in range(N):
        Fc[i][0] = w[i][0] * w[i][1]
        Fc[i][1] = w[i][1] * w[i][1] + vol[i] * w[i][3]
        Fc[i][2] = w[i][1] * w[i][2]

    # calculate the fluxes at the cell walls
    # assuming constant primitive variables
    for n in range(3):
        for i in range(1, N):
            Fl[i][n] = Fc[i - 1][n]
            Fr[i][n] = Fc[i][n]

    # calculate the flux differences at the cell walls
    for n in range(3):
        for i in range(1, N):
            fludif[i][n] = Fr[i][n] - Fl[i][n]

    # calculate the tilded U variables = mean values at the interfaces
    for i in range(1, N):
        rsumr[i] = 1 / (w[i - 1][0] + w[i][0])

        utilde[i] = (w[i - 1][1] + w[i][1]) * rsumr[i]
        htilde[i] = (w[i - 1][2] + w[i][2]) * rsumr[i]

        absvt[i] = 0.5 * utilde[i] * utilde[i]
        uvdif[i] = utilde[i] * fludif[i][1]

        ssc[i] = (gamma - 1) * (htilde[i] - absvt[i])
        if ssc[i] > 0.0:
            vsc[i] = math.sqrt(ssc[i])
        else:
            vsc[i] = math.sqrt(abs(ssc[i]))
            icntl += 1

    # calculate the eigenvalues and projection coefficients for each eigenvector
    for i in range(1, N):
        eiglam[i][0] = utilde[i] - vsc[i]
        eiglam[i][1] = utilde[i]
        eiglam[i][2] = utilde[i] + vsc[i]
        for n in range(3):
            if eiglam[i][n] < 0.0:
                sgn[i][n] = -1
            else:
                sgn[i][n] = 1
        a[i][0] = 0.5 * ((gamma - 1) * (absvt[i] * fludif[i][0] + fludif[i][2]
                  - uvdif[i]) - vsc[i] * (fludif[i][1] - utilde[i]
                  * fludif[i][0])) / ssc[i]
        a[i][1] = (gamma - 1) * ((htilde[i] - 2 * absvt[i]) * fludif[i][0]
                  + uvdif[i] - fludif[i][2]) / ssc[i]
        a[i][2] = 0.5 * ((gamma - 1) * (absvt[i] * fludif[i][0] + fludif[i][2]
                  - uvdif[i]) + vsc[i] * (fludif[i][1] - utilde[i]
                  * fludif[i][0])) / ssc[i]

    # divide the projection coefficients by the wave speeds
    # to evade expansion correction
    for n in range(3):
        for i in range(1, N):
            a[i][n] /= eiglam[i][n] + tiny

    # calculate the first order projection coefficients ac1
    for n in range(3):
        for i in range(1, N):
            ac1[i][n] = - sgn[i][n] * a[i][n] * eiglam[i][n]

    # apply the 'superbee' flux correction to made 2nd order projection
    # coefficients ac2
    for n in range(3):
        ac2[1][n] = ac1[1][n]
        ac2[N - 1][n] = ac1[N - 1][n]

    dtdx = tau / h
    for n in range(3):
        for i in range(2, N -1):
            isb[i][n] = i - int(sgn[i][n])
            ac2[i][n] = (ac1[i][n] + eiglam[i][n] *
                        ((max(0.0, min(sbpar1 * a[isb[i][n]][n], max(a[i][n],
                        min(a[isb[i][n]][n], sbpar2 * a[i][n])))) +
                        min(0.0, max(sbpar1 * a[isb[i][n]][n], min(a[i][n],
                        max(a[isb[i][n]][n], sbpar2 * a[i][n])))) ) *
                        (sgn[i][n] - dtdx * eiglam[i][n])))

    # calculate the final fluxes
    for i in range(1, N):
        F[i][0] = 0.5 * (Fl[i][0] + Fr[i][0] + ac2[i][0]
                     + ac2[i][1] + ac2[i][2])
        F[i][1] = 0.5 * (Fl[i][1] + Fr[i][1] +
                     eiglam[i][0] * ac2[i][0] + eiglam[i][1] * ac2[i][1] +
                     eiglam[i][2] * ac2[i][2])
        F[i][2] = 0.5 * (Fl[i][2] + Fr[i][2] +
                     (htilde[i] - utilde[i] * vsc[i]) * ac2[i][0] +
                     absvt[i] * ac2[i][1] +
                     (htilde[i] + utilde[i] * vsc[i]) * ac2[i][2])

    # calculate test variable for negative pressure check
    for i in range(1, N - 1):
        ptest[i] = (h * vol[i] * U[i][1] +
                   tau * (F[i][1] - F[i + 1][1]))
        ptest[i] = (- ptest[i] * ptest[i] + 2 * (h * vol[i] * U[i][0] +
                   tau * (F[i][0] - F[i + 1][0])) * (h * vol[i] *
                   U[i][2] + tau * (F[i][2] - F[i + 1][2])))

    # check for negative pressure/internal energy and set fluxes
    # left and right to first order if detected
    for i in range(1, N - 1):
        if (ptest[i] <= 0.0 or (h * vol[i] * U[i][0] + tau * (F[i][0]
                                - F[i + 1][0])) <= 0.0):

            F[i][0] = 0.5 * (Fl[i][0] + Fr[i][0] +
                ac1[i][0] + ac1[i][1] + ac1[i][2])
            F[i][1] = 0.5 * (Fl[i][1] + Fr[i][1] +
                eiglam[i][0] * ac1[i][0] + eiglam[i][1] * ac1[i][1] +
                eiglam[i][2] * ac1[i][2])
            F[i][2] = 0.5 * (Fl[i][2] + Fr[i][2] +
                (htilde[i]-utilde[i] * vsc[i]) * ac1[i][0] +
                absvt[i] * ac1[i][1] +
                (htilde[i] + utilde[i] * vsc[i]) * ac1[i][2])
            F[i + 1][0] = 0.5 * (Fl[i + 1][0] + Fr[i + 1][0] +
                 ac1[i + 1][0] + ac1[i + 1][1] + ac1[i + 1][2])
            F[i + 1][1] = 0.5 * (Fl[i + 1][1] + Fr[i + 1][1] +
                 eiglam[i + 1][0] * ac1[i + 1][0] + eiglam[i + 1][1] *
                 ac1[i + 1][1] + eiglam[i + 1][2] * ac1[i + 1][2])
            F[i + 1][2] = 0.5 * (Fl[i + 1][2] + Fr[i + 1][2] +
                 (htilde[i + 1] - utilde[i + 1] * vsc[i + 1]) * ac1[i + 1][0]
                 + absvt[i + 1] * ac1[i + 1][1] +
                 (htilde[i + 1] + utilde[i + 1] * vsc[i + 1]) * ac1[i + 1][2])

            # Check if it helped, set control variable if not

            ptest[i] = (h * vol[i] * U[i][1] +
                       tau * (F[i][1] - F[i + 1][1]))
            ptest[i] = (2.0 * (h * vol[i] * U[i][0]
                + tau * (F[i][0]-F[i + 1][0])) * (h * vol[i] *
                U[i][2] + tau * (F[i][2] - F[i + 1][2]))
                - ptest[i] * ptest[i])
            if (ptest[i] <= 0.0 or (h * vol[i] * U[i][0] + tau * (F[i][0] - F[i + 1][0])) <= 0.0):
                icntl += 1


    # update U
    for j in range(1, N - 1):   # 15N
        for i in range(3):
            U[j][i] -= tau / h * (F[j + 1][i] - F[j][i])
    return U
