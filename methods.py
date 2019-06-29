import numpy as np
import cupy as cp

def initial_U(N, gamma):
    #{{{
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
    return U_gpu
    #}}}

def boundary_condition_U(U_gpu):
    #{{{
    U_gpu[0, 0] = U_gpu[1, 0]
    U_gpu[0, 1] = -U_gpu[1, 1]
    U_gpu[0, 2] = U_gpu[1, 2]
    U_gpu[-1, 0] = U_gpu[-2, 0]
    U_gpu[-1, 1] = -U_gpu[-2, 1]
    U_gpu[-1, 2] = U_gpu[-2, 2]
    return U_gpu
    #}}}

def boundary_condition_rho_m_e(rho, m, e):
    #{{{
    rho[0] = rho[1]
    m[0] = -m[1]
    e[0] = e[1]
    rho[-1] = rho[-2]
    m[-1] = -m[-2]
    e[-1] = e[-2]
    return rho, m, e
    #}}}

def cmax(U_gpu, N, gamma):
    #{{{
    c_max = 0.0
    for j in range(N):
        if U_gpu[j][0] != 0.0:
            rho_gpu = U_gpu[j][0]
            v_gpu = U_gpu[j][1] / rho_gpu
            P_gpu = (U_gpu[j][2] - rho_gpu * v_gpu**2 / 2) * (gamma - 1)
            c_gpu = cp.sqrt(gamma * abs(P_gpu) / rho_gpu)
            if c_max < c_gpu + abs(v_gpu):
                c_max = c_gpu + abs(v_gpu)
    return c_max
    #}}}

#{{{ Lax_Wendroff_1st_half_step (ElementwiseKernel)
Lax_Wendroff_1st_half_step = cp.ElementwiseKernel(
        'raw T rho, raw T m, raw T e, T tau, T h, T gamma',
        'T rho_new, T m_new, T e_new',
        '''
            T Pi = (gamma - 1) * (e[i] - m[i] * m[i] / rho[i] / 2);
            T Pip1 = (gamma - 1) * (e[i + 1] - m[i + 1] * m[i + 1] / rho[i + 1] / 2);
            T F0i = m[i];
            T F0ip1 = m[i + 1];
            T F1i = m[i] * m[i] / rho[i] + Pi;
            T F1ip1 = m[i + 1] * m[i + 1] / rho[i + 1] + Pip1;
            T F2i = m[i] / rho[i] * (e[i] + Pi);
            T F2ip1 = m[i + 1] / rho[i + 1] * (e[i + 1] + Pip1);
            if (i > 0 && i < _ind.size() - 1)
            {
                rho_new = (rho[i + 1] + rho[i]) / 2 - tau / 2 / h * (F0ip1 - F0i);
                m_new = (m[i + 1] + m[i]) / 2 - tau / 2 / h * (F1ip1 - F1i);
                e_new = (e[i + 1] + e[i]) / 2 - tau / 2 / h * (F2ip1 - F2i);
            }
        ''',
        'Lax_Wendroff_1st')
#}}}

#{{{ Lax_Wendroff_2nd_half_step (ElementwiseKernel)
Lax_Wendroff_2nd_half_step = cp.ElementwiseKernel(
        'T rho, T m, T e, raw T rho2, raw T m2, raw T e2, T tau, T h, T gamma',
        'T rho_new, T m_new, T e_new',
        '''
            T Pi = (gamma - 1) * (e2[i] - m2[i] * m2[i] / rho2[i] / 2);
            T Pim1 = (gamma - 1) * (e2[i - 1] - m2[i - 1] * m2[i - 1] / rho2[i - 1] / 2);
            T F0i = m2[i];
            T F0im1 = m2[i - 1];
            T F1i = m2[i] * m2[i] / rho2[i] + Pi;
            T F1im1 = m2[i - 1] * m2[i - 1] / rho2[i - 1] + Pim1;
            T F2i = m2[i] / rho2[i] * (e2[i] + Pi);
            T F2im1 = m2[i - 1] / rho2[i - 1] * (e2[i - 1] + Pim1);
            if (i > 0 && i < _ind.size() - 1)
            {
                rho_new = rho - tau / h * (F0i - F0im1);
                m_new = m - tau / h * (F1i - F1im1);
                e_new = e - tau / h * (F2i - F2im1);
            }
        ''',
        'Lax_Wendroff_2nd')
#}}}
