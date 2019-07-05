import numpy as np
import cupy as cp
from cupy import prof

def initial_U(N, gamma):
    #{{{
    p_left = 1.0
    p_right = 0.1
    rho_left = 1.0
    rho_right = 0.125
    v_left = 0.0
    v_right = 0.0
    print(cp.sqrt(4.0))

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

#{{{ Roe_step (ElementwiseKernel)
Roe_step = cp.ElementwiseKernel(
        'raw T rho, raw T m, raw T e, T tau, T h, T gamma',
        'T rho_new, T m_new, T e_new',
        '''
            //T icntl = 0;
            const T vol = 1.0;
            T w0m2 = sqrt(vol * rho[i - 2]);
            T w1m2 = w0m2 * m[i - 2] / rho[i - 2];
            T w3m2 = (gamma - 1) * (e[i - 2] - 0.5 * (m[i - 2] * m[i - 2]) / rho[i - 2]);
            T w2m2 = w0m2 * (e[i - 2] + w3m2) / rho[i - 2];
            T w0m1 = sqrt(vol * rho[i - 1]);
            T w1m1 = w0m1 * m[i - 1] / rho[i - 1];
            T w3m1 = (gamma - 1) * (e[i - 1] - 0.5 * (m[i - 1] * m[i - 1]) / rho[i - 1]);
            T w2m1 = w0m1 * (e[i - 1] + w3m1) / rho[i - 1];
            T w0 = sqrt(vol * rho[i]);
            T w1 = w0 * m[i] / rho[i];
            T w3 = (gamma - 1) * (e[i] - 0.5 * (m[i] * m[i]) / rho[i]);
            T w2 = w0 * (e[i] + w3) / rho[i];
            T w0p1 = sqrt(vol * rho[i + 1]);
            T w1p1 = w0p1 * m[i + 1] / rho[i + 1];
            T w3p1 = (gamma - 1) * (e[i + 1] - 0.5 * m[i + 1] * m[i + 1] / rho[i + 1]);
            T w2p1 = w0p1 * (e[i + 1] + w3p1) / rho[i + 1];
            T w0p2 = sqrt(vol * rho[i + 2]);
            T w1p2 = w0p2 * m[i + 2] / rho[i + 2];
            T w3p2 = (gamma - 1) * (e[i + 2] - 0.5 * m[i + 2] * m[i + 2] / rho[i + 2]);
            T w2p2 = w0p2 * (e[i + 2] + w3p2) / rho[i + 2];
            T Fc0m2 = w0m2 * w1m2;
            T Fc1m2 = w1m2 * w1m2 + vol * w3m2;
            T Fc2m2 = w1m2 * w2m2;
            T Fc0m1 = w0m1 * w1m1;
            T Fc1m1 = w1m1 * w1m1 + vol * w3m1;
            T Fc2m1 = w1m1 * w2m1;
            T Fc0 = w0 * w1;
            T Fc1 = w1 * w1 + vol * w3;
            T Fc2 = w1 * w2;
            T Fc0p1 = w0p1 * w1p1;
            T Fc1p1 = w1p1 * w1p1 + vol * w3p1;
            T Fc2p1 = w1p1 * w2p1;
            T Fc0p2 = w0p2 * w1p2;
            T Fc1p2 = w1p2 * w1p2 + vol * w3p2;
            T Fc2p2 = w1p2 * w2p2;
            T Fl0m1;
            T Fl1m1;
            T Fl2m1;
            T Fr0m1;
            T Fr1m1;
            T Fr2m1;
            if (i > 1)
            {
                Fl0m1 = Fc0m2;
                Fl1m1 = Fc1m2;
                Fl2m1 = Fc2m2;
                Fr0m1 = Fc0m1;
                Fr1m1 = Fc1m1;
                Fr2m1 = Fc2m1;
            }
            T Fl0;
            T Fl1;
            T Fl2;
            T Fr0;
            T Fr1;
            T Fr2;
            if (i > 0)
            {
                Fl0 = Fc0m1;
                Fl1 = Fc1m1;
                Fl2 = Fc2m1;
                Fr0 = Fc0;
                Fr1 = Fc1;
                Fr2 = Fc2;
            }
            T Fl0p1;
            T Fl1p1;
            T Fl2p1;
            T Fr0p1;
            T Fr1p1;
            T Fr2p1;
            if (i < _ind.size() - 1)
            {
                Fl0p1 = Fc0;
                Fl1p1 = Fc1;
                Fl2p1 = Fc2;
                Fr0p1 = Fc0p1;
                Fr1p1 = Fc1p1;
                Fr2p1 = Fc2p1;
            }
            T Fl0p2;
            T Fl1p2;
            T Fl2p2;
            T Fr0p2;
            T Fr1p2;
            T Fr2p2;
            if (i < _ind.size() - 2)
            {
                Fl0p2 = Fc0p1;
                Fl1p2 = Fc1p1;
                Fl2p2 = Fc2p1;
                Fr0p2 = Fc0p2;
                Fr1p2 = Fc1p2;
                Fr2p2 = Fc2p2;
            }
            T fludif0m1;
            T fludif1m1;
            T fludif2m1;
            if (i > 1)
            {
                fludif0m1 = Fr0m1 - Fl0m1;
                fludif1m1 = Fr1m1 - Fl1m1;
                fludif2m1 = Fr2m1 - Fl2m1;
            }
            T fludif0;
            T fludif1;
            T fludif2;
            if (i > 0)
            {
                fludif0 = Fr0 - Fl0;
                fludif1 = Fr1 - Fl1;
                fludif2 = Fr2 - Fl2;
            }
            T fludif0p1;
            T fludif1p1;
            T fludif2p1;
            if (i < _ind.size() - 1)
            {
                fludif0p1 = Fr0p1 - Fl0p1;
                fludif1p1 = Fr1p1 - Fl1p1;
                fludif2p1 = Fr2p1 - Fl2p1;
            }
            T fludif0p2;
            T fludif1p2;
            T fludif2p2;
            if (i < _ind.size() - 2)
            {
                fludif0p2 = Fr0p2 - Fl0p2;
                fludif1p2 = Fr1p2 - Fl1p2;
                fludif2p2 = Fr2p2 - Fl2p2;
            }
            T rsumrm1;
            T utildem1;
            T htildem1;
            T absvtm1;
            T uvdifm1;
            T sscm1;
            T vscm1;
            if (i > 1)
            {
                rsumrm1 = 1 / (w0m2 + w0m1);
                utildem1 = (w1m2 + w1m1) * rsumrm1;
                htildem1 = (w2m2 + w2m1) * rsumrm1;
                absvtm1 = 0.5 * utildem1 * utildem1;
                uvdifm1 = utildem1 * fludif1m1;
                sscm1 = (gamma - 1) * (htildem1 - absvtm1);
                if (sscm1 > 0.0)
                    vscm1 = sqrt(sscm1);
                else
                    vscm1 = sqrt(abs(sscm1));
            }
            T rsumr;
            T utilde;
            T htilde;
            T absvt;
            T uvdif;
            T ssc;
            T vsc;
            if (i > 0)
            {
                rsumr = 1 / (w0m1 + w0);
                utilde = (w1m1 + w1) * rsumr;
                htilde = (w2m1 + w2) * rsumr;
                absvt = 0.5 * utilde * utilde;
                uvdif = utilde * fludif1;
                ssc = (gamma - 1) * (htilde - absvt);
                if (ssc > 0.0)
                    vsc = sqrt(ssc);
                else
                    vsc = sqrt(abs(ssc));
            }
            T rsumrp1;
            T utildep1;
            T htildep1;
            T absvtp1;
            T uvdifp1;
            T sscp1;
            T vscp1;
            if (i < _ind.size() - 1)
            {
                rsumrp1 = 1 / (w0 + w0p1);
                utildep1 = (w1 + w1p1) * rsumrp1;
                htildep1 = (w2 + w2p1) * rsumrp1;
                absvtp1 = 0.5 * utildep1 * utildep1;
                uvdifp1 = utildep1 * fludif1p1;
                sscp1 = (gamma - 1) * (htildep1 - absvtp1);
                if (sscp1 > 0.0)
                    vscp1 = sqrt(sscp1);
                else
                    vscp1 = sqrt(abs(sscp1));
            }
            T rsumrp2;
            T utildep2;
            T htildep2;
            T absvtp2;
            T uvdifp2;
            T sscp2;
            T vscp2;
            if (i < _ind.size() - 2)
            {
                rsumrp2 = 1 / (w0p1 + w0p2);
                utildep2 = (w1p1 + w1p2) * rsumrp2;
                htildep2 = (w2p1 + w2p2) * rsumrp2;
                absvtp2 = 0.5 * utildep2 * utildep2;
                uvdifp2 = utildep2 * fludif1p2;
                sscp2 = (gamma - 1) * (htildep2 - absvtp2);
                if (sscp2 > 0.0)
                    vscp2 = sqrt(sscp2);
                else
                    vscp2 = sqrt(abs(sscp2));
            }
            T eiglam0m1;
            T eiglam1m1;
            T eiglam2m1;
            if (i > 1)
            {
                eiglam0m1 = utildem1 - vscm1;
                eiglam1m1 = utildem1;
                eiglam2m1 = utildem1 + vscm1;
            }
            T eiglam0;
            T eiglam1;
            T eiglam2;
            T sgn0;
            T sgn1;
            T sgn2;
            if (i > 0)
            {
                eiglam0 = utilde - vsc;
                eiglam1 = utilde;
                eiglam2 = utilde + vsc;
                sgn0 = -2 * (eiglam0 < 0.0) + 1.0;
                sgn1 = -2 * (eiglam1 < 0.0) + 1.0;
                sgn2 = -2 * (eiglam2 < 0.0) + 1.0;
            }
            T eiglam0p1;
            T eiglam1p1;
            T eiglam2p1;
            T sgn0p1;
            T sgn1p1;
            T sgn2p1;
            if (i < _ind.size() - 1)
            {
                eiglam0p1 = utildep1 - vscp1;
                eiglam1p1 = utildep1;
                eiglam2p1 = utildep1 + vscp1;
                sgn0p1 = -2 * (eiglam0p1 < 0.0) + 1.0;
                sgn1p1 = -2 * (eiglam1p1 < 0.0) + 1.0;
                sgn2p1 = -2 * (eiglam2p1 < 0.0) + 1.0;
            }
            T eiglam0p2;
            T eiglam1p2;
            T eiglam2p2;
            if (i < _ind.size() - 2)
            {
                eiglam0p2 = utildep2 - vscp2;
                eiglam1p2 = utildep2;
                eiglam2p2 = utildep2 + vscp2;
            }
            const T tiny = 1e-30;
            T a0m1;
            T a1m1;
            T a2m1;
            if (i > 1)
            {
                a0m1 = 0.5 * ((gamma - 1) * (absvtm1 * fludif0m1 + fludif2m1 - uvdifm1) - vscm1 * (fludif1m1 - utildep1 * fludif0m1)) / sscm1;
                a1m1 = (gamma - 1) * ((htildem1 - 2 * absvtm1) * fludif0m1 + uvdifm1 - fludif2m1) / sscm1;
                a2m1 = 0.5 * ((gamma - 1) * (absvtm1 * fludif0m1 + fludif2m1 - uvdifm1) + vscm1 * (fludif1m1 - utildem1 * fludif0m1)) / sscm1;
                a0m1 /= eiglam0m1 + tiny;
                a1m1 /= eiglam1m1 + tiny;
                a2m1 /= eiglam2m1 + tiny;
            }
            T a0;
            T a1;
            T a2;
            if (i > 0)
            {
                a0 = 0.5 * ((gamma - 1) * (absvt * fludif0 + fludif2 - uvdif) - vsc * (fludif1 - utilde * fludif0)) / ssc;
                a1 = (gamma - 1) * ((htilde - 2 * absvt) * fludif0 + uvdif - fludif2) / ssc;
                a2 = 0.5 * ((gamma - 1) * (absvt * fludif0 + fludif2 - uvdif) + vsc * (fludif1 - utilde * fludif0)) / ssc;
                a0 /= eiglam0 + tiny;
                a1 /= eiglam1 + tiny;
                a2 /= eiglam2 + tiny;
            }
            T a0p1;
            T a1p1;
            T a2p1;
            if (i < _ind.size() - 1)
            {
                a0p1 = 0.5 * ((gamma - 1) * (absvtp1 * fludif0p1 + fludif2p1 - uvdifp1) - vscp1 * (fludif1p1 - utildep1 * fludif0p1)) / sscp1;
                a1p1 = (gamma - 1) * ((htildep1 - 2 * absvtp1) * fludif0p1 + uvdifp1 - fludif2p1) / sscp1;
                a2p1 = 0.5 * ((gamma - 1) * (absvtp1 * fludif0p1 + fludif2p1 - uvdifp1) + vscp1 * (fludif1p1 - utildep1 * fludif0p1)) / sscp1;
                a0p1 /= eiglam0p1 + tiny;
                a1p1 /= eiglam1p1 + tiny;
                a2p1 /= eiglam2p1 + tiny;
            }
            T a0p2;
            T a1p2;
            T a2p2;
            if (i < _ind.size() - 2)
            {
                a0p2 = 0.5 * ((gamma - 1) * (absvtp2 * fludif0p2 + fludif2p2 - uvdifp2) - vscp2 * (fludif1p2 - utildep2 * fludif0p2)) / sscp2;
                a1p2 = (gamma - 1) * ((htildep2 - 2 * absvtp2) * fludif0p2 + uvdifp2 - fludif2p2) / sscp2;
                a2p2 = 0.5 * ((gamma - 1) * (absvtp2 * fludif0p2 + fludif2p2 - uvdifp2) + vscp2 * (fludif1p2 - utildep2 * fludif0p2)) / sscp2;
                a0p2 /= eiglam0p2 + tiny;
                a1p2 /= eiglam1p2 + tiny;
                a2p2 /= eiglam2p2 + tiny;
            }
            T ac10;
            T ac11;
            T ac12;
            if (i > 0)
            {
                ac10 = -sgn0 * a0 * eiglam0;
                ac11 = -sgn1 * a1 * eiglam1;
                ac12 = -sgn2 * a2 * eiglam2;
            }
            T ac10p1;
            T ac11p1;
            T ac12p1;
            if (i < _ind.size() - 1)
            {
                ac10p1 = -sgn0p1 * a0p1 * eiglam0p1;
                ac11p1 = -sgn1p1 * a1p1 * eiglam1p1;
                ac12p1 = -sgn2p1 * a2p1 * eiglam2p1;
            }
            T sbpar1 = 2.0;
            T sbpar2 = 2.0;
            T ac20;
            T ac21;
            T ac22;
            if (i > 1 && i < _ind.size() - 1)
            {
                T a0_isb_index = ((sgn0 + 1) / 2 * a0m1 - (sgn0 - 1) / 2 * a0p1);
                T a1_isb_index = ((sgn1 + 1) / 2 * a1m1 - (sgn1 - 1) / 2 * a1p1);
                T a2_isb_index = ((sgn2 + 1) / 2 * a2m1 - (sgn2 - 1) / 2 * a2p1);
                ac20 = (ac10 + eiglam0 *
                    ((max(0.0, min(sbpar1 * a0_isb_index, max(a0, min(a0_isb_index, sbpar2 * a0)))) +
                    min(0.0, max(sbpar1 * a0_isb_index, min(a0, max(a0_isb_index, sbpar2 * a0))))) *
                    (sgn0 - tau / h * eiglam0)));
                ac21 = (ac11 + eiglam1 *
                    ((max(0.0, min(sbpar1 * a1_isb_index, max(a1, min(a1_isb_index, sbpar2 * a1)))) +
                    min(0.0, max(sbpar1 * a1_isb_index, min(a1, max(a1_isb_index, sbpar2 * a1))))) *
                    (sgn1 - tau / h * eiglam1)));
                ac22 = (ac12 + eiglam2 *
                    ((max(0.0, min(sbpar1 * a2_isb_index, max(a2, min(a2_isb_index, sbpar2 * a2)))) +
                    min(0.0, max(sbpar1 * a2_isb_index, min(a2, max(a2_isb_index, sbpar2 * a2))))) *
                    (sgn2 - tau / h * eiglam2)));
            }
            else
            {
                if (i > 0)
                {
                    ac20 = ac10;
                    ac21 = ac11;
                    ac22 = ac12;
                }
            }
            T ac20p1;
            T ac21p1;
            T ac22p1;
            if (i > 0 && i < _ind.size() - 2)
            {
                T a0_isbp1_index = ((sgn0p1 + 1) / 2 * a0 - (sgn0p1 - 1) / 2 * a0p2);
                T a1_isbp1_index = ((sgn1p1 + 1) / 2 * a1 - (sgn1p1 - 1) / 2 * a1p2);
                T a2_isbp1_index = ((sgn2p1 + 1) / 2 * a2 - (sgn2p1 - 1) / 2 * a2p2);
                ac20p1 = (ac10p1 + eiglam0p1 *
                    ((max(0.0, min(sbpar1 * a0_isbp1_index, max(a0p1, min(a0_isbp1_index, sbpar2 * a0p1)))) +
                    min(0.0, max(sbpar1 * a0_isbp1_index, min(a0p1, max(a0_isbp1_index, sbpar2 * a0p1))))) *
                    (sgn0p1 - tau / h * eiglam0p1)));
                ac21p1 = (ac11p1 + eiglam1p1 *
                    ((max(0.0, min(sbpar1 * a1_isbp1_index, max(a1p1, min(a1_isbp1_index, sbpar2 * a1p1)))) +
                    min(0.0, max(sbpar1 * a1_isbp1_index, min(a1p1, max(a1_isbp1_index, sbpar2 * a1p1))))) *
                    (sgn1p1 - tau / h * eiglam1p1)));
                ac22p1 = (ac12p1 + eiglam2p1 *
                    ((max(0.0, min(sbpar1 * a2_isbp1_index, max(a2p1, min(a2_isbp1_index, sbpar2 * a2p1)))) +
                    min(0.0, max(sbpar1 * a2_isbp1_index, min(a2p1, max(a2_isbp1_index, sbpar2 * a2p1))))) *
                    (sgn2p1 - tau / h * eiglam2p1)));
            }
            else
            {
                if (i == 0 || i == _ind.size() - 2)
                {
                    ac20p1 = ac10p1;
                    ac21p1 = ac11p1;
                    ac22p1 = ac12p1;
                }
            }
            T F0;
            T F1;
            T F2;
            if (i > 0)
            {
                F0 = 0.5 * (Fl0 + Fr0 + ac20 + ac21 + ac22);
                F1 = 0.5 * (Fl1 + Fr1 + eiglam0 * ac20 + eiglam1 *
                               ac21 + eiglam2 * ac22);
                F2 = 0.5 * (Fl2 + Fr2 + (htilde - utilde * vsc) * ac20 +
                               absvt * ac21 + (htilde + utilde * vsc) * ac22);
            }
            T F0p1;
            T F1p1;
            T F2p1;
            if (i > 0 && i < _ind.size() - 1)
            {
                F0p1 = 0.5 * (Fl0p1 + Fr0p1 + ac20p1 + ac21p1 + ac22p1);
                F1p1 = 0.5 * (Fl1p1 + Fr1p1 + eiglam0p1 * ac20p1 + eiglam1p1 *
                               ac21p1 + eiglam2p1 * ac22p1);
                F2p1 = 0.5 * (Fl2p1 + Fr2p1 + (htildep1 - utildep1 * vscp1) * ac20p1 +
                               absvtp1 * ac21p1 + (htildep1 + utildep1 * vscp1) * ac22p1);
            }
            T ptest;
            if (i > 0 && i < _ind.size() - 1)
                ptest = (h * vol * m[i] + tau * (F1 - F1p1));
                ptest = (- ptest * ptest + 2 * (h * vol * rho[i] +
                           tau * (F0 - F0p1)) * (h * vol * e[i] + tau * (F2 - F2p1)));
                if (ptest <= 0.0 || (h * vol * rho[i] + tau * (F0 - F0p1)) <= 0.0)
                    //F0 = 0.5 * (Fl0 + Fr0 + ac10 + ac11 + ac12);
                    //F1 = 0.5 * (Fl1 + Fr1 + eiglam0 * ac10 + eiglam1 * ac11 +
                    //    eiglam2 * ac12);
                    //F2 = 0.5 * (Fl2 + Fr2 + (htilde - utilde * vsc) * ac10 +
                    //    absvt * ac11 + (htilde + utilde * vsc) * ac12);
                    F0p1 = 0.5 * (Fl0p1 + Fr0p1 + ac10p1 + ac11p1 + ac12p1);
                    //F1p1 = 0.5 * (Fl1p1 + Fr1p1 + eiglam0p1 * ac10p1 + eiglam1p1 *
                    //     ac11p1 + eiglam2p1 * ac12p1);
                    //F2p1 = 0.5 * (Fl2p1 + Fr2p1 + (htildep1 - utildep1 * vscp1) * ac10p1
                    //     + absvtp1 * ac11p1 + (htildep1 + utildep1 * vscp1) * ac12p1);
                    //ptest = (h * vol * m[i] + tau * (F1 - F1p1));
                    //ptest = (2.0 * (h * vol * rho[i] + tau * (F0 - F0p1)) *
                    //        (h * vol * e[i] + tau * (F2 - F2p1)) - ptest * ptest);
                    //if (ptest <= 0.0 or (h * vol * rho[i] + tau * (F0 - F0p1)) <= 0.0)
                    //    icntl += 1
            //rho_new = ptest;
            //m_new = F1p1;
            //e_new = F2p1;
            if (i > 0 && i < _ind.size() - 1)
            {
                rho_new = rho[i] - tau / h * (F0p1 - F0);
                m_new = m[i] - tau / h * (F1p1 - F1);
                e_new = e[i] - tau / h * (F2p1 - F2);
            }
        ''',
        'Roe_step')
#}}}
