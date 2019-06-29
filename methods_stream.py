#'''
def check(U, N, n, i, label):
    for j in range(n):
        print("U[" + str(N + j) + "][" + str(i) + "] = " + str(U[N + j][i]))
#'''

def out_file(U, plot, file_name, t, gamma=1.4):
    """ write solution in plot files and print averages"""
    file = open(str(plot) + "-" + file_name, "w")
    rho_avg = 0.0
    v_avg = 0.0
    e_avg = 0.0
    P_avg = 0.0
    N = len(U)
    for j in range(N):
        rho = U[j][0]
        e = U[j][2]
        v = -1 # it means unknown
        P = -1 # it means unknown
        if U[j][0] != 0:
            v = U[j][1] / U[j][0]
            P = (U[j][2] - U[j][1] * U[j][1] / U[j][0] / 2) * (gamma - 1)
            rho_avg += rho
            v_avg += v
            e_avg += e
            P_avg += P
        else:
            print('rho is zero at j= ', j)
        file.write(str(j) + '\t' + str(rho) + '\t' + str(v) + '\t'
                   + str(e) + '\t' + str(P) + '\n')
    if rho_avg != 0.0:
        rho_avg /= N
    if v_avg != 0.0:
        v_avg /= N
    if e_avg != 0.0:
        e_avg /= N
    if P_avg != 0.0:
        P_avg /= N
    print(" t: \t" + str(t) + "  rho_avg: " + str(rho_avg) + "  v_avg: " + str(v_avg) + "  e_avg: " + str(e_avg) +  "  P_avg: " + str(P_avg))
    file.close()
