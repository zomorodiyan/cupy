'''
def check(U): print(f"U[0][0] = {U[0][0]}", end="  ")
    print(f"U[1][0] = {U[1][0]}", end="  ")
    print(f"U[2][0] = {U[2][0]}", end="  ")
    print(f"U[3][0] = {U[2][0]}", end="  ")
    print(f"U[4][0] = {U[3][0]}")
    print(f"U[-1][0] = {U[-1][0]}", end="  ")
    print(f"U[-2][0] = {U[-2][0]}", end="  ")
    print(f"U[-3][0] = {U[-3][0]}", end="  ")
    print(f"U[-4][0] = {U[-4][0]}", end="  ")
    print(f"U[-5][0] = {U[-5][0]}", end="\n")
'''

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
        v = U[j][1] / U[j][0]
        e = U[j][2]
        P = (U[j][2] - U[j][1] * U[j][1] / U[j][0] / 2) * (gamma - 1)
        rho_avg += rho
        v_avg += v
        e_avg += e
        P_avg += P
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
    print(" ", t, '\t', rho_avg, '\t', v_avg, '\t', e_avg, '\t', P_avg)
    file.close()
