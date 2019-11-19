import numpy as np
import gurobipy as gb
from import_data import *

I1_ALL = []
I2_ALL = []
I3_ALL = []
I4_ALL = []
for i in range(N):
    for k in range(len(I_ALL[i][0])):
        I1_ALL.append(I_ALL[i][0][k])
    for k in range(len(I_ALL[i][1])):
        I2_ALL.append(I_ALL[i][1][k])
    for k in range(len(I_ALL[i][2])):
        I3_ALL.append(I_ALL[i][2][k])
    for k in range(len(I_ALL[i][3])):
        I4_ALL.append(I_ALL[i][3][k])

w = np.zeros((N,4,T))
w[:,1,:] = 1
for t in range(int(T/60)):
    w[:,1,t*60:t*60+30]=1
    w[:,1,t*60+30:(t+1)*60]=0

m = gb.Model()
y = m.addVars(len(C_ALL), T, lb=0, vtype=gb.GRB.CONTINUOUS)
n = m.addVars(len(C_ALL), T+1, lb=0, vtype=gb.GRB.CONTINUOUS)
# m.setObjective(gb.quicksum(gb.quicksum(t * y[c,t] for c in D_ALL) for t in range(T))
#        + alpha * gb.quicksum(gb.quicksum(t* y[c,t] for c in list(set(C_ALL)-set(D_ALL))) for t in range(T)), gb.GRB.MINIMIZE)
m.setObjective(alpha * gb.quicksum(gb.quicksum((T-t) * y[c,t] for c in list(set(C_ALL)-set(D_ALL))) for t in range(T)), gb.GRB.MINIMIZE)

m.addConstrs(y[c,t]-n[c,t]<=0 for c in C_ALL for t in range(T))
m.addConstrs(y[c,t]<=Q[c] for c in list(set(C_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(D_ALL)) for t in range(T))
m.addConstrs(y[c,t]<=Q[c]/beta[c,0] for c in V_ALL for t in range(T))
m.addConstrs(y[c,t]<=Q[c]/beta[c,1] for c in V_ALL for t in range(T))
m.addConstrs(y[c,t]<=Q[c]/beta[c,2] for c in V_ALL for t in range(T))
m.addConstrs(y[c,t]-w[i,j,t]*Q[c]<=0 for i in range(N) for j in range(4) for c in I_ALL[i][j] for t in range(T))
m.addConstrs(y[c,t]+W*n[proc_all[c][0],t]<=W*jam[c] for c in list(set(C_ALL)-set(D_ALL)-set(V_ALL)-set(DU_ALL)) for t in range(T))

m.addConstrs(beta[c,0]*y[c,t]+W*n[c+1,t]<=W*jam[c+1] for c in V_ALL for t in range(T))
m.addConstrs(beta[c,1]*y[c,t]+W*n[c+2,t]<=W*jam[c+2] for c in V_ALL for t in range(T))
m.addConstrs(beta[c,2]*y[c,t]+W*n[c+3,t]<=W*jam[c+3] for c in V_ALL for t in range(T))
m.addConstrs(n[c,t+1] - n[c,t] - beta[int(pred_all[c][0]),int(c-pred_all[c][0]-1)]*y[int(pred_all[c][0]),t] + y[c,t] == 0 for c in I1_ALL+I2_ALL+I3_ALL+I4_ALL for t in range(T))

m.addConstrs(n[c,t+1] - n[c,t] - y[pred_all[c][0],t] + y[c,t] == 0 for c in list(set(C_ALL)-set(O_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(M_ALL)) for t in range(T))
m.addConstrs(n[c,t+1] - n[c,t] - gb.quicksum(y[d,t] for d in pred_all[c]) + y[c,t] == 0 for c in M_ALL for t in range(T))
m.addConstrs(n[c,t+1] - n[c,t] + y[c,t] == demand[c] for c in O_ALL for t in range(T))
m.addConstrs(n[c,0] == 0 for c in C_ALL)
# m.addConstrs(n[c,0] == 1 for c in C_ALL)

m.Params.TimeLimit=7200
m.Params.LogFile = "log/LP_fix_Gurobi_T240.log"
m.Params.LogToConsole = 0
m.optimize()
opt = m.objval

n_value = m.getAttr('X', n)
result_n = np.zeros((len(C_ALL), T))
for i in range(len(C_ALL)):
    for t in range(T):
        result_n[i, t] = n_value[i, t]

f = open('log/n.log', 'w+')
for i in range(len(C_ALL)):
    for t in range(T):
        print(result_n[i, t], end=",", file=f)
    print("\n", file=f)
f.close()



