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

""" w = np.zeros((N,4,T))
w[:,1,:] = 1
for t in range(int(T/60)):
    w[:,1,t*60:t*60+30]=1
    w[:,1,t*60+30:(t+1)*60]=0 """
print(T)
m = gb.Model()
y = m.addVars(len(C_ALL), T, lb=0, vtype=gb.GRB.CONTINUOUS)
n = m.addVars(len(C_ALL), T+1, lb=0, vtype=gb.GRB.CONTINUOUS)
w = m.addVars(N, 4, T, lb=0, ub=1,  vtype=gb.GRB.BINARY)      #binary variable
# w = m.addVars(N, 4, T, lb=0, ub=1,  vtype=gb.GRB.CONTINUOUS) 
new_w = m.addVars(N, T, lb=0, ub=1, vtype=gb.GRB.CONTINUOUS)
u = m.addVars(N, T-1, lb=0, ub=1,  vtype=gb.GRB.CONTINUOUS)
m.setObjective(gb.quicksum(gb.quicksum(t * y[c,t] for c in D_ALL) for t in range(T))
       + alpha * gb.quicksum(gb.quicksum((T-t) * y[c,t] for c in list(set(C_ALL)-set(D_ALL))) for t in range(T)), gb.GRB.MINIMIZE)
# m.setObjective(alpha * gb.quicksum(gb.quicksum((T-t) * y[c,t] for c in list(set(C_ALL)-set(D_ALL))) for t in range(T)), gb.GRB.MINIMIZE)

m.addConstrs(y[c,t]-n[c,t]<=0 for c in C_ALL for t in range(T))
m.addConstrs(y[c,t]<=Q[c] for c in list(set(C_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(D_ALL)) for t in range(T))
m.addConstrs(y[c,t]<=Q[c+1]/beta[c,0] for c in V_ALL for t in range(T))
m.addConstrs(y[c,t]<=Q[c+2]/beta[c,1] for c in V_ALL for t in range(T))
m.addConstrs(y[c,t]<=Q[c+3]/beta[c,2] for c in V_ALL for t in range(T))
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

m.addConstrs(new_w[i,t] == w[i,0,t] + w[i,1,t] for i in range(N) for t in range(T))
m.addConstrs(w[i,0,t]+w[i,1,t]+w[i,2,t]+w[i,3,t]==1 for i in range(N) for t in range(T))
m.addConstrs(new_w[i,t]-new_w[i,t-1]-u[i,t-1]<=0 for i in range(N) for t in range(1,T))
m.addConstrs( -new_w[i,t]+new_w[i,t-1]-u[i,t-1]<=0 for i in range(N) for t in range(1,T))
m.addConstrs(-new_w[i,t]-new_w[i,t-1]+u[i,t-1]<=0 for i in range(N) for t in range(1,T))
m.addConstrs(new_w[i,t]+new_w[i,t-1]+u[i,t-1]<=2 for  i in range(N) for t in range(1,T))
m.addConstrs(u[i,t]+u[i,t+1]+u[i,t+2]+u[i,t+3]<=1 for  i in range(N) for t in range(0,T-4))
m.addConstrs(new_w[i,t]+new_w[i,t+1]+new_w[i,t+2]+new_w[i,t+3]+new_w[i,t+4]+new_w[i,t+5]+new_w[i,t+6]+new_w[i,t+7] <= 8 for i in range(N) for t in range(0,T-7))
m.addConstrs(-(new_w[i,t]+new_w[i,t+1]+new_w[i,t+2]+new_w[i,t+3]+new_w[i,t+4]+new_w[i,t+5]+new_w[i,t+6]+new_w[i,t+7])<=-1 for i in range(N) for t in range(0,T-7))
# m.addConstrs(n[c,0] == 1 for c in C_ALL)

m.Params.TimeLimit=7200
m.Params.LogFile = "log/LP_test_Gurobi_T8.log"
m.Params.LogToConsole = 0
m.optimize()
opt = m.objval

n_value = m.getAttr('X', n)
result_n = np.zeros((len(C_ALL), T))
for i in range(len(C_ALL)):
    for t in range(T):
        result_n[i, t] = n_value[i, t]

f = open('log/n_opt.log', 'w+')
for i in range(len(C_ALL)):
    for t in range(T):
        print(result_n[i, t], end=",", file=f)
    print("\n", file=f)
f.close()

"""w_value = m.getAttr('X', w)
result_w = np.zeros((N, 4, T))
for i in range(N):
    for j in range(4):
        for t in range(T):
            result_w[i, j, t] = w_value[i, j, t]

f = open('log/w.log', 'w+')
for i in range(N):
    for j in range(4):
        for t in range(T):
            print(result_w[i, j, t], end=",", file=f)
    print("\n", file=f)
f.close()"""











