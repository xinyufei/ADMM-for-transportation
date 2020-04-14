import numpy as np
import gurobipy as gb
from data import Network
import matplotlib.pyplot as plt
from data_global import *

""" w = np.zeros((N, 4, T))
for i in range(N):
    for j in range(4):
        for t in range(T):
            w[i,j, t] = (t % 25 < 15)
single_signal = w[0,0,:]
print(np.matrix(single_signal))
print(np.shape(np.matrix(single_signal))) """
num_scenario = sample_size
m = gb.Model()
y = m.addVars(len(C_ALL), T, sample_size, lb=0, vtype=gb.GRB.CONTINUOUS)
n = m.addVars(len(C_ALL), T+1, sample_size, lb=0, vtype=gb.GRB.CONTINUOUS)
w = m.addVars(N, 4, T, lb=0, ub=1,  vtype=gb.GRB.BINARY)      #binary variable
# w = m.addVars(N, 4, T, lb=0, ub=1,  vtype=gb.GRB.CONTINUOUS) 
# new_w = m.addVars(N, T, lb=0, ub=1, vtype=gb.GRB.CONTINUOUS)
u = m.addVars(N, T-1, lb=0, ub=1,  vtype=gb.GRB.CONTINUOUS)
""" m.setObjective(1/sample_size * (gb.quicksum(gb.quicksum(gb.quicksum(n[c,t,xi] for c in D_ALL) for t in range(T)) for xi in range(sample_size))
        + alpha * gb.quicksum(gb.quicksum(gb.quicksum(t* y[c,t,xi] for c in list(set(C_ALL)-set(D_ALL))) for t in range(T)) for xi in range(sample_size))), gb.GRB.MINIMIZE) """
m.setObjective(1/num_scenario*(-gb.quicksum(gb.quicksum(gb.quicksum((T - t) * y[c,t,xi] for c in C_ALL) for t in range(T)) for xi in range(sample_size)) -
 alpha*gb.quicksum(gb.quicksum(gb.quicksum(n[c,t,xi] for c in D_ALL) for t in range(T)) for xi in range(sample_size))), gb.GRB.MINIMIZE)

# maximize the y for cell transmission model, minimize total number of vehicles in the network

m.addConstrs((n[c,0,xi] == n_init_all[c,xi] for c in C_ALL for xi in range(sample_size)), name = 'n0')
m.addConstrs(y[c,t,xi]-n[c,t,xi]<=0 for c in C_ALL for t in range(T) for xi in range(sample_size))
m.addConstrs(y[c,t,xi]<=Q[0][0] for c in list(set(C_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(D_ALL)) for t in range(T) for xi in range(sample_size))
for i in range(N):
    m.addConstrs(y[c+add[i],t,xi]<=Q[i][c+1]/beta[i][0,0] for c in V[i] for t in range(T) for xi in range(sample_size))
    m.addConstrs(y[c+add[i],t,xi]<=Q[i][c+2]/beta[i][1,0] for c in V[i] for t in range(T) for xi in range(sample_size))
    m.addConstrs(y[c+add[i],t,xi]<=Q[i][c+3]/beta[i][2,0] for c in V[i] for t in range(T) for xi in range(sample_size))
# m.addConstrs(y[c,t,xi]-w[i,j,t]*Q_ALL[c]<=0 for i in range(N) for j in range(4) for c in I[i][j] for t in range(T) for xi in range(sample_size))
m.addConstrs(y[c,t,xi]-w[i,j,t]*Q_ALL[c]<=0 for i in range(N) for j in range(4) for c in I[i][j] for t in range(T) for xi in range(sample_size))
# m.addConstrs(y[c,t,xi]-w[i,j,t - 1]*Q_ALL[c]<=0 for i in range(N) for j in range(4) for c in I[i][j] for t in range(1,T) for xi in range(sample_size))
m.addConstrs(y[c,t,xi]+W*n[proc_all[c],t,xi]<=W*Jam_N_ALL[c] for c in list(set(C_ALL)-set(D_ALL)-set(V_ALL)) for t in range(T) for xi in range(sample_size))

for i in range(N):
    m.addConstrs(beta[i][0,0]*y[c+add[i],t,xi]+W*n[c+add[i]+1,t,xi]<=W*Jam_N_ALL[int(c+add[i]+1)] for c in V[i] for t in range(T) for xi in range(sample_size))
    m.addConstrs(beta[i][1,0]*y[c+add[i],t,xi]+W*n[c+add[i]+2,t,xi]<=W*Jam_N_ALL[int(c+add[i]+2)] for c in V[i] for t in range(T) for xi in range(sample_size))
    if beta[i].shape[0]==3:
        m.addConstrs(beta[i][2,0]*y[c+add[i],t,xi]+W*n[c+add[i]+3,t,xi]<=W*Jam_N_ALL[int(c+add[i]+3)] for c in V[i] for t in range(T) for xi in range(sample_size))
    m.addConstrs(n[c+add[i],t+1,xi] - n[c+add[i],t,xi] - beta[i][c-pred[i][c]-1,0]*y[pred[i][c]+add[i],t,xi] + y[c+add[i],t,xi] == 0 for c in I1[i]+I2[i]+I3[i]+I4[i] for t in range(T) for xi in range(sample_size))

m.addConstrs(n[c,t+1,xi] - n[c,t,xi] - y[pred_all[c],t,xi] + y[c,t,xi] == 0 for c in list(set(C_ALL)-set(O_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(M_ALL)) for t in range(T) for xi in range(sample_size))
m.addConstrs(n[c,t+1,xi] - n[c,t,xi] - gb.quicksum(y[d,t,xi] for d in pred_all[c]) + y[c,t,xi] == 0 for c in M_ALL for t in range(T) for xi in range(sample_size))
# m.addConstrs(n[c,t+1] - n[c,t] + y[c,t] == Demand[0,0] for c in O_ALL for t in range(T))
m.addConstrs(n[O_ALL[i],t+1,xi] - n[O_ALL[i],t,xi] + y[O_ALL[i],t,xi] == Demand_ALL[i] for i in range(len(O_ALL)) for t in range(T) for xi in range(sample_size))

""" m.addConstrs(new_w[i,t] == w[i,0,t] + w[i,1,t] for i in range(N) for t in range(T))
m.addConstrs(w[i,0,t]+w[i,1,t]+w[i,2,t]+w[i,3,t]<=1 for i in range(N) for t in range(T))
m.addConstrs(new_w[i,t]-new_w[i,t-1]-u[i,t-1]<=0 for i in range(N) for t in range(1,T))
m.addConstrs( -new_w[i,t]+new_w[i,t-1]-u[i,t-1]<=0 for i in range(N) for t in range(1,T))
m.addConstrs(-new_w[i,t]-new_w[i,t-1]+u[i,t-1]<=0 for i in range(N) for t in range(1,T))
m.addConstrs(new_w[i,t]+new_w[i,t-1]+u[i,t-1]<=2 for  i in range(N) for t in range(1,T))
m.addConstrs(u[i,t]+u[i,t+1]+u[i,t+2]+u[i,t+3]<=1 for  i in range(N) for t in range(0,T-4))
m.addConstrs(new_w[i,t]+new_w[i,t+1]+new_w[i,t+2]+new_w[i,t+3]+new_w[i,t+4]+new_w[i,t+5]+new_w[i,t+6]+new_w[i,t+7] <= 8 for  i in range(N) for t in range(0,T-7))
m.addConstrs(-(new_w[i,t]+new_w[i,t+1]+new_w[i,t+2]+new_w[i,t+3]+new_w[i,t+4]+new_w[i,t+5]+new_w[i,t+6]+new_w[i,t+7])<=-1 for  i in range(N) for t in range(0,T-7)) """
m.addConstrs(w[i,0,t]+w[i,1,t]+w[i,2,t]+w[i,3,t]<=1 for i in range(N) for t in range(T))
m.addConstrs(w[i,0,t]+w[i,1,t]-(w[i,0,t-1]+w[i,1,t-1])-u[i,t-1]<=0 for i in range(N) for t in range(1,T))
m.addConstrs(-(w[i,0,t]+w[i,1,t])+w[i,0,t-1]+w[i,1,t-1]-u[i,t-1]<=0 for i in range(N) for t in range(1,T))
m.addConstrs(-(w[i,0,t]+w[i,1,t])-(w[i,0,t-1]+w[i,1,t-1])+u[i,t-1]<=0 for i in range(N) for t in range(1,T))
m.addConstrs(w[i,0,t]+w[i,1,t]+w[i,0,t-1]+w[i,1,t-1]+u[i,t-1]<=2 for  i in range(N) for t in range(1,T))
m.addConstrs(u[i,t]+u[i,t+1]+u[i,t+2]+u[i,t+3]<=1 for  i in range(N) for t in range(0,T-4))
m.addConstrs(w[i,0,t]+w[i,1,t]+w[i,0,t+1]+w[i,1,t+1]+w[i,0,t+2]+w[i,1,t+2]+w[i,0,t+3]+w[i,1,t+3]+w[i,0,t+4]+w[i,1,t+4]+w[i,0,t+5]+w[i,1,t+5]+w[i,0,t+6]+w[i,1,t+6]+w[i,0,t+7]+w[i,1,t+7] <= 8 for  i in range(N) for t in range(0,T-7))
m.addConstrs(-(w[i,0,t]+w[i,1,t]+w[i,0,t+1]+w[i,1,t+1]+w[i,0,t+2]+w[i,1,t+2]+w[i,0,t+3]+w[i,1,t+3]+w[i,0,t+4]+w[i,1,t+4]+w[i,0,t+5]+w[i,1,t+5]+w[i,0,t+6]+w[i,1,t+6]+w[i,0,t+7]+w[i,1,t+7])<=-1 for  i in range(N) for t in range(0,T-7))
# m.addConstrs(n[c,T]==0 for c in C_ALL)
# m.addConstrs(n[c,0] == 1 for c in C_ALL)
m.Params.TimeLimit=7200
m.Params.MIPGap = 1e-3
m.Params.LogFile = 'n_init_random/MIP_Gurobi_T' + str(T) + '_S' + str(num_scenario) + 'test_cons.log'
m.Params.LogToConsole = 1
m.update()
m.optimize()