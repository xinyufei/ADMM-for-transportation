import numpy as np
import gurobipy as gb
from data import Network

sample_size = 1
N_edge = 4
# network_data = Network(N_edge, True, sample_size)
network_data = Network(N_edge, True)
N = network_data.N
T = network_data.T
C = network_data.C
O = network_data.O
D = network_data.D
BI = network_data.BI
BO = network_data.BO
I1 = network_data.I1
I2 = network_data.I2
I3 = network_data.I3
I4 = network_data.I4
V = network_data.V
M = network_data.M
proc = network_data.proc
pred = network_data.pred
Demand = network_data.Demand
Jam_N = network_data.Jam_N
Q = network_data.Q
W = network_data.W 
alpha = network_data.alpha
beta = network_data.beta
n_init = network_data.n_init

add=np.zeros(N)
add[0] = 0
# add[0] = int(len(C[0]))
for i in range(1,N):
    add[i] = int(add[i-1] + len(C[i-1]))


C_ALL = list(range(int(add[N-1]+len(C[N-1]))))
O_ALL = O[0]
D_ALL = D[0]
V_ALL = V[0]
M_ALL = M[0]
BI_ALL = BI[0]
BO_ALL = BO[0]
I1_ALL = I1[0]
I2_ALL = I2[0]
I3_ALL = I3[0]
I4_ALL = I4[0]
Demand_ALL = Demand[0]
Q_ALL = Q[0]
Jam_N_ALL = Jam_N[0]
n_init_all = n_init[0]
for i in range(1,N):
    O_ALL = O_ALL + [int(d+add[i]) for d in O[i]]
    D_ALL = D_ALL + [int(d+add[i]) for d in D[i]]
    V_ALL = V_ALL + [int(d+add[i]) for d in V[i]]
    M_ALL = M_ALL + [int(d+add[i]) for d in M[i]]
    BI_ALL = BI_ALL + [int(d+add[i]) for d in BI[i]]
    BO_ALL = BO_ALL + [int(d+add[i]) for d in BO[i]]
    I1_ALL = I1_ALL + [int(d+add[i]) for d in I1[i]]
    I2_ALL = I2_ALL + [int(d+add[i]) for d in I2[i]]
    I3_ALL = I3_ALL + [int(d+add[i]) for d in I3[i]]
    I4_ALL = I4_ALL + [int(d+add[i]) for d in I4[i]]
    Demand_ALL = Demand_ALL + [de for de in Demand[i]]
    Q_ALL = np.hstack((Q_ALL, Q[i]))
    Jam_N_ALL = np.hstack((Jam_N_ALL, Jam_N[i]))
    n_init_all = np.concatenate((n_init_all, n_init[i]), axis=0)
I = [None] * N
for i in range(N):
    I[i] = [None] * 4
    I[i][0] = [int(d+add[i]) for d in I1[i]]
    I[i][1] = [int(d+add[i]) for d in I2[i]]
    I[i][2] = [int(d+add[i]) for d in I3[i]]
    I[i][3] = [int(d+add[i]) for d in I4[i]]

Demand = np.ones((len(O_ALL),T))
proc_all={}
for i in range(N):
    proc_all.update({k+add[i]: proc[i][k]+add[i] for k in proc[i]})

proc_all[8+add[0]] = 0 + add[1]
proc_all[31+add[0]] = 25 + add[N_edge]
proc_all[17+add[N_edge-1]] = 8 + add[N_edge-2]
proc_all[31+add[N_edge-1]] = 25 + add[2*N_edge-1]
proc_all[8+add[(N_edge-1)*(N_edge)]] = 0 + add[(N_edge-1)*N_edge+1]
proc_all[24+add[(N_edge-1)*(N_edge)]] = 18 + add[(N_edge-2)*N_edge]
proc_all[17+add[N_edge*N_edge-1]] = 8 + add[N_edge*N_edge-2]
proc_all[24+add[N_edge*N_edge-1]] = 18 + add[(N_edge-1)*N_edge-1]

proc_all[15+add[1]] = 9 + add[0]
proc_all[7+add[1]] = 0 + add[2]
proc_all[29+add[1]] = 23 + add[1+N_edge]
proc_all[15+add[(N_edge-1)*N_edge+1]] = 9 + add[(N_edge-1)*N_edge]
proc_all[7+add[(N_edge-1)*N_edge+1]] = 0 + add[(N_edge-1)*N_edge+2]
proc_all[22+add[(N_edge-1)*N_edge+1]] = 16 + add[(N_edge-2)*N_edge+1]

for i in range(1,N_edge-1):
    proc_all[8+add[i*N_edge]] = 0 + add[i*N_edge+1]
    proc_all[24+add[i*N_edge]] = 18 + add[(i-1)*N_edge]
    proc_all[31+add[i*N_edge]] = 25 + add[(i+1)*N_edge]
    proc_all[15+add[i*N_edge+1]] = 9 + add[i*N_edge]
    proc_all[7+add[i*N_edge+1]] = 0 + add[i*N_edge+2]
    proc_all[29+add[i*N_edge+1]] = 23 + add[(i+1)*N_edge+1]
    proc_all[22+add[i*N_edge+1]] = 16 + add[(i-1)*N_edge+1]
    proc_all[17+add[(i+1)*N_edge-1]] = 8 + add[(i+1)*N_edge-2]
    proc_all[24+add[(i+1)*N_edge-1]] = 18 + add[i*N_edge-1]
    proc_all[31+add[(i+1)*N_edge-1]] = 25 + add[(i+2)*N_edge-1]

for j in range(2,N_edge-1):
    proc_all[15+add[j]] = 8 + add[j-1]
    proc_all[7+add[j]] = 0 + add[j+1]
    proc_all[29+add[j]] = 23 + add[j+N_edge]
    for i in range(1,N_edge-1):
        proc_all[15+add[i*N_edge+j]] = 8 + add[i*N_edge+j-1]
        proc_all[7+add[i*N_edge+j]] = 0 + add[i*N_edge+j+1]
        proc_all[29+add[i*N_edge+j]] = 23 + add[(i+1)*N_edge+j]
        proc_all[22+add[i*N_edge+j]] = 16 + add[(i-1)*N_edge+j]
    proc_all[15+add[(N_edge-1)*N_edge+j]] = 8 + add[(N_edge-1)*N_edge+j-1]
    proc_all[7+add[(N_edge-1)*N_edge+j]] = 0 + add[(N_edge-1)*N_edge+j+1]
    proc_all[22+add[(N_edge-1)*N_edge+j]] = 16 + add[(N_edge-2)*N_edge+j]


pred_all={}
# pred_all.update(pred[0])
for i in range(N):
    for key, value in pred[i].items():
        if isinstance(value,int):
            pred_all.update({key+add[i]: value+add[i]})
        else:
            pred_all.update({key+add[i]: [v+add[i] for v in value]})

pred_all[9+add[0]] = 15 + add[1]
pred_all[18+add[0]] = 24 + add[N_edge]
pred_all[0+add[N_edge-1]] = 7 + add[N_edge-2]
pred_all[18+add[N_edge-1]] = 24 + add[2*N_edge-1]
pred_all[9+add[(N_edge-1)*(N_edge)]] = 15 + add[(N_edge-1)*N_edge+1]
pred_all[25+add[(N_edge-1)*(N_edge)]] = 31 + add[(N_edge-2)*N_edge]
pred_all[0+add[N_edge*N_edge-1]] = 7 + add[N_edge*N_edge-2]
pred_all[25+add[N_edge*N_edge-1]] = 31 + add[(N_edge-1)*N_edge-1]

pred_all[0+add[1]] = 8 + add[0]
pred_all[8+add[1]] = 15 + add[2]
pred_all[16+add[1]] = 22 + add[1+N_edge]
pred_all[0+add[N_edge-2]] = 7 + add[N_edge-3]
pred_all[8+add[N_edge-2]] = 17 + add[N_edge-1]
pred_all[16+add[N_edge-2]] = 22 + add[2*N_edge-2]
pred_all[0+add[(N_edge-1)*N_edge+1]] = 8 + add[(N_edge-1)*N_edge]
pred_all[8+add[(N_edge-1)*N_edge+1]] = 15 + add[(N_edge-1)*N_edge+2]
pred_all[23+add[(N_edge-1)*N_edge+1]] = 29 + add[(N_edge-2)*N_edge+1]
pred_all[0+add[N_edge*N_edge-2]] = 7 + add[N_edge*N_edge-3]
pred_all[8+add[N_edge*N_edge-2]] = 17 + add[N_edge*N_edge-1]
pred_all[23+add[N_edge*N_edge-2]] = 29 + add[(N_edge-1)*N_edge-2]

for i in range(1,N_edge-1):
    pred_all[9+add[i*N_edge]] = 15 + add[i*N_edge+1]
    pred_all[25+add[i*N_edge]] = 31 + add[(i-1)*N_edge]
    pred_all[18+add[i*N_edge]] = 24 + add[(i+1)*N_edge]
    pred_all[8+add[i*N_edge+1]] = 15 + add[i*N_edge+2]
    pred_all[0+add[i*N_edge+1]] = 8 + add[i*N_edge]
    pred_all[16+add[i*N_edge+1]] = 22 + add[(i+1)*N_edge+1]
    pred_all[23+add[i*N_edge+1]] = 29 + add[(i-1)*N_edge+1]
    pred_all[8+add[(i+1)*N_edge-2]] = 17 + add[(i+1)*N_edge-1]
    pred_all[0+add[(i+1)*N_edge-2]] = 7 + add[(i+1)*N_edge-3]
    pred_all[16+add[(i+1)*N_edge-2]] = 22 + add[(i+2)*N_edge-2]
    pred_all[23+add[(i+1)*N_edge-2]] = 29 + add[i*N_edge-2]
    pred_all[0+add[(i+1)*N_edge-1]] = 7 + add[(i+1)*N_edge-2]
    pred_all[25+add[(i+1)*N_edge-1]] = 31 + add[i*N_edge-1]
    pred_all[18+add[(i+1)*N_edge-1]] = 24 + add[(i+2)*N_edge-1]

for j in range(2,N_edge-2):
    pred_all[0+add[j]] = 7 + add[j-1]
    pred_all[8+add[j]] = 15 + add[j+1]
    pred_all[16+add[j]] = 22 + add[j+N_edge]
    for i in range(1,N_edge-1):
        pred_all[0+add[i*N_edge+j]] = 7 + add[i*N_edge+j-1]
        pred_all[8+add[i*N_edge+j]] = 15 + add[i*N_edge+j+1]
        pred_all[16+add[i*N_edge+j]] = 22 + add[(i+1)*N_edge+j]
        pred_all[23+add[i*N_edge+j]] = 29 + add[(i-1)*N_edge+j]
    pred_all[0+add[(N_edge-1)*N_edge+j]] = 7 + add[(N_edge-1)*N_edge+j-1]
    pred_all[8+add[(N_edge-1)*N_edge+j]] = 15 + add[(N_edge-1)*N_edge+j+1]
    pred_all[23+add[(N_edge-1)*N_edge+j]] = 29 + add[(N_edge-2)*N_edge+j]