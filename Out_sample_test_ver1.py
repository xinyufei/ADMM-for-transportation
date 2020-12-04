from data_global import *
from warmup import Warmup
import numpy as np
import gurobipy as gb
import time
import matplotlib.pyplot as plt
from plot import plot_vehicle
from tqdm import tqdm 

def Out_Sample_Test_Fixed(N_edge = 4, num_scenario = 500, T = 600, file_name = "benders/T600_S10_optimal_signal_fixed_length.log"):
    dataset = []
    file = open(file_name, mode='r')
    for line in file:
        line = line.strip('\n')
        line = line.split(',')
        if line[0] != '':
            dataset.append(line)
    file.close()
    for data in dataset:
        if len(data) == 1:
            data[0] = data[0].split(' ')
        else:
            data[4] = data[4].split(' ')
    # print(dataset)

    m_master = [None]*N
    z1=[None]*N
    z2=[None]*N
    e=[None]*N
    b=[None]*N
    l=[None]*N
    o=[None]*N
    g=[None]*N
    z1_tilde = [None]*N
    z2_tilde = [None]*N
    for inter in range(N):
        m_master[inter] = gb.Model()
        z1[inter] = m_master[inter].addVars(4, 2, T, vtype=gb.GRB.BINARY)
        z2[inter] = m_master[inter].addVars(4, 2, T, vtype=gb.GRB.BINARY)
        e[inter] = np.zeros((4, 2))
        b[inter] = np.zeros((4, 2))
        l[inter] = float(dataset[inter*2][4][2])
        o[inter] = float(dataset[inter*2+1][0][1])
        T_cycle = int(np.ceil(l[inter]))
        U = T_cycle*2
        g[inter] = np.zeros(4)
        for p in range(4):
            g[inter][p] = float(dataset[inter*2][p])
        for cy in range(2):
            b[inter][0,cy] = cy*l[inter] - o[inter]
            e[inter][0,cy] = b[inter][0,cy]+g[inter][0]
            b[inter][1,cy] = e[inter][0,cy]
            e[inter][1,cy] = b[inter][1,cy]+g[inter][1]
            b[inter][2,cy] = e[inter][1,cy]
            e[inter][2,cy] = b[inter][2,cy]+g[inter][2]
            b[inter][3,cy] = e[inter][2,cy]
            e[inter][3,cy] = b[inter][3,cy]+g[inter][3]
            m_master[inter].setObjective(0, gb.GRB.MINIMIZE)
            m_master[inter].update()
            m_master[inter].addConstrs(-U*z1[inter][p,cy,t]+epsilon <= t-e[inter][p,cy] for p in range(4) for t in range(T_cycle))
            m_master[inter].addConstrs(-U*z2[inter][p,cy,t] <= b[inter][p,cy]-t for p in range(4) for t in range(T_cycle))
            m_master[inter].addConstrs(t-e[inter][p,cy] - U*(1-z1[inter][p,cy,t]) <= 0 for p in range(4) for t in range(T_cycle))
            m_master[inter].addConstrs(b[inter][p,cy]-t - U*(1-z2[inter][p,cy,t]) <= 0 for p in range(4) for t in range(T_cycle))
            # m_master[inter].addConstrs(z1[p,t]+z2[p,t]-z[p,t] == 1 for p in range(4) for t in range(T))
            m_master[inter].addConstrs(gb.quicksum(z1[inter][p,cy,t]+z2[inter][p,cy,t] for p in range(4)) <= 5 for t in range(T_cycle))
        m_master[inter].optimize()
        z1_tilde[inter] = m_master[inter].getAttr('X', z1[inter])
        z2_tilde[inter] = m_master[inter].getAttr('X', z2[inter])

    network_data = Network(N_edge, True, num_scenario, T)
    n_init_all = Warmup(False)
    Demand_ALL = network_data.Demand[0]
    for i in range(1,N):
        Demand_ALL = Demand_ALL + [de for de in network_data.Demand[i]]
    beta = network_data.beta
    print(num_scenario)
    print(T)

    opt_sub = np.zeros(num_scenario)
    delay_sub = np.zeros(num_scenario)
    ctm_sub = np.zeros(num_scenario)
    for xi in tqdm(range(num_scenario)):
        # solve subproblem
        m_sub = gb.Model()
        y = m_sub.addVars(len(C_ALL), T, lb=0, vtype=gb.GRB.CONTINUOUS)
        n = m_sub.addVars(len(C_ALL), T+1, lb=0, vtype=gb.GRB.CONTINUOUS)
        """ m_sub.setObjective(gb.quicksum(gb.quicksum(t * y[c,t] for c in D_ALL) for t in range(T))
                + alpha * gb.quicksum(gb.quicksum(t* y[c,t] for c in list(set(C_ALL)-set(D_ALL))) for t in range(T)), gb.GRB.MINIMIZE) """
        m_sub.setObjective(-alpha*gb.quicksum(gb.quicksum((T - t) * y[c,t] for c in C_ALL) for t in range(T)) -
                gb.quicksum(gb.quicksum(n[c,t] for c in D_ALL) for t in range(T)), gb.GRB.MINIMIZE)
        m_sub.addConstrs(y[c,t]-gb.quicksum(z1_tilde[i][p,cy,int(t-np.floor(t/l[i])*l[i])]+z2_tilde[i][p,cy,int(t-np.floor(t/l[i])*l[i])]-1 for cy in range(2))*Q_ALL[c] <= 0 for i in range(N) for p in range(4) for c in I[i][p] for t in range(T))
        m_sub.addConstrs(y[c,t]-n[c,t]<=0 for c in C_ALL for t in range(T))
        m_sub.addConstrs(y[c,t]<=Q[0][0] for c in list(set(C_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(D_ALL)) for t in range(T))
        for i in range(N):
            m_sub.addConstrs(y[c+add[i],t]<=Q[i][c+1]/beta[i][xi][t][0,0] for c in V[i] for t in range(T))
            m_sub.addConstrs(y[c+add[i],t]<=Q[i][c+2]/beta[i][xi][t][1,0] for c in V[i] for t in range(T))
            m_sub.addConstrs(y[c+add[i],t]<=Q[i][c+3]/beta[i][xi][t][2,0] for c in V[i] for t in range(T))
        m_sub.addConstrs(y[c,t]+W*n[proc_all[c],t]<=W*Jam_N_ALL[int(proc_all[c])] for c in list(set(C_ALL)-set(D_ALL)-set(V_ALL)) for t in range(T))
        for i in range(N):
            m_sub.addConstrs(beta[i][xi][t][0,0]*y[c+add[i],t]+W*n[c+add[i]+1,t]<=W*Jam_N_ALL[int(c+add[i]+1)] for c in V[i] for t in range(T))
            m_sub.addConstrs(beta[i][xi][t][1,0]*y[c+add[i],t]+W*n[c+add[i]+2,t]<=W*Jam_N_ALL[int(c+add[i]+2)] for c in V[i] for t in range(T))
            m_sub.addConstrs(beta[i][xi][t][2,0]*y[c+add[i],t]+W*n[c+add[i]+3,t]<=W*Jam_N_ALL[int(c+add[i]+3)] for c in V[i] for t in range(T))
            m_sub.addConstrs(n[c+add[i],t+1] - n[c+add[i],t] - beta[i][xi][t][c-pred[i][c]-1,0]*y[pred[i][c]+add[i],t] + y[c+add[i],t] == 0 for c in I1[i]+I2[i]+I3[i]+I4[i] for t in range(T))
        m_sub.addConstrs(n[c,t+1] - n[c,t] - y[pred_all[c],t] + y[c,t] == 0 for c in list(set(C_ALL)-set(O_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(M_ALL)) for t in range(T))
        m_sub.addConstrs(n[c,t+1] - n[c,t] - gb.quicksum(y[d,t] for d in pred_all[c]) + y[c,t] == 0 for c in M_ALL for t in range(T))
        m_sub.addConstrs(n[O_ALL[i],t+1] - n[O_ALL[i],t] + y[O_ALL[i],t] == Demand_ALL[i][xi][t] for i in range(len(O_ALL)) for t in range(T))
        m_sub.addConstrs((n[c,0] == n_init_all[c] for c in C_ALL), name = 'n0')
        m_sub.addConstrs(y[c,t] == 0 for c in D_ALL for t in range(T))
        m_sub.Params.LogToConsole = 0
        # m_sub.Params.TimeLimit=600
        m_sub.Params.LogFile = 'benders/T' + str(T) + '_S' + str(num_scenario) + '_sub_evaluate.log'
        m_sub.Params.InfUnbdInfo = 0
        m_sub.optimize()
        y_value = m_sub.getAttr('X', y)
        n_value = m_sub.getAttr('X', n)
        ctm_sub[xi] = -alpha*sum(sum((T - t) * y_value[c,t] for c in C_ALL) for t in range(T))
        delay_sub[xi] = -sum(sum(n_value[c,t] for c in D_ALL) for t in range(T))
        opt_sub[xi] = m_sub.objval
    
    f = open('benders/T' + str(T) + '_S' + str(num_scenario) + '_out_of_sample_fixed_length.txt', 'w+')
    print("throughput term is %f" % (sum(delay_sub)/num_scenario), file = f)
    print("ctm term is %f" % (sum(ctm_sub)/num_scenario), file = f)
    print("objective value is %f" % (sum(opt_sub)/num_scenario), file = f)
if __name__ == '__main__':
    T = 600
    num_scenario = 2
    file_name = "benders/T600_S10_optimal_signal_fixed_length.log"
    Out_Sample_Test_Fixed(N_edge, num_scenario, T, file_name)