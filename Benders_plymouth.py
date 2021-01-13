from data_global import *
from warmup import Warmup
import numpy as np
import gurobipy as gb
import time
import matplotlib.pyplot as plt 
from plot import plot_vehicle

def Build_master(l_tilde, num_scenario):
    m_master = [None]*N
    z1_pre = [None]*N
    z2_pre = [None]*N
    z1 = [None]*N
    z2 = [None]*N
    e = [None]*N
    b = [None]*N
    g = [None]*N
    l = [None]*N
    theta = [None]*N
    o = [None]*N
    print(T)
    print(num_cycle)
    start_build = time.time()
    for inter in range(N):
        m_master[inter] = gb.Model()
        T_cycle = int(np.ceil(l_tilde[inter]))
        U = T_cycle*2
        z1[inter] = m_master[inter].addVars(4,2,T_cycle, vtype=gb.GRB.BINARY)
        z2[inter] = m_master[inter].addVars(4,2,T_cycle, vtype=gb.GRB.BINARY)
        # z = m_master[inter].addVars(4, T, vtype=gb.GRB.BINARY)
        e[inter] = m_master[inter].addVars(4,2)
        b[inter] = m_master[inter].addVars(4,2)
        for p in range(4):
            b[inter][p,0].lb = -np.infty
        g[inter] = m_master[inter].addVars(4, lb = 2, ub = 25)
        # l[inter]  = m_master[inter].addVar()
        theta[inter] = m_master[inter].addVars(num_scenario, lb=0)
        o[inter] = m_master[inter].addVar()
        m_master[inter].setObjective(1/num_scenario*gb.quicksum(theta[inter][xi] for xi in range(num_scenario)), gb.GRB.MINIMIZE)
        m_master[inter].update()
        for t in range(T_cycle):
            m_master[inter].addConstrs(-U*z1[inter][p,cy,t]+epsilon <= t-e[inter][p,cy] for p in range(4) for cy in range(2))
            m_master[inter].addConstrs(-U*z2[inter][p,cy,t] <= b[inter][p,cy]-t for p in range(4) for cy in range(2))
            m_master[inter].addConstrs(t-e[inter][p,cy] <= U*(1-z1[inter][p,cy,t]) for p in range(4) for cy in range(2))
            m_master[inter].addConstrs(b[inter][p,cy]-t <= U*(1-z2[inter][p,cy,t]) for p in range(4) for cy in range(2))
            m_master[inter].addConstrs(z1[inter][p,cy,t]+z2[inter][p,cy,t] >= 1 for p in range(4) for cy in range(2))
            m_master[inter].addConstrs(gb.quicksum(z1[inter][p,cy,t]+z2[inter][p,cy,t] for p in range(4)) <= 5 for cy in range(2))
        m_master[inter].addConstr(o[inter] <= l_tilde[inter])
        m_master[inter].addConstrs(b[inter][0,cy] == l_tilde[inter]*cy - o[inter] for cy in range(2))
        m_master[inter].addConstrs(e[inter][0,cy] == b[inter][0,cy] + g[inter][0] for cy in range(2))
        m_master[inter].addConstrs(b[inter][1,cy] == e[inter][0,cy] for cy in range(2))
        m_master[inter].addConstrs(e[inter][1,cy] == b[inter][1,cy] + g[inter][1] for cy in range(2))
        m_master[inter].addConstrs(b[inter][2,cy] == e[inter][1,cy] for cy in range(2))
        m_master[inter].addConstrs(e[inter][2,cy] == b[inter][2,cy] + g[inter][2] for cy in range(2))
        m_master[inter].addConstrs(b[inter][3,cy] == e[inter][2,cy] for cy in range(2))
        m_master[inter].addConstrs(e[inter][3,cy] == b[inter][3,cy] + g[inter][3] for cy in range(2))
        m_master[inter].addConstr(gb.quicksum(g[inter][p] for p in range(4)) == l_tilde[inter])
        m_master[inter].Params.LogToConsole = 1
        m_master[inter].Params.TimeLimit = 7200
        m_master[inter].Params.LogFile = 'Plymouth/T' + str(T) + '_S' + str(num_scenario) + '_master.log'
        m_master[inter].Params.LazyConstraints = 1
        m_master[inter].update()
        # m_master[inter].optimize()
    # initialized integer variables (fist-stage variables)
    # w_tilde = np.zeros((N,4,T))
    # theta_tilde = np.ones((N,num_scenario))*(-np.infty)
    end_build = time.time()
    print(end_build - start_build)
    master_model = [m_master, z1, z2, e, b, g, theta, o]
    return master_model

def Benders(epsilon = 0.01, num_scenario = 100, T = 20):
    ub = np.infty
    lb = -np.infty
    f = open('Plymouth/T' + str(T) + '_S' + str(num_scenario) + '_bound_LP_fixed_length_decentralize.log', 'w+')
    f = open('Plymouth/T' + str(T) + '_S' + str(num_scenario) + '_w_fixed_length_decentralize.log', 'w+')
    f = open('Plymouth/T' + str(T) + '_S' + str(num_scenario) + '_y_fixed_length_decentralize.log', 'w+')
    f = open('Plymouth/T' + str(T) + '_S' + str(num_scenario) + '_n_fixed_length_decentralize.log', 'w+')
    f = open('debug.txt', 'w+')
    # generalize all samples
    network_data = Network(N_edge, True, num_scenario, T)
    n_init_all = Warmup(False)
    Demand_ALL = network_data.Demand[0]
    for i in range(1,N):
        Demand_ALL = Demand_ALL + [de for de in network_data.Demand[i]]
    beta = network_data.beta
    num_cycle = network_data.num_cycle
    U = network_data.U
    Demand = network_data.Demand
    # theta_tilde = np.ones(num_scenario)*m_init.objval
    # build master model
    l_tilde = np.zeros(N)
    for i in range(N):
        # l_tilde[i] = 20/network_data.Y
        l_tilde[i] = 40
    T_cycle = int(np.ceil(l_tilde[0]))
    [m_master, z1, z2, e, b, g, theta, o] = Build_master(l_tilde, num_scenario)

    l_optimal = np.ones(N)
    for i in range(N):
        l_optimal[i] = 40
    o_optimal = np.zeros(N)
    g_optimal = np.zeros((N,4))
    z1_optimal = np.zeros((N,4,num_cycle,T))
    z2_optimal = np.zeros((N,4,num_cycle,T))

    z1_tilde = [None]*N
    z2_tilde = [None]*N
    theta_tilde = [None]*N
    g_tilde = [None]*N
    o_tilde = [None]*N
    e_tilde = [None]*N
    start_global = time.time()
    start_ite = time.time()
    start_master = time.time()
    for i in range(N):
        same = True
        m_master[i].optimize()
        z1_tilde[i] = m_master[i].getAttr('X', z1[i])
        z2_tilde[i] = m_master[i].getAttr('X', z2[i])
        g_tilde[i] = m_master[i].getAttr('X', g[i])
        # l_tilde[i] = l[i].x
        o_tilde[i] = o[i].x
        e_tilde[i] = m_master[i].getAttr('X', e[i])
        theta_tilde[i] = np.ones(num_scenario)*(-np.infty)
        for xi in range(num_scenario):
            theta[i][xi].lb = -np.infty
    end_master = time.time()

   # build subproblem
    m_sub = [None]*num_scenario
    n= [None]*num_scenario
    y = [None]*num_scenario
    n_value= [None]*num_scenario
    y_value = [None]*num_scenario
    n_opt= [None]*num_scenario
    y_opt = [None]*num_scenario
    cons1 = [None]*num_scenario
    cons1_1 = [None]*num_scenario
    cons2 = [None]*num_scenario
    cons3 = [None]*num_scenario
    cons4 = [None]*num_scenario
    cons5 = [None]*num_scenario
    cons6 = [None]*num_scenario
    cons7 = [None]*num_scenario
    cons8 = [None]*num_scenario
    cons9 = [None]*num_scenario
    cons10 = [None]*num_scenario
    cons11 = [None]*num_scenario
    for xi in range(num_scenario):
        m_sub[xi] = gb.Model()
        y[xi] = m_sub[xi].addVars(len(C_ALL), T, lb=-np.infty, vtype=gb.GRB.CONTINUOUS)
        n[xi] = m_sub[xi].addVars(len(C_ALL), T+1, lb=-np.infty, vtype=gb.GRB.CONTINUOUS)
        m_sub[xi].setObjective(-alpha*gb.quicksum(gb.quicksum((T - t) * y[xi][c,t] for c in C_ALL) for t in range(T)) -
                gb.quicksum(gb.quicksum(n[xi][c,t] for c in D_ALL) for t in range(T)), gb.GRB.MINIMIZE)
        cons1[xi] = m_sub[xi].addConstrs(y[xi][c,t] <= gb.quicksum(z1_tilde[i][p,cy,int(t-np.floor(t/l_tilde[i])*l_tilde[i])]+z2_tilde[i][p,cy,int(t-np.floor(t/l_tilde[i])*l_tilde[i])]-1 for cy in range(2))*Q_ALL[c] for i in range(N) for p in range(4) for c in I[i][p] for t in range(T))
        cons1_1[xi] = m_sub[xi].addConstrs(y[xi][c,t] <= Q_ALL[c] for c in C_ALL for t in range(T))
        cons2[xi] = m_sub[xi].addConstrs(y[xi][c,t]-n[xi][c,t]<=0 for c in C_ALL for t in range(T))
        cons3[xi] = m_sub[xi].addConstrs(y[xi][c+add[i],t]<=Q_ALL[proc_all[int(c+add[i])]] for i in range(N) for c in list(set(C[i])-set(I1[i])-set(I2[i])-set(I3[i])-set(I4[i])-set(D[i])) for t in range(T))
        # cons4[xi] = [None]*3*N
        cons4[xi] = [None]*N
        for i in range(N):
            cons4[xi][i] = [None]*2
            if beta[i][xi][0].shape[0] == 3:
                cons4[xi][i] = [None]*3
                cons4[xi][i][2] = m_sub.addConstrs(y[c+add[i],t]<=Q[i][c+3]/beta[i][xi][t][2,V[i].index(c)] for c in V[i] for t in range(T))
            cons4[xi][i][0] = m_sub.addConstrs(y[c+add[i],t]<=Q[i][c+1]/beta[i][xi][t][0,V[i].index(c)] for c in V[i] for t in range(T))
            cons4[xi][i][1] = m_sub.addConstrs(y[c+add[i],t]<=Q[i][c+2]/beta[i][xi][t][1,V[i].index(c)] for c in V[i] for t in range(T))
        """ for i in range(N):
            cons4[xi][i*3] = m_sub[xi].addConstrs(y[xi][c+add[i],t]<=Q[i][c+1]/beta[i][xi][t][0,0] for c in V[i] for t in range(T))
            cons4[xi][i*3+1] = m_sub[xi].addConstrs(y[xi][c+add[i],t]<=Q[i][c+2]/beta[i][xi][t][1,0] for c in V[i] for t in range(T))
            cons4[xi][i*3+2] = m_sub[xi].addConstrs(y[xi][c+add[i],t]<=Q[i][c+3]/beta[i][xi][t][2,0] for c in V[i] for t in range(T)) """
        cons5[xi] = m_sub[xi].addConstrs(y[xi][c+add[i],t]+W*n[xi][proc_all[c+add[i]],t]<=W*Jam_N_ALL[int(proc_all[int(c+add[i])])] for i in range(N) for c in list(set(C[i])-set(D[i])-set(V[i])) for t in range(T))
        # cons6[xi] = [None]*4*N
        cons6[xi] = [None]*N
        for i in range(N):
            cons6[xi][i] = [None]*3
            if beta[i][xi][0].shape[0] == 3:
                cons6[xi][i] = [None]*4
                cons6[xi][i][3] = m_sub.addConstrs(beta[i][xi][t][2,V[i].index(c)]*y[c+add[i],t]+W*n[c+add[i]+3,t]<=W*Jam_N_ALL[int(c+add[i]+3)] for c in V[i] for t in range(T))
            cons6[xi][i][1] = m_sub.addConstrs(beta[i][xi][t][0,V[i].index(c)]*y[c+add[i],t]+W*n[c+add[i]+1,t]<=W*Jam_N_ALL[int(c+add[i]+1)] for c in V[i] for t in range(T))
            cons6[xi][i][2] = m_sub.addConstrs(beta[i][xi][t][1,V[i].index(c)]*y[c+add[i],t]+W*n[c+add[i]+2,t]<=W*Jam_N_ALL[int(c+add[i]+2)] for c in V[i] for t in range(T))
            cons6[xi][i][0] = m_sub.addConstrs(n[c+add[i],t+1] - n[c+add[i],t] - beta[i][xi][t][c-pred[i][c]-1,V[i].index(pred[i][c])]*y[pred[i][c]+add[i],t] + y[c+add[i],t] == 0 for c in I1[i]+I2[i]+I3[i]+I4[i] for t in range(T))
            """ cons6[xi][i*4] = m_sub[xi].addConstrs(beta[i][xi][t][0,0]*y[xi][c+add[i],t]+W*n[xi][c+add[i]+1,t]<=W*Jam_N_ALL[int(c+add[i]+1)] for c in V[i] for t in range(T))
            cons6[xi][i*4+1] = m_sub[xi].addConstrs(beta[i][xi][t][1,0]*y[xi][c+add[i],t]+W*n[xi][c+add[i]+2,t]<=W*Jam_N_ALL[int(c+add[i]+2)] for c in V[i] for t in range(T))
            cons6[xi][i*4+2] = m_sub[xi].addConstrs(beta[i][xi][t][2,0]*y[xi][c+add[i],t]+W*n[xi][c+add[i]+3,t]<=W*Jam_N_ALL[int(c+add[i]+3)] for c in V[i] for t in range(T))
            cons6[xi][i*4+3] = m_sub[xi].addConstrs(n[xi][c+add[i],t+1] - n[xi][c+add[i],t] - beta[i][xi][t][c-pred[i][c]-1,0]*y[xi][pred[i][c]+add[i],t] + y[xi][c+add[i],t] == 0 for c in I1[i]+I2[i]+I3[i]+I4[i] for t in range(T)) """
        cons7[xi] = m_sub[xi].addConstrs(n[xi][c,t+1] - n[xi][c,t] - y[xi][pred_all[c],t] + y[xi][c,t] == 0 for c in list(set(C_ALL)-set(O_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(M_ALL)) for t in range(T))
        cons8[xi] = m_sub[xi].addConstrs(n[xi][c,t+1] - n[xi][c,t] - gb.quicksum(y[xi][d,t] for d in pred_all[c]) + y[xi][c,t] == 0 for c in M_ALL for t in range(T))
        cons9[xi] = m_sub[xi].addConstrs(n[xi][int(O[i][c]+add[i]),t+1] - n[xi][int(O[i][c]+add[i]),t] + y[xi][int(O[i][c]+add[i]),t] == Demand[i][c][xi][t] for i in range(N) for c in range(len(O[i])) for t in range(T))
        cons10[xi] = m_sub[xi].addConstrs((n[xi][c,0] == n_init_all[c] for c in C_ALL), name = 'n0')
        cons11[xi] = m_sub[xi].addConstrs(y[xi][c,t] == 0 for c in D_ALL for t in range(T))
        m_sub[xi].Params.LogToConsole = 1
        m_sub[xi].Params.TimeLimit=7200
        m_sub[xi].Params.LogFile = 'Plymouth/T' + str(T) + '_S' + str(num_scenario) + '_sub.log'
        m_sub[xi].Params.InfUnbdInfo = 1
        m_sub[xi].update()
        # print(cons1[xi][-1].lhs)
    ub_all = np.infty
    ub_all_array = []
    num_ite_all = 0
    I_delay = np.zeros(N)
    start_global = time.time()
    while 1:
        num_ite_all += 1
        if num_ite_all > 1:
            # l_tilde = [None]*N
            # update l_tilde
            for i in range(N):
                if I_delay[i] == 0:
                    l_tilde[i] -= 1
                else:
                    l_tilde[i] += I_delay[i]/2
            # for i in range(N):
            #     l_tilde[i] = 20
            [m_master, z1, z2, e, b, g, theta, o] = Build_master(l_tilde, num_scenario)
            start_ite = time.time()
            start_master = time.time()
            z1_tilde = [None]*N
            z2_tilde = [None]*N
            theta_tilde = [None]*N
            g_tilde = [None]*N
            o_tilde = [None]*N
            e_tilde = [None]*N
            for i in range(N):
                same = True
                m_master[i].optimize()
                z1_tilde[i] = m_master[i].getAttr('X', z1[i])
                z2_tilde[i] = m_master[i].getAttr('X', z2[i])
                g_tilde[i] = m_master[i].getAttr('X', g[i])
                # l_tilde[i] = l[i].x
                o_tilde[i] = o[i].x
                e_tilde[i] = m_master[i].getAttr('X', e[i])
                theta_tilde[i] = np.ones(num_scenario)*(-np.infty)
                for xi in range(num_scenario):
                    theta[i][xi].lb = -np.infty
            end_master = time.time()
        num_ite = 0
        gap = []
        throughput = []
        obj_array = []
        lb_array = []
        ub_array = []
        time_sub = []
        time_master = []
        time_ite = []
        m_master_obj_array = np.ones(N)*(-np.infty)
        m_master_obj = -np.infty
        while 1:
            start_ite = time.time()
            num_ite = num_ite + 1
            num_optimal_sub = 0
            delay_sub = np.zeros((N,num_scenario))
            ctm_sub = np.zeros((N,num_scenario))
            opt_sub = np.zeros((N,num_scenario))
            lb_sub = np.zeros(num_scenario)
            if num_ite > 1:
                start_ite = time.time()
                start_master = time.time()
                m_master_obj = 0
                for i in range(N):
                    # z1[i].start = z1_opt[i]
                    # z2[i].start = z2_opt[i]
                    m_master[i].update()
                    m_master[i].printStats()
                    m_master[i].optimize()
                    print(m_master[i].status)
                    if m_master[i].status == 2:
                        z1_tilde[i] = m_master[i].getAttr('X',z1[i])
                        z2_tilde[i] = m_master[i].getAttr('X',z2[i])
                        g_tilde[i] = m_master[i].getAttr('X', g[i])
                        o_tilde[i] = o[i].x
                        e_tilde[i] = m_master[i].getAttr('X', e[i])
                        theta_tilde[i] = m_master[i].getAttr('X',theta[i])
                        m_master_obj_array[i] = m_master[i].objval
                m_master_obj = sum(m_master_obj_array)
                end_master = time.time()
            start_sub = time.time()
            # m_sub[xi].printStats()
            num_constr = 0
            for xi in range(num_scenario):
                m_sub[xi].remove(cons1[xi])
                cons1[xi] = m_sub[xi].addConstrs(y[xi][c,t] <= gb.quicksum(z1_tilde[i][p,cy,int(t-np.floor(t/l_tilde[i])*l_tilde[i])]+z2_tilde[i][p,cy,int(t-np.floor(t/l_tilde[i])*l_tilde[i])]-1 for cy in range(2))*Q_ALL[c] for i in range(N) for p in range(4) for c in I[i][p] for t in range(T))
                m_sub[xi].optimize()
                if m_sub[xi].status == 3:
                    m_sub[xi].computeIIS()
                    m_sub[xi].write("model1.ilp")
                    f = open('debug.txt','a+')
                    print('infeasible!', file = f)
                if m_sub[xi].status == 2:
                    num_optimal_sub = num_optimal_sub + 1
                    n_value[xi] = m_sub[xi].getAttr('X',n[xi])
                    y_value[xi] = m_sub[xi].getAttr('X',y[xi])
                    for i in range(N):
                        opt_sub[i,xi] = -sum(sum(n_value[xi][c+add[i],t] for c in D[i]) for t in range(T)) - alpha*sum(sum((T-t)* y_value[xi][c+add[i],t] for c in C[i]) for t in range(T))
                        ctm_sub[i,xi] = -alpha*sum(sum((T - t) * y_value[xi][c,t] for c in C[i]) for t in range(T))
                        delay_sub[i,xi] = -sum(sum(n_value[xi][c+add[i],t] for c in D[i]) for t in range(T))
                    lb_sub[xi] = m_sub[xi].objval
                    constant = [0]*N
                    scope = [None]*N
                    for i in range(N):
                        scope[i] = gb.quicksum(-gb.quicksum(z1[i][p,cy,int(t-np.floor(t/l_tilde[i])*l_tilde[i])]+z2[i][p,cy,int(t-np.floor(t/l_tilde[i])*l_tilde[i])] for cy in range(2))*Q_ALL[c]*cons1[xi][i,p,c,t].Pi for p in range(4) for c in I[i][p] for t in range(T))
                        constant[i] = constant[i] - gb.quicksum(Q_ALL[c]*2*cons1[xi][i,p,c,t].Pi for p in range(4) for c in I[i][p] for t in range(T))
                        """ constant[i] = constant[i] + gb.quicksum(Q[i][0]*cons3[xi][i,c,t].Pi for c in list(set(C[i])-set(I1[i])-set(I2[i])-set(I3[i])-set(I4[i])-set(D[i])) for t in range(T))
                        constant[i] = constant[i] + gb.quicksum(Q[i][c+1]/beta[i][xi][t][0,0]*cons4[xi][i*3][c,t].Pi + Q[i][c+2]/beta[i][xi][t][1,0]*cons4[xi][i*3+1][c,t].Pi + Q[i][c+3]/beta[i][xi][t][2,0]*cons4[xi][i*3+2][c,t].Pi for c in V[i] for t in range(T))
                        constant[i] = constant[i] + gb.quicksum(W*Jam_N_ALL[int(proc_all[int(c+add[i])])]*cons5[xi][i,c,t].Pi for c in list(set(C[i])-set(D[i])-set(V[i])) for t in range(T))
                        constant[i] = constant[i] + gb.quicksum(W*Jam_N_ALL[int(c+add[i]+1)]*cons6[xi][i*4][c,t].Pi + W*Jam_N_ALL[int(c+add[i]+2)]*cons6[xi][i*4+1][c,t].Pi + W*Jam_N_ALL[int(c+add[i]+3)]*cons6[xi][i*4+2][c,t].Pi for c in V[i] for t in range(T))
                        constant[i] = constant[i] + gb.quicksum(Demand[i][c][xi][t]*cons9[xi][i,c,t].Pi for c in range(len(O[i])) for t in range(T))
                        constant[i] = constant[i] + gb.quicksum(n_init_all[int(c+add[i])]*cons10[xi][int(c+add[i])].Pi for c in C[i]) """
                        constant[i] = constant[i] + gb.quicksum(Q_ALL[c]*cons1_1[xi][c,t].Pi for c in C_ALL for t in range(T))
                        constant[i] = constant[i] + gb.quicksum(Q_ALL[proc_all[c]]*cons3[xi][c,t].Pi for c in list(set(C_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(D_ALL)) for t in range(T))
                        constant[i] = constant[i] + gb.quicksum(Q[i][c+1]/beta[i][xi][t][0,V[i].index(c)]*cons4[xi][i][0][c,t].Pi + Q[i][c+2]/beta[i][xi][t][1,V[i].index(c)]*cons4[xi][i][1][c,t].Pi for c in V[i] for t in range(T))
                        if beta[i][xi][0].shape[0] == 3:
                            constant[i] = constant[i] + gb.quicksum(Q[i][c+3]/beta[i][xi][t][2,V[i].index(c)]*cons4[xi][i][2][c,t].Pi for c in V[i] for t in range(T))
                        constant[i] = constant[i] + gb.quicksum(W*Jam_N_ALL[int(proc_all[int(c+add[i])])]*cons5[xi][i,c,t].Pi for c in list(set(C[i])-set(D[i])-set(V[i])) for t in range(T))
                        constant[i] = constant[i] + gb.quicksum(W*Jam_N_ALL[int(c+add[i]+1)]*cons6[xi][i][1][c,t].Pi + W*Jam_N_ALL[int(c+add[i]+2)]*cons6[xi][i][2][c,t].Pi for c in V[i] for t in range(T))
                        if beta[i][xi][0].shape[0] == 3:
                            constant[i] = constant[i] + gb.quicksum(W*Jam_N_ALL[int(c+add[i]+3)]*cons6[xi][i][3][c,t].Pi for c in V[i] for t in range(T))
                        constant[i] = constant[i] + gb.quicksum(Demand_ALL[o_index+sum(len(O[j]) for j in range(i))][xi][t]*cons9[xi][o_index+sum(len(O[j]) for j in range(i)),t].Pi for o_index in range(len(O[i])) for t in range(T))
                        constant[i] = constant[i] + gb.quicksum(n_init_all[int(c+add[i])]*cons10[xi][int(c+add[i])].Pi for c in C[i])
                        if theta_tilde[i][xi] < opt_sub[i,xi]:
                            # theta[i][xi] = m_master[i].getVarByName('theta'+str(i)+'['+str(xi)+']')
                            # m_master[i].addConstr(theta[i][xi] >= constant[i] - scope[i])
                            add_const = m_master[i].addConstr(theta[i][xi] >= constant[i] - scope[i])
                            add_const.Lazy = 1
                        # add_const = m_master[i].addConstrs(y_value[c,t] - gb.quicksum(z1[i][p,cy,t]+z2[i][p,cy,t]-1 for cy in range(2))*Q_ALL[c] <= Q_ALL[c] for p in range(4) for c in I[i][p] for t in range(T))
                        # add_const.Lazy = 1

            end_sub = time.time()
                
            if lb < m_master_obj:
                lb = m_master_obj
            if num_ite>0:
                lb_array.append(m_master_obj)
            if ub > sum(lb_sub)/num_scenario:
                ub = sum(lb_sub)/num_scenario
                for i in range(N):
                    l_optimal[i] = l_tilde[i]
                    o_optimal[i] = o_tilde[i]
                    for p in range(4):
                        g_optimal[i,p] = g_tilde[i][p]
                        for cy in range(2):
                            for t in range(T_cycle):
                                z1_optimal[i,p,cy,t] = z1_tilde[i][p,cy,t]
                                z2_optimal[i,p,cy,t] = z2_tilde[i][p,cy,t]
                #compute delay of intersection cell 
                """ for i in range(N):
                    for xi in range(num_scenario):
                        delay = [0]*4
                        for p in range(4):
                            if e_tilde[i][p,0] >= 0:
                                t_delay = np.floor(e_tilde[i][p,0])
                            else:
                                t_delay = np.floor(e_tilde[i][p,1]) 
                            num_cycle = 0
                            while np.floor(t_delay) < T:
                                num_cycle += 1
                                for c in I[i][p]:
                                    delay[p] += n_value[xi][c,int(np.floor(t_delay))]
                                t_delay += l_tilde[i]
                            delay[p] = delay[p]/num_cycle
                    I_delay[i] = sum(delay)/num_scenario """
            """ if ub_all > ub:
                ub_all = ub
                g_optimal = g_tilde
                l_optimal = l_tilde
                o_optimal = o_tilde
                z1_optimal = z1_tilde
                z2_optimal = z2_tilde """
            # ub_array.append(ub)
            start_plot = time.time()
            y_value_average = np.zeros((len(C_ALL), T))
            n_value_average = np.zeros((len(C_ALL), T))
            for c in C_ALL:
                for t in range(T):
                    y_value_average[c,t] = sum(y_value[xi][c,t] for xi in range(num_scenario))/num_scenario
                    n_value_average[c,t] = sum(n_value[xi][c,t] for xi in range(num_scenario))/num_scenario
            plot_vehicle(num_ite, T, num_scenario, "fixed_length_decentralize", add, len(C_ALL), n_value_average, y_value_average)
            end_plot = time.time()   
            end_global = time.time()
            time_sub.append(end_sub-start_sub)
            time_master.append(end_master-start_master)
            time_ite.append(end_global-start_ite)
            gap.append((ub-lb)/abs(lb))
            obj_array.append(sum(lb_sub)/num_scenario)
            ub_array.append(ub)
            print("iteration " + str(num_ite))
            print("upper bound is ", ub)
            print("lower bound is ", lb)
            print("gap is ", ((ub-lb)/abs(lb)))
            print("time to solve master problem is ", (end_master - start_master))
            print("time to solve sub problem is ", (end_sub - start_sub))
            print("time for this iteration is  ", (end_global-start_ite))
            print("all time to solve problem is ", (end_global - start_global))
            f = open('Plymouth/T' + str(T) + '_S' + str(num_scenario) + '_bound_LP_fixed_length_decentralize.log', 'a+')
            print("iteration " + str(num_ite), file = f)
            print("upper bound is " , ub, file = f)
            print("lower bound is " , lb, file = f)
            print("gap is ", ((ub-lb)/abs(lb)), file = f)
            print("throughput term is ", (sum(delay_sub)/num_scenario), file = f)
            print("ctm term is ", (sum(ctm_sub)/num_scenario), file = f)
            print("time to solve master problem is ", (end_master - start_master), file = f)
            print("time to solve sub problem is ", (end_sub - start_sub), file = f)
            print("time to plot the graph is ", (end_plot - start_plot), file = f)
            print("time for this iteration is ", (end_global-start_ite), file = f)
            print("all time to solve problem is ", (end_global - start_global), file = f)
            f = open('Plymouth/T' + str(T) + '_S' + str(num_scenario) + '_signal_fixed_length_decentralize.log', 'a+')
            print("iteration " + str(num_ite), file = f)
            for i in range(N):
                for p in range(4):
                    print(g_tilde[i][p], end = ",", file=f)
                print("cycle length", l_tilde[i], file=f)
                print("offset", o_tilde[i], file=f)
                print("\n", file=f)
            f = open('Plymouth/T' + str(T) + '_S' + str(num_scenario) + '_binary_fixed_length_decentralize.log', 'a+')


            if num_ite == 20:
                break
        ub_all_array.append(ub_all)
        """ f = open('Plymouth/T' + str(T) + '_S' + str(num_scenario) + '_bound_LP_fixed_length_decentralize.log', 'a+')
        print("all iteration " + str(num_ite_all), file = f)
        print("best upper bound is %f" % ub_all, file = f) """
        if len(ub_all_array) == 1:
            break
        """ if ub_all_array >= 3:
            if abs(ub_all_array[-1] - ub_all_array[-2]) < 0.01*abs(ub_all_array[-1]) and abs(ub_all_array[-3] - ub_all_array[-2]) < 0.01*abs(ub_all_array[-2]):
                break """
    f = open('Plymouth/T' + str(T) + '_S' + str(num_scenario) + '_optimal_signal_fixed_length_decentralize.log', 'a+')
    for i in range(N):
        for p in range(4):
            print(g_optimal[i,p], end = ",", file=f)
        print("cycle length", l_optimal[i], file=f)
        print("offset", o_optimal[i], file = f)
        print("\n", file=f)
    f.close()
        # evaluate solution
    eval_delay = np.zeros(num_scenario)
    eval_ctm = np.zeros(num_scenario)
    for xi in range(num_scenario):
        m_eval = gb.Model()
        y = m_eval.addVars(len(C_ALL), T, lb=0, vtype=gb.GRB.CONTINUOUS)
        n = m_eval.addVars(len(C_ALL), T+1, lb=0, vtype=gb.GRB.CONTINUOUS)
        m_eval.setObjective(-gb.quicksum(gb.quicksum((T - t) * y[c,t] for c in C_ALL) for t in range(T)), gb.GRB.MINIMIZE)
        m_eval.addConstrs(y[c,t]-gb.quicksum(z1_optimal[i][p,cy,int(t-np.floor(t/l_optimal[i])*l_optimal[i])]+z2_optimal[i][p,cy,int(t-np.floor(t/l_optimal[i])*l_optimal[i])]-1 for cy in range(2))*Q_ALL[c]<=0 for i in range(N) for p in range(4) for c in I[i][p] for t in range(T))
        m_eval.addConstrs(y[c,t]-n[c,t]<=0 for c in C_ALL for t in range(T))
        m_eval.addConstrs(y[c,t]<=Q[0][0] for c in list(set(C_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(D_ALL)) for t in range(T))
        for i in range(N):
            m_eval.addConstrs(y[c+add[i],t]<=Q[i][c+1]/beta[i][xi][t][0,0] for c in V[i] for t in range(T))
            m_eval.addConstrs(y[c+add[i],t]<=Q[i][c+2]/beta[i][xi][t][1,0] for c in V[i] for t in range(T))
            m_eval.addConstrs(y[c+add[i],t]<=Q[i][c+3]/beta[i][xi][t][2,0] for c in V[i] for t in range(T))
        m_eval.addConstrs(y[c,t]+W*n[proc_all[c],t]<=W*Jam_N_ALL[int(proc_all[c])] for c in list(set(C_ALL)-set(D_ALL)-set(V_ALL)) for t in range(T))
        for i in range(N):
            m_eval.addConstrs(beta[i][xi][t][0,0]*y[c+add[i],t]+W*n[c+add[i]+1,t]<=W*Jam_N_ALL[int(c+add[i]+1)] for c in V[i] for t in range(T))
            m_eval.addConstrs(beta[i][xi][t][1,0]*y[c+add[i],t]+W*n[c+add[i]+2,t]<=W*Jam_N_ALL[int(c+add[i]+2)] for c in V[i] for t in range(T))
            m_eval.addConstrs(beta[i][xi][t][2,0]*y[c+add[i],t]+W*n[c+add[i]+3,t]<=W*Jam_N_ALL[int(c+add[i]+3)] for c in V[i] for t in range(T))
            m_eval.addConstrs(n[c+add[i],t+1] - n[c+add[i],t] - beta[i][xi][t][c-pred[i][c]-1,0]*y[pred[i][c]+add[i],t] + y[c+add[i],t] == 0 for c in I1[i]+I2[i]+I3[i]+I4[i] for t in range(T))
        m_eval.addConstrs(n[c,t+1] - n[c,t] - y[pred_all[c],t] + y[c,t] == 0 for c in list(set(C_ALL)-set(O_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(M_ALL)) for t in range(T))
        m_eval.addConstrs(n[c,t+1] - n[c,t] - gb.quicksum(y[d,t] for d in pred_all[c]) + y[c,t] == 0 for c in M_ALL for t in range(T))
        m_eval.addConstrs(n[O_ALL[i],t+1] - n[O_ALL[i],t] + y[O_ALL[i],t] == Demand_ALL[i][xi][t] for i in range(len(O_ALL)) for t in range(T))
        m_eval.addConstrs((n[c,0] == n_init_all[c] for c in C_ALL))
        m_eval.addConstrs(y[c,t] == 0 for c in D_ALL for t in range(T))
        m_eval.optimize()
        n_value = m_eval.getAttr('X', n)
        y_value = m_eval.getAttr('X', y)
        eval_delay[xi] = -sum(sum(n_value[c,t] for c in D_ALL) for t in range(T))
        eval_ctm[xi] = m_eval.objval
    throughput.append(sum(eval_delay)/num_scenario)
    print("the throughput is ", (sum(eval_delay)/num_scenario))
    print("the ctm objective value is ", (sum(eval_ctm)/num_scenario))
    f = open('Plymouth/T' + str(T) + '_S' + str(num_scenario) + '_bound_LP_fixed_length_decentralize.log', 'a+')
    print("the throughput is ", (sum(eval_delay)/num_scenario), file = f)
    print("the ctm objective value is ", (sum(eval_ctm)/num_scenario), file = f)
    # draw the figure of time
    plt.figure()
    # plt.plot(time_sub, label = 'time of sub problem')
    # plt.plot(time_master, label = 'time of master problem')
    # plt.plot(time_ite, label = 'time of iteration')
    plt.plot(obj_array, label = 'objective value')
    plt.xlabel('iterations')
    plt.ylabel('objective value')
    plt.legend()
    plt.savefig('Plymouth/T' + str(T) + '_S' + str(num_scenario) + '_obj_fixed_length_decentralize.jpg')

    plt.figure()
    plt.plot(ub_array, label = 'throughput')
    plt.xlabel('iterations')
    plt.ylabel('upper bound')
    plt.legend()
    plt.savefig('Plymouth/T' + str(T) + '_S' + str(num_scenario) + '_ub_fixed_length_decentralize.jpg')

    plt.figure()
    plt.plot(lb_array, label = 'lb')
    plt.xlabel('iterations')
    plt.ylabel('lower bound')
    plt.legend()
    plt.savefig('Plymouth/T' + str(T) + '_S' + str(num_scenario) + '_lb_fixed_length_decentralize.jpg')

    return obj_array, ub_array, lb_array
    
if __name__ == '__main__':
    T = 600
    num_scenario = 10
    obj_array, ub_array, lb_array = Benders(0.0001, num_scenario, T)
    f1 = open('Plymouth/T' + str(T) + '_S' + str(num_scenario) + '_obj_fixed_length_decentralize.txt', 'w+')
    f2 = open('Plymouth/T' + str(T) + '_S' + str(num_scenario) + '_ub_fixed_length_decentralize.txt', 'w+')
    f3 = open('Plymouth/T' + str(T) + '_S' + str(num_scenario) + '_lb_fixed_length_decentralize.txt', 'w+')
    for num_ite in range(len(obj_array)):
        print(str(obj_array[num_ite]), file = f1)
    for num_ite in range(len(ub_array)):
        print(str(ub_array[num_ite]), file = f2)
    for num_ite in range(len(lb_array)):
        print(str(lb_array[num_ite]), file = f3)
    