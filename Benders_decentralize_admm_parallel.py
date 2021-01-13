# from data_plymouth import *
from data_global import *
from warmup import Warmup
import numpy as np
import gurobipy as gb
import time
import matplotlib.pyplot as plt 

def Benders(epsilon = 0.01, num_scenario = 100, T = 20):
    ub = np.infty
    lb = -np.infty
    f = open('H_4/T' + str(T) + '_S' + str(num_scenario) + '_bound_LP_alpha0.001_grid_admm.log', 'w+')
    f = open('H_4/T' + str(T) + '_S' + str(num_scenario) + '_w_alpha0.001_grid_admm.log', 'w+')
    f = open('H_4/T' + str(T) + '_S' + str(num_scenario) + '_y_alpha0.001_grid_admm.log', 'w+')
    f = open('H_4/T' + str(T) + '_S' + str(num_scenario) + '_n_alpha0.001_grid_admm.log', 'w+')
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
    l_tilde = [None]*N
    for i in range(N):
        l_tilde[i] = 20
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
    U = 2*T
    start_build = time.time()
    for inter in range(N):
        m_master[inter] = gb.Model()
        z1[inter] = m_master[inter].addVars(4, 2, T, vtype=gb.GRB.BINARY)
        z2[inter] = m_master[inter].addVars(4, 2, T, vtype=gb.GRB.BINARY)
        # z = m_master[inter].addVars(4, T, vtype=gb.GRB.BINARY)
        e[inter] = m_master[inter].addVars(4, num_cycle)
        b[inter] = m_master[inter].addVars(4, num_cycle)
        b[inter][0,0].lb = -np.infty
        g[inter] = m_master[inter].addVars(4, lb = 3, ub = 8)
        l[inter]  = m_master[inter].addVar()
        theta[inter] = m_master[inter].addVars(num_scenario, lb=0)
        o[inter] = m_master[inter].addVar()
        m_master[inter].setObjective(1/num_scenario*gb.quicksum(theta[inter][xi] for xi in range(num_scenario)), gb.GRB.MINIMIZE)
        m_master[inter].update()
        for t in range(T):
            cycle_l = max(int(np.floor(t/l_tilde[inter])),0)
            cycle_l = min(cycle_l, num_cycle-1)
            cycle_u = min(int(np.floor(t/l_tilde[inter])+1), num_cycle-1)
            m_master[inter].addConstrs(-U*z1[inter][p,cy-cycle_l,t]+epsilon <= t-e[inter][p,cy] for p in range(4) for cy in range(cycle_l, cycle_u+1))
            m_master[inter].addConstrs(-U*z2[inter][p,cy-cycle_l,t] <= b[inter][p,cy]-t for p in range(4) for cy in range(cycle_l, cycle_u+1))
            m_master[inter].addConstrs(t-e[inter][p,cy] <= U*(1-z1[inter][p,cy-cycle_l,t]) for p in range(4) for cy in range(cycle_l, cycle_u+1))
            m_master[inter].addConstrs(b[inter][p,cy]-t <= U*(1-z2[inter][p,cy-cycle_l,t])-epsilon for p in range(4) for cy in range(cycle_l, cycle_u+1))
            m_master[inter].addConstrs(z1[inter][p,cy,t]+z2[inter][p,cy,t] >= 1 for p in range(4) for cy in range(2))
            m_master[inter].addConstrs(gb.quicksum(z1[inter][p,cy,t]+z2[inter][p,cy,t] for p in range(4)) <= 5 for cy in range(2))
        m_master[inter].addConstr(o[inter] <= l[inter])
        m_master[inter].addConstrs(b[inter][0,cy] == l[inter]*cy - o[inter] for cy in range(num_cycle))
        m_master[inter].addConstrs(e[inter][0,cy] == b[inter][0,cy] + g[inter][0] for cy in range(num_cycle))
        m_master[inter].addConstrs(b[inter][1,cy] == e[inter][0,cy] for cy in range(num_cycle))
        m_master[inter].addConstrs(e[inter][1,cy] == b[inter][1,cy] + g[inter][1] for cy in range(num_cycle))
        m_master[inter].addConstrs(b[inter][2,cy] == e[inter][1,cy] for cy in range(num_cycle))
        m_master[inter].addConstrs(e[inter][2,cy] == b[inter][2,cy] + g[inter][2] for cy in range(num_cycle))
        m_master[inter].addConstrs(b[inter][3,cy] == e[inter][2,cy] for cy in range(num_cycle))
        m_master[inter].addConstrs(e[inter][3,cy] == b[inter][3,cy] + g[inter][3] for cy in range(num_cycle))
        m_master[inter].addConstr(gb.quicksum(g[inter][p] for p in range(4)) == l[inter])
        m_master[inter].Params.LogToConsole = 1
        m_master[inter].Params.TimeLimit = 7200
        m_master[inter].Params.LogFile = 'H_4/T' + str(T) + '_S' + str(num_scenario) + '_master.log'
        m_master[inter].Params.LazyConstraints = 1
        m_master[inter].update()
        # m_master[inter].optimize()
    # initialized integer variables (fist-stage variables)
    # w_tilde = np.zeros((N,4,T))
    # theta_tilde = np.ones((N,num_scenario))*(-np.infty)
    end_build = time.time()
    print(end_build - start_build)
    z1_tilde = [None]*N
    z2_tilde = [None]*N
    theta_tilde = [None]*N
    g_tilde = [None]*N
    l_tilde = [None]*N
    o_tilde = [None]*N
    start_global = time.time()
    start_ite = time.time()
    start_master = time.time()
    for i in range(N):
        m_master[i].optimize()
        z1_tilde[i] = m_master[i].getAttr('X', z1[i])
        z2_tilde[i] = m_master[i].getAttr('X', z2[i])
        g_tilde[i] = m_master[i].getAttr('X', g[i])
        l_tilde[i] = l[i].x
        o_tilde[i] = o[i].x
        theta_tilde[i] = np.ones(num_scenario)*(-np.infty)
        for xi in range(num_scenario):
            theta[i][xi].lb = -np.infty
        z1_pre[i] = z1_tilde[i]
        z2_pre[i] = z2_tilde[i]
    end_master = time.time()
    # build subproblem
    m_sub = [None]*num_scenario
    m_sub_aux = [None]*num_scenario
    n = [None]*num_scenario
    y = [None]*num_scenario
    s = [None]*num_scenario
    n_value = [None]*num_scenario
    y_value = [None]*num_scenario
    s_value = [None]*num_scenario
    aux_n = [None]*num_scenario
    aux_y = [None]*num_scenario
    aux_n_value = [None]*num_scenario
    aux_y_value = [None]*num_scenario
    mlambda = [None]*num_scenario
    mnu_y_1 = [None]*num_scenario
    mnu_n_1 = [None]*num_scenario
    mnu_y_2 = [None]*num_scenario
    mnu_n_2 = [None]*num_scenario
    mmu = [None]*num_scenario
    cons1 = [None]*num_scenario
    cons2 = [None]*num_scenario
    cons3 = [None]*num_scenario
    cons4 = [None]*num_scenario
    cons5 = [None]*num_scenario
    cons5_b = [None]*num_scenario
    cons6 = [None]*num_scenario
    cons7 = [None]*num_scenario
    cons7_b = [None]*num_scenario
    cons8 = [None]*num_scenario
    cons9 = [None]*num_scenario
    cons10 = [None]*num_scenario
    cons11 = [None]*num_scenario
    rho_y = N*T
    rho_n = N*T
    rho_cons5 = N*T
    rho_cons7 = N*T
    for xi in range(num_scenario):
        m_sub[xi] = [None]*N
        m_sub_aux[xi] = [None]*N
        y[xi] = [None]*N
        n[xi] = [None]*N
        s[xi] = [None]*N
        y_value[xi] = [None]*N
        n_value[xi] = [None]*N
        s_value[xi] = [None]*N
        aux_n[xi] = [None]*N
        aux_y[xi] = [None]*N
        aux_y_value[xi] = [None]*N
        aux_n_value[xi] = [None]*N
        mlambda[xi] = [None]*N
        mnu_y_1[xi] = [None]*N
        mnu_n_1[xi] = [None]*N
        mnu_y_2[xi] = [None]*N
        mnu_n_2[xi] = [None]*N
        mmu[xi] = [None]*N
        cons1[xi] = [None]*N
        cons2[xi] = [None]*N
        cons3[xi] = [None]*N
        cons4[xi] = [None]*N
        cons5[xi] = [None]*N
        cons5_b[xi] = [None]*N
        cons6[xi] = [None]*N
        cons7[xi] = [None]*N
        cons7_b[xi] = [None]*N
        cons8[xi] = [None]*N
        cons9[xi] = [None]*N
        cons10[xi] = [None]*N
        cons11[xi] = [None]*N
        for i in range(N):
            aux_y_value[xi][i] = np.ones((len(BI[i]),T))
            aux_n_value[xi][i] = np.ones((len(BO[i]),T))
            # y_value[xi][i] = np.ones((len(C[i]),T))
            # n_value[xi][i] = np.ones((len(C[i]),T))
            mnu_y_1[xi][i] = np.zeros((len(BI[i]),T))
            mnu_y_2[xi][i] = np.zeros((len(BO[i]),T))
            mnu_n_1[xi][i] = np.zeros((len(BO[i]),T+1))
            mnu_n_2[xi][i] = np.zeros((len(BI[i]),T+1))
            mmu[xi][i] = np.zeros((len(BI[i]),T))
        for i in range(N):
            m_sub[xi][i] = gb.Model()
            y[xi][i] = m_sub[xi][i].addVars(C[i], T, lb=0, vtype=gb.GRB.CONTINUOUS)
            n[xi][i] = m_sub[xi][i].addVars(C[i], T+1, lb=0, vtype=gb.GRB.CONTINUOUS)
            # aux_y[xi][i] = m_sub[xi][i].addVars(len(BI[i]), T, lb=0, vtype=gb.GRB.CONTINUOUS)
            # aux_n[xi][i] = m_sub[xi][i].addVars(len(BO[i]), T+1, lb=0, vtype=gb.GRB.CONTINUOUS)
            s[xi][i] = m_sub[xi][i].addVars(len(BO[i]), T, lb=0, vtype=gb.GRB.CONTINUOUS)
            """ m_sub[xi][i].setObjective(-alpha*gb.quicksum(gb.quicksum((T - t) * y[xi][i][c,t] for c in C[i]) for t in range(T))
                - gb.quicksum(gb.quicksum(n[xi][i][c,t] for c in D[i]) for t in range(T))
                + rho*gb.quicksum(gb.quicksum(mnu_y_1[xi][i][BI[i].index(c),t]*(y_value[xi][BI_IN[i][BI[i].index(c)]][int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]]),t]-aux_y[xi][i][BI[i].index(c),t]) for c in BI[i]) for t in range(T))
                + rho*gb.quicksum(gb.quicksum(mnu_y_2[xi][i][BO[i].index(c),t]*(y[xi][i][c,t]-aux_y_value[xi][BO_IN[i][BO[i].index(c)]][BI[BO_IN[i][BO[i].index(c)]].index(int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])),t]) for c in BO[i]) for t in range(T))
                + rho*gb.quicksum(gb.quicksum(mnu_n_1[xi][i][BO[i].index(c),t]*(n_value[xi][BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]]),t]-aux_n[xi][i][BO[i].index(c),t]) for c in BO[i]) for t in range(T))
                + rho*gb.quicksum(gb.quicksum(mnu_n_2[xi][i][BI[i].index(c),t]*(n[xi][i][c,t]-aux_n_value[xi][BI_IN[i][BI[i].index(c)]][BO[BI_IN[i][BI[i].index(c)]].index(int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]])),t]) for c in BI[i]) for t in range(T))
                + rho/2*gb.quicksum(gb.quicksum((y_value[xi][BI_IN[i][BI[i].index(c)]][int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]]),t]-aux_y[xi][i][BI[i].index(c),t])*(y_value[xi][BI_IN[i][BI[i].index(c)]][int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]]),t]-aux_y[xi][i][BI[i].index(c),t]) for c in BI[i]) for t in range(T))
                + rho/2*gb.quicksum(gb.quicksum((y[xi][i][c,t]-aux_y_value[xi][BO_IN[i][BO[i].index(c)]][BI[BO_IN[i][BO[i].index(c)]].index(int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])),t])*(y[xi][i][c,t]-aux_y_value[xi][BO_IN[i][BO[i].index(c)]][BI[BO_IN[i][BO[i].index(c)]].index(int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])),t]) for c in BO[i]) for t in range(T))
                + rho/2*gb.quicksum(gb.quicksum((n_value[xi][BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]]),t]-aux_n[xi][i][BO[i].index(c),t])*(n_value[xi][BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]]),t]-aux_n[xi][i][BO[i].index(c),t]) for c in BO[i]) for t in range(T))
                + rho/2*gb.quicksum(gb.quicksum((n[xi][i][c,t]-aux_n_value[xi][BI_IN[i][BI[i].index(c)]][BO[BI_IN[i][BI[i].index(c)]].index(int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]])),t])*(n[xi][i][c,t]-aux_n_value[xi][BI_IN[i][BI[i].index(c)]][BO[BI_IN[i][BI[i].index(c)]].index(int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]])),t]) for c in BI[i]) for t in range(T))) """
            cons1[xi][i] = m_sub[xi][i].addConstrs(y[xi][i][int(c-add[i]),t] <= gb.quicksum(z1_tilde[i][p,cy,t]+z2_tilde[i][p,cy,t]-1 for cy in range(2))*Q[i][int(c-add[i])] for p in range(4) for c in I[i][p] for t in range(T))
            cons2[xi][i] = m_sub[xi][i].addConstrs(y[xi][i][c,t]-n[xi][i][c,t]<=0 for c in C[i] for t in range(T))
            cons3[xi][i] = m_sub[xi][i].addConstrs(y[xi][i][c,t]<=Q[i][c] for c in list(set(C[i])-set(I1[i])-set(I2[i])-set(I3[i])-set(I4[i])-set(D[i])) for t in range(T))
            cons4[xi][i] = [None]*3
            cons4[xi][i][0] = m_sub[xi][i].addConstrs(y[xi][i][c,t]<=Q[i][c+1]/beta[i][xi][t][0,0] for c in V[i] for t in range(T))
            cons4[xi][i][1] = m_sub[xi][i].addConstrs(y[xi][i][c,t]<=Q[i][c+2]/beta[i][xi][t][1,0] for c in V[i] for t in range(T))
            cons4[xi][i][2] = m_sub[xi][i].addConstrs(y[xi][i][c,t]<=Q[i][c+3]/beta[i][xi][t][2,0] for c in V[i] for t in range(T))
            cons5[xi][i] = m_sub[xi][i].addConstrs(y[xi][i][c,t]+W*n[xi][i][proc[i][c],t]<=W*Jam_N[i][proc[i][c]] for c in list(set(C[i])-set(D[i])-set(V[i])-set(BO[i])) for t in range(T))
            # cons5_b[xi][i] = m_sub[xi][i].addConstrs(y[xi][i][c,t] <= -W*aux_n_value[xi][i][BO[i].index(c),t] + W*Jam_N[BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])] for c in BO[i] for t in range(T))
            cons6[xi][i] = [None]*4
            cons6[xi][i][0] = m_sub[xi][i].addConstrs(beta[i][xi][t][0,0]*y[xi][i][c,t]+W*n[xi][i][c+1,t]<=W*Jam_N[i][int(c+1)] for c in V[i] for t in range(T))
            cons6[xi][i][1] = m_sub[xi][i].addConstrs(beta[i][xi][t][1,0]*y[xi][i][c,t]+W*n[xi][i][c+2,t]<=W*Jam_N[i][int(c+2)] for c in V[i] for t in range(T))
            cons6[xi][i][2] = m_sub[xi][i].addConstrs(beta[i][xi][t][2,0]*y[xi][i][c,t]+W*n[xi][i][c+3,t]<=W*Jam_N[i][int(c+3)] for c in V[i] for t in range(T))
            cons6[xi][i][3] = m_sub[xi][i].addConstrs(n[xi][i][c,t+1] - n[xi][i][c,t] - beta[i][xi][t][c-pred[i][c]-1,0]*y[xi][i][pred[i][c],t] + y[xi][i][c,t] == 0 for c in I1[i]+I2[i]+I3[i]+I4[i] for t in range(T))
            cons7[xi][i] = m_sub[xi][i].addConstrs(n[xi][i][c,t+1] - n[xi][i][c,t] - y[xi][i][pred[i][c],t] + y[xi][i][c,t] == 0 for c in list(set(C[i])-set(O[i])-set(I1[i])-set(I2[i])-set(I3[i])-set(I4[i])-set(M[i])-set(BI[i])) for t in range(T))
            # cons7_b[xi][i] = m_sub[xi][i].addConstrs(n[xi][i][c,t+1] - n[xi][i][c,t] - aux_y_value[xi][i][BI[i].index(c),t] + y[xi][i][c,t] == 0 for c in BI[i] for t in range(T))
            cons8[xi][i] = m_sub[xi][i].addConstrs(n[xi][i][c,t+1] - n[xi][i][c,t] - gb.quicksum(y[xi][i][d,t] for d in pred[i][c]) + y[xi][i][c,t] == 0 for c in M[i] for t in range(T))
            cons9[xi][i] = m_sub[xi][i].addConstrs(n[xi][i][int(O[i][c]),t+1] - n[xi][i][int(O[i][c]),t] + y[xi][i][int(O[i][c]),t] == Demand[i][c][xi][t] for c in range(len(O[i])) for t in range(T))
            cons10[xi][i] = m_sub[xi][i].addConstrs((n[xi][i][c,0] == n_init_all[int(c+add[i])] for c in C[i]), name = 'n0')
            cons11[xi][i] = m_sub[xi][i].addConstrs(y[xi][i][c,t] == 0 for c in D[i] for t in range(T))
            m_sub[xi][i].Params.LogToConsole = 1
            m_sub[xi][i].Params.TimeLimit=7200
            m_sub[xi][i].Params.LogFile = 'H_4/T' + str(T) + '_S' + str(num_scenario) + '_sub.log'
            m_sub[xi][i].Params.InfUnbdInfo = 1
            m_sub[xi][i].update()

        # print(cons1[xi][-1].lhs)
            m_sub_aux[xi][i] = gb.Model()
            aux_y[xi][i] = m_sub_aux[xi][i].addVars(len(BI[i]), T, lb=0, vtype=gb.GRB.CONTINUOUS)
            aux_n[xi][i] = m_sub_aux[xi][i].addVars(len(BO[i]), T+1, lb=0, ub=Jam_N[0][0], vtype=gb.GRB.CONTINUOUS)
            m_sub_aux[xi][i].update()
    num_ite_benders = 0
    time_sub = []
    time_master = []
    time_ite = []
    gap = []
    m_master_obj_array = np.ones(N)*(-np.infty)
    m_master_obj = -np.infty
    throughput = []
    obj_array = []
    lb_array = []
    start_global = time.time()
    num_ite = 0
    while 1:
        start_ite = time.time()
        num_ite += 1
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
                    l_tilde[i] = l[i].x
                    o_tilde[i] = o[i].x
                    theta_tilde[i] = m_master[i].getAttr('X',theta[i])
                    m_master_obj_array[i] = m_master[i].objval
            m_master_obj = sum(m_master_obj_array)
            end_master = time.time()
            f = open("debug.txt", "a+")
            i=0
            for i in range(N):
                z1_pre[i] = z1_tilde[i]
                z2_pre[i] = z2_tilde[i]
        start_sub = time.time()
        for xi in range(num_scenario):
            for i in range(N):
                aux_y_value[xi][i] = np.zeros((len(BI[i]),T))
                aux_n_value[xi][i] = np.zeros((len(BO[i]),T+1))
                y_value[xi][i] = np.zeros((len(C[i]),T))
                n_value[xi][i] = np.zeros((len(C[i]),T+1))
                s_value[xi][i] = np.zeros((len(BO[i]), T))
                # aux_s[xi][i] = np.ones((len(BO[i]),T))
                # mlambda[xi][i] = np.zeros((len(BO[i]),T))
                mnu_y_1[xi][i] = np.zeros((len(BI[i]),T)) # corresponding to cons7
                mnu_y_2[xi][i] = np.zeros((len(BO[i]),T))
                # mnu_y_2[xi][i] = np.zeros((len(BI[i]),T))
                mnu_n_1[xi][i] = np.zeros((len(BO[i]),T+1)) # corresponding to cons5
                mnu_n_2[xi][i] = np.zeros((len(BI[i]),T+1))
                # mnu_n_2[xi][i] = np.zeros((len(BO[i]),T+1))
                m_sub[xi][i].remove(cons1[xi][i])
                cons1[xi][i] = m_sub[xi][i].addConstrs(y[xi][i][int(c-add[i]),t] <= gb.quicksum(z1_tilde[i][p,cy,t]+z2_tilde[i][p,cy,t]-1 for cy in range(2))*Q[i][int(c-add[i])] for p in range(4) for c in I[i][p] for t in range(T))
            num_ite_admm = 0
            obj_pre = np.zeros(N)
                # m_sub[xi].printStats()
            while 1:
                num_ite_admm += 1
                """ constant = [0]*N
                scope = [None]*N
                scope_test = [None]*N
                early_terminate = True
                for i in range(N):
                    m_sub[xi][i].setObjective(-alpha*gb.quicksum(gb.quicksum((T - t) * y[xi][i][c,t] for c in C[i]) for t in range(T))
                        - gb.quicksum(gb.quicksum(n[xi][i][c,t] for c in D[i]) for t in range(T)))
                    m_sub[xi][i].optimize() 
                    if m_sub[xi][i].objval > theta_tilde[xi][i]:
                        if theta_tilde[i][xi] < m_sub[xi][i].objval:
                            scope[i] = gb.quicksum(-gb.quicksum(z1[i][p,cy,t]+z2[i][p,cy,t] for cy in range(2))*Q_ALL[c]*cons1[xi][i][p,c,t].Pi for p in range(4) for c in I[i][p] for t in range(T))
                            scope_test[i] = gb.quicksum(gb.quicksum(-z1_tilde[i][p,cy,t]-z2_tilde[i][p,cy,t] for cy in range(2))*Q_ALL[c]*cons1[xi][i][p,c,t].Pi for p in range(4) for c in I[i][p] for t in range(T))
                            constant[i] = m_sub[xi][i].objval + scope_test[i].getValue()
                            add_const = m_master[i].addConstr(theta[i][xi] >= constant[i] - scope[i])
                    add_const.Lazy = 1 """
                for i in range(N):
                    m_sub[xi][i].setObjective(-alpha*gb.quicksum(gb.quicksum((T - t) * y[xi][i][c,t] for c in C[i]) for t in range(T))
                        - gb.quicksum(gb.quicksum(n[xi][i][c,t] for c in D[i]) for t in range(T))
                        # + gb.quicksum(gb.quicksum(mnu_y_2[xi][BI_IN[i][BI[i].index(c)]][BO[BI_IN[i][BI[i].index(c)]].index(int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]])),t]*(y_value[xi][BI_IN[i][BI[i].index(c)]][int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]]),t]-aux_y[xi][i][BI[i].index(c),t]) for c in BI[i]) for t in range(T))
                        + gb.quicksum(gb.quicksum(mnu_y_1[xi][i][BI[i].index(c),t]*(n[xi][i][c,t+1]-n[xi][i][c,t]-aux_y_value[xi][i][BI[i].index(c),t] + y[xi][i][c,t]) for c in BI[i]) for t in range(T))
                        + gb.quicksum(gb.quicksum(mnu_y_2[xi][i][BO[i].index(c),t]*(y[xi][i][c,t]-aux_y_value[xi][BO_IN[i][BO[i].index(c)]][BI[BO_IN[i][BO[i].index(c)]].index(int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])),t]) for c in BO[i]) for t in range(T))
                        # + gb.quicksum(gb.quicksum(mnu_n_2[xi][BO_IN[i][BO[i].index(c)]][BI[BO_IN[i][BO[i].index(c)]].index(int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])),t]*(n_value[xi][BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]]),t]-aux_n[xi][i][BO[i].index(c),t]) for c in BO[i]) for t in range(T))
                        # + gb.quicksum(gb.quicksum(mnu_y_2[xi][i][BI[i].index(c),t]*(y[xi][i][c,t]-aux_y_value[xi][BI_IN[i][BI[i].index(c)]][BO[BI_IN[BI[i].index(c)]].index(int(pred_all[int(c+add[i])]-add[BI[i].index(c)])),t]) for c in BO[i]) for t in range(T))
                        + gb.quicksum(gb.quicksum(mnu_n_1[xi][i][BO[i].index(c),t]*(y[xi][i][c,t]+s[xi][i][BO[i].index(c),t]+W*aux_n_value[xi][i][BO[i].index(c),t] - W*Jam_N[BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])]) for c in BO[i]) for t in range(T))
                        + gb.quicksum(gb.quicksum(mnu_n_2[xi][i][BI[i].index(c),t]*(n[xi][i][c,t]-aux_n_value[xi][BI_IN[i][BI[i].index(c)]][BO[BI_IN[i][BI[i].index(c)]].index(int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]])),t]) for c in BI[i]) for t in range(T))
                        # + rho_y/2*gb.quicksum(gb.quicksum((y_value[xi][BI_IN[i][BI[i].index(c)]][int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]]),t]-aux_y[xi][i][BI[i].index(c),t])*(y_value[xi][BI_IN[i][BI[i].index(c)]][int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]]),t]-aux_y[xi][i][BI[i].index(c),t]) for c in BI[i]) for t in range(T))
                        # + gb.quicksum(gb.quicksum(mnu_n_2[xi][i][BO[i].index(c),t]*(n[xi][i][c,t]-aux_n_value[xi][BO_IN[i][BO[i].index(c)]][BI[BO_IN[BO[i].index(c)]].index(int(proc_all[int(c+add[i])]-add[BI[i].index(c)])),t]))
                        + rho_cons7/2*gb.quicksum(gb.quicksum((n[xi][i][c,t+1]-n[xi][i][c,t]-aux_y_value[xi][i][BI[i].index(c),t] + y[xi][i][c,t])*(n[xi][i][c,t+1]-n[xi][i][c,t]-aux_y_value[xi][i][BI[i].index(c),t] + y[xi][i][c,t]) for c in BI[i]) for t in range(T))
                        + rho_y/2*gb.quicksum(gb.quicksum((y[xi][i][c,t]-aux_y_value[xi][BO_IN[i][BO[i].index(c)]][BI[BO_IN[i][BO[i].index(c)]].index(int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])),t])*(y[xi][i][c,t]-aux_y_value[xi][BO_IN[i][BO[i].index(c)]][BI[BO_IN[i][BO[i].index(c)]].index(int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])),t]) for c in BO[i]) for t in range(T))
                        # + rho_n/2*gb.quicksum(gb.quicksum((n_value[xi][BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]]),t]-aux_n[xi][i][BO[i].index(c),t])*(n_value[xi][BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]]),t]-aux_n[xi][i][BO[i].index(c),t]) for c in BO[i]) for t in range(T))
                        + rho_cons5/2*gb.quicksum(gb.quicksum((y[xi][i][c,t]+s[xi][i][BO[i].index(c),t]+W*aux_n_value[xi][i][BO[i].index(c),t] - W*Jam_N[BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])])*(y[xi][i][c,t]+s[xi][i][BO[i].index(c),t]+W*aux_n_value[xi][i][BO[i].index(c),t] - W*Jam_N[BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])]) for c in BO[i]) for t in range(T))
                        + rho_n/2*gb.quicksum(gb.quicksum((n[xi][i][c,t]-aux_n_value[xi][BI_IN[i][BI[i].index(c)]][BO[BI_IN[i][BI[i].index(c)]].index(int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]])),t])*(n[xi][i][c,t]-aux_n_value[xi][BI_IN[i][BI[i].index(c)]][BO[BI_IN[i][BI[i].index(c)]].index(int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]])),t]) for c in BI[i]) for t in range(T)))
                    """ m_sub[xi][i].remove(cons5_b[xi][i])
                    cons5_b[xi][i] = m_sub[xi][i].addConstrs(y[xi][i][c,t] <= -W*aux_n_value[xi][i][BO[i].index(c),t] + W*Jam_N[BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])] for c in BO[i] for t in range(T))
                    m_sub[xi][i].remove(cons7_b[xi][i])
                    cons7_b[xi][i] = m_sub[xi][i].addConstrs(n[xi][i][c,t+1] - n[xi][i][c,t] - aux_y_value[xi][i][BI[i].index(c),t] + y[xi][i][c,t] == 0 for c in BI[i] for t in range(T)) """
                    m_sub[xi][i].optimize()  
                    """ for i in range(N):            
                    if m_sub[xi][i].status == 3:
                        m_sub[xi][i].computeIIS()
                        m_sub[xi][i].write("model1.ilp")
                        f = open('debug.txt','a+')
                        print('infeasible!', file = f) """
                    if m_sub[xi][i].status == 2:
                            # num_optimal_sub = num_optimal_sub + 1
                        n_value[xi][i] = m_sub[xi][i].getAttr('X',n[xi][i])
                        y_value[xi][i] = m_sub[xi][i].getAttr('X',y[xi][i])
                        s_value[xi][i] = m_sub[xi][i].getAttr('X',s[xi][i])
                        # aux_n_value[xi][i] = m_sub[xi][i].getAttr('X',aux_n[xi][i])
                        # aux_y_value[xi][i] = m_sub[xi][i].getAttr('X',aux_y[xi][i])
                for i in range(N):
                    """ if m_sub[xi][i].status == 2:
                        for t in range(T):
                            for c in BI[i]:
                                aux_y_value[xi][i][BI[i].index(c),t] = max(1/(2*rho_y)*(mnu_y_2[xi][BI_IN[i][BI[i].index(c)]][int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]]),t]+rho_y*y_value[xi][BI_IN[i][BI[i].index(c)]][int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]]),t]
                                    +mnu_y_1[xi][i][BI[i].index(c)]+rho_y*(n_value[xi][i][c,t+1]-n_value[xi][i][c,t])+rho_y*y_value[xi][i][c,t]), 0)
                                # aux_n_value[xi][BI_IN[i][BI[i].index(c)]][BO[BI_IN[i][BI[i].index(c)]].index(int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]])),t] = min(max(mnu_n_2[xi][i][BI[i].index(c),t]/rho_n + n_value[xi][i][c,t], 0), Jam_N[i][c])
                            for c in BO[i]:
                                # aux_y_value[xi][BO_IN[i][BO[i].index(c)]][BI[BO_IN[i][BO[i].index(c)]].index(int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])),t] = max(mnu_y_2[xi][i][BO[i].index(c),t]/rho_y + y_value[xi][i][c,t], 0)
                                aux_n_value[xi][BO[i].index(c),t] = min(max()) """
                    m_sub_aux[xi][i].setObjective(
                        gb.quicksum(gb.quicksum(mnu_y_2[xi][BI_IN[i][BI[i].index(c)]][BO[BI_IN[i][BI[i].index(c)]].index(int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]])),t]*(y_value[xi][BI_IN[i][BI[i].index(c)]][int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]]),t]-aux_y[xi][i][BI[i].index(c),t]) for c in BI[i]) for t in range(T))
                        + gb.quicksum(gb.quicksum(mnu_y_1[xi][i][BI[i].index(c),t]*(n_value[xi][i][c,t+1]-n_value[xi][i][c,t]-aux_y[xi][i][BI[i].index(c),t] + y_value[xi][i][c,t]) for c in BI[i]) for t in range(T))
                        # + gb.quicksum(gb.quicksum(mnu_y_2[xi][i][BO[i].index(c),t]*(y[xi][i][c,t]-aux_y_value[xi][BO_IN[i][BO[i].index(c)]][BI[BO_IN[i][BO[i].index(c)]].index(int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])),t]) for c in BO[i]) for t in range(T))
                        + gb.quicksum(gb.quicksum(mnu_n_2[xi][BO_IN[i][BO[i].index(c)]][BI[BO_IN[i][BO[i].index(c)]].index(int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])),t]*(n_value[xi][BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]]),t]-aux_n[xi][i][BO[i].index(c),t]) for c in BO[i]) for t in range(T))
                        # + gb.quicksum(gb.quicksum(mnu_y_2[xi][i][BI[i].index(c),t]*(y[xi][i][c,t]-aux_y_value[xi][BI_IN[i][BI[i].index(c)]][BO[BI_IN[BI[i].index(c)]].index(int(pred_all[int(c+add[i])]-add[BI[i].index(c)])),t]) for c in BO[i]) for t in range(T))
                        + gb.quicksum(gb.quicksum(mnu_n_1[xi][i][BO[i].index(c),t]*(y_value[xi][i][c,t]+s_value[xi][i][BO[i].index(c),t]+W*aux_n[xi][i][BO[i].index(c),t] - W*Jam_N[BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])]) for c in BO[i]) for t in range(T))
                        # + gb.quicksum(gb.quicksum(mnu_n_2[xi][i][BI[i].index(c),t]*(n[xi][i][c,t]-aux_n_value[xi][BI_IN[i][BI[i].index(c)]][BO[BI_IN[i][BI[i].index(c)]].index(int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]])),t]) for c in BI[i]) for t in range(T))
                        + rho_y/2*gb.quicksum(gb.quicksum((y_value[xi][BI_IN[i][BI[i].index(c)]][int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]]),t]-aux_y[xi][i][BI[i].index(c),t])*(y_value[xi][BI_IN[i][BI[i].index(c)]][int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]]),t]-aux_y[xi][i][BI[i].index(c),t]) for c in BI[i]) for t in range(T))
                        # + gb.quicksum(gb.quicksum(mnu_n_2[xi][i][BO[i].index(c),t]*(n[xi][i][c,t]-aux_n_value[xi][BO_IN[i][BO[i].index(c)]][BI[BO_IN[BO[i].index(c)]].index(int(proc_all[int(c+add[i])]-add[BI[i].index(c)])),t]))
                        + rho_cons7/2*gb.quicksum(gb.quicksum((n_value[xi][i][c,t+1]-n_value[xi][i][c,t]-aux_y[xi][i][BI[i].index(c),t] + y_value[xi][i][c,t])*(n_value[xi][i][c,t+1]-n_value[xi][i][c,t]-aux_y[xi][i][BI[i].index(c),t] + y_value[xi][i][c,t]) for c in BI[i]) for t in range(T))
                        # + rho_y/2*gb.quicksum(gb.quicksum((y[xi][i][c,t]-aux_y_value[xi][BO_IN[i][BO[i].index(c)]][BI[BO_IN[i][BO[i].index(c)]].index(int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])),t])*(y[xi][i][c,t]-aux_y_value[xi][BO_IN[i][BO[i].index(c)]][BI[BO_IN[i][BO[i].index(c)]].index(int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])),t]) for c in BO[i]) for t in range(T))
                        + rho_n/2*gb.quicksum(gb.quicksum((n_value[xi][BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]]),t]-aux_n[xi][i][BO[i].index(c),t])*(n_value[xi][BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]]),t]-aux_n[xi][i][BO[i].index(c),t]) for c in BO[i]) for t in range(T))
                        + rho_cons5/2*gb.quicksum(gb.quicksum((y_value[xi][i][c,t]+s_value[xi][i][BO[i].index(c),t]+W*aux_n[xi][i][BO[i].index(c),t]-W*Jam_N[BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])])*(y_value[xi][i][c,t]+s_value[xi][i][BO[i].index(c),t]+W*aux_n[xi][i][BO[i].index(c),t]-W*Jam_N[BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])]) for c in BO[i]) for t in range(T)))
                        # + rho_n/2*gb.quicksum(gb.quicksum((n[xi][i][c,t]-aux_n_value[xi][BI_IN[i][BI[i].index(c)]][BO[BI_IN[i][BI[i].index(c)]].index(int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]])),t])*(n[xi][i][c,t]-aux_n_value[xi][BI_IN[i][BI[i].index(c)]][BO[BI_IN[i][BI[i].index(c)]].index(int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]])),t]) for c in BI[i]) for t in range(T)))
                    m_sub_aux[xi][i].optimize()
                    if m_sub_aux[xi][i].status == 2:
                        aux_n_value[xi][i] = m_sub_aux[xi][i].getAttr('X', aux_n[xi][i])
                        aux_y_value[xi][i] = m_sub_aux[xi][i].getAttr('X', aux_y[xi][i])
                for i in range(N):
                    if m_sub[xi][i].status == 2:
                        for t in range(T):
                            for c in BI[i]:
                                # mnu_y_1[xi][i][BI[i].index(c),t] += rho*(y_value[xi][BI_IN[i][BI[i].index(c)]][int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]]),t]-aux_y_value[xi][i][BI[i].index(c),t])
                                # print(n_value[xi][i][c,t]-aux_n_value[xi][BI_IN[i][BI[i].index(c)]][BO[BI_IN[i][BI[i].index(c)]].index(int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]])),t])
                                mnu_y_1[xi][i][BI[i].index(c),t] = rho_cons7*(n_value[xi][i][c,t+1]-n_value[xi][i][c,t]-aux_y_value[xi][i][BI[i].index(c),t] + y_value[xi][i][c,t]) + mnu_y_1[xi][i][BI[i].index(c),t]
                                mnu_n_2[xi][i][BI[i].index(c),t] = rho_n*(n_value[xi][i][c,t]-aux_n_value[xi][BI_IN[i][BI[i].index(c)]][BO[BI_IN[i][BI[i].index(c)]].index(int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]])),t]) + mnu_n_2[xi][i][BI[i].index(c),t]
                            for c in BO[i]:
                                mnu_n_1[xi][i][BO[i].index(c),t] = rho_cons5*(y_value[xi][i][c,t]+s_value[xi][i][BO[i].index(c),t]+W*aux_n_value[xi][i][BO[i].index(c),t] - W*Jam_N[BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])]) + mnu_n_1[xi][i][BO[i].index(c),t]
                                mnu_y_2[xi][i][BO[i].index(c),t] = rho_y*(y_value[xi][i][c,t]-aux_y_value[xi][BO_IN[i][BO[i].index(c)]][BI[BO_IN[i][BO[i].index(c)]].index(int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])),t]) + mnu_y_2[xi][i][BO[i].index(c),t]
                                # mnu_n_1[xi][i][BO[i].index(c),t] += rho*(n_value[xi][BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]]),t]-aux_n_value[xi][i][BO[i].index(c),t])
                        """ for c in BI[i]:
                            mnu_n_2[xi][i][BI[i].index(c),T] += rho*(n_value[xi][i][c,T]-aux_n_value[xi][BI_IN[i][BI[i].index(c)]][BO[BI_IN[i][BI[i].index(c)]].index(int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]])),T]) """
                        """ for c in BO[i]:
                            mnu_n_1[xi][i][BO[i].index(c),t] += rho*(n_value[xi][BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]]),T]-aux_n_value[xi][i][BO[i].index(c),T]) """
                
                f = open("debug.txt","a+")
                print(num_ite_admm, file = f)
                # print(sum(m_sub[xi][i].objval for i in range(N)))
                # print(sum(sum(y_value[xi][BI_IN[i][BI[i].index(c)]][int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]]),t]-aux_y_value[xi][i][BI[i].index(c),t] for c in BI[i]) for t in range(T)))
                # print(sum(m_sub[xi][i].objval for i in range(N)), file = f)
                # print(sum(sum(sum(y_value[xi][BI_IN[i][BI[i].index(c)]][int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]]),t]-aux_y_value[xi][i][BI[i].index(c),t] for c in BI[i]) for t in range(T)) for i in range(N)), file = f)
                # print(sum(sum(sum(n_value[xi][i][c,t]-aux_n_value[xi][BI_IN[i][BI[i].index(c)]][BO[BI_IN[i][BI[i].index(c)]].index(int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]])),t] for c in BI[i]) for t in range(T)) for i in range(N)), file=f)
                print(m_sub[xi][i].objval, file = f)
                print(sum(-alpha*sum(sum((T - t) * y_value[xi][i][c,t] for c in C[i]) for t in range(T))
                        - sum(sum(n_value[xi][i][c,t] for c in D[i]) for t in range(T)) for i in range(N)),file = f)
                print(sum(m_sub[xi][i].objval for i in range(N)), file = f)
                """ print("=================================", file = f) """
                # print("=================================")
                y_error = sum(sum(sum(abs(y_value[xi][BI_IN[i][BI[i].index(c)]][int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]]),t]-aux_y_value[xi][i][BI[i].index(c),t]) for c in BI[i]) for t in range(T)) for i in range(N))
                n_error = sum(sum(sum(abs(n_value[xi][i][c,t]-aux_n_value[xi][BI_IN[i][BI[i].index(c)]][BO[BI_IN[i][BI[i].index(c)]].index(int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]])),t]) for c in BI[i]) for t in range(T)) for i in range(N))
                cons5_error = sum(sum(sum(abs(y_value[xi][i][c,t]+s_value[xi][i][BO[i].index(c),t]+W*aux_n_value[xi][i][BO[i].index(c),t] - W*Jam_N[BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])]) for c in BO[i]) for t in range(T)) for i in range(N))
                cons7_error = sum(sum(sum(abs(n_value[xi][i][c,t]-aux_n_value[xi][BI_IN[i][BI[i].index(c)]][BO[BI_IN[i][BI[i].index(c)]].index(int(pred_all[int(c+add[i])]-add[BI_IN[i][BI[i].index(c)]])),t]) for c in BI[i]) for t in range(T)) for i in range(N))
                print(y_error, n_error, cons5_error, cons7_error, file = f)
                f.close()
                """ if abs(y_error) < T/10 and abs(n_error) < T/10:
                    # print(num_ite_admm, file = f)
                    break """
                # if abs(obj_pre[i] - m_sub[xi][i].objval)/abs(m_sub[xi][i].objval) < 0.005:
                if num_ite_admm == 100:
                    break
                obj_pre[i] = m_sub[xi][i].objval
       
            lb_sub[xi] = sum(m_sub[xi][i].objval for i in range(N))
            f = open("debug.txt","a+")
            print("=================================", file = f)    
            f.close()
            constant = [0]*N
            scope = [None]*N
            scope_test = [None]*N
            for i in range(N):
                if theta_tilde[i][xi] < m_sub[xi][i].objval:
                    scope[i] = gb.quicksum(-gb.quicksum(z1[i][p,cy,t]+z2[i][p,cy,t] for cy in range(2))*Q_ALL[c]*cons1[xi][i][p,c,t].Pi for p in range(4) for c in I[i][p] for t in range(T))
                    scope_test[i] = gb.quicksum(gb.quicksum(-z1_tilde[i][p,cy,t]-z2_tilde[i][p,cy,t] for cy in range(2))*Q_ALL[c]*cons1[xi][i][p,c,t].Pi for p in range(4) for c in I[i][p] for t in range(T))
                    """ constant[i] = constant[i] - gb.quicksum(Q_ALL[c]*2*cons1[xi][i][p,c,t].Pi for p in range(4) for c in I[i][p] for t in range(T))
                    constant[i] = constant[i] + gb.quicksum(Q[i][c]*cons3[xi][i][c,t].Pi for c in list(set(C[i])-set(I1[i])-set(I2[i])-set(I3[i])-set(I4[i])-set(D[i])) for t in range(T))
                    constant[i] = constant[i] + gb.quicksum(Q[i][c+1]/beta[i][xi][t][0,0]*cons4[xi][i][0][c,t].Pi + Q[i][c+2]/beta[i][xi][t][1,0]*cons4[xi][i][1][c,t].Pi + Q[i][c+3]/beta[i][xi][t][2,0]*cons4[xi][i][2][c,t].Pi for c in V[i] for t in range(T))
                    constant[i] = constant[i] + gb.quicksum(W*Jam_N[i][proc[i][c]]*cons5[xi][i][c,t].Pi for c in list(set(C[i])-set(D[i])-set(V[i])-set(BO[i])) for t in range(T))
                    constant[i] = constant[i] + gb.quicksum(W*Jam_N[BO_IN[i][BO[i].index(c)]][int(proc_all[int(c+add[i])]-add[BO_IN[i][BO[i].index(c)]])]*cons5_b[xi][i][c,t].Pi for c in BO[i] for t in range(T))
                    constant[i] = constant[i] + gb.quicksum(W*Jam_N[i][int(c+1)]*cons6[xi][i][0][c,t].Pi + W*Jam_N[i][int(c+2)]*cons6[xi][i][1][c,t].Pi + W*Jam_N[i][int(c+3)]*cons6[xi][i][2][c,t].Pi for c in V[i] for t in range(T))
                    constant[i] = constant[i] + gb.quicksum(Demand[i][c][xi][t]*cons9[xi][i][c,t].Pi for c in range(len(O[i])) for t in range(T))
                    constant[i] = constant[i] + gb.quicksum(n_init_all[int(c+add[i])]*cons10[xi][i][c].Pi for c in C[i]) """
                    # constant[i] = -alpha*sum(sum((T - t) * y_value[xi][i][c,t] for c in C[i]) for t in range(T))- sum(sum(n_value[xi][i][c,t] for c in D[i]) for t in range(T))
                    # theta[i][xi] = m_master[i].getVarByName('theta'+str(i)+'['+str(xi)+']')
                    # m_master[i].addConstr(theta[i][xi] >= constant[i] - scope[i])
                    constant[i] = m_sub[xi][i].objval + scope_test[i].getValue()
                    # constant[i] =  -alpha*sum(sum((T - t) * y_value[xi][i][c,t] for c in C[i]) for t in range(T))- sum(sum(n_value[xi][i][c,t] for c in D[i]) for t in range(T)) + scope_test[i].getValue()
                    add_const = m_master[i].addConstr(theta[i][xi] >= constant[i] - scope[i])
                    add_const.Lazy = 1
                    """ f = open("debug.txt","a+")
                    print(constant[i].getValue()-scope_test[i].getValue(), file = f)
                    print(m_sub[xi][i].objval, file = f)
                    print("===============================", file = f)
                    f.close()
        f = open("debug.txt","a+")
        print(sum(constant[i].getValue()-scope_test[i].getValue() for i in range(N)), file = f)
        print(sum(m_sub[xi][i].objval for i in range(N)), file = f)
        print("===============================", file = f)
        f.close() """
                    
        end_sub = time.time()
        
        if lb < m_master_obj:
            lb = m_master_obj
        if num_ite>0:
            lb_array.append(m_master_obj)
        if ub > sum(lb_sub)/num_scenario:
            ub = sum(lb_sub)/num_scenario
            """ for i in range(N):
                z1_opt[i] = z1_tilde[i]
                z2_opt[i] = z2_tilde[i] """
                # w_optimal = w_tilde
        end_global = time.time()
        time_sub.append(end_sub-start_sub)
        time_master.append(end_master-start_master)
        time_ite.append(end_global-start_ite)
        gap.append((ub-lb)/abs(lb))
        obj_array.append(sum(lb_sub)/num_scenario)
        print("iteration " + str(num_ite))
        print("upper bound is %f" % ub)
        print("lower bound is %f" % lb)
        print("gap is %f" % ((ub-lb)/abs(lb)))
        print("time to solve master problem is %f" % (end_master - start_master))
        print("time to solve sub problem is %f" % (end_sub - start_sub))
        print("time for this iteration is  %f" % (end_global-start_ite))
        print("all time to solve problem is %f" % (end_global - start_global))
        f = open('H_4/T' + str(T) + '_S' + str(num_scenario) + '_bound_LP_alpha0.001_grid_admm.log', 'a+')
        print("iteration " + str(num_ite), file = f)
        print("upper bound is %f" % ub, file = f)
        print("lower bound is %f" % lb, file = f)
        print("objective value is %f" % (sum(lb_sub)/num_scenario), file = f)
        print("gap is %f" % ((ub-lb)/abs(lb)), file = f)
        print("throughput term is %f" % (sum(sum(delay_sub))/num_scenario), file = f)
        print("ctm term is %f" % (sum(sum(ctm_sub))/num_scenario), file =f)
        print("time to solve master problem is %f" % (end_master - start_master), file = f)
        print("time to solve sub problem is %f" % (end_sub - start_sub), file = f)
        print("time for this iteration is  %f" % (end_global-start_ite), file = f)
        print("all time to solve problem is %f" % (end_global - start_global), file = f)
        f = open('H_4/T' + str(T) + '_S' + str(num_scenario) + '_signal_alpha0.001_grid_admm.log', 'a+')
        print("iteration " + str(num_ite), file = f)
        for i in range(N):
            for p in range(4):
                print(g_tilde[i][p], end = ",", file=f)
            print("cycle length", l_tilde[i], file=f)
            print("_coridor_admm_start", o_tilde[i], file=f)
            print("\n", file=f)
        f = open('H_4/T' + str(T) + '_S' + str(num_scenario) + '_binary_alpha0.001_grid_admm.log', 'a+')

        if num_ite == 10:
            break
        # evaluate solution
    eval_delay = np.zeros(num_scenario)
    eval_ctm = np.zeros(num_scenario)
    for xi in range(num_scenario):
        m_eval = gb.Model()
        y = m_eval.addVars(len(C_ALL), T, lb=0, vtype=gb.GRB.CONTINUOUS)
        n = m_eval.addVars(len(C_ALL), T+1, lb=0, vtype=gb.GRB.CONTINUOUS)
        m_eval.setObjective(-gb.quicksum(gb.quicksum((T - t) * y[c,t] for c in C_ALL) for t in range(T)), gb.GRB.MINIMIZE)
        m_eval.addConstrs(y[c,t]-gb.quicksum(z1_tilde[i][p,cy,t]+z2_tilde[i][p,cy,t]-1 for cy in range(2))*Q_ALL[c]<=0 for i in range(N) for p in range(4) for c in I[i][p] for t in range(T))
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
        """ f = open('H_4/T' + str(T) + '_S' + str(num_scenario) + '_n_alpha0.001.log', 'a+')
        for c in C_ALL:
            for t in range(T+1):
                print(n_value[c,t], end = ",", file=f)
            print("\n", file=f)
        f = open('H_4/T' + str(T) + '_S' + str(num_scenario) + '_y_alpha0.001.log', 'a+')
        for c in C_ALL:
            for t in range(T):
                print(y_value[c,t], end = ",", file=f)
            print("\n", file=f) """
        eval_delay[xi] = -sum(sum(n_value[c,t] for c in D_ALL) for t in range(T))
        eval_ctm[xi] = m_eval.objval
    throughput.append(sum(eval_delay)/num_scenario)
    print("the throughput is %f" % (sum(eval_delay)/num_scenario))
    print("the ctm objective value is %f" % (sum(eval_ctm)/num_scenario))
    f = open('H_4/T' + str(T) + '_S' + str(num_scenario) + '_bound_LP_alpha0.001_grid_admm.log', 'a+')
    print("the throughput is %f" % (sum(eval_delay)/num_scenario), file = f)
    print("the ctm objective value is %f" % (sum(eval_ctm)/num_scenario), file = f)
    # draw the figure of time
    """ plt.figure()
    plt.plot(time_sub, label = 'time of sub problem')
    plt.plot(time_master, label = 'time of master problem')
    plt.plot(time_ite, label = 'time of iteration')
    plt.xlabel('iterations')
    plt.ylabel('time(s)')
    plt.legend()
    plt.savefig('H_4/T' + str(T) + '_S' + str(num_scenario) + '_time.jpg')

    plt.figure()
    plt.plot(gap)
    plt.xlabel('iteration')
    plt.ylabel('gap')
    plt.savefig('H_4/T' + str(T) + '_S' + str(num_scenario) + '_gap.jpg') """
    plt.figure()
    # plt.plot(time_sub, label = 'time of sub problem')
    # plt.plot(time_master, label = 'time of master problem')
    # plt.plot(time_ite, label = 'time of iteration')
    plt.plot(obj_array, label = 'objective value')
    plt.xlabel('iterations')
    plt.ylabel('objective value')
    plt.legend()
    plt.savefig('H_4/T' + str(T) + '_S' + str(num_scenario) + '_obj_grid_admm.jpg')

    """ plt.figure()
    plt.plot(throughput, label = 'throughput')
    plt.xlabel('iterations')
    plt.ylabel('throughput')
    plt.legend()
    plt.savefig('H_4/T' + str(T) + '_S' + str(num_scenario) + '_throughput_grid_admm.jpg') """

    plt.figure()
    plt.plot(lb_array, label = 'lb')
    plt.xlabel('iterations')
    plt.ylabel('lb')
    plt.legend()
    plt.savefig('H_4/T' + str(T) + '_S' + str(num_scenario) + '_lb_grid_admm.jpg')

    return obj_array, throughput, lb_array
    
if __name__ == '__main__':
    num_scenario = 1
    obj_array, throughput, lb_array = Benders(0.0001, 1, 50)
    f1 = open('H_4_coridor_admm_ub_T50.txt', 'w+')
    f2 = open('H_4_coridor_admm_throughput_T50.txt', 'w+')
    f3 = open('H_4_coridor_admm_lb_T50.txt', 'w+')
    num_ite = 10
    for num_ite in range(num_ite):
        print(str(obj_array[num_ite]), file = f1)
        # print(str(throughput[num_ite]), file = f2)
        if num_ite < num_ite-1:
            print(str(lb_array[num_ite]), file = f3)
    