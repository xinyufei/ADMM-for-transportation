from data_global import *
from warmup import Warmup
import numpy as np
import gurobipy as gb
import time
import matplotlib.pyplot as plt
from plot import plot_vehicle

def Build_master(l_tilde, num_scenario):
    print(T)
    print(num_cycle)
    start_build = time.time()
    m_master = gb.Model()
    T_cycle = int(np.ceil(l_tilde[0]))
    U = T_cycle*2
    epsilon = 0.001
    z1 = m_master.addVars(N,4,2,T_cycle, vtype=gb.GRB.BINARY)
    z2 = m_master.addVars(N,4,2,T_cycle, vtype=gb.GRB.BINARY)
    # z = m_master.addVars(4, T, vtype=gb.GRB.BINARY)
    e = m_master.addVars(N,4,2)
    b = m_master.addVars(N,4, 2)
    for i in range(N):
        for p in range(4):
            b[i,p,0].lb = -np.infty
    g = m_master.addVars(N,4, lb = 2, ub = 25)
    # l  = m_master.addVar()
    print(num_scenario)
    theta = m_master.addVars(num_scenario, lb=0)
    o = m_master.addVars(N)
    m_master.setObjective(1/num_scenario*gb.quicksum(theta[xi] for xi in range(num_scenario)), gb.GRB.MINIMIZE)
    m_master.update()
    for i in range(N):
        for t in range(T_cycle):
            m_master.addConstrs(-U*z1[i,p,cy,t]+epsilon <= t-e[i,p,cy] for p in range(4) for cy in range(2))
            m_master.addConstrs(-U*z2[i,p,cy,t] <= b[i,p,cy]-t for p in range(4) for cy in range(2))
            m_master.addConstrs(t-e[i,p,cy] <= U*(1-z1[i,p,cy,t]) for p in range(4) for cy in range(2))
            m_master.addConstrs(b[i,p,cy]-t <= U*(1-z2[i,p,cy,t]) for p in range(4) for cy in range(2))
            m_master.addConstrs(z1[i,p,cy,t]+z2[i,p,cy,t] >= 1 for p in range(4) for cy in range(2))
            m_master.addConstrs(gb.quicksum(z1[i,p,cy,t]+z2[i,p,cy,t] for p in range(4)) <= 5 for cy in range(2))
        m_master.addConstr(o[i] <= l_tilde[i])
        m_master.addConstrs(b[i,0,cy] == l_tilde[i]*cy - o[i] for cy in range(2))
        m_master.addConstrs(e[i,0,cy] == b[i,0,cy] + g[i,0] for cy in range(2))
        m_master.addConstrs(b[i,1,cy] == e[i,0,cy] for cy in range(2))
        m_master.addConstrs(e[i,1,cy] == b[i,1,cy] + g[i,1] for cy in range(2))
        m_master.addConstrs(b[i,2,cy] == e[i,1,cy] for cy in range(2))
        m_master.addConstrs(e[i,2,cy] == b[i,2,cy] + g[i,2] for cy in range(2))
        m_master.addConstrs(b[i,3,cy] == e[i,2,cy] for cy in range(2))
        m_master.addConstrs(e[i,3,cy] == b[i,3,cy] + g[i,3] for cy in range(2))
        m_master.addConstr(gb.quicksum(g[i,p] for p in range(4)) == l_tilde[i])
    m_master.Params.LogToConsole = 1
    m_master.Params.TimeLimit = 7200
    m_master.Params.LogFile = 'benders/T' + str(T) + '_S' + str(num_scenario) + '_master.log'
    m_master.Params.LazyConstraints = 1
    m_master.update()
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
    f = open('benders/T' + str(T) + '_S' + str(num_scenario) + '_bound_LP_alpha0.001_corridor.log', 'w+')
    f = open('benders/T' + str(T) + '_S' + str(num_scenario) + '_w_alpha0.001_corridor.log', 'w+')
    f = open('benders/T' + str(T) + '_S' + str(num_scenario) + '_y_alpha0.001_corridor.log', 'w+')
    f = open('benders/T' + str(T) + '_S' + str(num_scenario) + '_n_alpha0.001_corridor.log', 'w+')
    # generalize all samples
    network_data = Network(N_edge, True, num_scenario, T)
    n_init_all = Warmup(False)
    Demand_ALL = network_data.Demand[0]
    for i in range(1,N):
        Demand_ALL = Demand_ALL + [de for de in network_data.Demand[i]]
    beta = network_data.beta
    num_cycle = network_data.num_cycle
    # build master model
    l_tilde = np.zeros(N)
    for i in range(N):
        # l_tilde[i] = 20/network_data.Y
        l_tilde[i] = 40
    T_cycle = int(np.ceil(l_tilde[0]))
    [m_master, z1, z2, e, b, g, theta, o] = Build_master(l_tilde,num_scenario)
    # get initialized solution
    # w_tilde = np.zeros((N,4,T))
    # theta_tilde = np.ones(num_scenario)*(-np.infty)
    l_optimal = np.ones(N)
    for i in range(N):
        l_optimal[i] = 40
    o_optimal = np.zeros(N)
    g_optimal = np.zeros((N,4))
    z1_optimal = np.zeros((N,4,num_cycle,T))
    z2_optimal = np.zeros((N,4,num_cycle,T))
    
    m_sub = [None]*num_scenario
    n = [None]*num_scenario
    y = [None]*num_scenario
    n_value = [None]*num_scenario
    y_value = [None]*num_scenario
    cons1 = [None]*num_scenario
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
    start_global = time.time()
    start_ite = time.time()
    start_master = time.time()
    m_master.optimize()
    end_master = time.time()
    # w_tilde = m_master.getAttr('X', w)
    z1_tilde = m_master.getAttr('X', z1)
    z2_tilde = m_master.getAttr('X', z2)
    g_tilde = m_master.getAttr('X', g)
    o_tilde = m_master.getAttr('X', o)
    # l_tilde = m_master.getAttr('X', l)
    for xi in range(num_scenario):
        theta[xi].lb = -np.infty
    theta_tilde = np.ones(num_scenario)*(-np.infty)
    m_master_obj = -np.infty
    for xi in range(num_scenario):
        # solve subproblem
        m_sub[xi] = gb.Model()
        y[xi] = m_sub[xi].addVars(len(C_ALL), T, lb=0, vtype=gb.GRB.CONTINUOUS)
        n[xi] = m_sub[xi].addVars(len(C_ALL), T+1, lb=0, vtype=gb.GRB.CONTINUOUS)
        """ m_sub[xi].setObjective(gb.quicksum(gb.quicksum(t * y[c,t] for c in D_ALL) for t in range(T))
                + alpha * gb.quicksum(gb.quicksum(t* y[c,t] for c in list(set(C_ALL)-set(D_ALL))) for t in range(T)), gb.GRB.MINIMIZE) """
        m_sub[xi].setObjective(-alpha*gb.quicksum(gb.quicksum((T - t) * y[xi][c,t] for c in C_ALL) for t in range(T)) -
                gb.quicksum(gb.quicksum(n[xi][c,t] for c in D_ALL) for t in range(T)), gb.GRB.MINIMIZE)
        cons1[xi] = m_sub[xi].addConstrs(y[xi][c,t]-gb.quicksum(z1_tilde[i,p,cy,int(t-np.floor(t/l_optimal[i])*l_optimal[i])]+z2_tilde[i,p,cy,int(t-np.floor(t/l_optimal[i])*l_optimal[i])]-1 for cy in range(2))*Q_ALL[c] <= 0 for i in range(N) for p in range(4) for c in I[i][p] for t in range(T))
        cons2[xi] = m_sub[xi].addConstrs(y[xi][c,t]-n[xi][c,t]<=0 for c in C_ALL for t in range(T))
        cons3[xi] = m_sub[xi].addConstrs(y[xi][c,t]<=Q[0][0] for c in list(set(C_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(D_ALL)) for t in range(T))
        cons4[xi] = [None]*3*N
        for i in range(N):
            cons4[xi][i*3] = m_sub[xi].addConstrs(y[xi][c+add[i],t]<=Q[i][c+1]/beta[i][xi][t][0,0] for c in V[i] for t in range(T))
            cons4[xi][i*3+1] = m_sub[xi].addConstrs(y[xi][c+add[i],t]<=Q[i][c+2]/beta[i][xi][t][1,0] for c in V[i] for t in range(T))
            cons4[xi][i*3+2] = m_sub[xi].addConstrs(y[xi][c+add[i],t]<=Q[i][c+3]/beta[i][xi][t][2,0] for c in V[i] for t in range(T))
        cons5[xi] = m_sub[xi].addConstrs(y[xi][c,t]+W*n[xi][proc_all[c],t]<=W*Jam_N_ALL[int(proc_all[c])] for c in list(set(C_ALL)-set(D_ALL)-set(V_ALL)) for t in range(T))
        cons6[xi] = [None]*4*N
        for i in range(N):
            cons6[xi][i*4] = m_sub[xi].addConstrs(beta[i][xi][t][0,0]*y[xi][c+add[i],t]+W*n[xi][c+add[i]+1,t]<=W*Jam_N_ALL[int(c+add[i]+1)] for c in V[i] for t in range(T))
            cons6[xi][i*4+1] = m_sub[xi].addConstrs(beta[i][xi][t][1,0]*y[xi][c+add[i],t]+W*n[xi][c+add[i]+2,t]<=W*Jam_N_ALL[int(c+add[i]+2)] for c in V[i] for t in range(T))
            cons6[xi][i*4+2] = m_sub[xi].addConstrs(beta[i][xi][t][2,0]*y[xi][c+add[i],t]+W*n[xi][c+add[i]+3,t]<=W*Jam_N_ALL[int(c+add[i]+3)] for c in V[i] for t in range(T))
            cons6[xi][i*4+3] = m_sub[xi].addConstrs(n[xi][c+add[i],t+1] - n[xi][c+add[i],t] - beta[i][xi][t][c-pred[i][c]-1,0]*y[xi][pred[i][c]+add[i],t] + y[xi][c+add[i],t] == 0 for c in I1[i]+I2[i]+I3[i]+I4[i] for t in range(T))
        cons7[xi] = m_sub[xi].addConstrs(n[xi][c,t+1] - n[xi][c,t] - y[xi][pred_all[c],t] + y[xi][c,t] == 0 for c in list(set(C_ALL)-set(O_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(M_ALL)) for t in range(T))
        cons8[xi] = m_sub[xi].addConstrs(n[xi][c,t+1] - n[xi][c,t] - gb.quicksum(y[xi][d,t] for d in pred_all[c]) + y[xi][c,t] == 0 for c in M_ALL for t in range(T))
        cons9[xi] = m_sub[xi].addConstrs(n[xi][O_ALL[i],t+1] - n[xi][O_ALL[i],t] + y[xi][O_ALL[i],t] == Demand_ALL[i][xi][t] for i in range(len(O_ALL)) for t in range(T))
        cons10[xi] = m_sub[xi].addConstrs((n[xi][c,0] == n_init_all[c] for c in C_ALL), name = 'n0')
        cons11[xi] = m_sub[xi].addConstrs(y[xi][c,t] == 0 for c in D_ALL for t in range(T))
        m_sub[xi].Params.LogToConsole = 0
        # m_sub[xi].Params.TimeLimit=600
        m_sub[xi].Params.LogFile = 'benders/T' + str(T) + '_S' + str(num_scenario) + '_sub.log'
        m_sub[xi].Params.InfUnbdInfo = 0
        m_sub[xi].update()
    # calculate number of constraints in sub problem 
    num_ite = 0
    time_sub = []
    time_master = []
    time_ite = []
    gap = []
    throughput = []
    obj_array = []
    lb_array = []
    ub_array = []
    # start_global = time.time()
    while 1:
        start_ite = time.time()
        num_ite = num_ite + 1
        num_optimal_sub = 0
        opt_sub = np.zeros(num_scenario)
        delay_sub = np.zeros(num_scenario)
        ctm_sub = np.zeros(num_scenario)
        if num_ite > 1:
            m_master.update()
            m_master.printStats()
            start_master = time.time()
            m_master.optimize()
            end_master = time.time()
            if m_master.status == 2:
                m_master_obj = m_master.objval
                z1_tilde = m_master.getAttr('X', z1)
                z2_tilde = m_master.getAttr('X', z2)
                g_tilde = m_master.getAttr('X', g)
                # l_tilde = m_master.getAttr('X', l)
                o_tilde = m_master.getAttr('X', o)
                theta_tilde = m_master.getAttr('X',theta)
            
        start_sub = time.time()
        for xi in range(num_scenario):
            m_sub[xi].remove(cons1[xi])
            cons1[xi] = m_sub[xi].addConstrs(y[xi][c,t]-gb.quicksum(z1_tilde[i,p,cy,int(t-np.floor(t/l_optimal[i])*l_optimal[i])]+z2_tilde[i,p,cy,int(t-np.floor(t/l_optimal[i])*l_optimal[i])]-1 for cy in range(2))*Q_ALL[c] <= 0 for i in range(N) for p in range(4) for c in I[i][p] for t in range(T))
            m_sub[xi].optimize()
            # m_sub.optimize()
            if m_sub[xi].status == 2:
                num_optimal_sub = num_optimal_sub + 1
                opt_sub[xi] = m_sub[xi].objval
                constant = 0
                scope = 0
                scope = gb.quicksum(-gb.quicksum(z1[i,p,cy,int(t-np.floor(t/l_optimal[i])*l_optimal[i])]+z2[i,p,cy,int(t-np.floor(t/l_optimal[i])*l_optimal[i])] for cy in range(2))*Q_ALL[c]*cons1[xi][i,p,c,t].Pi for i in range(N) for p in range(4) for c in I[i][p] for t in range(T))
                constant = constant - gb.quicksum(Q_ALL[c]*2*cons1[xi][i,p,c,t].Pi for i in range(N) for p in range(4) for c in I[i][p] for t in range(T))
                constant = constant + gb.quicksum(Q[0][0]*cons3[xi][c, t].Pi for c in list(set(C_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(D_ALL)) for t in range(T))
                for i in range(N):
                    constant = constant + gb.quicksum(Q[i][c+1]/beta[i][xi][t][0,0]*cons4[xi][i*3][c,t].Pi + Q[i][c+2]/beta[i][xi][t][1,0]*cons4[xi][i*3+1][c,t].Pi + Q[i][c+3]/beta[i][xi][t][2,0]*cons4[xi][i*3+2][c,t].Pi for c in V[i] for t in range(T))
                constant = constant + gb.quicksum(W*Jam_N_ALL[int(proc_all[c])]*cons5[xi][c,t].Pi for c in list(set(C_ALL)-set(D_ALL)-set(V_ALL)) for t in range(T))
                for i in range(N):
                    constant = constant + gb.quicksum(W*Jam_N_ALL[int(c+add[i]+1)]*cons6[xi][i*4][c,t].Pi + W*Jam_N_ALL[int(c+add[i]+2)]*cons6[xi][i*4+1][c,t].Pi + W*Jam_N_ALL[int(c+add[i]+3)]*cons6[xi][i*4+2][c,t].Pi for c in V[i] for t in range(T))
                constant = constant + gb.quicksum(Demand_ALL[i][xi][t]*cons9[xi][i,t].Pi for i in range(len(O_ALL)) for t in range(T))
                constant = constant + gb.quicksum(n_init_all[c]*cons10[xi][c].Pi for c in C_ALL)
                if theta_tilde[xi] < opt_sub[xi]:
                    add_const = m_master.addConstr(theta[xi] >= constant - scope)
                    add_const.Lazy = 1
            y_value[xi] = m_sub[xi].getAttr('X', y[xi])
            n_value[xi] = m_sub[xi].getAttr('X', n[xi])
            ctm_sub[xi] = -alpha*sum(sum((T - t) * y_value[xi][c,t] for c in C_ALL) for t in range(T))
            delay_sub[xi] = -sum(sum(n_value[xi][c,t] for c in D_ALL) for t in range(T))
        end_sub = time.time()
        
        if lb < m_master_obj:
            lb = m_master_obj
        if num_ite > 0:
            lb_array.append(m_master_obj)
        if ub > sum(opt_sub)/num_scenario:
            ub = sum(opt_sub)/num_scenario
            for i in range(N):
                l_optimal[i] = l_tilde[i]
                o_optimal[i] = o_tilde[i]
                for p in range(4):
                    g_optimal[i,p] = g_tilde[i,p]
                    for cy in range(2):
                        for t in range(T_cycle):
                            z1_optimal[i,p,cy,t] = z1_tilde[i,p,cy,t]
                            z2_optimal[i,p,cy,t] = z2_tilde[i,p,cy,t]
        ub_array.append(ub)
        start_plot = time.time()
        y_value_average = np.zeros((len(C_ALL), T))
        n_value_average = np.zeros((len(C_ALL), T))
        for c in C_ALL:
            for t in range(T):
                y_value_average[c,t] = sum(y_value[xi][c,t] for xi in range(num_scenario))/num_scenario
                n_value_average[c,t] = sum(n_value[xi][c,t] for xi in range(num_scenario))/num_scenario
        # plot_vehicle(num_ite, T, num_scenario, "fixed_length", add, len(C_ALL), n_value_average, y_value_average)
        end_plot = time.time()
        end_global = time.time()
        time_sub.append(end_sub-start_sub)
        time_master.append(end_master-start_master)
        time_ite.append(end_global-start_ite)
        gap.append((ub-lb)/abs(lb))
        obj_array.append(sum(opt_sub)/num_scenario)
        print("iteration " + str(num_ite))
        print("upper bound is ", ub)
        print("lower bound is ", lb)
        print("gap is ", ((ub-lb)/abs(lb)))
        print("time to solve master problem is ", (end_master - start_master))
        print("time to solve sub problem is ", (end_sub - start_sub))
        print("time for this iteration is  ", (end_global-start_ite))
        print("all time to solve problem is ", (end_global - start_global))
        f = open('benders/T' + str(T) + '_S' + str(num_scenario) + '_bound_LP_fixed_length.log', 'a+')
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
        f = open('benders/T' + str(T) + '_S' + str(num_scenario) + '_signal_alpha0.001_corridor.log', 'a+')
        print("iteration " + str(num_ite), file = f)
        for i in range(N):
            for p in range(4):
                print(g_tilde[i,p], end = ",", file=f)
            print("cycle length", l_tilde[i], file=f)
            print("offset", o_tilde[i], file = f)
            print("\n", file=f)
        if num_ite == 3:
            break

    f = open('benders/T' + str(T) + '_S' + str(num_scenario) + '_signal_fixed_length.log', 'a+')
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
        m_eval.addConstrs(y[c,t]-gb.quicksum(z1_optimal[i,p,cy,int(t-np.floor(t/l_optimal[i])*l_optimal[i])]+z2_optimal[i,p,cy,int(t-np.floor(t/l_optimal[i])*l_optimal[i])]-1 for cy in range(2))*Q_ALL[c]<=0 for i in range(N) for p in range(4) for c in I[i][p] for t in range(T))
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
        print("the objective value is %f" % sum(sum(n_value[c,t] for c in D_ALL) for t in range(T)))
        f = open('benders/T' + str(T) + '_S' + str(num_scenario) + '_bound_LP_fixed_length.log', 'a+')
        print("the objective value is %f" % sum(sum(n_value[c,t] for c in D_ALL) for t in range(T)), file = f)
        f = open('benders/T' + str(T) + '_S' + str(num_scenario) + '_n_fixed_length.log', 'a+')
        for c in C_ALL:
            for t in range(T+1):
                print(n_value[i,t], end = ",", file=f)
            print("\n", file=f)
        f = open('benders/T' + str(T) + '_S' + str(num_scenario) + '_y_fixed_length.log', 'a+')
        for c in C_ALL:
            for t in range(T):
                print(y_value[i,t], end = ",", file=f)
            print("\n", file=f)
        eval_delay[xi] = -sum(sum(n_value[c,t] for c in D_ALL) for t in range(T))
        eval_ctm[xi] = alpha*m_eval.objval
    throughput.append(sum(eval_delay)/num_scenario)
    print("the throughput is %f" % (sum(eval_delay)/num_scenario))
    print("the ctm objective value is %f" % (sum(eval_ctm)/num_scenario))
    f = open('benders/T' + str(T) + '_S' + str(num_scenario) + '_bound_LP_fixed_length.log', 'a+')
    print("the throughput is %f" % (sum(eval_delay)/num_scenario), file = f)
    print("the ctm objective value is %f" % (sum(eval_ctm)/num_scenario), file = f)
    # if num_ite == 50 or (end_master-start_master) >= 7200:
    # draw the figure of time
    """ plt.figure()
    plt.plot(time_sub, label = 'time of sub problem')
    plt.plot(time_master, label = 'time of master problem')
    plt.plot(time_ite, label = 'time of iteration')
    plt.xlabel('iterations')
    plt.ylabel('time(s)')
    plt.legend()
    plt.savefig('benders/T' + str(T) + '_S' + str(num_scenario) + '_time.jpg')

    plt.figure()
    plt.plot(gap)
    plt.xlabel('iteration')
    plt.ylabel('gap')
    plt.savefig('benders/T' + str(T) + '_S' + str(num_scenario) + '_gap.jpg') """
    plt.figure()
    # plt.plot(time_sub, label = 'time of sub problem')
    # plt.plot(time_master, label = 'time of master problem')
    # plt.plot(time_ite, label = 'time of iteration')
    plt.plot(obj_array, label = 'objective value')
    plt.xlabel('iterations')
    plt.ylabel('objective value')
    plt.legend()
    plt.savefig('benders/T' + str(T) + '_S' + str(num_scenario) + '_obj_fixed_length.jpg')

    plt.figure()
    plt.plot(ub_array, label = 'throughput')
    plt.xlabel('iterations')
    plt.ylabel('upper bound')
    plt.legend()
    plt.savefig('benders/T' + str(T) + '_S' + str(num_scenario) + '_ub_fixed_length.jpg')

    plt.figure()
    plt.plot(lb_array, label = 'lb')
    plt.xlabel('iterations')
    plt.ylabel('lower bound')
    plt.legend()
    plt.savefig('benders/T' + str(T) + '_S' + str(num_scenario) + '_lb_fixed_length.jpg')

    return obj_array, ub_array, lb_array
    
if __name__ == '__main__':
    T = 600
    num_scenario = 1
    obj_array, ub_array, lb_array = Benders(0.0001, num_scenario, T)
    f1 = open('benders/T' + str(T) + '_S' + str(num_scenario) + '_obj_fixed_length.txt', 'w+')
    f2 = open('benders/T' + str(T) + '_S' + str(num_scenario) + '_ub_fixed_length.txt', 'w+')
    f3 = open('benders/T' + str(T) + '_S' + str(num_scenario) + '_lb_fixed_length.txt', 'w+')
    for num_ite in range(len(obj_array)):
        print(str(obj_array[num_ite]), file = f1)
    for num_ite in range(len(ub_array)):
        print(str(ub_array[num_ite]), file = f2)
    for num_ite in range(len(lb_array)):
        print(str(lb_array[num_ite]), file = f3)