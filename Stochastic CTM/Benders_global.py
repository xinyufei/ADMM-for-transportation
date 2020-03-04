from data_global import *
import numpy as np
import gurobipy as gb
import time

def Benders(epsilon = 0.01, num_scenario = 10):
    ub = np.infty
    lb = -np.infty
    f = open('benders/T' + str(T) + '_S' + str(num_scenario) + '_bound.log', 'w+')
    # generalize all samples
    network_data = Network(N_edge, True, num_scenario)
    n_init = network_data.n_init
    n_init_all = n_init[0]
    for i in range(1,N):
        n_init_all = np.concatenate((n_init_all, n_init[i]), axis=0)
    # build master model
    m_master = gb.Model()
    w = m_master.addVars(N, 4, T, lb=0, ub=1, vtype=gb.GRB.BINARY)
    theta = m_master.addVars(num_scenario, lb=-np.infty)
    m_master.setObjective(1/num_scenario*gb.quicksum(theta[xi] for xi in range(num_scenario)), gb.GRB.MINIMIZE)
    new_w = m_master.addVars(N, T, lb=0, ub=1, vtype=gb.GRB.CONTINUOUS)
    u = m_master.addVars(N, T-1, lb=0, ub=1,  vtype=gb.GRB.CONTINUOUS)
    m_master.addConstrs(new_w[i,t] == w[i,0,t] + w[i,1,t] for i in range(N) for t in range(T))
    m_master.addConstrs(w[i,0,t]+w[i,1,t]+w[i,2,t]+w[i,3,t]==1 for i in range(N) for t in range(T))
    m_master.addConstrs(new_w[i,t]-new_w[i,t-1]-u[i,t-1]<=0 for i in range(N) for t in range(1,T))
    m_master.addConstrs( -new_w[i,t]+new_w[i,t-1]-u[i,t-1]<=0 for i in range(N) for t in range(1,T))
    m_master.addConstrs(-new_w[i,t]-new_w[i,t-1]+u[i,t-1]<=0 for i in range(N) for t in range(1,T))
    m_master.addConstrs(new_w[i,t]+new_w[i,t-1]+u[i,t-1]<=2 for  i in range(N) for t in range(1,T))
    m_master.addConstrs(u[i,t]+u[i,t+1]+u[i,t+2]+u[i,t+3]<=1 for  i in range(N) for t in range(0,T-4))
    m_master.addConstrs(new_w[i,t]+new_w[i,t+1]+new_w[i,t+2]+new_w[i,t+3]+new_w[i,t+4]+new_w[i,t+5]+new_w[i,t+6]+new_w[i,t+7] <= 8 for  i in range(N) for t in range(0,T-7))
    m_master.addConstrs(-(new_w[i,t]+new_w[i,t+1]+new_w[i,t+2]+new_w[i,t+3]+new_w[i,t+4]+new_w[i,t+5]+new_w[i,t+6]+new_w[i,t+7])<=-1 for  i in range(N) for t in range(0,T-7))
    m_master.Params.LogToConsole = 0
    m_master.Params.TimeLimit = 7200
    m_master.Params.LogFile = 'benders/T' + str(T) + '_S' + str(num_scenario) + '_master.log'
    # initialized integer variables (fist-stage variables)
    w_tilde = np.zeros((N,4,T))
    theta_tilde = np.ones(num_scenario)*(-np.infty)
    # calculate number of constraints in sub problem 
    num_cons1 = sum(len(I[i][j]) for i in range(N) for j in range(4))*T
    num_cons2 = len(C_ALL)*T
    num_cons3 = (len(C_ALL)-len(I1_ALL)-len(I2_ALL)-len(I3_ALL)-len(I4_ALL)-len(D_ALL))*T
    num_cons4 = 3*len(V_ALL)*T
    num_cons5 = (len(C_ALL)-len(D_ALL)-len(V_ALL))*T
    num_cons6 = len(V_ALL)*T*3 + (len(I1_ALL)+len(I2_ALL)+len(I3_ALL)+len(I4_ALL))*T
    num_cons7 = len(list(set(C_ALL)-set(O_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(M_ALL)))*T
    num_cons8 = len(M_ALL)*T
    num_cons9 = len(O_ALL)*T
    num_cons10 = len(C_ALL)
    num_ite = 0
    while ub-lb > epsilon:
        num_ite = num_ite + 1
        num_optimal_sub = 0
        opt_sub = np.zeros(num_scenario)
        start_sub = time.time()
        for xi in range(num_scenario):
            # solve subproblem
            m_sub = gb.Model()
            y = m_sub.addVars(len(C_ALL), T, lb=0, vtype=gb.GRB.CONTINUOUS)
            n = m_sub.addVars(len(C_ALL), T+1, lb=0, vtype=gb.GRB.CONTINUOUS)
            m_sub.setObjective(gb.quicksum(gb.quicksum(t * y[c,t] for c in D_ALL) for t in range(T))
                    + alpha * gb.quicksum(gb.quicksum(t* y[c,t] for c in list(set(C_ALL)-set(D_ALL))) for t in range(T)), gb.GRB.MINIMIZE)
            cons1 = m_sub.addConstrs(y[c,t]-w_tilde[i,j,t]*Q_ALL[c]<=0 for i in range(N) for j in range(4) for c in I[i][j] for t in range(T))
            cons2 = m_sub.addConstrs(y[c,t]-n[c,t]<=0 for c in C_ALL for t in range(T))
            cons3 = m_sub.addConstrs(y[c,t]<=Q[0][0] for c in list(set(C_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(D_ALL)) for t in range(T))
            cons4 = [None]*3*N
            for i in range(N):
                cons4[i*3] = m_sub.addConstrs(y[c+add[i],t]<=Q[i][c+1]/beta[i][0,0] for c in V[i] for t in range(T))
                cons4[i*3+1] = m_sub.addConstrs(y[c+add[i],t]<=Q[i][c+2]/beta[i][1,0] for c in V[i] for t in range(T))
                cons4[i*3+2] = m_sub.addConstrs(y[c+add[i],t]<=Q[i][c+3]/beta[i][2,0] for c in V[i] for t in range(T))
            cons5 = m_sub.addConstrs(y[c,t]+W*n[proc_all[c],t]<=W*Jam_N_ALL[c] for c in list(set(C_ALL)-set(D_ALL)-set(V_ALL)) for t in range(T))
            cons6 = [None]*4*N
            for i in range(N):
                cons6[i*4] = m_sub.addConstrs(beta[i][0,0]*y[c+add[i],t]+W*n[c+add[i]+1,t]<=W*Jam_N_ALL[int(c+add[i]+1)] for c in V[i] for t in range(T))
                cons6[i*4+1] = m_sub.addConstrs(beta[i][1,0]*y[c+add[i],t]+W*n[c+add[i]+2,t]<=W*Jam_N_ALL[int(c+add[i]+2)] for c in V[i] for t in range(T))
                cons6[i*4+2] = m_sub.addConstrs(beta[i][2,0]*y[c+add[i],t]+W*n[c+add[i]+3,t]<=W*Jam_N_ALL[int(c+add[i]+3)] for c in V[i] for t in range(T))
                cons6[i*4+3] = m_sub.addConstrs(n[c+add[i],t+1] - n[c+add[i],t] - beta[i][c-pred[i][c]-1,0]*y[pred[i][c]+add[i],t] + y[c+add[i],t] == 0 for c in I1[i]+I2[i]+I3[i]+I4[i] for t in range(T))
            cons7 = m_sub.addConstrs(n[c,t+1] - n[c,t] - y[pred_all[c],t] + y[c,t] == 0 for c in list(set(C_ALL)-set(O_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(M_ALL)) for t in range(T))
            cons8 = m_sub.addConstrs(n[c,t+1] - n[c,t] - gb.quicksum(y[d,t] for d in pred_all[c]) + y[c,t] == 0 for c in M_ALL for t in range(T))
            cons9 = m_sub.addConstrs(n[O_ALL[i],t+1] - n[O_ALL[i],t] + y[O_ALL[i],t] == Demand_ALL[i] for i in range(len(O_ALL)) for t in range(T))
            cons10 = m_sub.addConstrs((n[c,0] == n_init_all[c,xi] for c in C_ALL), name = 'n0')
            m_sub.Params.LogToConsole = 0
            m_sub.Params.TimeLimit=7200
            m_sub.Params.LogFile = 'benders/T' + str(T) + '_S' + str(num_scenario) + '_sub.log'
            m_sub.Params.InfUnbdInfo = 1
            m_sub.update()
            m_sub.printStats()
            m_sub.optimize()
            if m_sub.status == 3:
                rho = m_sub.FarkasDual()
                constant = 0
                scope = 0
                cons = 0
                for i in range(N):
                    for j in range(4):
                        for c in I[i][j]:
                            for t in range(T):
                                scope = scope - w[i,j,t]*Q_ALL[c]*rho[cons]
                                cons = cons + 1
                print(num_cons1)
                print(cons)
                for cons in range(num_cons3):
                    constant = constant + rho[num_cons2+cons]*Q[0][0]
                cons = 0
                for i in range(N):
                    for c in range(V[i]):
                        for t in range(T):
                            constant = constant + rho[num_cons3+cons]*Q[i][c+1]/beta[i][0,0] + rho[num_cons3+cons]*Q[i][c+2]/beta[i][1,0] + rho[num_cons3+cons]*Q[i][c+3]/beta[i][2,0]
                            cons = cons + 3
                cons = 0
                for c in list(set(C_ALL)-set(D_ALL)-set(V_ALL)):
                    for t in range(T):
                        constant = constant + rho[num_cons4+cons]*W*Jam_N_ALL[c]
                        cons = cons + 1
                cons = 0
                for i in range(N):
                    for c in V[i]:
                        for t in range(T):
                            constant = constant + rho[num_cons5+cons]*W*Jam_N_ALL[int(c+add[i]+1)] + rho[num_cons5+cons+1]*W*Jam_N_ALL[int(c+add[i]+2)] + rho[num_cons5+cons+2]*W*Jam_N_ALL[int(c+add[i]+3)]
                            cons = cons + 4
                cons = 0
                for i in range(len(O_ALL)):
                    for t in range(T):
                        constant = constant + rho[num_cons8+cons]*Demand_ALL[i]
                        cons = cons + 1
                cons = 0
                for c in C_ALL:
                    constant = constant + rho[num_cons9+cons]*n_init_all[c,xi]
                    cons = cons + 1
                m_master.addConstr(constant - scope <= 0)
            if m_sub.status == 2:
                num_optimal_sub = num_optimal_sub + 1
                opt_sub[xi] = m_sub.objval
                constant = 0
                scope = 0
                scope = gb.quicksum(-w[i,j,t]*Q_ALL[c]*cons1[i,j,c,t].Pi for i in range(N) for j in range(4) for c in I[i][j] for t in range(T))
                constant = constant + gb.quicksum(Q[0][0]*cons3[c, t].Pi for c in list(set(C_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(D_ALL)) for t in range(T))
                for i in range(N):
                    constant = constant + gb.quicksum(Q[i][c+1]/beta[i][0,0]*cons4[i*3][c,t].Pi + Q[i][c+2]/beta[i][1,0]*cons4[i*3+1][c,t].Pi + Q[i][c+3]/beta[i][2,0]*cons4[i*3+2][c,t].Pi for c in V[i] for t in range(T))
                constant = constant + gb.quicksum(W*Jam_N_ALL[c]*cons5[c,t].Pi for c in list(set(C_ALL)-set(D_ALL)-set(V_ALL)) for t in range(T))
                for i in range(N):
                    constant = constant + gb.quicksum(W*Jam_N_ALL[int(c+add[i]+1)]*cons6[i*4][c,t].Pi + W*Jam_N_ALL[int(c+add[i]+2)]*cons6[i*4+1][c,t].Pi + W*Jam_N_ALL[int(c+add[i]+3)]*cons6[i*4+2][c,t].Pi for c in V[i] for t in range(T))
                constant = constant + gb.quicksum(Demand_ALL[i]*cons9[i,t].Pi for i in range(len(O_ALL)) for t in range(T))
                constant = constant + gb.quicksum(n_init_all[c,xi]*cons10[c].Pi for c in C_ALL)
                if theta_tilde[xi] < opt_sub[xi]:
                    m_master.addConstr(theta[xi] >= constant - scope)
        end_sub = time.time()
        m_master.update()
        m_master.printStats()
        start_master = time.time()
        m_master.optimize()
        end_master = time.time()
        w_tilde = m_master.getAttr('X',w)
        theta_tilde = m_master.getAttr('X',theta)
        if lb < m_master.objval:
            lb = m_master.objval
        if num_optimal_sub == num_scenario:
            if ub > sum(opt_sub)/num_scenario:
                ub = sum(opt_sub)/num_scenario
        print("iteration " + str(num_ite))
        print("upper bound is %f" % ub)
        print("lower bound is %f" % lb)
        print("time to solve master problem is %f" % (end_master - start_master))
        print("time to solve sub problem is %f" % (end_sub - start_sub))
        f = open('benders/T' + str(T) + '_S' + str(num_scenario) + '_bound.log', 'a+')
        print("iteration " + str(num_ite), file = f)
        print("upper bound is %f" % ub, file = f)
        print("lower bound is %f" % lb, file = f)
        print("time to solve master problem is %f" % (end_master - start_master), file = f)
        print("time to solve sub problem is %f" % (end_sub - start_sub), file = f)
if __name__ == '__main__':
    Benders()