import numpy as np
import gurobipy as gb
from data import Network
from data_global import *

def LB (num_group = 10, num_scenario = 100):
    # for every sample set, solve two-stage stochastic problem to get lower bound
    opt_value = [None]*num_group
    opt_value_delay = [None]*num_group
    opt_solution = [None]*num_group
    sample_size = num_scenario
    # build model
    m = gb.Model()
    y = m.addVars(len(C_ALL), T, sample_size, lb=0, vtype=gb.GRB.CONTINUOUS)
    n = m.addVars(len(C_ALL), T+1, sample_size, lb=0, vtype=gb.GRB.CONTINUOUS)
    # w = m.addVars(N, 4, T, lb=0, ub=1,  vtype=gb.GRB.BINARY)      #binary variable
    w = m.addVars(N, 4, T, lb=0, ub=1,  vtype=gb.GRB.CONTINUOUS) 
    new_w = m.addVars(N, T, lb=0, ub=1, vtype=gb.GRB.CONTINUOUS)
    u = m.addVars(N, T-1, lb=0, ub=1,  vtype=gb.GRB.CONTINUOUS)
    m.setObjective(1/sample_size * (gb.quicksum(gb.quicksum(gb.quicksum(t * y[c,t,xi] for c in D_ALL) for t in range(T)) for xi in range(sample_size))
            + alpha * gb.quicksum(gb.quicksum(gb.quicksum(t* y[c,t,xi] for c in list(set(C_ALL)-set(D_ALL))) for t in range(T)) for xi in range(sample_size))), gb.GRB.MINIMIZE)

    m.addConstrs((n[c,0,xi] == 0 for c in C_ALL for xi in range(sample_size)), name = 'n0')
    m.addConstrs(y[c,t,xi]-n[c,t,xi]<=0 for c in C_ALL for t in range(T) for xi in range(sample_size))
    m.addConstrs(y[c,t,xi]<=Q[0][0] for c in list(set(C_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(D_ALL)) for t in range(T) for xi in range(sample_size))
    for i in range(N):
        m.addConstrs(y[c+add[i],t,xi]<=Q[i][c+1]/beta[i][0,0] for c in V[i] for t in range(T) for xi in range(sample_size))
        m.addConstrs(y[c+add[i],t,xi]<=Q[i][c+2]/beta[i][1,0] for c in V[i] for t in range(T) for xi in range(sample_size))
        m.addConstrs(y[c+add[i],t,xi]<=Q[i][c+3]/beta[i][2,0] for c in V[i] for t in range(T) for xi in range(sample_size))
    m.addConstrs(y[c,t,xi]-w[i,j,t]*Q_ALL[c]<=0 for i in range(N) for j in range(4) for c in I[i][j] for t in range(T) for xi in range(sample_size))
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

    m.addConstrs(new_w[i,t] == w[i,0,t] + w[i,1,t] for i in range(N) for t in range(T))
    m.addConstrs(w[i,0,t]+w[i,1,t]+w[i,2,t]+w[i,3,t]==1 for i in range(N) for t in range(T))
    m.addConstrs(new_w[i,t]-new_w[i,t-1]-u[i,t-1]<=0 for i in range(N) for t in range(1,T))
    m.addConstrs( -new_w[i,t]+new_w[i,t-1]-u[i,t-1]<=0 for i in range(N) for t in range(1,T))
    m.addConstrs(-new_w[i,t]-new_w[i,t-1]+u[i,t-1]<=0 for i in range(N) for t in range(1,T))
    m.addConstrs(new_w[i,t]+new_w[i,t-1]+u[i,t-1]<=2 for  i in range(N) for t in range(1,T))
    m.addConstrs(u[i,t]+u[i,t+1]+u[i,t+2]+u[i,t+3]<=1 for  i in range(N) for t in range(0,T-4))
    m.addConstrs(new_w[i,t]+new_w[i,t+1]+new_w[i,t+2]+new_w[i,t+3]+new_w[i,t+4]+new_w[i,t+5]+new_w[i,t+6]+new_w[i,t+7] <= 8 for  i in range(N) for t in range(0,T-7))
    m.addConstrs(-(new_w[i,t]+new_w[i,t+1]+new_w[i,t+2]+new_w[i,t+3]+new_w[i,t+4]+new_w[i,t+5]+new_w[i,t+6]+new_w[i,t+7])<=-1 for  i in range(N) for t in range(0,T-7))
    # m.addConstrs(n[c,T]==0 for c in C_ALL)
    # m.addConstrs(n[c,0] == 1 for c in C_ALL)
    m.Params.TimeLimit=7200
    m.Params.LogFile = 'n_init_random/LP_Gurobi_T' + str(T) + '_S' + str(num_scenario) + '_LB.log'
    m.Params.LogToConsole = 0
    m.update()

    for m_group in range(num_group):
        network_data = Network(N_edge, True, sample_size)
        n_init = network_data.n_init
        n_init_all = n_init[0]
        for i in range(1,N):
            n_init_all = np.concatenate((n_init_all, n_init[i]), axis=0)
        # update model
        m.update()
        m.printStats()
        for c in range(len(C_ALL)):
            for xi in range(sample_size):
                m.remove(m.getConstrByName("n0[%d,%d]" %(c,xi)))
        # m.remove(m.getConstrs()[0:len(C_ALL)*sample_size])
        m.addConstrs((n[c,0,xi] == n_init_all[c,xi] for c in C_ALL for xi in range(sample_size)), name = 'n0')
        # m.addConstrs((n[c,0,xi] == 8 for c in C_ALL for xi in range(sample_size)), name = 'n0')
        m.update()
        m.printStats()
        m.optimize()
        opt_value[m_group] = m.objval
        y_value = m.getAttr('X',y)
        opt_value_delay[m_group] = 1/sample_size * sum(sum(sum(t * y_value[c,t,xi] for c in D_ALL) for t in range(T)) for xi in range(sample_size))
        opt_solution[m_group] = m.getAttr('X', w)
        print(m.status)

    lb = sum(opt_value)/num_group
    lb_delay = sum(opt_value_delay)/num_group
    return opt_value, opt_solution, lb, lb_delay


def UB(opt_value, opt_solution, num_scenario = 1000):
    # calculate upper bound by testing on large size of sample set
    num_group = len(opt_value)
    # sample set
    network_data = Network(N_edge, True, num_scenario)
    n_init = network_data.n_init
    n_init_all = n_init[0]
    for i in range(1,N):
        n_init_all = np.concatenate((n_init_all, n_init[i]), axis=0)
    ub = [None]*num_group
    ub_delay = [None]*num_group
    for m_group in range(num_group):
        optval = [None]*num_scenario
        optval_delay = [None]*num_scenario
        for xi in range(num_scenario):
            w = opt_solution[m_group]
            m = gb.Model()
            y = m.addVars(len(C_ALL), T, lb=0, vtype=gb.GRB.CONTINUOUS)
            n = m.addVars(len(C_ALL), T+1, lb=0, vtype=gb.GRB.CONTINUOUS)
            m.setObjective(gb.quicksum(gb.quicksum(t * y[c,t] for c in D_ALL) for t in range(T))
                    + alpha * gb.quicksum(gb.quicksum(t* y[c,t] for c in list(set(C_ALL)-set(D_ALL))) for t in range(T)), gb.GRB.MINIMIZE)
            m.addConstrs(y[c,t]-n[c,t]<=0 for c in C_ALL for t in range(T))
            m.addConstrs(y[c,t]<=Q[0][0] for c in list(set(C_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(D_ALL)) for t in range(T))
            for i in range(N):
                m.addConstrs(y[c+add[i],t]<=Q[i][c+1]/beta[i][0,0] for c in V[i] for t in range(T))
                m.addConstrs(y[c+add[i],t]<=Q[i][c+2]/beta[i][1,0] for c in V[i] for t in range(T))
                m.addConstrs(y[c+add[i],t]<=Q[i][c+3]/beta[i][2,0] for c in V[i] for t in range(T))
            m.addConstrs(y[c,t]-w[i,j,t]*Q_ALL[c]<=0 for i in range(N) for j in range(4) for c in I[i][j] for t in range(T))
            m.addConstrs(y[c,t]+W*n[proc_all[c],t]<=W*Jam_N_ALL[c] for c in list(set(C_ALL)-set(D_ALL)-set(V_ALL)) for t in range(T))

            for i in range(N):
                m.addConstrs(beta[i][0,0]*y[c+add[i],t]+W*n[c+add[i]+1,t]<=W*Jam_N_ALL[int(c+add[i]+1)] for c in V[i] for t in range(T))
                m.addConstrs(beta[i][1,0]*y[c+add[i],t]+W*n[c+add[i]+2,t]<=W*Jam_N_ALL[int(c+add[i]+2)] for c in V[i] for t in range(T))
                if beta[i].shape[0]==3:
                    m.addConstrs(beta[i][2,0]*y[c+add[i],t]+W*n[c+add[i]+3,t]<=W*Jam_N_ALL[int(c+add[i]+3)] for c in V[i] for t in range(T))
                m.addConstrs(n[c+add[i],t+1] - n[c+add[i],t] - beta[i][c-pred[i][c]-1,0]*y[pred[i][c]+add[i],t] + y[c+add[i],t] == 0 for c in I1[i]+I2[i]+I3[i]+I4[i] for t in range(T))

            m.addConstrs(n[c,t+1] - n[c,t] - y[pred_all[c],t] + y[c,t] == 0 for c in list(set(C_ALL)-set(O_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(M_ALL)) for t in range(T))
            m.addConstrs(n[c,t+1] - n[c,t] - gb.quicksum(y[d,t] for d in pred_all[c]) + y[c,t] == 0 for c in M_ALL for t in range(T))
            # m.addConstrs(n[c,t+1] - n[c,t] + y[c,t] == Demand[0,0] for c in O_ALL for t in range(T))
            m.addConstrs(n[O_ALL[i],t+1] - n[O_ALL[i],t] + y[O_ALL[i],t] == Demand_ALL[i] for i in range(len(O_ALL)) for t in range(T))
            m.addConstrs((n[c,0] == n_init_all[c,xi] for c in C_ALL), name = 'n0')
            m.Params.LogToConsole = 0
            m.Params.TimeLimit=7200
            m.Params.LogFile = 'n_init_random/LP_Gurobi_T' + str(T) + '_S' + str(num_scenario) + '_UB.log'
            m.optimize()
            optval[xi] = m.objval
            y_value = m.getAttr('X',y)
            optval_delay[xi] = sum(sum(t * y_value[c,t] for c in D_ALL) for t in range(T))
        ub[m_group] = sum(optval)/num_scenario
        ub_delay[m_group] = sum(optval_delay)/num_scenario
    return min(ub), min(ub_delay)

            
if __name__ == '__main__':
    opt_val, opt_solution, lb, lb_delay = LB(1,1)
    ub, ub_delay = UB(opt_val, opt_solution, 1)
    print('upper bound:' + str(ub))
    print('upper bound of delay term: ' + str(ub_delay))
    print('lower bound:' + str(lb))
    print('lower bound of delay term:' + str(lb_delay))
    print('gap:' + str(ub-lb))
    print('gap of delay term:' + str(ub_delay - lb_delay))
