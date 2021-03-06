from data_global import *
from warmup import Warmup
import numpy as np
import gurobipy as gb
import time
import matplotlib.pyplot as plt

def Benders(epsilon = 0.01, num_scenario = 100, T = 20):
    ub = np.infty
    lb = -np.infty
    f = open('benders_decentralize/T' + str(T) + '_S' + str(num_scenario) + '_bound_LP.log', 'w+')
    f = open('benders_decentralize/T' + str(T) + '_S' + str(num_scenario) + '_w.log', 'w+')
    f = open('benders_decentralize/T' + str(T) + '_S' + str(num_scenario) + '_y.log', 'w+')
    f = open('benders_decentralize/T' + str(T) + '_S' + str(num_scenario) + '_n.log', 'w+')
    # generalize all samples
    network_data = Network(N_edge, True, num_scenario, T)
    n_init_all = Warmup(False)
    Demand_ALL = network_data.Demand[0]
    for i in range(1,N):
        Demand_ALL = Demand_ALL + [de for de in network_data.Demand[i]]
    beta = network_data.beta
    # build master model
    m_master = [None]*N
    w = [None]*N
    theta = [None]*N
    for inter in range(N):
        m_master[inter] = gb.Model()
        w[inter] = m_master[inter].addVars(4, T, lb=0, ub=1, vtype=gb.GRB.CONTINUOUS)
        theta[inter] = m_master[inter].addVars(num_scenario, lb=0, name = 'theta' + str(inter))
        m_master[inter].setObjective(1/num_scenario*gb.quicksum(theta[inter][xi] for xi in range(num_scenario)), gb.GRB.MINIMIZE)
        new_w = m_master[inter].addVars(T, lb=0, ub=1, vtype=gb.GRB.CONTINUOUS)
        u = m_master[inter].addVars(T-1, lb=0, ub=1,  vtype=gb.GRB.CONTINUOUS)
        m_master[inter].addConstrs(new_w[t] == w[inter][0,t] + w[inter][1,t] for t in range(T))
        m_master[inter].addConstrs(w[inter][0,t]+w[inter][1,t]+w[inter][2,t]+w[inter][3,t]==1 for t in range(T))
        m_master[inter].addConstrs(new_w[t]-new_w[t-1]-u[t-1]<=0 for t in range(1,T))
        m_master[inter].addConstrs( -new_w[t]+new_w[t-1]-u[t-1]<=0 for t in range(1,T))
        m_master[inter].addConstrs(-new_w[t]-new_w[t-1]+u[t-1]<=0 for t in range(1,T))
        m_master[inter].addConstrs(new_w[t]+new_w[t-1]+u[t-1]<=2 for t in range(1,T))
        m_master[inter].addConstrs(u[t]+u[t+1]+u[t+2]+u[t+3]<=1 for t in range(0,T-4))
        m_master[inter].addConstrs(new_w[t]+new_w[t+1]+new_w[t+2]+new_w[t+3]+new_w[t+4]+new_w[t+5]+new_w[t+6]+new_w[t+7] <= 8 for t in range(0,T-7))
        m_master[inter].addConstrs(-(new_w[t]+new_w[t+1]+new_w[t+2]+new_w[t+3]+new_w[t+4]+new_w[t+5]+new_w[t+6]+new_w[t+7])<=-1 for t in range(0,T-7))
        m_master[inter].Params.LogToConsole = 0
        m_master[inter].Params.TimeLimit = 7200
        m_master[inter].Params.LogFile = 'benders_decentralize/T' + str(T) + '_S' + str(num_scenario) + '_master.log'
        m_master[inter].update()
    # initialized integer variables (fist-stage variables)
    # w_tilde = np.zeros((N,4,T))
    # theta_tilde = np.ones((N,num_scenario))*(-np.infty)
    w_tilde = [None]*N
    theta_tilde = [None]*N
    for i in range(N):
        m_master[i].optimize()
        w_tilde[i] = m_master[i].getAttr('X', w[i])
        theta_tilde[i] = np.ones(num_scenario)*(-np.infty)
        for xi in range(num_scenario):
            theta[i][xi].lb = -np.infty
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
    time_sub = []
    time_master = []
    time_ite = []
    gap = []
    start_global = time.time()
    while (ub-lb)/abs(lb) > epsilon or lb == -np.infty:
        start_ite = time.time()
        num_ite = num_ite + 1
        num_optimal_sub = 0
        delay_sub = np.zeros((N,num_scenario))
        ctm_sub = np.zeros((N,num_scenario))
        opt_sub = np.zeros((N,num_scenario))
        lb_sub = np.zeros(num_scenario)
        start_sub = time.time()
        for xi in range(num_scenario):
            # solve subproblem
            m_sub = gb.Model()
            y = m_sub.addVars(len(C_ALL), T, lb=0, vtype=gb.GRB.CONTINUOUS)
            n = m_sub.addVars(len(C_ALL), T+1, lb=0, vtype=gb.GRB.CONTINUOUS)
            m_sub.setObjective(-gb.quicksum(gb.quicksum((T - t) * y[c,t] for c in C_ALL) for t in range(T)) -
                    alpha*gb.quicksum(gb.quicksum(n[c,t] for c in D_ALL) for t in range(T)), gb.GRB.MINIMIZE)
            cons1 = m_sub.addConstrs(y[c,t]-w_tilde[i][j,t]*Q_ALL[c]<=0 for i in range(N) for j in range(4) for c in I[i][j] for t in range(T))
            cons2 = m_sub.addConstrs(y[c,t]-n[c,t]<=0 for c in C_ALL for t in range(T))
            cons3 = m_sub.addConstrs(y[c+add[i],t]<=Q[i][0] for i in range(N) for c in list(set(C[i])-set(I1[i])-set(I2[i])-set(I3[i])-set(I4[i])-set(D[i])) for t in range(T))
            cons4 = [None]*3*N
            for i in range(N):
                cons4[i*3] = m_sub.addConstrs(y[c+add[i],t]<=Q[i][c+1]/beta[i][xi][t][0,0] for c in V[i] for t in range(T))
                cons4[i*3+1] = m_sub.addConstrs(y[c+add[i],t]<=Q[i][c+2]/beta[i][xi][t][1,0] for c in V[i] for t in range(T))
                cons4[i*3+2] = m_sub.addConstrs(y[c+add[i],t]<=Q[i][c+3]/beta[i][xi][t][2,0] for c in V[i] for t in range(T))
            cons5 = m_sub.addConstrs(y[c+add[i],t]+W*n[proc_all[c+add[i]],t]<=W*Jam_N_ALL[int(c+add[i])] for i in range(N) for c in list(set(C[i])-set(D[i])-set(V[i])) for t in range(T))
            cons6 = [None]*4*N
            for i in range(N):
                cons6[i*4] = m_sub.addConstrs(beta[i][xi][t][0,0]*y[c+add[i],t]+W*n[c+add[i]+1,t]<=W*Jam_N_ALL[int(c+add[i]+1)] for c in V[i] for t in range(T))
                cons6[i*4+1] = m_sub.addConstrs(beta[i][xi][t][1,0]*y[c+add[i],t]+W*n[c+add[i]+2,t]<=W*Jam_N_ALL[int(c+add[i]+2)] for c in V[i] for t in range(T))
                cons6[i*4+2] = m_sub.addConstrs(beta[i][xi][t][2,0]*y[c+add[i],t]+W*n[c+add[i]+3,t]<=W*Jam_N_ALL[int(c+add[i]+3)] for c in V[i] for t in range(T))
                cons6[i*4+3] = m_sub.addConstrs(n[c+add[i],t+1] - n[c+add[i],t] - beta[i][xi][t][c-pred[i][c]-1,0]*y[pred[i][c]+add[i],t] + y[c+add[i],t] == 0 for c in I1[i]+I2[i]+I3[i]+I4[i] for t in range(T))
            cons7 = m_sub.addConstrs(n[c,t+1] - n[c,t] - y[pred_all[c],t] + y[c,t] == 0 for c in list(set(C_ALL)-set(O_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(M_ALL)) for t in range(T))
            cons8 = m_sub.addConstrs(n[c,t+1] - n[c,t] - gb.quicksum(y[d,t] for d in pred_all[c]) + y[c,t] == 0 for c in M_ALL for t in range(T))
            cons9 = m_sub.addConstrs(n[O_ALL[i],t+1] - n[O_ALL[i],t] + y[O_ALL[i],t] == Demand_ALL[i][xi][t] for i in range(len(O_ALL)) for t in range(T))
            cons10 = m_sub.addConstrs((n[c,0] == n_init_all[c] for c in C_ALL), name = 'n0')
            m_sub.Params.LogToConsole = 0
            m_sub.Params.TimeLimit=7200
            m_sub.Params.LogFile = 'benders_decentralize/T' + str(T) + '_S' + str(num_scenario) + '_sub.log'
            m_sub.Params.InfUnbdInfo = 1
            m_sub.update()
            # m_sub.printStats()
            m_sub.optimize()
            if m_sub.status == 3:
                rho = m_sub.FarkasDual()
                constant = [0]*N
                scope = [0]*N
                cons = 0
                for i in range(N):
                    for j in range(4):
                        for c in I[i][j]:
                            for t in range(T):
                                scope[i] = scope[i] - w[i][j,t]*Q_ALL[c]*rho[cons]
                                cons = cons + 1
                print(num_cons1)
                print(cons)
                cons = 0
                for i in range(N):
                    constant[i] = constant[i] + rho[num_cons2+cons]*Q[i][0]
                    cons =  (len(C[i])-len(I1[i])-len(I2[i])-len(I3[i])-len(I4[i])-len(D[i]))*T
                cons = 0
                for i in range(N):
                    for c in range(V[i]):
                        for t in range(T):
                            constant[i] = constant[i] + rho[num_cons3+cons]*Q[i][c+1]/beta[i][0,0] + rho[num_cons3+cons]*Q[i][c+2]/beta[i][1,0] + rho[num_cons3+cons]*Q[i][c+3]/beta[i][2,0]
                            cons = cons + 3
                cons = 0
                for i in range(N):
                    for c in list(set(C[i])-set(D[i])-set(V[i])):
                        for t in range(T):
                            constant[i] = constant[i] + rho[num_cons4+cons]*W*Jam_N_ALL[int(c+add[i])]
                            cons = cons + 1
                cons = 0
                for i in range(N):
                    for c in V[i]:
                        for t in range(T):
                            constant = constant + rho[num_cons5+cons]*W*Jam_N_ALL[int(c+add[i]+1)] + rho[num_cons5+cons+1]*W*Jam_N_ALL[int(c+add[i]+2)] + rho[num_cons5+cons+2]*W*Jam_N_ALL[int(c+add[i]+3)]
                            cons = cons + 4
                cons = 0
                for i in range(N):
                    for o_index in O[i]:
                        for t in range(T):
                            constant[i] = constant[i] + rho[num_cons8+cons]*Demand_ALL[o_index + sum(len(O[j]) for j in range(i))]
                            cons = cons + 1
                cons = 0
                for i in range(N):
                    for c in C[i]:
                        constant[i] = constant[i] + rho[num_cons9+cons]*n_init_all[int(c+add[i]),xi]
                        cons = cons + 1
                m_master.addConstr(constant[i] - scope[i] <= 0)
            if m_sub.status == 2:
                num_optimal_sub = num_optimal_sub + 1
                n_value = m_sub.getAttr('X',n)
                y_value = m_sub.getAttr('X',y)
                for i in range(N):
                    opt_sub[i,xi] = -alpha * sum(sum(n_value[c+add[i],t] for c in D[i]) for t in range(T)) - sum(sum((T-t)* y_value[c+add[i],t] for c in C[i]) for t in range(T))
                    ctm_sub[i,xi] = -sum(sum((T - t) * y_value[c,t] for c in C[i]) for t in range(T))
                    delay_sub[i,xi] = -sum(sum(n_value[c,t] for c in D[i]) for t in range(T))
                lb_sub[xi] = m_sub.objval
                constant = [0]*N
                scope = [None]*N
                for i in range(N):
                    scope[i] = gb.quicksum(-w[i][j,t]*Q_ALL[c]*cons1[i,j,c,t].Pi for j in range(4) for c in I[i][j] for t in range(T))
                    constant[i] = constant[i] + gb.quicksum(Q[i][0]*cons3[i,c,t].Pi for c in list(set(C[i])-set(I1[i])-set(I2[i])-set(I3[i])-set(I4[i])-set(D[i])) for t in range(T))
                    constant[i] = constant[i] + gb.quicksum(Q[i][c+1]/beta[i][xi][t][0,0]*cons4[i*3][c,t].Pi + Q[i][c+2]/beta[i][xi][t][1,0]*cons4[i*3+1][c,t].Pi + Q[i][c+3]/beta[i][xi][t][2,0]*cons4[i*3+2][c,t].Pi for c in V[i] for t in range(T))
                    constant[i] = constant[i] + gb.quicksum(W*Jam_N_ALL[int(c+add[i])]*cons5[i,c,t].Pi for c in list(set(C[i])-set(D[i])-set(V[i])) for t in range(T))
                    constant[i] = constant[i] + gb.quicksum(W*Jam_N_ALL[int(c+add[i]+1)]*cons6[i*4][c,t].Pi + W*Jam_N_ALL[int(c+add[i]+2)]*cons6[i*4+1][c,t].Pi + W*Jam_N_ALL[int(c+add[i]+3)]*cons6[i*4+2][c,t].Pi for c in V[i] for t in range(T))
                    constant[i] = constant[i] + gb.quicksum(Demand_ALL[o_index+sum(len(O[j]) for j in range(i))][xi][t]*cons9[o_index+sum(len(O[j]) for j in range(i)),t].Pi for o_index in range(len(O[i])) for t in range(T))
                    constant[i] = constant[i] + gb.quicksum(n_init_all[int(c+add[i])]*cons10[int(c+add[i])].Pi for c in C[i])
                    if theta_tilde[i][xi] < opt_sub[i,xi]:
                        # theta[i][xi] = m_master[i].getVarByName('theta'+str(i)+'['+str(xi)+']')
                        # m_master[i].addConstr(theta[i][xi] >= constant[i] - scope[i])
                        add_const = m_master[i].addConstr(theta[i][xi] >= constant[i] - scope[i])
                        add_const.Lazy = 1            
        end_sub = time.time()
        start_master = time.time()
        m_master_obj = 0
        for i in range(N):
            m_master[i].update()
            # m_master[i].printStats()
            m_master[i].optimize()
            w_tilde[i] = m_master[i].getAttr('X',w[i])
            theta_tilde[i] = m_master[i].getAttr('X',theta[i])
            m_master_obj = m_master_obj + m_master[i].objval
        end_master = time.time()
        
        if lb < m_master_obj:
            lb = m_master_obj
        if num_optimal_sub == num_scenario:
            if ub > sum(lb_sub)/num_scenario:
                ub = sum(lb_sub)/num_scenario
        end_global = time.time()
        time_sub.append(end_sub-start_sub)
        time_master.append(end_master-start_master)
        time_ite.append(end_global-start_ite)
        gap.append((ub-lb)/abs(lb))
        print("iteration " + str(num_ite))
        print("upper bound is %f" % ub)
        print("lower bound is %f" % lb)
        print("gap is %f" % ((ub-lb)/abs(lb)))
        print("time to solve master problem is %f" % (end_master - start_master))
        print("time to solve sub problem is %f" % (end_sub - start_sub))
        print("time for this iteration is  %f" % (end_global-start_ite))
        print("all time to solve problem is %f" % (end_global - start_global))
        f = open('benders_decentralize/T' + str(T) + '_S' + str(num_scenario) + '_bound_LP.log', 'a+')
        print("iteration " + str(num_ite), file = f)
        print("upper bound is %f" % ub, file = f)
        print("lower bound is %f" % lb, file = f)
        print("gap is %f" % ((ub-lb)/abs(lb)), file = f)
        print("delay term is %f" % (sum(sum(delay_sub))/num_scenario), file = f)
        print("ctm term is %f" % (sum(sum(ctm_sub))/num_scenario), file = f)
        print("time to solve master problem is %f" % (end_master - start_master), file = f)
        print("time to solve sub problem is %f" % (end_sub - start_sub), file = f)
        print("time for this iteration is  %f" % (end_global-start_ite), file = f)
        print("all time to solve problem is %f" % (end_global - start_global), file = f)
        f = open('benders_decentralize/T' + str(T) + '_S' + str(num_scenario) + '_w.log', 'a+')
        print("iteration " + str(num_ite), file = f)
        for i in range(N):
            for j in range(4):
                for t in range(T):
                    print(w_tilde[i][j,t], end = ",", file=f)
            print("\n", file=f)
        
        if end_global - start_global > 7200:
            break
    
    # evaluate solution
    m_eval = gb.Model()
    y = m_eval.addVars(len(C_ALL), T, lb=0, vtype=gb.GRB.CONTINUOUS)
    n = m_eval.addVars(len(C_ALL), T+1, lb=0, vtype=gb.GRB.CONTINUOUS)
    m_eval.setObjective(-gb.quicksum(gb.quicksum((T - t) * y[c,t] for c in C_ALL) for t in range(T)), gb.GRB.MINIMIZE)
    m_eval.addConstrs(y[c,t]-w_tilde[i][j,t]*Q_ALL[c]<=0 for i in range(N) for j in range(4) for c in I[i][j] for t in range(T))
    m_eval.addConstrs(y[c,t]-n[c,t]<=0 for c in C_ALL for t in range(T))
    m_eval.addConstrs(y[c,t]<=Q[0][0] for c in list(set(C_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(D_ALL)) for t in range(T))
    for i in range(N):
        m_eval.addConstrs(y[c+add[i],t]<=Q[i][c+1]/beta[i][xi][t][0,0] for c in V[i] for t in range(T))
        m_eval.addConstrs(y[c+add[i],t]<=Q[i][c+2]/beta[i][xi][t][1,0] for c in V[i] for t in range(T))
        m_eval.addConstrs(y[c+add[i],t]<=Q[i][c+3]/beta[i][xi][t][2,0] for c in V[i] for t in range(T))
    m_eval.addConstrs(y[c,t]+W*n[proc_all[c],t]<=W*Jam_N_ALL[c] for c in list(set(C_ALL)-set(D_ALL)-set(V_ALL)) for t in range(T))
    for i in range(N):
        m_eval.addConstrs(beta[i][xi][t][0,0]*y[c+add[i],t]+W*n[c+add[i]+1,t]<=W*Jam_N_ALL[int(c+add[i]+1)] for c in V[i] for t in range(T))
        m_eval.addConstrs(beta[i][xi][t][1,0]*y[c+add[i],t]+W*n[c+add[i]+2,t]<=W*Jam_N_ALL[int(c+add[i]+2)] for c in V[i] for t in range(T))
        m_eval.addConstrs(beta[i][xi][t][2,0]*y[c+add[i],t]+W*n[c+add[i]+3,t]<=W*Jam_N_ALL[int(c+add[i]+3)] for c in V[i] for t in range(T))
        m_eval.addConstrs(n[c+add[i],t+1] - n[c+add[i],t] - beta[i][xi][t][c-pred[i][c]-1,0]*y[pred[i][c]+add[i],t] + y[c+add[i],t] == 0 for c in I1[i]+I2[i]+I3[i]+I4[i] for t in range(T))
    m_eval.addConstrs(n[c,t+1] - n[c,t] - y[pred_all[c],t] + y[c,t] == 0 for c in list(set(C_ALL)-set(O_ALL)-set(I1_ALL)-set(I2_ALL)-set(I3_ALL)-set(I4_ALL)-set(M_ALL)) for t in range(T))
    m_eval.addConstrs(n[c,t+1] - n[c,t] - gb.quicksum(y[d,t] for d in pred_all[c]) + y[c,t] == 0 for c in M_ALL for t in range(T))
    m_eval.addConstrs(n[O_ALL[i],t+1] - n[O_ALL[i],t] + y[O_ALL[i],t] == Demand_ALL[i][xi][t] for i in range(len(O_ALL)) for t in range(T))
    m_eval.addConstrs((n[c,0] == n_init_all[c] for c in C_ALL))
    m_eval.optimize()
    n_value = m_eval.getAttr('X', n)
    print("the objective value is %f" % sum(sum(n_value[c,t] for c in D_ALL) for t in range(T)))
    f = open('benders_decentralize/T' + str(T) + '_S' + str(num_scenario) + '_bound_LP.log', 'a+')
    print("the objective value is %f" % sum(sum(n_value[c,t] for c in D_ALL) for t in range(T)), file = f)
    f = open('benders_decentralize/T' + str(T) + '_S' + str(num_scenario) + '_n.log', 'a+')
    for i in range(N):
        for t in range(T+1):
            print(n_value[i,t], end = ",", file=f)
        print("\n", file=f)
    f = open('benders_decentralize/T' + str(T) + '_S' + str(num_scenario) + '_y.log', 'a+')
    for i in range(N):
        for t in range(T):
            print(y_value[i,t], end = ",", file=f)
        print("\n", file=f)
    # draw the figure of time
    plt.figure()
    plt.plot(time_sub, label = 'time of sub problem')
    plt.plot(time_master, label = 'time of master problem')
    plt.plot(time_ite, label = 'time of iteration')
    plt.xlabel('iterations')
    plt.ylabel('time(s)')
    plt.legend()
    plt.savefig('benders_decentralize/T' + str(T) + '_S' + str(num_scenario) + '_time.jpg')

    plt.figure()
    plt.plot(gap)
    plt.xlabel('iteration')
    plt.ylabel('gap')
    plt.savefig('benders_decentralize/T' + str(T) + '_S' + str(num_scenario) + '_gap.jpg')
    
if __name__ == '__main__':
    num_scenario = 100
    Benders(0.0001, 100, 20)
    # Benders(0.0001, 100, 50)
    # Benders(0.0001, 100, 100)
    