import numpy as np
from numpy.linalg import inv
#from numpy.linalg import norm
from cvxpy.atoms.norm import norm
from cvxpy.atoms.elementwise.square import square
from cvxpy.atoms.sum_squares import sum_squares
from cvxpy.atoms.affine.reshape import reshape
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Process, Manager, cpu_count, Pool
import cvxpy as cp
from numpy import linalg as LA
import time
#import mosek
import gurobipy as gb

class ADMM:
    def __init__(self, input_data, parallel = False, rounding = True):

        self.parallel = parallel
        self.rounding = rounding
        self.data={}
        self.data['num_inter'] = input_data['num_inter'] #number of intersections
        self.data['opt_time'] = int(input_data['optimization_time'])
        self.data['ratio_W'] = input_data['ratio_W']
        self.data['jam_N'] = input_data['jam_N']
        self.data['capacity_Q'] = input_data['capacity_Q']
        self.data['rho'] = input_data['penalty_rho']
        self.data['alpha'] = input_data['weight_alpha']
        self.data['phases'] = input_data['phases']

        self.data['C'] = input_data['C']
        self.data['O'] = input_data['O']
        self.data['D'] = input_data['D']
        self.data['I1'] = input_data['I1']
        self.data['I2'] = input_data['I2']
        self.data['I3'] = input_data['I3']
        self.data['I4'] = input_data['I4']
        self.data['BO'] = input_data['BO']
        self.data['BI'] = input_data['BI']
        self.data['V'] = input_data['V']
        self.data['M'] = input_data['M']
        self.data['beta'] = input_data['beta']
        self.data['proc'] = input_data['proc']
        self.data['pred'] = input_data['pred']

        self.Demand=[None] * self.data['num_inter']
        for i in range(self.data['num_inter']):
            self.Demand[i] = np.ones((len(self.data['O'][i]),self.data['opt_time']))  #will update later
            for j in range(len(self.Demand[i])):
                self.Demand[i][j,:] = input_data['Demand'][i][j]

        self.n_init = [None] * self.data['num_inter']
        self.n = [None] * self.data['num_inter']
        self.y = [None] * self.data['num_inter']
        self.w = np.zeros((self.data['num_inter'], self.data['phases'], self.data['opt_time']))
        self.w_B = np.zeros((self.data['num_inter'], self.data['phases'], self.data['opt_time'])) # Binary variables 
        self.u = np.zeros((self.data['num_inter'], self.data['opt_time']-1))
        self.s = [None] * self.data['num_inter']
        self.lamb = [None] * self.data['num_inter']
        self.mu = [None] * self.data['num_inter']
        self.out_extra_y = [None] * self.data['num_inter']
        self.out_extra_n = [None] * self.data['num_inter']
        self.out_extra_mu = [None] * self.data['num_inter']
        self.in_extra_y = [None] * self.data['num_inter']
        self.in_extra_s = [None] * self.data['num_inter']
        self.in_extra_lamb = [None] * self.data['num_inter']
        self.in_extra_xi_y = [None] * self.data['num_inter']
        self.out_extra_xi_n = [None] * self.data['num_inter']

        self.xi_n = [None] * self.data['num_inter']
        self.xi_y = [None] * self.data['num_inter']
        self.xi_s = [None] * self.data['num_inter']
        self.nu_n = [None] * self.data['num_inter']
        self.nu_y = [None] * self.data['num_inter']
        self.nu_s = [None] * self.data['num_inter']

        self.zeta_1 = [None] * self.data['num_inter']
        self.zeta_21 = [None] * self.data['num_inter']
        self.zeta_22 = [None] * self.data['num_inter']
        self.zeta_23 = [None] * self.data['num_inter']
        self.zeta_24 = [None] * self.data['num_inter']
        self.zeta_31 = [None] * self.data['num_inter']
        self.zeta_32 = [None] * self.data['num_inter']
        self.zeta_33 = [None] * self.data['num_inter']
        self.zeta_34 = [None] * self.data['num_inter']
        self.zeta_41 = [None] * self.data['num_inter']
        self.zeta_42 = [None] * self.data['num_inter']
        self.zeta_43 = [None] * self.data['num_inter']
        self.zeta_5 = [None] * self.data['num_inter']
        self.kappa_1 = [None] * self.data['num_inter']
        self.kappa_21 = [None] * self.data['num_inter']
        self.kappa_22 = [None] * self.data['num_inter']
        self.kappa_23 = [None] * self.data['num_inter']
        self.kappa_24 = [None] * self.data['num_inter']
        self.kappa_31 = [None] * self.data['num_inter']
        self.kappa_32 = [None] * self.data['num_inter']
        self.kappa_33 = [None] * self.data['num_inter']
        self.kappa_34 = [None] * self.data['num_inter']
        self.kappa_41 = [None] * self.data['num_inter']
        self.kappa_42 = [None] * self.data['num_inter']
        self.kappa_43 = [None] * self.data['num_inter']
        self.kappa_5 = [None] * self.data['num_inter']


        for i in range(self.data['num_inter']):
            # self.n_init[i] = np.ones((len(self.data['C'][i])))
            self.n_init[i] = np.zeros((len(self.data['C'][i])))
            self.n[i] = np.zeros((len(self.data['C'][i]), self.data['opt_time'] + 1))
            self.y[i] = np.zeros((len(self.data['C'][i]), self.data['opt_time']))
            self.s[i] = np.zeros((len(self.data['BO'][i]), self.data['opt_time'])) 

            self.xi_n[i] = np.zeros_like(self.n[i])# same size with all elements equal to 0
            self.xi_y[i] = np.zeros_like(self.y[i])
            self.xi_s[i] = np.zeros_like(self.s[i])
            self.nu_n[i] = np.zeros_like(self.n[i])
            self.nu_y[i] = np.zeros_like(self.y[i])
            self.nu_s[i] = np.zeros_like(self.s[i])

            self.out_extra_y[i] = np.zeros((len(self.data['BO'][i]), self.data['opt_time']))
            self.out_extra_xi_n[i] = np.zeros((len(self.data['BO'][i]), self.data['opt_time']+1))
            self.out_extra_n[i] = np.zeros((len(self.data['BO'][i]), self.data['opt_time'] + 1))
            self.out_extra_mu[i] = np.zeros((len(self.data['BO'][i]), self.data['opt_time']))

            self.in_extra_xi_y[i] = np.zeros((len(self.data['BI'][i]), self.data['opt_time']))
            self.in_extra_y[i] = np.zeros((len(self.data['BI'][i]), self.data['opt_time']))
            self.in_extra_s[i] = np.zeros(( len(self.data['BI'][i]), self.data['opt_time']))
            self.in_extra_lamb[i] = np.zeros((len(self.data['BI'][i]), self.data['opt_time']))#used in updating dual
            self.lamb[i] = np.zeros([len(self.data['BO'][i]), self.data['opt_time']]) 
            self.mu[i] = np.zeros([len(self.data['BI'][i]), self.data['opt_time']]) 
            self.zeta_1[i] = np.zeros((len(self.data['C'][i]), self.data['opt_time']))
            self.zeta_21[i] = np.zeros([len(self.data['C'][i])-len(self.data['I1'][i])-len(self.data['I2'][i])-len(self.data['I3'][i])
                                -len(self.data['I4'][i])-len(self.data['D'][i]), self.data['opt_time']])
            self.zeta_22[i] = np.zeros([len(self.data['V'][i]), self.data['opt_time']])
            self.zeta_23[i] = np.zeros([len(self.data['V'][i]), self.data['opt_time']])
            self.zeta_24[i] = np.zeros([len(self.data['V'][i]), self.data['opt_time']])

            self.zeta_31[i] = np.zeros([len(self.data['I1'][i]), self.data['opt_time']])
            self.zeta_32[i] = np.zeros([len(self.data['I2'][i]), self.data['opt_time']])
            self.zeta_33[i] = np.zeros([len(self.data['I3'][i]), self.data['opt_time']])
            self.zeta_34[i] = np.zeros([len(self.data['I4'][i]), self.data['opt_time']])

            self.zeta_41[i] = np.zeros([len(self.data['V'][i]), self.data['opt_time']])
            self.zeta_42[i] = np.zeros([len(self.data['V'][i]), self.data['opt_time']])
            self.zeta_43[i] = np.zeros([len(self.data['V'][i]), self.data['opt_time']])
        
            self.zeta_5[i] = np.zeros([len(self.data['C'][i])-len(self.data['D'][i])-len(self.data['V'][i])-len(self.data['BO'][i]), self.data['opt_time']])

            self.kappa_1[i] = np.zeros_like(self.zeta_1[i])
            self.kappa_21[i] = np.zeros_like(self.zeta_21[i])
            self.kappa_22[i] = np.zeros_like(self.zeta_22[i])
            self.kappa_23[i] = np.zeros_like(self.zeta_23[i])
            self.kappa_24[i] = np.zeros_like(self.zeta_24[i])
            self.kappa_31[i] = np.zeros_like(self.zeta_31[i])
            self.kappa_32[i] = np.zeros_like(self.zeta_32[i])
            self.kappa_33[i] = np.zeros_like(self.zeta_33[i])
            self.kappa_34[i] = np.zeros_like(self.zeta_34[i])
            self.kappa_41[i] = np.zeros_like(self.zeta_41[i])
            self.kappa_42[i] = np.zeros_like(self.zeta_42[i]) 
            self.kappa_43[i] = np.zeros_like(self.zeta_43[i])
            self.kappa_5[i] = np.zeros_like(self.zeta_5[i])

        self.zeta_6 = np.zeros((self.data['num_inter'], self.data['opt_time']-1))
        self.zeta_7 = np.zeros((self.data['num_inter'], self.data['opt_time']-1))
        self.zeta_8 = np.zeros((self.data['num_inter'], self.data['opt_time']-1))
        self.zeta_9 = np.zeros((self.data['num_inter'], self.data['opt_time']-1))
        self.zeta_10 = np.zeros((self.data['num_inter'], self.data['opt_time']-4))
        self.zeta_11 = np.zeros((self.data['num_inter'], self.data['opt_time']-7))
        self.zeta_12 = np.zeros((self.data['num_inter'], self.data['opt_time']-7))

        self.xi_w = np.zeros_like(self.w)
        self.xi_u = np.zeros_like(self.u)

        self.kappa_6 = np.zeros_like(self.zeta_6)
        self.kappa_7 = np.zeros_like(self.zeta_7)
        self.kappa_8 = np.zeros_like(self.zeta_8)
        self.kappa_9 = np.zeros_like(self.zeta_9)
        self.kappa_10 = np.zeros_like(self.zeta_10)
        self.kappa_11 = np.zeros_like(self.zeta_11)
        self.kappa_12 = np.zeros_like(self.zeta_12)

        self.nu_w = np.zeros_like(self.w)
        self.nu_u = np.zeros_like(self.u)

        self.parallel = parallel
        self.numberOfThreads = cpu_count()
        self.time = np.zeros(4) # time to solve X, solve Z, update dual and rounding(using DDVR)

    def step(self):
        if self.parallel:
            return self.step_parallel() # if compute in parallel
        else: # if compute in sequential
            # Solve for X_t+1
            start1 = time.time()
            for i in range(self.data['num_inter']):
                self.n[i], self.y[i], self.w[i], self.u[i], self.s[i] = self.solveForX(i, None)
            end1 = time.time()
            self.time[0] = end1 - start1

            for i in range(self.data['num_inter']):
            # communicate to neighborhoods
                if i == 0:
                    self.out_extra_n[i][0] = self.n[i+1][0]
                    self.out_extra_y[i][0] = self.y[i+1][0]
                    self.out_extra_mu[i][0] = self.mu[i+1][self.data['BI'][i+1].index(0)] # mu is only defined for BI
                    self.in_extra_y[i][0] = self.y[i+1][65]
                    self.in_extra_s[i][0] = self.s[i+1][self.data['BO'][i+1].index(65)] 
                    self.in_extra_lamb[i][0] = self.lamb[i+1][self.data['BO'][i+1].index(65)]
                elif i == 1:
                    self.out_extra_n[i][0] = self.n[i+1][0]
                    self.out_extra_n[i][1] = self.n[i-1][29] 
                    self.out_extra_y[i][0] = self.y[i+1][0]
                    self.out_extra_y[i][1] = self.y[i-1][29]
                    self.out_extra_mu[i][0] = self.mu[i+1][self.data['BI'][i+1].index(0)] 
                    self.out_extra_mu[i][1] = self.mu[i-1][self.data['BI'][i-1].index(29)] 
                    self.in_extra_y[i][0] = self.y[i-1][28]
                    self.in_extra_y[i][1] = self.y[i+1][41]
                    self.in_extra_s[i][0] = self.s[i-1][self.data['BO'][i-1].index(28)]
                    self.in_extra_s[i][1] = self.s[i+1][self.data['BO'][i+1].index(41)]
                    self.in_extra_lamb[i][0] = self.lamb[i-1][self.data['BO'][i-1].index(28)]
                    self.in_extra_lamb[i][1] = self.lamb[i+1][self.data['BO'][i+1].index(41)]
                elif i == 2:
                    self.out_extra_n[i][0] = self.n[i+1][0]
                    self.out_extra_n[i][1] = self.n[i-1][33] 
                    self.out_extra_y[i][0] = self.y[i+1][0]
                    self.out_extra_y[i][1] = self.y[i-1][33]
                    self.out_extra_mu[i][0] = self.mu[i+1][self.data['BI'][i+1].index(0)]
                    self.out_extra_mu[i][1] = self.mu[i-1][self.data['BI'][i-1].index(33)] 
                    self.in_extra_y[i][0] = self.y[i-1][32]
                    self.in_extra_y[i][1] = self.y[i+1][25]
                    self.in_extra_s[i][0] = self.s[i-1][self.data['BO'][i-1].index(32)]
                    self.in_extra_s[i][1] = self.s[i+1][self.data['BO'][i+1].index(25)]
                    self.in_extra_lamb[i][0] = self.lamb[i-1][self.data['BO'][i-1].index(32)]
                    self.in_extra_lamb[i][1] = self.lamb[i+1][self.data['BO'][i+1].index(25)]
                elif i == 3:
                    self.out_extra_n[i][0] = self.n[i+1][0]
                    self.out_extra_n[i][1] = self.n[i-1][21] 
                    self.out_extra_y[i][0] = self.y[i+1][0]
                    self.out_extra_y[i][1] = self.y[i-1][21]
                    self.out_extra_mu[i][0] = self.mu[i+1][self.data['BI'][i+1].index(0)] 
                    self.out_extra_mu[i][1] = self.mu[i-1][self.data['BI'][i-1].index(21)] 
                    self.in_extra_y[i][0] = self.y[i-1][20]
                    self.in_extra_y[i][1] = self.y[i+1][43]
                    self.in_extra_s[i][0] = self.s[i-1][self.data['BO'][i-1].index(20)]
                    self.in_extra_s[i][1] = self.s[i+1][self.data['BO'][i+1].index(43)]
                    self.in_extra_lamb[i][0] = self.lamb[i-1][self.data['BO'][i-1].index(20)]
                    self.in_extra_lamb[i][1] = self.lamb[i+1][self.data['BO'][i+1].index(43)]
                elif i == 4:
                    self.out_extra_n[i][0] = self.n[i+1][0]
                    self.out_extra_n[i][1] = self.n[i-1][13] 
                    self.out_extra_y[i][0] = self.y[i+1][0]
                    self.out_extra_y[i][1] = self.y[i-1][13]
                    self.out_extra_mu[i][0] = self.mu[i+1][self.data['BI'][i+1].index(0)] 
                    self.out_extra_mu[i][1] = self.mu[i-1][self.data['BI'][i-1].index(13)] 
                    self.in_extra_y[i][0] = self.y[i-1][12]
                    self.in_extra_y[i][1] = self.y[i+1][57]
                    self.in_extra_s[i][0] = self.s[i-1][self.data['BO'][i-1].index(12)]
                    self.in_extra_s[i][1] = self.s[i+1][self.data['BO'][i+1].index(57)]
                    self.in_extra_lamb[i][0] = self.lamb[i-1][self.data['BO'][i-1].index(12)]
                    self.in_extra_lamb[i][1] = self.lamb[i+1][self.data['BO'][i+1].index(57)]
                elif i == 5:
                    self.out_extra_n[i][0] = self.n[i-1][22]
                    self.out_extra_y[i][0] = self.y[i-1][22]
                    self.out_extra_mu[i][0] = self.mu[i-1][self.data['BI'][i-1].index(22)] 
                    self.in_extra_y[i][0] = self.y[i-1][21]
                    self.in_extra_s[i][0] = self.s[i-1][self.data['BO'][i-1].index(21)] 
                    self.in_extra_lamb[i][0] = self.lamb[i-1][self.data['BO'][i-1].index(21)]


            # Solve for Z_t+1
            start2 = time.time()
            for i in range(self.data['num_inter']):
                self.zeta_1[i], self.zeta_21[i], self.zeta_22[i],self.zeta_23[i], self.zeta_24[i], self.zeta_31[i],self.zeta_32[i],self.zeta_33[i],self.zeta_34[i],\
                self.zeta_41[i], self.zeta_42[i], self.zeta_43[i], self.zeta_5[i], self.zeta_6[i], self.zeta_7[i], self.zeta_8[i], self.zeta_9[i], self.zeta_10[i], self.zeta_11[i], self.zeta_12[i],\
                self.xi_n[i], self.xi_y[i], self.xi_w[i], self.xi_u[i], self.xi_s[i] = self.solveForZ(i, None)
            end2 = time.time()
            self.time[1] = end2 - start2
            
            for i in range(self.data['num_inter']):
                if i == 0:
                    self.out_extra_xi_n[i][0] = self.xi_n[i+1][0]
                    self.in_extra_xi_y[i][0] = self.xi_y[i+1][65]
                elif i == 1:
                    self.out_extra_xi_n[i][0] = self.xi_n[i+1][0]
                    self.out_extra_xi_n[i][1] = self.xi_n[i-1][29]
                    self.in_extra_xi_y[i][0] = self.xi_y[i-1][28]
                    self.in_extra_xi_y[i][1] = self.xi_y[i+1][41]
                elif i == 2:
                    self.out_extra_xi_n[i][0] = self.xi_n[i+1][0]
                    self.out_extra_xi_n[i][1] = self.xi_n[i-1][33]
                    self.in_extra_xi_y[i][0] = self.xi_y[i-1][32]
                    self.in_extra_xi_y[i][1] = self.xi_y[i+1][25]
                elif i == 3:
                    self.out_extra_xi_n[i][0] = self.xi_n[i+1][0]
                    self.out_extra_xi_n[i][1] = self.xi_n[i-1][21]
                    self.in_extra_xi_y[i][0] = self.xi_y[i-1][20]
                    self.in_extra_xi_y[i][1] = self.xi_y[i+1][43]
                elif i == 4:
                    self.out_extra_xi_n[i][0] = self.xi_n[i+1][0]
                    self.out_extra_xi_n[i][1] = self.xi_n[i-1][13]
                    self.in_extra_xi_y[i][0] = self.xi_y[i-1][12]
                    self.in_extra_xi_y[i][1] = self.xi_y[i+1][57]
                elif i == 5:
                    self.out_extra_xi_n[i][0] = self.xi_n[i-1][22]
                    self.in_extra_xi_y[i][0] = self.xi_y[i-1][21]


            
            # update dual
            start3 = time.time()
            for i in range(self.data['num_inter']):
                self.kappa_1[i], self.kappa_21[i], self.kappa_22[i], self.kappa_23[i], self.kappa_24[i], self.kappa_31[i],self.kappa_32[i],self.kappa_33[i],self.kappa_34[i], self.kappa_41[i],self.kappa_42[i],self.kappa_43[i], self.kappa_5[i], self.kappa_6[i], self.kappa_7[i], self.kappa_8[i], self.kappa_9[i],\
                self.kappa_10[i], self.kappa_11[i], self.kappa_12[i], \
                self.nu_n[i], self.nu_y[i], self.nu_w[i], self.nu_u[i], self.nu_s[i], self.lamb[i], self.mu[i]    = self.updateDual(i, None)
            end3 = time.time()
            self.time[2] = end3 - start3

    def solveForX(self, i, output):
        return self.solveIndividualX(i, self.data['beta'][i], self.data['proc'][i], self.data['pred'][i], self.data['C'][i], self.data['O'][i], self.data['D'][i], self.data['I1'][i], self.data['I2'][i],  self.data['I3'][i],  self.data['I4'][i], self.data['BO'][i], self.data['BI'][i],  self.data['V'][i],  self.data['M'][i], self.data['jam_N'][i], self.data['capacity_Q'][i], 
        self.zeta_1[i], self.zeta_21[i], self.zeta_22[i], self.zeta_23[i], self.zeta_24[i], self.zeta_31[i], self.zeta_32[i], self.zeta_33[i], self.zeta_34[i], self.zeta_41[i], self.zeta_42[i], self.zeta_43[i], self.zeta_5[i], self.zeta_6[i], self.zeta_7[i], self.zeta_8[i], self.zeta_9[i], self.zeta_10[i], self.zeta_11[i], self.zeta_12[i],
        self.xi_n[i], self.xi_y[i], self.xi_w[i], self.xi_u[i], self.xi_s[i], 
        self.kappa_1[i], self.kappa_21[i], self.kappa_22[i], self.kappa_23[i], self.kappa_24[i], self.kappa_31[i], self.kappa_32[i], self.kappa_33[i], self.kappa_34[i], self.kappa_41[i], self.kappa_42[i], self.kappa_43[i], self.kappa_5[i], self.kappa_6[i], self.kappa_7[i], self.kappa_8[i], self.kappa_9[i], self.kappa_10[i], self.kappa_11[i], self.kappa_12[i], 
        self.lamb[i], self.mu[i], self.nu_n[i], self.nu_y[i], self.nu_w[i], self.nu_u[i], self.nu_s[i], self.out_extra_xi_n[i], self.in_extra_xi_y[i], self.Demand[i], output)

    
    def solveForZ(self, i, output):
        return self.solveIndividualZ(i, self.n_init[i], self.data['proc'][i], self.data['beta'][i],self.data['C'][i], self.data['O'][i], self.data['D'][i], self.data['I1'][i], self.data['I2'][i],  self.data['I3'][i],  self.data['I4'][i], self.data['BO'][i], self.data['BI'][i],  self.data['V'][i],  self.data['M'][i], self.data['jam_N'][i], self.data['capacity_Q'][i], 
        self.n[i], self.y[i], self.w[i], self.u[i], self.s[i],
         self.kappa_1[i], self.kappa_21[i], self.kappa_22[i], self.kappa_23[i], self.kappa_24[i], self.kappa_31[i], self.kappa_32[i], self.kappa_33[i], self.kappa_34[i], self.kappa_41[i], self.kappa_42[i], self.kappa_43[i],  self.kappa_5[i], self.kappa_6[i], self.kappa_7[i], self.kappa_8[i], self.kappa_9[i],
        self.kappa_10[i], self.kappa_11[i], self.kappa_12[i], self.lamb[i], self.mu[i], self.nu_n[i], self.nu_y[i], self.nu_w[i], self.nu_u[i], self.nu_s[i],
         self.in_extra_y[i], self.in_extra_s[i], self.in_extra_lamb[i], self.out_extra_mu[i], self.out_extra_n[i], self.out_extra_y[i], output)

    def updateDual(self, i, output):
        return self.updateIndividualDual(i, self.data['proc'][i], self.data['beta'][i], self.data['C'][i], self.data['O'][i], self.data['D'][i], self.data['I1'][i], self.data['I2'][i],  self.data['I3'][i],  self.data['I4'][i], self.data['BO'][i], self.data['BI'][i],  self.data['V'][i],  self.data['M'][i], self.data['jam_N'][i], self.data['capacity_Q'][i], 
        self.n[i], self.y[i], self.w[i], self.u[i], self.s[i],
        self.zeta_1[i], self.zeta_21[i], self.zeta_22[i], self.zeta_23[i], self.zeta_24[i], self.zeta_31[i], self.zeta_32[i], self.zeta_33[i], self.zeta_34[i], self.zeta_41[i], self.zeta_42[i], self.zeta_43[i], self.zeta_5[i], self.zeta_6[i], self.zeta_7[i], self.zeta_8[i], self.zeta_9[i], self.zeta_10[i], self.zeta_11[i], self.zeta_12[i],
        self.xi_n[i], self.xi_y[i], self.xi_w[i], self.xi_u[i], self.xi_s[i], 
        self.kappa_1[i], self.kappa_21[i], self.kappa_22[i], self.kappa_23[i], self.kappa_24[i], self.kappa_31[i], self.kappa_32[i], self.kappa_33[i], self.kappa_34[i], self.kappa_41[i], self.kappa_42[i], self.kappa_43[i],  self.kappa_5[i], self.kappa_6[i], self.kappa_7[i], self.kappa_8[i], self.kappa_9[i],
        self.kappa_10[i], self.kappa_11[i], self.kappa_12[i],  self.lamb[i], self.mu[i], self.nu_n[i], self.nu_y[i], self.nu_w[i], self.nu_u[i], self.nu_s[i], self.out_extra_xi_n[i], self.in_extra_xi_y[i], output)

    def step_parallel(self):
        # Solve for X_t+1 in parallel
        start = time.time()
        output = multiprocessing.Queue()

        process = [Process(target= self.solveForX, args=(i, output)) for i in range(self.data['num_inter'])]

        for p in process:
            p.start()

        results = [output.get() for p in process]

        for i in range(len(results)): # get the results from the parallel processes, indexed by i
            self.n[results[i][0]] = results[i][1]
            self.y[results[i][0]] = results[i][2]
            self.w[results[i][0]] = results[i][3]
            self.u[results[i][0]] = results[i][4]
            self.s[results[i][0]] = results[i][5]

        for p in process:
            p.join()
        
        end = time.time()
        self.time[0] = end - start

        for i in range(self.data['num_inter']):
            # communicate to neighborhood
                if i == 0:
                    self.out_extra_n[i][0] = self.n[i+1][0]
                    self.out_extra_y[i][0] = self.y[i+1][0]
                    self.out_extra_mu[i][0] = self.mu[i+1][self.data['BI'][i+1].index(0)] # mu is only defined for BI. BI1={0,3}
                    self.in_extra_y[i][0] = self.y[i+1][65]
                    self.in_extra_s[i][0] = self.s[i+1][self.data['BO'][i+1].index(65)] 
                    self.in_extra_lamb[i][0] = self.lamb[i+1][self.data['BO'][i+1].index(65)]
                elif i == 1:
                    self.out_extra_n[i][0] = self.n[i+1][0]
                    self.out_extra_n[i][1] = self.n[i-1][29] # BO1={2,5} -> C+1 = {0 IN 2, 3 IN 0}
                    self.out_extra_y[i][0] = self.y[i+1][0]
                    self.out_extra_y[i][1] = self.y[i-1][29]
                    self.out_extra_mu[i][0] = self.mu[i+1][self.data['BI'][i+1].index(0)] #BI2={0}
                    self.out_extra_mu[i][1] = self.mu[i-1][self.data['BI'][i-1].index(29)] #BI0={3}
                    self.in_extra_y[i][0] = self.y[i-1][28]
                    self.in_extra_y[i][1] = self.y[i+1][41]
                    self.in_extra_s[i][0] = self.s[i-1][self.data['BO'][i-1].index(28)]
                    self.in_extra_s[i][1] = self.s[i+1][self.data['BO'][i+1].index(41)]
                    self.in_extra_lamb[i][0] = self.lamb[i-1][self.data['BO'][i-1].index(28)]
                    self.in_extra_lamb[i][1] = self.lamb[i+1][self.data['BO'][i+1].index(41)]
                elif i == 2:
                    self.out_extra_n[i][0] = self.n[i+1][0]
                    self.out_extra_n[i][1] = self.n[i-1][33] # BO1={2,5} -> C+1 = {0 IN 2, 3 IN 0}
                    self.out_extra_y[i][0] = self.y[i+1][0]
                    self.out_extra_y[i][1] = self.y[i-1][33]
                    self.out_extra_mu[i][0] = self.mu[i+1][self.data['BI'][i+1].index(0)] #BI2={0}
                    self.out_extra_mu[i][1] = self.mu[i-1][self.data['BI'][i-1].index(33)] #BI0={3}
                    self.in_extra_y[i][0] = self.y[i-1][32]
                    self.in_extra_y[i][1] = self.y[i+1][25]
                    self.in_extra_s[i][0] = self.s[i-1][self.data['BO'][i-1].index(32)]
                    self.in_extra_s[i][1] = self.s[i+1][self.data['BO'][i+1].index(25)]
                    self.in_extra_lamb[i][0] = self.lamb[i-1][self.data['BO'][i-1].index(32)]
                    self.in_extra_lamb[i][1] = self.lamb[i+1][self.data['BO'][i+1].index(25)]
                elif i == 3:
                    self.out_extra_n[i][0] = self.n[i+1][0]
                    self.out_extra_n[i][1] = self.n[i-1][21] # BO1={2,5} -> C+1 = {0 IN 2, 3 IN 0}
                    self.out_extra_y[i][0] = self.y[i+1][0]
                    self.out_extra_y[i][1] = self.y[i-1][21]
                    self.out_extra_mu[i][0] = self.mu[i+1][self.data['BI'][i+1].index(0)] #BI2={0}
                    self.out_extra_mu[i][1] = self.mu[i-1][self.data['BI'][i-1].index(21)] #BI0={3}
                    self.in_extra_y[i][0] = self.y[i-1][20]
                    self.in_extra_y[i][1] = self.y[i+1][43]
                    self.in_extra_s[i][0] = self.s[i-1][self.data['BO'][i-1].index(20)]
                    self.in_extra_s[i][1] = self.s[i+1][self.data['BO'][i+1].index(43)]
                    self.in_extra_lamb[i][0] = self.lamb[i-1][self.data['BO'][i-1].index(20)]
                    self.in_extra_lamb[i][1] = self.lamb[i+1][self.data['BO'][i+1].index(43)]
                elif i == 4:
                    self.out_extra_n[i][0] = self.n[i+1][0]
                    self.out_extra_n[i][1] = self.n[i-1][13] # BO1={2,5} -> C+1 = {0 IN 2, 3 IN 0}
                    self.out_extra_y[i][0] = self.y[i+1][0]
                    self.out_extra_y[i][1] = self.y[i-1][13]
                    self.out_extra_mu[i][0] = self.mu[i+1][self.data['BI'][i+1].index(0)] #BI2={0}
                    self.out_extra_mu[i][1] = self.mu[i-1][self.data['BI'][i-1].index(13)] #BI0={3}
                    self.in_extra_y[i][0] = self.y[i-1][12]
                    self.in_extra_y[i][1] = self.y[i+1][57]
                    self.in_extra_s[i][0] = self.s[i-1][self.data['BO'][i-1].index(12)]
                    self.in_extra_s[i][1] = self.s[i+1][self.data['BO'][i+1].index(57)]
                    self.in_extra_lamb[i][0] = self.lamb[i-1][self.data['BO'][i-1].index(12)]
                    self.in_extra_lamb[i][1] = self.lamb[i+1][self.data['BO'][i+1].index(57)]
                elif i == 5:
                    self.out_extra_n[i][0] = self.n[i-1][22]
                    self.out_extra_y[i][0] = self.y[i-1][22]
                    self.out_extra_mu[i][0] = self.mu[i-1][self.data['BI'][i-1].index(22)] 
                    self.in_extra_y[i][0] = self.y[i-1][21]
                    self.in_extra_s[i][0] = self.s[i-1][self.data['BO'][i-1].index(21)] 
                    self.in_extra_lamb[i][0] = self.lamb[i-1][self.data['BO'][i-1].index(21)]
        
        start = time.time()
        output2 = multiprocessing.Queue()

        process2 = [Process(target= self.solveForZ, args=(i, output2)) for i in range(self.data['num_inter'])]

        for p in process2:
            p.start()

        results2 = [output2.get() for p in process2]

        for i in range(len(results2)):
            self.zeta_1[results2[i][0]] = results2[i][1]
            # self.zeta_2[results2[i][0]] = results2[i][2]
            self.zeta_31[results2[i][0]] = results2[i][3]
            self.zeta_32[results2[i][0]] = results2[i][4]
            self.zeta_33[results2[i][0]] = results2[i][5]
            self.zeta_34[results2[i][0]] = results2[i][6]
            self.zeta_41[results2[i][0]] = results2[i][7]
            self.zeta_42[results2[i][0]] = results2[i][8]
            self.zeta_43[results2[i][0]] = results2[i][9]
            self.zeta_5[results2[i][0]] = results2[i][10]
            self.zeta_6[results2[i][0]] = results2[i][11]
            self.zeta_7[results2[i][0]] = results2[i][12]
            self.zeta_8[results2[i][0]] = results2[i][13]
            self.zeta_9[results2[i][0]] = results2[i][14]
            self.zeta_10[results2[i][0]] = results2[i][15]
            self.zeta_11[results2[i][0]] = results2[i][16]
            self.zeta_12[results2[i][0]] = results2[i][17]
            self.xi_n[results2[i][0]] = results2[i][18]
            self.xi_y[results2[i][0]] = results2[i][19]
            self.xi_w[results2[i][0]] = results2[i][20]
            self.xi_u[results2[i][0]] = results2[i][21]
            self.xi_s[results2[i][0]] = results2[i][22]

        for p in process2:
            p.join()

        end = time.time()
        self.time[1] = end - start

        for i in range(self.data['num_inter']):
            #communicate to neighborhood
                if i == 0:
                    self.out_extra_xi_n[i][0] = self.xi_n[i+1][0]
                    self.in_extra_xi_y[i][0] = self.xi_y[i+1][65]
                elif i == 1:
                    self.out_extra_xi_n[i][0] = self.xi_n[i+1][0]
                    self.out_extra_xi_n[i][1] = self.xi_n[i-1][29]
                    self.in_extra_xi_y[i][0] = self.xi_y[i-1][28]
                    self.in_extra_xi_y[i][1] = self.xi_y[i+1][41]
                elif i == 2:
                    self.out_extra_xi_n[i][0] = self.xi_n[i+1][0]
                    self.out_extra_xi_n[i][1] = self.xi_n[i-1][33]
                    self.in_extra_xi_y[i][0] = self.xi_y[i-1][32]
                    self.in_extra_xi_y[i][1] = self.xi_y[i+1][25]
                elif i == 3:
                    self.out_extra_xi_n[i][0] = self.xi_n[i+1][0]
                    self.out_extra_xi_n[i][1] = self.xi_n[i-1][21]
                    self.in_extra_xi_y[i][0] = self.xi_y[i-1][20]
                    self.in_extra_xi_y[i][1] = self.xi_y[i+1][43]
                elif i == 4:
                    self.out_extra_xi_n[i][0] = self.xi_n[i+1][0]
                    self.out_extra_xi_n[i][1] = self.xi_n[i-1][13]
                    self.in_extra_xi_y[i][0] = self.xi_y[i-1][12]
                    self.in_extra_xi_y[i][1] = self.xi_y[i+1][57]
                elif i == 5:
                    self.out_extra_xi_n[i][0] = self.xi_n[i-1][22]
                    self.in_extra_xi_y[i][0] = self.xi_y[i-1][21]
        
        start = time.time()
        output3 = multiprocessing.Queue()
        process3 = [Process(target= self.updateDual, args=(i, output3)) for i in range(self.data['num_inter'])]

        for p in process3:
            p.start()

        results3 = [output3.get() for p in process3]

        end = time.time()
        self.time[2] = end - start
        
        for i in range(len(results3)):
            self.kappa_1[results3[i][0]] = results3[i][1]
            # self.kappa_2[results3[i][0]] = results3[i][2]
            self.kappa_31[results3[i][0]] = results3[i][3]
            self.kappa_32[results3[i][0]] = results3[i][4]
            self.kappa_33[results3[i][0]] = results3[i][5]
            self.kappa_34[results3[i][0]] = results3[i][6]
            self.kappa_41[results3[i][0]] = results3[i][7]
            self.kappa_42[results3[i][0]] = results3[i][8]
            self.kappa_43[results3[i][0]] = results3[i][9]
            self.kappa_5[results3[i][0]] = results3[i][10]
            self.kappa_6[results3[i][0]] = results3[i][11]
            self.kappa_7[results3[i][0]] = results3[i][12]
            self.kappa_8[results3[i][0]] = results3[i][13]
            self.kappa_9[results3[i][0]] = results3[i][14]
            self.kappa_10[results3[i][0]] = results3[i][15]
            self.kappa_11[results3[i][0]] = results3[i][16]
            self.kappa_12[results3[i][0]] = results3[i][17]
            self.nu_n[results3[i][0]] = results3[i][18]
            self.nu_y[results3[i][0]] = results3[i][19]
            self.nu_w[results3[i][0]] = results3[i][20]
            self.nu_u[results3[i][0]] = results3[i][21]
            self.nu_s[results3[i][0]] = results3[i][22]
            self.lamb[results3[i][0]] = results3[i][23]
            self.mu[results3[i][0]] = results3[i][24]

        for p in process3:
            p.join()
    

        
    def solveIndividualX(self, i, beta, proc, pred, C, O, D, I1, I2, I3, I4, BO, BI, V, M, Jam_N, Q, 
    zeta_1, zeta_21, zeta_22, zeta_23, zeta_24, zeta_31, zeta_32, zeta_33, zeta_34,  zeta_41, zeta_42, zeta_43, zeta_5, zeta_6, zeta_7, zeta_8, zeta_9, zeta_10, zeta_11, zeta_12, 
    xi_n, xi_y, xi_w, xi_u, xi_s, 
    kappa_1, kappa_21, kappa_22, kappa_23, kappa_24, kappa_31, kappa_32, kappa_33, kappa_34, kappa_41, kappa_42, kappa_43, kappa_5, kappa_6, kappa_7, kappa_8, kappa_9, kappa_10, kappa_11, kappa_12, 
    lamb, mu, nu_n, nu_y, nu_w, nu_u, nu_s, out_extra_xi_n, in_extra_xi_y, Demand, output):
        
        T = self.data['opt_time']
        n = cp.Variable((len(C), T+1))
        y = cp.Variable((len(C), T))
        # w = cp.Variable((self.data['phases'],T),boolean=True)
        w = cp.Variable((self.data['phases'],T))   #relax the binary variable
        u = cp.Variable(T-1)
        s = cp.Variable((len(BO),T))

        new_w = w[0,:] + w[1,:]

        I = np.sort(I1 + I2 + I3 + I4)
        ND = np.sort(list(set(C) - set(D)))
        EO = np.sort(list((set(C) - set(I)) - set(D)))
        OI = np.sort(list(set(C) - set(D) - set(V) - set(BO))) 
        DO = np.sort(list(set(C) - set(O) - set(I) - set(BI)-set(M))) 
        f_i = (np.sum(np.sum([t * y[i,t] for i in D]) for t in range(T)) 
        + self.data['alpha'] *  np.sum(np.sum([(T-t) * y[i,t] for i in ND]) for t in range(T)))
        # f_i = self.data['alpha'] * np.sum(np.sum([(T-t) * y[i,t] for i in ND]) for t in range(T))
        norm1 = (sum_squares(y - n[:,range(T)] + zeta_1 + kappa_1)
        + sum_squares(y[EO,:] + zeta_21 - np.repeat(Q.reshape([-1,1])[EO], T, axis = 1) + kappa_21)
        # + sum_squares(y[V,:] + zeta_22 - np.repeat(Q.reshape([-1,1])[[c+1 for c in V]]/beta[0,:].reshape([-1,1]), T, axis = 1) + kappa_22)
        # + sum_squares(y[V,:] + zeta_23 - np.repeat(Q.reshape([-1,1])[[c+2 for c in V]]/beta[1,:].reshape([-1,1]), T, axis = 1) + kappa_23)
        + np.sum(np.sum(square(y[I1[i],t] - w[0,t]*Q[I1[i]] + zeta_31[i,t] + kappa_31[i,t]) for i in range(len(I1))) for t in range(T))
        + np.sum(np.sum(square(y[I2[i],t] - w[1,t]*Q[I2[i]] + zeta_32[i,t] + kappa_32[i,t]) for i in range(len(I2))) for t in range(T))
        + np.sum(np.sum(square(y[I3[i],t] - w[2,t]*Q[I3[i]] + zeta_33[i,t] + kappa_33[i,t]) for i in range(len(I3))) for t in range(T))
        + np.sum(np.sum(square(y[I4[i],t] - w[3,t]*Q[I4[i]] + zeta_34[i,t] + kappa_34[i,t]) for i in range(len(I4))) for t in range(T))
        + sum_squares(y[OI,:] + self.data['ratio_W'] * n[[proc[d] for d in OI],0:T] + zeta_5 - self.data['ratio_W'] * np.repeat(Jam_N.reshape([-1,1])[OI], T, axis = 1) + kappa_5)
        + sum_squares(new_w[1:T] - new_w[0:T-1] - u[0:T-1] + zeta_6 + kappa_6)
        + sum_squares(-new_w[1:T] + new_w[0:T-1] - u[0:T-1] + zeta_7 + kappa_7)
        + sum_squares(-new_w[1:T] - new_w[0:T-1] + u[0:T-1] + zeta_8 + kappa_8)
        + sum_squares(new_w[1:T] + new_w[0:T-1] + u[0:T-1] -2 + zeta_9+ kappa_9)
        + sum_squares(u[0:T-4] + u[1:T-3] + u[2:T-2] + u[3:T-1] -1 + zeta_10 + kappa_10)
        + sum_squares(new_w[0:T-7] + new_w[1:T-6] + new_w[2:T-5] + new_w[3:T-4] + new_w[4:T-3] + new_w[5:T-2] + new_w[6:T-1] + new_w[7:T] -8 + zeta_11 + kappa_11)
        + sum_squares(-new_w[0:T-7] - new_w[1:T-6] - new_w[2:T-5] - new_w[3:T-4] - new_w[4:T-3]  - new_w[5:T-2] - new_w[6:T-1] - new_w[7:T] +1 + zeta_12 + kappa_12)
        + sum_squares(n - xi_n + nu_n)
        + sum_squares(y - xi_y + nu_y)
        + sum_squares(w - xi_w + nu_w)
        + sum_squares(u - xi_u + nu_u)
        + sum_squares(s - xi_s + nu_s)
        + sum_squares(y[BO,:] + s - np.repeat(self.data['ratio_W']*Jam_N.reshape([-1,1])[BO], T, axis = 1) + self.data['ratio_W']* out_extra_xi_n[:,0:T] + lamb)
        + sum_squares(n[BI,1:T+1] - n[BI,0:T] + y[BI,:] - in_extra_xi_y + mu))

        norm1 = (norm1 + np.sum(np.sum(square(beta[0,i]*y[V[i],t] + self.data['ratio_W']*n[[V[i]+1],t] + zeta_41[i,t] - self.data['ratio_W'] * Jam_N[V[i]] + kappa_41[i,t]) for i in range(len(V))) for t in range(T))
                    + np.sum(np.sum(square(beta[1,i]*y[V[i],t] + self.data['ratio_W']*n[[V[i]+2],t] + zeta_42[i,t] - self.data['ratio_W'] * Jam_N[V[i]] + kappa_42[i,t]) for i in range(len(V))) for t in range(T))
                    + sum_squares(y[V,:] + zeta_22 - np.repeat(Q.reshape([-1,1])[[c+1 for c in V]]/beta[0,:].reshape([-1,1]), T, axis = 1) + kappa_22)
                    + sum_squares(y[V,:] + zeta_23 - np.repeat(Q.reshape([-1,1])[[c+2 for c in V]]/beta[1,:].reshape([-1,1]), T, axis = 1) + kappa_23))

        if beta.shape[0] == 3:
            norm1 = (norm1 + np.sum(np.sum(square(beta[2,i]*y[V[i],t] + self.data['ratio_W']*n[[V[i]+3],t] + zeta_43[i,t] - self.data['ratio_W'] * Jam_N[V[i]] + kappa_43[i,t]) for i in range(len(V))) for t in range(T))
                    + sum_squares(y[V,:] + zeta_24 - np.repeat(Q.reshape([-1,1])[[c+3 for c in V]]/beta[2,:].reshape([-1,1]), T, axis = 1) + kappa_24))
     
       
        objective = cp.Minimize(f_i + self.data['rho']/2 * norm1)
        constraints = [n[O,1:T+1] - n[O,0:T] + y[O,0:T] == Demand]
        constraints += [w[0,] + w[1,] + w[2,] + w[3,] == 1]

        for c in M:
            constraints += [n[c,1:T+1] - n[c,0:T] - np.sum(y[p,0:T] for p in pred[c]) + y[c,0:T] == 0]
        for c in DO:
            constraints += [n[c,1:T+1] - n[c,0:T] - y[pred[c],0:T] + y[c,0:T] == 0]
        for c in I:
            constraints += [n[c,1:T+1] - n[c,0:T] - reshape(beta[c-pred[c]-1, V.index(pred[c])]*y[[pred[c]],0:T],(T,))+ y[c,0:T] ==0] 
        
        #constraints += [n >= 0, y >= 0, w<=1, w>=0]
        
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        #print("Optimal value", prob.solve())
        #print(n.value)

        # output.put([i, n.value, y.value, w.value, u.value, s.value]) #work for parallel computing
        return n.value, y.value, w.value, u.value, s.value


    def solveIndividualZ(self, i, n_init, proc, beta, C, O, D, I1, I2, I3, I4, BO, BI, V, M, Jam_N, Q, n, y, w, u, s, 
    kappa_1, kappa_21, kappa_22, kappa_23, kappa_24, kappa_31, kappa_32, kappa_33, kappa_34, kappa_41, kappa_42, kappa_43, kappa_5, kappa_6, kappa_7, kappa_8, kappa_9, kappa_10, kappa_11, kappa_12, 
    lamb, mu, nu_n, nu_y, nu_w, nu_u, nu_s, in_extra_y, in_extra_s, in_extra_lamb, out_extra_mu, out_extra_n, out_extra_y, output):

        T = self.data['opt_time']
        
        xi_n = np.zeros_like(n)
        xi_y = np.zeros_like(y)
        xi_w = np.zeros_like(w)
        xi_u = np.zeros_like(u)
        xi_s = np.zeros_like(s)
        
        EO = np.sort(list(set(C)-set(I1)-set(I2)-set(I3)-set(I4)-set(D)))
        OI = np.sort(list(set(C)-set(D)-set(V)-set(BO)))
        NBI = np.sort(list(set(C)-set(BI)))
        NBO = np.sort(list(set(C)-set(BO)))
        new_w = w[0,]+w[1,]
        
        #closed form solution of variable update
        zeta_1 = np.maximum(0, -(y - n[:,range(T)] + kappa_1))
        
        zeta_21 = np.maximum(0,-(y[EO,] - np.repeat(Q.reshape([-1,1])[EO], T, axis = 1) + kappa_21))
        zeta_22 = np.maximum(0,-(y[V,] - np.repeat(Q.reshape([-1,1])[[c+1 for c in V]]/beta[0,:].reshape([-1,1]), T, axis = 1) + kappa_22))
        zeta_23 = np.maximum(0,-(y[V,] - np.repeat(Q.reshape([-1,1])[[c+2 for c in V]]/beta[1,:].reshape([-1,1]), T, axis = 1) + kappa_23))
        if beta.shape[0] == 3:
            zeta_24 = np.maximum(0,-(y[V,] - np.repeat(Q.reshape([-1,1])[[c+3 for c in V]]/beta[2,:].reshape([-1,1]), T, axis = 1) + kappa_24))
        else: 
            zeta_24 = None

        zeta_31 = np.maximum(0, -(y[I1,] - np.repeat(w[0,].reshape([1,-1]), len(I1), axis = 0) * np.repeat(Q.reshape([-1,1])[I1], T, axis = 1) + kappa_31))
        zeta_32 = np.maximum(0, -(y[I2,] - np.repeat(w[1,].reshape([1,-1]), len(I2), axis = 0) * np.repeat(Q.reshape([-1,1])[I2], T, axis = 1) + kappa_32))
        zeta_33 = np.maximum(0, -(y[I3,] - np.repeat(w[2,].reshape([1,-1]), len(I3), axis = 0) * np.repeat(Q.reshape([-1,1])[I3], T, axis = 1) + kappa_33))
        zeta_34 = np.maximum(0, -(y[I4,] - np.repeat(w[3,].reshape([1,-1]), len(I4), axis = 0) * np.repeat(Q.reshape([-1,1])[I4], T, axis = 1) + kappa_34))

        zeta_41 = np.maximum(0, -(np.repeat(beta[0,].reshape([-1,1]), T, axis=1)*y[V,:] + self.data['ratio_W']* n[[x+1 for x in V],0:T] - np.repeat(self.data['ratio_W'] * Jam_N.reshape([-1,1])[V], T, axis = 1) + kappa_41))
        zeta_42 = np.maximum(0, -(np.repeat(beta[1,].reshape([-1,1]), T, axis=1)*y[V,:] + self.data['ratio_W']* n[[x+2 for x in V],0:T] - np.repeat(self.data['ratio_W'] * Jam_N.reshape([-1,1])[V], T, axis = 1) + kappa_42))
        if beta.shape[0]==3:
            zeta_43 = np.maximum(0, -(np.repeat(beta[2,].reshape([-1,1]), T, axis=1)*y[V,:] + self.data['ratio_W']* n[[x+3 for x in V],0:T] - np.repeat(self.data['ratio_W'] * Jam_N.reshape([-1,1])[V], T, axis = 1) + kappa_43))
        else:
            zeta_43 = None
        zeta_5 = np.maximum(0, -(y[OI,] + self.data['ratio_W'] * n[[proc[x] for x in OI],0:T] - np.repeat(self.data['ratio_W']* Jam_N.reshape([-1,1])[OI], T, axis = 1) + kappa_5))

        
        zeta_6 = np.maximum(0, -(new_w[1:T] - new_w[0:T-1] - u[0:T-1] + kappa_6))
        zeta_7 = np.maximum(0, -(-new_w[1:T] + new_w[0:T-1] - u[0:T-1] + kappa_7))
        zeta_8 = np.maximum(0, -(-new_w[1:T] - new_w[0:T-1] + u[0:T-1] + kappa_8))
        zeta_9 = np.maximum(0, -(new_w[1:T] + new_w[0:T-1] + u[0:T-1] -2 + kappa_9))
        

        zeta_10 = np.maximum(0, -(u[0:T-4] + u[1:T-3] + u[2:T-2] + u[3:T-1] + kappa_10))

        zeta_11 = np.maximum(0, -(new_w[0:T-7] + new_w[1:T-6] + new_w[2:T-5] + new_w[3:T-4] + new_w[4:T-3] + new_w[5:T-2] + new_w[6:T-1] + new_w[7:T] -8  + kappa_11))
        zeta_12 = np.maximum(0, -(-new_w[0:T-7] - new_w[1:T-6] - new_w[2:T-5] - new_w[3:T-4] - new_w[4:T-3]  - new_w[5:T-2] - new_w[6:T-1] - new_w[7:T] +1 + kappa_12))
        

        #xi_n[:,T] = 0        #add if T is big enough, otherwise, problem is infeasible
        xi_n[:,T] = n[:,T] + nu_n[:,T]
        xi_n[:,0] = n_init

        xi_n[BI,1:T] = (1+ self.data['ratio_W']**2)**(-1) * (n[BI,1:T] + nu_n[BI,1:T]- self.data['ratio_W']*(in_extra_y[:,1:T].reshape((len(BI),T-1)) + in_extra_s[:,1:T].reshape((len(BI),T-1)) - np.repeat(self.data['ratio_W']*Q.reshape([-1,1])[BI], T-1, axis = 1) + in_extra_lamb[:,1:T].reshape((len(BI),T-1))))

        xi_n[NBI,1:T+1] = n[NBI,1:T+1] + nu_n[NBI,1:T+1] #change to T if T is big enough
        

        xi_y[BO,] = np.maximum(0, 1/2*(out_extra_n[:,1:T+1] - out_extra_n[:,0:T] + out_extra_y[:,0:T] + out_extra_mu[:,0:T] + y[BO,0:T] +nu_y[BO,0:T]))
        xi_y[NBO,] = np.maximum(0, y[NBO,0:T] + nu_y[NBO,0:self.data['opt_time']])# T='opt_time'
        
        # xi_w = 1 * (abs(w+nu_w)>=abs(w+nu_w-1)) # work for heuristic 1: ADMM with projection

        xi_w = np.minimum(1, np.maximum(0, w + nu_w)) # work for vanilla ADMM solving the relaxed LP
        xi_u = u + nu_u
        xi_s = s + nu_s

        '''this is the original optimization model for the variable update, can be used to check the above solution
        zeta_1_var = cp.Variable((len(C), T))
        zeta_2_var = cp.Variable((len(EO), T))
        zeta_31_var = cp.Variable((len(I1),T))
        zeta_32_var = cp.Variable((len(I2),T))
        zeta_33_var = cp.Variable((len(I3),T))
        zeta_34_var = cp.Variable((len(I4), T))
        zeta_41_var  = cp.Variable((len(V), T))
        zeta_42_var = cp.Variable((len(V), T))
        zeta_43_var = cp.Variable((len(V), T))
        zeta_5_var = cp.Variable((len(OI),T))
        zeta_6_var = cp.Variable(T-1)
        zeta_7_var = cp.Variable(T-1)
        zeta_8_var = cp.Variable(T-1)
        zeta_9_var = cp.Variable(T-1)
        zeta_10_var = cp.Variable(T-4)
        zeta_11_var = cp.Variable(T-7)
        zeta_12_var = cp.Variable(T-7)

        xi_n_var = cp.Variable((len(C), T+1))
        xi_y_var = cp.Variable((len(C), T))
        xi_s_var = cp.Variable((len(BO),T))
        xi_w_var = cp.Variable((4,T))
        xi_u_var = cp.Variable(T-1)

        norm1 = (sum_squares(y - n[:,range(T)] + zeta_1_var + kappa_1)
        + sum_squares(y[EO,:] + zeta_2_var - self.data['capacity_Q'] + kappa_2)
        + np.sum(np.sum(square(y[I1[i],t] - w[0,t]*self.data['capacity_Q'] + zeta_31_var[i,t] + kappa_31[i,t]) for i in range(len(I1))) for t in range(T))
        + np.sum(np.sum(square(y[I2[i],t] - w[1,t]*self.data['capacity_Q'] + zeta_32_var[i,t] + kappa_32[i,t]) for i in range(len(I2))) for t in range(T))
        + np.sum(np.sum(square(y[I3[i],t] - w[2,t]*self.data['capacity_Q'] + zeta_33_var[i,t] + kappa_33[i,t]) for i in range(len(I3))) for t in range(T))
        + np.sum(np.sum(square(y[I4[i],t] - w[3,t]*self.data['capacity_Q'] + zeta_34_var[i,t] + kappa_34[i,t]) for i in range(len(I4))) for t in range(T))
        + sum_squares(y[OI,:] + self.data['ratio_W'] * n[[proc[d] for d in OI],0:T] + zeta_5_var - self.data['ratio_W'] * self.data['jam_N'] + kappa_5)
        + sum_squares(new_w[1:T] - new_w[0:T-1] - u[0:T-1] + zeta_6_var + kappa_6[0:T-1])
        + sum_squares(-new_w[1:T] + new_w[0:T-1] - u[0:T-1] + zeta_7_var + kappa_7[0:T-1])
        + sum_squares(-new_w[1:T] - new_w[0:T-1] + u[0:T-1] + zeta_8_var + kappa_8[0:T-1])
        + sum_squares(new_w[1:T] + new_w[0:T-1] + u[0:T-1] -2 + zeta_9_var + kappa_9[0:T-1])
        + sum_squares(u[0:T-4] + u[1:T-3] + u[2:T-2] + u[3:T-1] -1 + zeta_10_var + kappa_10[0:T-4])
        + sum_squares(new_w[0:T-7] + new_w[1:T-6] + new_w[2:T-5] + new_w[3:T-4] + new_w[4:T-3] + new_w[5:T-2] + new_w[6:T-1] + new_w[7:T] -8 + zeta_11_var + kappa_11[0:T-7])
        + sum_squares(-new_w[0:T-7] - new_w[1:T-6] - new_w[2:T-5] - new_w[3:T-4] - new_w[4:T-3]  - new_w[5:T-2] - new_w[6:T-1] - new_w[7:T] +1 + zeta_12_var + kappa_12[0:T-7])
        + sum_squares(n - xi_n_var + nu_n)
        + sum_squares(y - xi_y_var + nu_y)
        + sum_squares(w - xi_w_var + nu_w)
        + sum_squares(u - xi_u_var + nu_u)
        + sum_squares(s - xi_s_var + nu_s)
        + sum_squares(in_extra_y + in_extra_s - self.data['ratio_W']*self.data['jam_N'] + self.data['ratio_W']* xi_n_var[BI,0:T] + in_extra_lamb)
        + sum_squares(out_extra_n[:,1:T+1] - out_extra_n[:,0:T] + out_extra_y - xi_y_var[BO,:] + out_extra_mu))
        
        if beta.shape[0]==3:
            norm1 = (norm1 + sum_squares(beta[0,0]*y[V,:] + self.data['ratio_W']*n[[x+1 for x in V],0:T] + zeta_41_var - self.data['ratio_W'] * self.data['jam_N'] + kappa_41)
                        + sum_squares(beta[1,0]*y[V,:] + self.data['ratio_W']*n[[x+2 for x in V],0:T] + zeta_42_var - self.data['ratio_W'] * self.data['jam_N'] + kappa_42)
                        + sum_squares(beta[2,0]*y[V,:] + self.data['ratio_W']*n[[x+3 for x in V],0:T] + zeta_43_var - self.data['ratio_W'] * self.data['jam_N'] + kappa_43))
        else:
            norm1 = (norm1 + sum_squares(beta[0,0]*y[V,:] + self.data['ratio_W']*n[[x+1 for x in V],0:T] + zeta_41_var - self.data['ratio_W'] * self.data['jam_N'] + kappa_41)
                        + sum_squares(beta[1,0]*y[V,:] + self.data['ratio_W']*n[[x+2 for x in V],0:T] + zeta_42_var - self.data['ratio_W'] * self.data['jam_N'] + kappa_42))
        
        objective = cp.Minimize(self.data['rho']/2 * norm1)
        constraints = [xi_n_var[:,0] == n_init, xi_y_var>=0, xi_w_var>=0, xi_w_var<=1, zeta_1_var>=0, zeta_2_var>=0,zeta_31_var>=0,
        zeta_32_var>=0,zeta_33_var>=0,zeta_34_var>=0,zeta_41_var>=0,zeta_42_var>=0,zeta_43_var>=0,zeta_5_var>=0,zeta_6_var>=0,
        zeta_7_var>=0,zeta_8_var>=0,zeta_9_var>=0,zeta_10_var>=0,zeta_11_var>=0, zeta_12_var>=0]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()

        return zeta_1_var.value, zeta_2_var.value, zeta_31_var.value, zeta_32_var.value, zeta_33_var.value, zeta_34_var.value, zeta_41_var.value, zeta_42_var.value, zeta_43_var.value, zeta_5_var.value, zeta_6_var.value, zeta_7_var.value, zeta_8_var.value, zeta_9_var.value, zeta_10_var.value, zeta_11_var.value, zeta_12_var.value, xi_n_var.value, xi_y_var.value, xi_w_var.value, xi_u_var.value, xi_s_var.value
        '''
        # output.put([i, zeta_1, zeta_2, zeta_31, zeta_32, zeta_33, zeta_34, zeta_41, zeta_42, zeta_43, zeta_5, zeta_6, zeta_7, zeta_8, zeta_9, zeta_10, zeta_11, zeta_12, xi_n, xi_y, xi_w, xi_u, xi_s])
        return zeta_1, zeta_21, zeta_22, zeta_23, zeta_24, zeta_31, zeta_32, zeta_33, zeta_34, zeta_41, zeta_42, zeta_43, zeta_5, zeta_6, zeta_7, zeta_8, zeta_9, zeta_10, zeta_11, zeta_12, xi_n, xi_y, xi_w, xi_u, xi_s


    def updateIndividualDual(self, i, proc, beta, C, O, D, I1, I2, I3, I4, BO, BI, V, M, Jam_N, Q, n, y, w, u, s, 
    zeta_1, zeta_21, zeta_22, zeta_23, zeta_24, zeta_31, zeta_32, zeta_33, zeta_34,  zeta_41, zeta_42, zeta_43, zeta_5, zeta_6, zeta_7, zeta_8, zeta_9, zeta_10, zeta_11, zeta_12, 
    xi_n, xi_y, xi_w, xi_u, xi_s, 
    current_kappa_1, current_kappa_21, current_kappa_22, current_kappa_23, current_kappa_24, current_kappa_31,current_kappa_32,current_kappa_33,current_kappa_34, current_kappa_41,current_kappa_42,current_kappa_43, 
    current_kappa_5, current_kappa_6, current_kappa_7, current_kappa_8, current_kappa_9, current_kappa_10, current_kappa_11, current_kappa_12,  
    current_lamb, current_mu, current_nu_n, current_nu_y, current_nu_w, current_nu_u, current_nu_s, out_extra_xi_n, in_extra_xi_y, output):
        
        EO = np.sort(list(set(C)-set(I1)-set(I2)-set(I3)-set(I4)-set(D)))
        OI = np.sort(list(set(C)-set(D)-set(V)-set(BO)))
        NBI = np.sort(list(set(C)-set(BI)))
        NBO = np.sort(list(set(C)-set(BO)))

        T = self.data['opt_time']
        nu_n = np.zeros_like(n)
        nu_y = np.zeros_like(y)
        nu_w = np.zeros_like(w)
        nu_u = np.zeros_like(u)
        nu_s = np.zeros_like(s)

        lamb = np.zeros_like(current_lamb)
        mu = np.zeros_like(current_mu) 

        kappa_1 = y - n[:,range(T)] + zeta_1 + current_kappa_1
        

        kappa_21 = y[EO,] + zeta_21 - Q.reshape([-1,1])[EO] + current_kappa_21
        kappa_22 = y[V,] + zeta_22 - np.repeat(Q.reshape([-1,1])[[c+1 for c in V]]/beta[0,:].reshape([-1,1]), T, axis = 1) + current_kappa_22
        kappa_23 = y[V,] + zeta_23 - np.repeat(Q.reshape([-1,1])[[c+2 for c in V]]/beta[1,:].reshape([-1,1]), T, axis = 1) + current_kappa_23
        if beta.shape[0] == 3:
            kappa_24 = y[V,] + zeta_24 - np.repeat(Q.reshape([-1,1])[[c+3 for c in V]]/beta[2,:].reshape([-1,1]), T, axis = 1) + current_kappa_24
        else: 
            kappa_24 = None
        
        kappa_31 = y[I1,] -  np.repeat(w[0,].reshape([1,-1]), len(I1), axis = 0) * np.repeat(Q.reshape([-1,1])[I1], T, axis = 1) + zeta_31 + current_kappa_31
        kappa_32 = y[I2,] -  np.repeat(w[1,].reshape([1,-1]), len(I2), axis = 0) * np.repeat(Q.reshape([-1,1])[I2], T, axis = 1) + zeta_32 + current_kappa_32
        kappa_33 = y[I3,] -  np.repeat(w[2,].reshape([1,-1]), len(I3), axis = 0) * np.repeat(Q.reshape([-1,1])[I3], T, axis = 1) + zeta_33 + current_kappa_33
        kappa_34 = y[I4,] -  np.repeat(w[3,].reshape([1,-1]), len(I4), axis = 0) * np.repeat(Q.reshape([-1,1])[I4], T, axis = 1) + zeta_34 + current_kappa_34

        kappa_41 = np.repeat(beta[0,].reshape([-1,1]), T, axis=1)*y[V,:] + self.data['ratio_W']* n[[x+1 for x in V],0:T] - np.repeat(self.data['ratio_W'] * Jam_N.reshape([-1,1])[V], T, axis = 1) + zeta_41 + current_kappa_41
        kappa_42 = np.repeat(beta[1,].reshape([-1,1]), T, axis=1)*y[V,:] + self.data['ratio_W']* n[[x+2 for x in V],0:T] - np.repeat(self.data['ratio_W'] * Jam_N.reshape([-1,1])[V], T, axis = 1) + zeta_42 + current_kappa_42
        if beta.shape[0]==3:
            kappa_43 = np.repeat(beta[2,].reshape([-1,1]), T, axis=1)*y[V,:] + self.data['ratio_W']* n[[x+3 for x in V],0:T] - np.repeat(self.data['ratio_W'] * Jam_N.reshape([-1,1])[V], T, axis = 1) + zeta_43 + current_kappa_43
        else:
            kappa_43=None

        kappa_5 = y[OI,] + self.data['ratio_W'] * n[[proc[d] for d in OI],0:T] + zeta_5 - np.repeat(self.data['ratio_W'] * Jam_N.reshape([-1,1])[OI], T, axis = 1) + current_kappa_5

        new_w = w[0,]+w[1,]
        kappa_6 = new_w[1:T] - new_w[0:T-1] - u[0:T-1] + zeta_6 + current_kappa_6

        kappa_7 = -new_w[1:T] + new_w[0:T-1] - u[0:T-1] + zeta_7 + current_kappa_7
        kappa_8 = -new_w[1:T] - new_w[0:T-1] + u[0:T-1] + zeta_8 + current_kappa_8
        kappa_9 = new_w[1:T] + new_w[0:T-1] + u[0:T-1] -2 + zeta_9 + current_kappa_9
        

        kappa_10 = u[0:T-4] + u[1:T-3] + u[2:T-2] + u[3:T-1] + zeta_10 + current_kappa_10

        kappa_11 = new_w[0:T-7] + new_w[1:T-6] + new_w[2:T-5] + new_w[3:T-4] + new_w[4:T-3] + new_w[5:T-2] + new_w[6:T-1] + new_w[7:T] -8  + zeta_11 + current_kappa_11
        kappa_12 = -new_w[0:T-7] - new_w[1:T-6] - new_w[2:T-5] - new_w[3:T-4] - new_w[4:T-3]  - new_w[5:T-2] - new_w[6:T-1] - new_w[7:T] +1 + zeta_12 + current_kappa_12

        nu_n = n - xi_n + current_nu_n
        nu_y = y - xi_y + current_nu_y
        nu_w = w - xi_w + current_nu_w
        nu_u = u - xi_u + current_nu_u
        nu_s = s - xi_s + current_nu_s

        lamb = y[BO,] + s + self.data['ratio_W'] * out_extra_xi_n[:,0:T] - np.repeat(self.data['ratio_W'] * Jam_N.reshape([-1,1])[BO], T, axis = 1) + current_lamb
        mu = n[BI,1:T+1] - n[BI,0:T] + y[BI, 0:T] - in_extra_xi_y + current_mu

        
        # output.put([i, kappa_1, kappa_2, kappa_31,kappa_32,kappa_33,kappa_34, kappa_41,kappa_42,kappa_43, kappa_5, kappa_6, kappa_7, kappa_8, kappa_9, kappa_10, kappa_11, kappa_12, nu_n, nu_y, nu_w, nu_u, nu_s, lamb, mu])
        return kappa_1, kappa_21, kappa_22, kappa_23, kappa_24, kappa_31,kappa_32,kappa_33,kappa_34, kappa_41,kappa_42,kappa_43, kappa_5, kappa_6, kappa_7, kappa_8, kappa_9, kappa_10, kappa_11, kappa_12, nu_n, nu_y, nu_w, nu_u, nu_s, lamb, mu

        #Rounding by DDVR

    def DDVR_step(self):
        if self.parallel:
            self.DDVR_parallel()
        else:
            for i in range(self.data['num_inter']):
                self.w[i], self.u[i] = self.DDVR(i, None)
    def DDVR_parallel(self):

        output = multiprocessing.Queue()

        process_r = [Process(target= self.DDVR, args=(i, output)) for i in range(self.data['num_inter'])]

        for p in process_r:
            p.start()

        results = [output.get() for p in process_r]

        for i in range(len(results)): # get the results from the parallel processes, indexed by i
            self.w[results[i][0]] = results[i][1]
            self.u[results[i][0]] = results[i][2]

        for p in process_r:
            p.join()


    def DDVRIndividual(self, i, beta, proc, pred, C, O, D, I1, I2, I3, I4, BO, BI, V, M, w, Demand, output):
        m = gb.Model('Rounding')
        T = self.data['opt_time']

        w_B = m.addVars(self.data['phases'], T, lb=0, ub=1, vtype = gb.GRB.BINARY, name = 'w_B')
        new_w_B = m.addVars(T, lb=0, ub=1, vtype = gb.GRB.CONTINUOUS)
        u_B = m.addVars(T-1, lb=0, ub=1, vtype = gb.GRB.CONTINUOUS)

        m.setObjective(gb.quicksum(gb.quicksum(w_B[p, t]*abs(1-w[p, t])+(1-w_B[p, t])*abs(w[p, t]) for p in range(self.data['phases'])) for t in range(T)), gb.GRB.MINIMIZE)
        
        m.addConstrs(new_w_B[t] == w_B[0, t] + w_B[1, t] for t in range(T))
        m.addConstrs(new_w_B[t] - new_w_B[t-1] - u_B[t-1] <= 0 for t in range(1, T))
        m.addConstrs(-new_w_B[t] + new_w_B[t-1] - u_B[t-1] <= 0 for t in range(1, T))
        m.addConstrs(-new_w_B[t] - new_w_B[t-1] + u_B[t-1] <= 0 for t in range(1, T))
        m.addConstrs(new_w_B[t] + new_w_B[t-1] + u_B[t-1] <= 2 for t in range(1, T))

        m.addConstrs(u_B[t] + u_B[t+1] + u_B[t+2] + u_B[t+3] <= 1 for t in range(T-4))  ## we should define u_B as a variable
        m.addConstrs(new_w_B[t]+new_w_B[t+1]+new_w_B[t+2]+new_w_B[t+3]+new_w_B[t+4]+new_w_B[t+5]+new_w_B[t+6]+new_w_B[t+7] <= 8 for t in range(0,T-7))
        m.addConstrs(-(new_w_B[t]+new_w_B[t+1]+new_w_B[t+2]+new_w_B[t+3]+new_w_B[t+4]+new_w_B[t+5]+new_w_B[t+6]+new_w_B[t+7])<=-1 for t in range(0,T-7))

        m.Params.LogToConsole = 0        
        m.Params.TimeLimit=7200
        m.Params.LogFile = "./log/DDVR_rounding.log"
        m.optimize()

        result_w = m.getAttr('x', w_B)
        result_u = m.getAttr('x', u_B)
        w_B_result = np.zeros((self.data['phases'], T))
        u_B_result = np.zeros(T-1)

        for t in range(T-1):
            for p in range(self.data['phases']):
                w_B_result[p, t] = result_w[p, t]
            u_B_result[t] = result_u[t]
        for p in range(self.data['phases']):
            w_B_result[p, T-1] = result_w[p, T-1]

        # output.put([i, w_B_result, u_B_result])
        return w_B_result, u_B_result
        # return w_B_result, u_B_result, y_result, n_result

    def DDVR(self, i, output):
        return self.DDVRIndividual(i, self.data['beta'][i], self.data['proc'][i], self.data['pred'][i], self.data['C'][i], self.data['O'][i], \
        self.data['D'][i], self.data['I1'][i], self.data['I2'][i], self.data['I3'][i], self.data['I4'][i], self.data['BO'][i], self.data['BI'][i], \
        self.data['V'][i], self.data['M'][i], self.w[i], self.Demand[i], output)

    # update y and n by resolving linear problem
    def UpdateYN_step(self):
        if self.parallel:
            self.UpdateYN_parallel()
        else:
            for i in range(self.data['num_inter']):
                self.n[i], self.y[i], self.s[i] = self.UpdateYN(i, None)

    def UpdateYN_parallel(self):

        output = multiprocessing.Queue()

        process_r = [Process(target= self.UpdateYN, args=(i, output)) for i in range(self.data['num_inter'])]

        for p in process_r:
            p.start()

        results = [output.get() for p in process_r]

        for i in range(len(results)): # get the results from the parallel processes, indexed by i

            self.n[results[i][0]] = results[i][1]
            self.y[results[i][0]] = results[i][2]
            self.s[results[i][0]] = results[i][3]

        for p in process_r:
            p.join()

    def UpdateYN(self, i, output):
        return self.UpdateIndividualYN(i, self.data['beta'][i], self.data['proc'][i], self.data['pred'][i], self.data['C'][i], self.data['O'][i], self.data['D'][i], self.data['I1'][i], self.data['I2'][i],  self.data['I3'][i],  self.data['I4'][i], 
        self.data['BO'][i], self.data['BI'][i],  self.data['V'][i],  self.data['M'][i], self.data['jam_N'][i], self.data['capacity_Q'][i], 
        self.w[i], self.u[i], self.zeta_1[i], self.zeta_21[i], self.zeta_22[i], self.zeta_23[i], self.zeta_24[i], self.zeta_31[i], self.zeta_32[i], self.zeta_33[i], self.zeta_34[i], self.zeta_41[i], self.zeta_42[i], self.zeta_43[i], self.zeta_5[i], 
        self.xi_n[i], self.xi_y[i], self.xi_s[i], 
        self.kappa_1[i], self.kappa_21[i], self.kappa_22[i], self.kappa_23[i], self.kappa_24[i], self.kappa_31[i], self.kappa_32[i], self.kappa_33[i], self.kappa_34[i], self.kappa_41[i], self.kappa_42[i], self.kappa_43[i], self.kappa_5[i], 
        self.lamb[i], self.mu[i], self.nu_n[i], self.nu_y[i], self.nu_s[i], self.out_extra_xi_n[i], self.in_extra_xi_y[i], self.Demand[i], output)

    def UpdateIndividualYN(self, i, beta, proc, pred, C, O, D, I1, I2, I3, I4, BO, BI, V, M, Jam_N, Q, w, u, 
    zeta_1, zeta_21, zeta_22, zeta_23, zeta_24, zeta_31, zeta_32, zeta_33, zeta_34,  zeta_41, zeta_42, zeta_43, zeta_5, 
    xi_n, xi_y, xi_s, 
    kappa_1, kappa_21, kappa_22, kappa_23, kappa_24, kappa_31, kappa_32, kappa_33, kappa_34, kappa_41, kappa_42, kappa_43, kappa_5, 
    lamb, mu, nu_n, nu_y, nu_s, out_extra_xi_n, in_extra_xi_y, Demand, output):
        
        T = self.data['opt_time']
        n = cp.Variable((len(C), T+1))
        y = cp.Variable((len(C), T))
        s = cp.Variable((len(BO),T))

        new_w = w[0,:] + w[1,:]

        I = np.sort(I1 + I2 + I3 + I4)
        ND = np.sort(list(set(C) - set(D)))
        EO = np.sort(list((set(C) - set(I)) - set(D)))
        OI = np.sort(list(set(C) - set(D) - set(V) - set(BO))) 
        DO = np.sort(list(set(C) - set(O) - set(I) - set(BI)-set(M))) 

        f_i = (np.sum(np.sum([t * y[i,t] for i in D]) for t in range(T)) 
        + self.data['alpha'] *  np.sum(np.sum([(T-t) * y[i,t] for i in ND]) for t in range(T)))
        norm1 = (sum_squares(y - n[:,range(T)] + zeta_1 + kappa_1)
        + sum_squares(y[EO,:] + zeta_21 - np.repeat(Q.reshape([-1,1])[EO], T, axis = 1) + kappa_21)
        + np.sum(np.sum(square(y[I1[i],t] - w[0,t]*Q[I1[i]] + zeta_31[i,t] + kappa_31[i,t]) for i in range(len(I1))) for t in range(T))
        + np.sum(np.sum(square(y[I2[i],t] - w[1,t]*Q[I2[i]] + zeta_32[i,t] + kappa_32[i,t]) for i in range(len(I2))) for t in range(T))
        + np.sum(np.sum(square(y[I3[i],t] - w[2,t]*Q[I3[i]] + zeta_33[i,t] + kappa_33[i,t]) for i in range(len(I3))) for t in range(T))
        + np.sum(np.sum(square(y[I4[i],t] - w[3,t]*Q[I4[i]] + zeta_34[i,t] + kappa_34[i,t]) for i in range(len(I4))) for t in range(T))
        + sum_squares(y[OI,:] + self.data['ratio_W'] * n[[proc[d] for d in OI],0:T] + zeta_5 - self.data['ratio_W'] * np.repeat(Jam_N.reshape([-1,1])[OI], T, axis = 1) + kappa_5)
        + sum_squares(n - xi_n + nu_n)
        + sum_squares(y - xi_y + nu_y)
        + sum_squares(s - xi_s + nu_s)
        + sum_squares(y[BO,:] + s - np.repeat(self.data['ratio_W']*Jam_N.reshape([-1,1])[BO], T, axis = 1) + self.data['ratio_W']* out_extra_xi_n[:,0:T] + lamb)
        + sum_squares(n[BI,1:T+1] - n[BI,0:T] + y[BI,:] - in_extra_xi_y + mu))

        norm1 = (norm1 + np.sum(np.sum(square(beta[0,i]*y[V[i],t] + self.data['ratio_W']*n[[V[i]+1],t] + zeta_41[i,t] - self.data['ratio_W'] * Jam_N[V[i]] + kappa_41[i,t]) for i in range(len(V))) for t in range(T))
                    + np.sum(np.sum(square(beta[1,i]*y[V[i],t] + self.data['ratio_W']*n[[V[i]+2],t] + zeta_42[i,t] - self.data['ratio_W'] * Jam_N[V[i]] + kappa_42[i,t]) for i in range(len(V))) for t in range(T))
                    + sum_squares(y[V,:] + zeta_22 - np.repeat(Q.reshape([-1,1])[[c+1 for c in V]]/beta[0,:].reshape([-1,1]), T, axis = 1) + kappa_22)
                    + sum_squares(y[V,:] + zeta_23 - np.repeat(Q.reshape([-1,1])[[c+2 for c in V]]/beta[1,:].reshape([-1,1]), T, axis = 1) + kappa_23))

        if beta.shape[0] == 3:
            norm1 = (norm1 + np.sum(np.sum(square(beta[2,i]*y[V[i],t] + self.data['ratio_W']*n[[V[i]+3],t] + zeta_43[i,t] - self.data['ratio_W'] * Jam_N[V[i]] + kappa_43[i,t]) for i in range(len(V))) for t in range(T))
                    + sum_squares(y[V,:] + zeta_24 - np.repeat(Q.reshape([-1,1])[[c+3 for c in V]]/beta[2,:].reshape([-1,1]), T, axis = 1) + kappa_24))

       
        objective = cp.Minimize(f_i + self.data['rho']/2 * norm1)
        constraints = [n[O,1:T+1] - n[O,0:T] + y[O,0:T] == Demand]

        for c in M:
            constraints += [n[c,1:T+1] - n[c,0:T] - np.sum(y[p,0:T] for p in pred[c]) + y[c,0:T] == 0]
        for c in DO:
            constraints += [n[c,1:T+1] - n[c,0:T] - y[pred[c],0:T] + y[c,0:T] == 0]
        for c in I:
            constraints += [n[c,1:T+1] - n[c,0:T] - reshape(beta[c-pred[c]-1, V.index(pred[c])]*y[[pred[c]],0:T],(T,))+ y[c,0:T] ==0] 
        
        #constraints += [n >= 0, y >= 0, w<=1, w>=0]
        
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        #print("Optimal value", prob.solve())
        #print(n.value)

        # output.put([i, n.value, y.value, s.value]) #work for parallel computing
        return n.value, y.value, s.value
    
    def Objective(self):
        return (np.sum(np.sum(np.sum([t * self.y[i][c,t] for c in self.data['D'][i]]) for t in range(self.data['opt_time']))  for i in range(self.data['num_inter']))
        + self.data['alpha'] *  np.sum(np.sum(np.sum([(self.data['opt_time']-t) * self.y[i][c,t] for c in list(set(self.data['C'][i])-set(self.data['D'][i]))]) for t in range(self.data['opt_time'])) for i in range(self.data['num_inter'])))
        # return self.data['alpha'] *  np.sum(np.sum(np.sum([(self.data['opt_time']-t) * self.y[i][c,t] for c in list(set(self.data['C'][i])-set(self.data['D'][i]))]) for t in range(self.data['opt_time'])) for i in range(self.data['num_inter']))

    def residual(self):
        # return LA.norm(self.y[i] - self.n[i][:,range(self.data['opt_time'])] + self.zeta_1[i])
        T = self.data['opt_time']
        res = np.zeros(self.data['num_inter'])
        for inter in range(self.data['num_inter']):
            I1 = self.data['I1'][inter]
            I2 = self.data['I2'][inter] 
            I3 = self.data['I3'][inter]
            I4 = self.data['I4'][inter]
            C = self.data['C'][inter]
            D = self.data['D'][inter]
            V = self.data['V'][inter]
            M = self.data['M'][inter]
            O = self.data['O'][inter]
            BO = self.data['BO'][inter]
            BI = self.data['BI'][inter]
            Jam_N = self.data['jam_N'][inter]
            Q = self.data['capacity_Q'][inter]
            I = np.sort(I1 + I2 + I3 + I4)
            ND = np.sort(list(set(C) - set(D)))
            EO = np.sort(list((set(C) - set(I)) - set(D)))
            OI = np.sort(list(set(C) - set(D) - set(V) - set(BO))) 
            DO = np.sort(list(set(C) - set(O) - set(I) - set(BI)-set(M)))
            beta = self.data['beta'][inter]
            proc = self.data['proc'][inter]
            new_w = self.w[inter][0,] + self.w[inter][1,]
            
            res_temp = max([LA.norm(self.y[inter] - self.n[inter][:,range(self.data['opt_time'])] + self.zeta_1[inter]), 
            LA.norm(self.y[inter][EO,:] + self.zeta_21[inter] - np.repeat(Q.reshape([-1,1])[EO], T, axis = 1)), 
            LA.norm(self.y[inter][V] + self.zeta_22[inter] - np.repeat(Q.reshape([-1,1])[[c+1 for c in V]]/beta[0,:].reshape([-1,1]), T, axis = 1)),
            LA.norm(self.y[inter][V] + self.zeta_23[inter] - np.repeat(Q.reshape([-1,1])[[c+2 for c in V]]/beta[1,:].reshape([-1,1]), T, axis = 1)),
            LA.norm(self.y[inter][I1,] - np.repeat(self.w[inter][0,].reshape([1,-1]), len(I1), axis = 0) * np.repeat(Q.reshape([-1,1])[I1], T, axis = 1) + self.zeta_31[inter]),
            LA.norm(self.y[inter][I2,] - np.repeat(self.w[inter][1,].reshape([1,-1]), len(I2), axis = 0) * np.repeat(Q.reshape([-1,1])[I2], T, axis = 1) + self.zeta_32[inter]),
            LA.norm(self.y[inter][I3,] - np.repeat(self.w[inter][2,].reshape([1,-1]), len(I3), axis = 0) * np.repeat(Q.reshape([-1,1])[I3], T, axis = 1) + self.zeta_33[inter]),
            LA.norm(self.y[inter][I4,] - np.repeat(self.w[inter][3,].reshape([1,-1]), len(I4), axis = 0) * np.repeat(Q.reshape([-1,1])[I4], T, axis = 1) + self.zeta_34[inter]),
            LA.norm(np.repeat(beta[0,].reshape([-1,1]), T, axis=1)*self.y[inter][V,:] + self.data['ratio_W']* self.n[inter][[x+1 for x in V],0:T] - np.repeat(self.data['ratio_W'] * Jam_N.reshape([-1,1])[V], T, axis = 1) + self.zeta_41[inter]),
            LA.norm(np.repeat(beta[1,].reshape([-1,1]), T, axis=1)*self.y[inter][V,:] + self.data['ratio_W']* self.n[inter][[x+1 for x in V],0:T] - np.repeat(self.data['ratio_W'] * Jam_N.reshape([-1,1])[V], T, axis = 1) + self.zeta_42[inter]),
            LA.norm(self.y[inter][OI,] + self.data['ratio_W'] * self.n[inter][[proc[x] for x in OI],0:T] - np.repeat(self.data['ratio_W']* Jam_N.reshape([-1,1])[OI], T, axis = 1) + self.zeta_5[inter]),
            LA.norm(new_w[1:T] - new_w[0:T-1] - self.u[inter][0:T-1] + self.zeta_6[inter]),
            LA.norm(-new_w[1:T] + new_w[0:T-1] - self.u[inter][0:T-1] + self.zeta_7[inter]),
            LA.norm(-new_w[1:T] - new_w[0:T-1] + self.u[inter][0:T-1] + self.zeta_8[inter]),
            LA.norm(new_w[1:T] + new_w[0:T-1] + self.u[inter][0:T-1] -2 + self.zeta_9[inter]),
            LA.norm(self.u[inter][0:T-4] + self.u[inter][1:T-3] + self.u[inter][2:T-2] + self.u[inter][3:T-1] -1 + self.zeta_10[inter]),
            LA.norm(new_w[0:T-7] + new_w[1:T-6] + new_w[2:T-5] + new_w[3:T-4] + new_w[4:T-3] + new_w[5:T-2] + new_w[6:T-1] + new_w[7:T] -8 + self.zeta_11[inter]),
            LA.norm(-new_w[0:T-7] - new_w[1:T-6] - new_w[2:T-5] - new_w[3:T-4] - new_w[4:T-3]  - new_w[5:T-2] - new_w[6:T-1] - new_w[7:T] +1 + self.zeta_12[inter])])
            if beta.shape[0] == 3:
                res[inter] = max([res_temp, LA.norm(self.y[inter][V] + self.zeta_24[inter] - np.repeat(Q.reshape([-1,1])[[c+3 for c in V]]/beta[2,:].reshape([-1,1]), T, axis = 1)), 
                LA.norm(np.repeat(beta[2,].reshape([-1,1]), T, axis=1)*self.y[inter][V,:] + self.data['ratio_W']* self.n[inter][[x+1 for x in V],0:T] - np.repeat(self.data['ratio_W'] * Jam_N.reshape([-1,1])[V], T, axis = 1) + self.zeta_43[inter])])
            else:
                res[inter] = res_temp
        print(LA.norm(self.y[0] - self.n[0][:,range(self.data['opt_time'])] + self.zeta_1[0]))
        return max(res)
