import numpy as np
import math
from admm_cvx import ADMM
from admm_cvxDDVR import ADMMDDVR
import multiprocessing
import time
import matplotlib.pyplot as plt
from data import *

multiprocessing.set_start_method('spawn', True)

if __name__ == '__main__':

    input_data={}
    input_data['C'] = C
    input_data['O'] = O
    input_data['D'] = D
    input_data['BI'] = BI
    input_data['BO'] = BO
    input_data['I1'] = I1
    input_data['I2'] = I2
    input_data['I3'] = I3
    input_data['I4'] = I4
    input_data['V'] = V
    input_data['M'] = M
    input_data['beta'] = beta
    input_data['proc'] = proc
    input_data['pred'] = pred
    input_data['jam_N'] = Jam_N
    input_data['capacity_Q'] = Q
    input_data['Demand'] = Demand

    input_data['ratio_W'] = W
    input_data['num_inter'] = N
    input_data['penalty_rho'] = N
    input_data['weight_alpha'] = alpha
    input_data['optimization_time'] = T
    input_data['phases'] = 4

    #lp_opt = -2.125199230769e+04
    lp_opt = -4.550933333e+02
    mip_opt = -6.562000838750e+06
    rounding = True
    admm = ADMM(input_data, parallel = False)
    obj = np.zeros(num_iterations)
    res = np.zeros(num_iterations)
    # admmDDVR = ADMMDDVR(input_data, parallel=True)
    time_solvex = np.zeros(num_iterations)
    time_solvez = np.zeros(num_iterations)
    time_updual = np.zeros(num_iterations)
    time_DDVR = np.zeros(num_iterations)
    ite_time = np.zeros(num_iterations)
    print(admm.Objective())
    f = open("log/ADMM_LP_T8.log", 'w+')
    f.close()
    ms_start = time.time()
    for i in range(0, num_iterations):
        ite_start = time.time()
        admm.step()

        ite_end = time.time()
        ite_time[i] = ite_end - ite_start
        obj[i] = admm.Objective()
        res[i] = admm.residual()
        # obj_r[i] = admmDDVR.Objective()
        # res_r[i] = admmDDVR.residual()
        time_solvex[i] = admm.time[0]
        time_solvez[i] = admm.time[1]
        time_updual[i] = admm.time[2]
        # time_DDVR[i] = admm.time[3]
        f = open("log/ADMM_LP_T8.log", "a+")
        print('iter %5d: gap %.3f' %
        (i + 1, (mip_opt-obj[i])/abs(mip_opt)))
        print('iter %5d: gap %.3f' %
        (i + 1, (mip_opt-obj[i])/abs(mip_opt)), file=f)
        print('iter %5d: gap %.3f' %
        (i + 1, (lp_opt-obj[i])/abs(lp_opt)))
        print('iter %5d: gap %.3f' %
        (i + 1, (lp_opt-obj[i])/abs(lp_opt)), file=f)
        print('iter %5d: res %.3f' %
        (i + 1, res[i]))
        print('iter %5d: res %.3f' %
        (i + 1, res[i]), file=f)
        print('iter %5d: time to solve X %.3f' % 
        (i + 1, time_solvex[i]))
        print('iter %5d: time to solve X %.3f' % 
        (i + 1, time_solvex[i]), file=f)
        print('iter %5d: time to solve Z %.3f' % 
        (i + 1, time_solvez[i]))
        print('iter %5d: time to solve Z %.3f' % 
        (i + 1, time_solvez[i]), file=f)
        print('iter %5d: time to update dual %.3f' % 
        (i + 1, time_updual[i]))
        print('iter %5d: time to update dual %.3f' % 
        (i + 1, time_updual[i]), file=f)
        print('iter %5d: time per iteration %.3f' % 
        (i + 1, ite_time[i]))
        print('iter %5d: time per iteration %.3f' % 
        (i + 1, ite_time[i]), file=f)
        f.close()

    ms_end1 = time.time()
    """f = open('log/ADMM_MIP_T8.log',"a+")
    print('time to solve MIP: ', ms_end1 - ms_start)
    print('time to solve MIP: ', ms_end1 - ms_start, file=f)
    f.close()
    admmDDVR.step_parallel(admm.w, admm.u)
    obj_r = np.zeros(1)"""
    DDVR_start = time.time()
    admm.DDVR_step()
    admm.UpdateYN_step()
    obj_r = admm.Objective()
    DDVR_end = time.time()
    time_DDVR = DDVR_end - DDVR_start
    ms_end = time.time()
    f = open('log/ADMM_LP_rounding_T8.log',"w+")
    print('time to rounding variables %.3f' % 
    (time_DDVR))
    print('time to rounding variables %.3f' % 
    (time_DDVR), file=f)
    print('objective value after rounding %.3f' % 
    obj_r)
    print('objective value after rounding %.3f' % 
    obj_r, file = f)
    print('gap after rounding %.3f' % 
    ((mip_opt - obj_r)/abs(mip_opt)))
    print('gap after rounding %.3f' % 
    ((mip_opt - obj_r)/abs(mip_opt)), file = f)
    f.close()
    print('elapsed time', ms_end - ms_start)
    f = open("log/ADMM_LP_rounding_T8.log", "a+")
    print('elapsed time', ms_end - ms_start, file=f)
    f.close()
    plt.plot(obj)
    plt.show()
    plt.plot((mip_opt-obj)/abs(mip_opt))
    plt.show()

    plt.plot(res)
    plt.show()

