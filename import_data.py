import xdrlib ,sys
import xlrd
import numpy as np
# specify the traffic network in plymouth road
num_iterations = 1000
T = 8
N = 6 # number of intersetions
 
 
file= 'Plymouth_Structure2.xlsx'
data = xlrd.open_workbook(file)
table = data.sheet_by_index(0)
nrows = table.nrows
ncols = table.ncols
excel_matrix = np.zeros((nrows-1, ncols))
print(nrows)
print(ncols)
for row in range(1, nrows):
    for col in range(ncols):
        cell_value = table.cell(row, col).value
        excel_matrix[row - 1, col] = cell_value

#type 1: ordinary
#type 2: merge
#type 3: diverge
#type 4: intersection?
#type 5: origin
#type 6: destination?
#type 7: dummy


C_ALL = list(range(nrows - 1))
O_ALL = []
M_ALL = []
V_ALL = []
D_ALL = []
DU_ALL = []
I_ALL = [None]*N
#I_ALL: num_intersection * num_phase * []
for ind in range(N):
    I_ALL[ind] = [None]*4
    for jnd in range(4):
        I_ALL[ind][jnd] = []

pred_all = [None]*len(C_ALL)
proc_all = [None]*len(C_ALL)
jam = np.zeros(len(C_ALL))
Q = np.zeros(len(C_ALL))
beta = np.zeros((len(C_ALL),3))
demand = np.zeros(len(C_ALL))

for c in range(nrows-1):
    type_c = excel_matrix[c,1]
    if type_c == 2:
        M_ALL.append(c)
    elif type_c == 3:
        V_ALL.append(c)
    elif type_c == 4:
        phase = int(excel_matrix[c,15]-1)
        if phase == -1:
            phase = int(excel_matrix[c-1,15]-1)
        inter = int(excel_matrix[c, 17]-1)
        I_ALL[inter][phase].append(c)
    elif type_c == 5:
        O_ALL.append(c)
    elif type_c == 6:
        D_ALL.append(c)
    elif type_c == 7:
        DU_ALL.append(c)

    pred = []
    for ind in range(int(excel_matrix[c,2])):
        pred.append(excel_matrix[c,3+ind] - 1)
    pred_all[c] = pred
    
    proc = []
    for ind in range(int(excel_matrix[c,6])):
        proc.append(excel_matrix[c,7+ind] - 1)
    proc_all[c] = proc

    jam[c] = excel_matrix[c,10]

    Q[c] = excel_matrix[c,11]

    beta[c,:] = excel_matrix[c, 12:15]

    demand[c] = excel_matrix[c, 16]/1800


alpha = -1.6
W = 1/3
