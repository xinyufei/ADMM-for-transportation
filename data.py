import numpy as np
# specify the traffic network in plymouth road
num_iterations = 100
T = 8
N = 6 # number of intersetions

C = [None] * N # cells in each intersection   all the cells
O = [None] * N # origins in each inter
D = [None] * N # destinations
BI = [None] * N # same as the ADMM paper boundary cells of inflow
BO = [None] * N
I1 = [None] * N #intersection cells
I2 = [None] * N
I3 = [None] * N
I4 = [None] * N
V = [None] * N #diverging cells  
M = [None] * N #merging cells
beta = [None] * N #ratio to turn left/right
proc = [None] * N #proceding cell relationships
pred = [None] * N #preceding cell relationships
Jam_N = [None] * N
Q = [None] * N
Demand = [None] * N


C[0] = list(range(80))
O[0] = [0, 68]
D[0] = [57, 67]
BI[0] = [29]
BO[0] = [28]
I1[0] = [10]
I2[0] = [11,46,47]
I3[0] = [78]
I4[0] = [79]
V[0] = [9, 45, 77]
M[0] = [12, 48, 58]
# beta[0] = np.array([[0.181, 0, 0.73], [0.819, 0.604, 0], [0, 0.396, 0.27]])
beta[0] = np.array([[0.181, 0.604, 0.73], [0.819, 0.396, 0.27]])
proc[0] = {}
proc[0][10] = 58
proc[0][11] = 12
proc[0][46] = 48
proc[0][47] = 58
proc[0][78] = 12
proc[0][79] = 48
proc[0].update({c: c+1 for c in list(set(C[0])-set(I1[0])-set(I2[0])-set(I3[0])-set(I4[0]))})
pred[0] = {}
pred[0][10] = 9
pred[0][11] = 9
pred[0][46] = 45
pred[0][47] = 45
pred[0][78] = 77
pred[0][79] = 77
pred[0][12] = [11, 78]
pred[0][48] = [46, 79]
pred[0][58] = [10, 47]
pred[0].update({c: c-1 for c in list(set(C[0])-set(I1[0])-set(I2[0])-set(I3[0])-set(I4[0])-set(M[0]))})

Jam_N[0] = np.zeros(len(C[0]))
for c in O[0]+D[0]: 
    Jam_N[0][c] = 99999
for c in C[0][1:9] + C[0][12:45] + C[0][48:57]:
    Jam_N[0][c] = 8
for c in C[0][58:67] + C[0][69:77]:
    Jam_N[0][c] = 4

for c in I1[0] + I2[0] + I3[0] + I4[0]:
    Jam_N[0][c] = 4
Jam_N[0][11] = 8
Jam_N[0][46] = 8
Jam_N[0][9] = 12
Jam_N[0][45] = 12
Jam_N[0][77] = 8

Q[0] = Jam_N[0]/4
Q[0][0] = 2
Q[0][68] = 1
for c in D[0]:
    Q[0][c] = 99999
Demand[0] = np.zeros(len(O[0]))
Demand[0][0] = 857
Demand[0][1] = 200

C[1] = list(range(112))
O[1] = [66, 89]
D[1] = [88, 111]
BI[1] = [0, 33]
BO[1] = [32, 65]
I1[1] = [16, 47]
I2[1] = [17,18,48,49]
I3[1] = [76, 99]
I4[1] = [77,78,100,101]
V[1] = [15,46,75,98]
M[1] = [19,50,79,102]
beta[1] = np.array([[0.006,0.048,0.595,0.603],[0.954,0.928,0.055,0.111],[0.04,0.024,0.35,0.286]])
proc[1] = {}
proc[1][16] = 79
proc[1][17] = 19
proc[1][18] = 102
proc[1][49] = 79
proc[1][48] = 50
proc[1][47] = 102
proc[1][76] = 50
proc[1][77] = 79
proc[1][78] = 19
proc[1][99] = 19
proc[1][100] = 102
proc[1][101] = 50
proc[1].update({c: c+1 for c in list(set(C[1])-set(I1[1])-set(I2[1])-set(I3[1])-set(I4[1]))})

pred[1] = {}
pred[1].update(dict([i, 15] for i in [16,17,18]))
pred[1].update(dict([i, 46] for i in [47,48,49]))
pred[1].update(dict([i, 75] for i in [76,77,78]))
pred[1].update(dict([i, 98] for i in [99,100,101]))
pred[1][19] = [17, 78, 99]
pred[1][50] = [48, 101, 76]
pred[1][79] = [49, 77, 16]
pred[1][102] = [100, 18, 47]
pred[1].update({c: c-1 for c in list(set(C[1])-set(I1[1])-set(I2[1])-set(I3[1])-set(I4[1])-set(M[1]))})

Jam_N[1] = np.zeros(len(C[1]))
for c in O[1]+D[1]: 
    Jam_N[1][c] = 99999
for c in C[1][0:15] + C[1][19:46] + C[1][50:66]:
    Jam_N[1][c] = 8
for c in C[1][67:75] + C[1][79:88] + C[1][90:98] + C[1][102:111]:
    Jam_N[1][c] = 4

for c in I1[1] + I2[1] + I3[1] + I4[1]:
    Jam_N[1][c] = 4
Jam_N[1][17] = 8
Jam_N[1][48] = 8
Jam_N[1][15] = 12
Jam_N[1][46] = 12
Jam_N[1][75] = 8
Jam_N[1][98] = 8

Q[1] = Jam_N[1]/4
Q[1][66] = 1
Q[1][89] = 1
for c in D[1]:
    Q[1][c] = 99999
Demand[1] = np.zeros(len(O[1]))
Demand[1][0] = 326
Demand[1][1] = 63


C[2] = list(range(64))
O[2] = [52]
D[2] = [51]
BI[2] = [0, 21]
BO[2] = [20,41]
I1[2] = [13]
I2[2] = [14,27,28]
I3[2] = [62]
I4[2] = [63]
V[2] = [12,26,61]
M[2] = [15,29,42]
# beta[2] = np.array([[0.059,0,0.378],[0.941,0.969,0],[0,0.031,0.622]])
beta[2] = np.array([[0.059, 0.969, 0.378], [0.941, 0.031, 0.622]])
proc[2]={}
proc[2][13] = 42
proc[2][14] = 15
proc[2][27] = 29
proc[2][28] = 42
proc[2][62] = 15
proc[2][63] = 29
proc[2].update({c: c+1 for c in list(set(C[2])-set(I1[2])-set(I2[2])-set(I3[2])-set(I4[2]))})
pred[2] = {}
pred[2][15] = [14,62]
pred[2][29] = [27,63]
pred[2][42] = [13, 28]
pred[2].update(dict([i, 12] for i in [13,14]))
pred[2].update(dict([i, 26] for i in [27,28]))
pred[2].update(dict([i, 61] for i in [62,63]))
pred[2].update({c: c-1 for c in list(set(C[2])-set(I1[2])-set(I2[2])-set(I3[2])-set(I4[2])-set(M[2]))})

Jam_N[2] = np.zeros(len(C[2]))
for c in O[2]+D[2]: 
    Jam_N[2][c] = 99999
for c in C[2][0:12] + C[2][15:26] + C[2][29:42]:
    Jam_N[2][c] = 8
for c in C[2][42:51] + C[2][53:61]:
    Jam_N[2][c] = 4

for c in I1[2] + I2[2] + I3[2] + I4[2]:
    Jam_N[2][c] = 4
Jam_N[2][14] = 8
Jam_N[2][27] = 8
Jam_N[2][12] = 12
Jam_N[2][26] = 12
Jam_N[2][61] = 8

Q[2] = Jam_N[2]/4
Q[2][52] = 1
for c in D[2]:
    Q[2][c] = 99999
Demand[2] = np.zeros(len(O[2]))
Demand[2][0] = 180



C[3] = list(range(72))
O[3] = [26,49]
D[3] = [48,71]
BI[3] = [0,13]
BO[3] = [12,25]
I1[3] = [6,17]
I2[3] = [7,8,18,19]
I3[3] = [36,59]
I4[3] = [37,38,60,61]
V[3] = [5,16,35,58]
M[3] = [9,20,39,62]
beta[3] = np.array([[0.193,0.016,0.448,0.423],[0.774,0.88,0.362,0.051],[0.033,0.104,0.19,0.526]])
proc[3]= {}
proc[3][6] = 39
proc[3][7] = 9
proc[3][8] = 62
proc[3][17] = 62
proc[3][18] = 20
proc[3][19] = 39
proc[3][36] = 20
proc[3][37] = 39
proc[3][38] = 9
proc[3][59] = 9
proc[3][60] = 62
proc[3][61] = 20
proc[3].update({c: c+1 for c in list(set(C[3])-set(I1[3])-set(I2[3])-set(I3[3])-set(I4[3]))})
pred[3] = {}
pred[3][9] = [7,38,59]
pred[3][20] = [18,61,36]
pred[3][39] = [37,19,6]
pred[3][62] = [8,60,17]
pred[3].update(dict([i, 5] for i in [6,7,8]))
pred[3].update(dict([i, 16] for i in [17,18,19]))
pred[3].update(dict([i, 35] for i in [36,37,38]))
pred[3].update(dict([i, 58] for i in [59,60,61]))
pred[3].update({c: c-1 for c in list(set(C[3])-set(I1[3])-set(I2[3])-set(I3[3])-set(I4[3])-set(M[3]))})

Jam_N[3] = np.zeros(len(C[3]))
for c in O[3]+D[3]: 
    Jam_N[3][c] = 99999
for c in C[3][0:5] + C[3][9:16] + C[3][20:26]:
    Jam_N[3][c] = 8
for c in C[3][27:35] + C[3][39:48] + C[3][50:58] + C[3][62:71]:
    Jam_N[3][c] = 4

for c in I1[3] + I2[3] + I3[3] + I4[3]:
    Jam_N[3][c] = 4
Jam_N[3][7] = 8
Jam_N[3][18] = 8
Jam_N[3][5] = 12
Jam_N[3][16] = 12
Jam_N[3][35] = 8
Jam_N[3][58] = 8

Q[3] = Jam_N[3]/4
Q[3][26] = 1
Q[3][49] = 1
for c in D[3]:
    Q[3][c] = 99999
Demand[3] = np.zeros(len(O[3]))
Demand[3][0] = 174
Demand[3][1] = 390



C[4] = list(range(90))
O[4] = [44,67]
D[4] = [66,89]
BI[4] = [0,22]
BO[4] = [21,43]
I1[4] = [3,38]
I2[4] = [4,5,39,40]
I3[4] = [54,77]
I4[4] = [55,56,78,79]
V[4] = [2,37,53,76]
M[4] = [6,41,57,80]
beta[4] = np.array([[0.012,0.189,0.28,0.444],[0.812,0.711,0.343,0.526],[0.176,0.1,0.377,0.03]])
proc[4] = {}
proc[4].update(dict([i, 6] for i in [4,56,77]))
proc[4].update(dict([i, 41] for i in [39,79,54]))
proc[4].update(dict([i, 57] for i in [55,40,3]))
proc[4].update(dict([i, 80] for i in [78,5,38]))
proc[4].update({c: c+1 for c in list(set(C[4])-set(I1[4])-set(I2[4])-set(I3[4])-set(I4[4]))})
pred[4] = {}
pred[4][6] = [4,56,77]
pred[4][41] = [39,79,54]
pred[4][57] = [55,40,3]
pred[4][80] = [78,5,38]
pred[4].update(dict([i, 2] for i in [3,4,5]))
pred[4].update(dict([i, 37] for i in [38,39,40]))
pred[4].update(dict([i, 53] for i in [54,55,56]))
pred[4].update(dict([i, 76] for i in [77,78,79]))
pred[4].update({c: c-1 for c in list(set(C[4])-set(I1[4])-set(I2[4])-set(I3[4])-set(I4[4])-set(M[4]))})

Jam_N[4] = np.zeros(len(C[4]))
for c in O[4]+D[4]: 
    Jam_N[4][c] = 99999
for c in C[4][0:2] + C[4][6:37] + C[4][41:44]:
    Jam_N[4][c] = 8
for c in C[4][45:53] + C[4][57:66] + C[4][68:76] + C[4][80:89]:
    Jam_N[4][c] = 8

for c in I1[4] + I2[4] + I3[4] + I4[4]:
    Jam_N[4][c] = 4
Jam_N[4][4] = 8
Jam_N[4][39] = 8
Jam_N[4][55] = 8
Jam_N[4][78] = 8
Jam_N[4][2] = 12
Jam_N[4][37] = 12
Jam_N[4][53] = 12
Jam_N[4][76] = 12

Q[4] = Jam_N[4]/4
Q[4][44] = 2
Q[4][67] = 2
for c in D[4]:
    Q[4][c] = 99999
Demand[4] = np.zeros(len(O[4]))
Demand[4][0] = 897
Demand[4][1] = 432


C[5] = list(range(104))
O[5] = [29,58,81]
D[5] = [28,80,103]
BI[5] = [0]
BO[5] = [57]
I1[5] = [16,39]
I2[5] = [17,18,40,41]
I3[5] = [68,91]
I4[5] = [69,70,92,93]
V[5] = [15,38,67,90]
M[5] = [19,42,71,94]   
beta[5] = np.array([[0.05,0.068,0.226,0.717],[0.905,0.708,0.252,0.15],[0.045,0.224,0.522,0.133]])
proc[5] = {}
proc[5].update(dict([i, 19] for i in [17,70,91]))
proc[5].update(dict([i, 42] for i in [40,93,68]))
proc[5].update(dict([i, 71] for i in [69,41,16]))
proc[5].update(dict([i, 94] for i in [92,18,39]))
proc[5].update({c: c+1 for c in list(set(C[5])-set(I1[5])-set(I2[5])-set(I3[5])-set(I4[5]))})
pred[5] = {}
pred[5][19] = [17,70,91]
pred[5][42] = [40,93,68]
pred[5][71] = [69,41,16]
pred[5][94] = [92,18,39]
pred[5].update(dict([i, 15] for i in [16,17,18]))
pred[5].update(dict([i, 38] for i in [39,40,41]))
pred[5].update(dict([i, 67] for i in [68,69,70]))
pred[5].update(dict([i, 90] for i in [91,92,93]))
pred[5].update({c: c-1 for c in list(set(C[5])-set(I1[5])-set(I2[5])-set(I3[5])-set(I4[5])-set(M[5]))})

Jam_N[5] = np.zeros(len(C[5]))
for c in O[5]+D[5]: 
    Jam_N[5][c] = 99999
for c in C[5][0:15] + C[5][30:38] + C[5][42:58]:
    Jam_N[5][c] = 8
for c in C[5][19:28]:
    Jam_N[5][c] = 12
for c in C[5][59:67]:
    Jam_N[5][c] = 4
for c in C[5][71:80] + C[5][82:90] + C[5][94:103]:
    Jam_N[5][c] = 8

for c in I1[5] + I2[5] + I3[5] + I4[5]:
    Jam_N[5][c] = 4
Jam_N[5][17] = 12
Jam_N[5][40] = 8
Jam_N[5][91] = 8
Jam_N[4][15] = 16
Jam_N[5][38] = 12
Jam_N[5][67] = 8
Jam_N[5][90] = 12

Q[5] = Jam_N[5]/4
Q[5][29] = 2
Q[5][58] = 1
Q[5][81] = 1
for c in D[5]:
    Q[5][c] = 99999
Demand[5] = np.zeros(len(O[5]))
Demand[5][0] = 1112
Demand[5][1] = 699
Demand[5][2] = 765


W=1/3
alpha=-1.6