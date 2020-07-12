# generate network model given size n
import numpy as np
# specify the traffic network in plymouth road
class Network:
	def __init__(self, size = (4,4), random = True, sample_size = 20, T = 20):
		Max_Jam_N = 16
		Max_Q = 4
		self.num_iterations = 1000
		self.T = T # num of time steps
		if type(size) == tuple:
			m = size[0]
			n = size[1]
			self.N = m*n
		else:
			m = size
			n = size
			self.N = n*n
		self.C = [None] * self.N # self.Cells in each intersection   all the cells
		self.O = [None] * self.N # origins in each inter
		self.D = [None] * self.N # destinations
		self.BI = [None] * self.N # same as the ADMM paper boundary cells of inflow
		self.BO = [None] * self.N
		self.I1 = [None] * self.N #intersection cells
		self.I2 = [None] * self.N
		self.I3 = [None] * self.N
		self.I4 = [None] * self.N
		self.V = [None] * self.N #diverging cells  
		self.M = [None] * self.N #merging cells
		self.proc = [None] * self.N #proceding cell relationships
		self.pred = [None] * self.N #preceding cell relationships
		self.Demand = [None] * self.N
		self.Jam_N = [None] * self.N # different for intersection cells and line cells
		self.Q = [None] * self.N # different for intersection cells and line cells
		self.beta = [None]*self.N # turning ratio
		self.n_init = [None]*self.N # initialization of vehicles in cells
		self.W = 1/3
		self.alpha = 1.6

		# generate network
		for i in range(m):
			self.C[i*n] = list(range(32))
			self.O[i*n] = [0]
			self.Demand[i*n]=[]
			self.D[i*n] = [17]
			self.BI[i*n] = [9,18,25]
			self.BO[i*n] = [8,24,31]
			self.I1[i*n] = [4,11]
			self.I2[i*n] = [5,6,12,13]
			self.I3[i*n] = [20,27]
			self.I4[i*n] = [21,22,28,29]
			self.V[i*n] = [3,10,19,26]
			self.M[i*n] = [7,14,23,30]
			# self.beta[i*n] = np.array([[0.15,0.15,0.15,0.15],[0.7,0.7,0.7,0.7],[0.15,0.15,0.15,0.15]])
			self.proc[i*n] = {}
			self.proc[i*n][4] = 23
			self.proc[i*n][5] = 7
			self.proc[i*n][6] = 30
			self.proc[i*n][11] = 30
			self.proc[i*n][12] = 14
			self.proc[i*n][13] = 23
			self.proc[i*n][20] = 14
			self.proc[i*n][21] = 23
			self.proc[i*n][22] = 7
			self.proc[i*n][27] = 7
			self.proc[i*n][28] = 30
			self.proc[i*n][29] = 14
			self.proc[i*n].update({c: c+1 for c in list(set(self.C[i*n])-set(self.I1[i*n])-set(self.I2[i*n])-set(self.I3[i*n])-set(self.I4[i*n]))})
			self.pred[i*n] = {}
			self.pred[i*n][7] = [5,22,27]
			self.pred[i*n][14] = [12,20,29]
			self.pred[i*n][23] = [4,13,21]
			self.pred[i*n][30] = [6,11,28]
			self.pred[i*n].update(dict([j,3] for j in [4,5,6]))
			self.pred[i*n].update(dict([j,10] for j in [11,12,13]))
			self.pred[i*n].update(dict([j,19] for j in [20,21,22]))
			self.pred[i*n].update(dict([j,26] for j in [27,28,29]))
			self.pred[i*n].update({c: c-1 for c in list(set(self.C[i*n])-set(self.I1[i*n])-set(self.I2[i*n])-set(self.I3[i*n])-set(self.I4[i*n])-set(self.M[i*n]))})
			self.n_init[i*n] = np.zeros(len(self.C[i*n]))
			if random == False:
				self.Demand[i*n].append([0.3*Max_Q]*self.T)
			if random == True:
				self.Demand[i*n].append([None]*sample_size)
				for xi in range(sample_size):
					self.Demand[i*n][-1][xi] = np.random.poisson(0.3*Max_Q, self.T)


			for j in range(1,n-1,2):
				self.C[i*n+j] = list(range(30))
				self.O[i*n+j] = []
				self.Demand[i*n+j] = []
				self.D[i*n+j] = []
				self.BI[i*n+j] = [0,8,16,23]
				self.BO[i*n+j] = [7,15,22,29]
				self.I1[i*n+j] = [2,11]
				self.I2[i*n+j] = [3,4,12,13]
				self.I3[i*n+j] = [18,25]
				self.I4[i*n+j] = [19,20,26,27]
				self.V[i*n+j] = [1,10,17,24]
				self.M[i*n+j] = [5,14,21,28]
				# self.beta[i*n+j] = np.array([[0.15,0.15,0.15,0.15],[0.7,0.7,0.7,0.7],[0.15,0.15,0.15,0.15]])
				self.proc[i*n+j] = {}
				self.proc[i*n+j][2] = 21
				self.proc[i*n+j][3] = 5
				self.proc[i*n+j][4] = 28
				self.proc[i*n+j][11] = 28
				self.proc[i*n+j][12] = 14
				self.proc[i*n+j][13] = 21
				self.proc[i*n+j][18] = 14
				self.proc[i*n+j][19] = 21
				self.proc[i*n+j][20] = 5
				self.proc[i*n+j][25] = 5
				self.proc[i*n+j][26] = 28
				self.proc[i*n+j][27] = 14
				self.proc[i*n+j].update({c: c+1 for c in list(set(self.C[i*n+j])-set(self.I1[i*n+j])-set(self.I2[i*n+j])-set(self.I3[i*n+j])-set(self.I4[i*n+j]))})
				self.pred[i*n+j] = {}
				self.pred[i*n+j][5] = [3,20,25]
				self.pred[i*n+j][14] = [12,18,27]
				self.pred[i*n+j][21] = [2,13,19]
				self.pred[i*n+j][28] = [4,11,26]
				self.pred[i*n+j].update(dict([j,1] for j in [2,3,4]))
				self.pred[i*n+j].update(dict([j,10] for j in [11,12,13]))
				self.pred[i*n+j].update(dict([j,17] for j in [18,19,20]))
				self.pred[i*n+j].update(dict([j,24] for j in [25,26,27]))
				self.pred[i*n+j].update({c: c-1 for c in list(set(self.C[i*n+j])-set(self.I1[i*n+j])-set(self.I2[i*n+j])-set(self.I3[i*n+j])-set(self.I4[i*n+j])-set(self.M[i*n+j]))})
				self.n_init[i*n+j] = np.zeros(len(self.C[i*n]))
				""" if random == False:
					# self.Demand[i*n] = [0.3*Max_Q*4]*self.T
					self.beta[i*n+j] = np.array([[0.15,0.15,0.15,0.15],[0.7,0.7,0.7,0.7],[0.15,0.15,0.15,0.15]])
				if random == True:
					# self.Demand[i*n] = np.random.poisson(0.3*Max_Q*4, sample_size)
					for xi in sample_size:
						pick_prob = np.random.rand()
						if pick_prob < 1/3:
							self.beta[i*n+j][xi] = np.array([[0.1,0.1,0.1,0.1],[0.8,0.8,0.8,0.8],[0.1,0.1,0.1,0.1]])
						if pick_prob >= 1/3 and pick_prob < 2/3:
							self.beta[i*n+j][xi] = np.array([[0.1,0.1,0.1,0.1],[0.7,0.7,0.7,0.7],[0.2,0.2,0.2,0.2]])
						if pick_prob >= 2/3:
							self.beta[i*n+j][xi] = np.array([[0.2,0.2,0.2,0.2],[0.75,0.75,0.75,0.75],[0.05,0.05,0.05,0.05]]) """

				self.C[i*n+j+1] = list(range(30))
				self.O[i*n+j+1] = []
				self.Demand[i*n+j+1] = []
				self.D[i*n+j+1] = []
				self.BI[i*n+j+1] = [0,8,16,23]
				self.BO[i*n+j+1] = [7,15,22,29]
				self.I1[i*n+j+1] = [3,10]
				self.I2[i*n+j+1] = [4,5,11,12]
				self.I3[i*n+j+1] = [18,25]
				self.I4[i*n+j+1] = [19,20,26,27]
				self.V[i*n+j+1] = [2,9,17,24]
				self.M[i*n+j+1] = [6,13,21,28]
				# self.beta[i*n+j+1] = np.array([[0.15,0.15,0.15,0.15],[0.7,0.7,0.7,0.7],[0.15,0.15,0.15,0.15]])
				self.proc[i*n+j+1] = {}
				self.proc[i*n+j+1][3] = 21
				self.proc[i*n+j+1][4] = 6
				self.proc[i*n+j+1][5] = 28
				self.proc[i*n+j+1][10] = 28
				self.proc[i*n+j+1][11] = 13
				self.proc[i*n+j+1][12] = 21
				self.proc[i*n+j+1][18] = 13
				self.proc[i*n+j+1][19] = 21
				self.proc[i*n+j+1][20] = 6
				self.proc[i*n+j+1][25] = 6
				self.proc[i*n+j+1][26] = 28
				self.proc[i*n+j+1][27] = 13
				self.proc[i*n+j+1].update({c: c+1 for c in list(set(self.C[i*n+j+1])-set(self.I1[i*n+j+1])-set(self.I2[i*n+j+1])-set(self.I3[i*n+j+1])-set(self.I4[i*n+j+1]))})
				self.pred[i*n+j+1] = {}
				self.pred[i*n+j+1][6] = [4,20,25]
				self.pred[i*n+j+1][13] = [11,18,27]
				self.pred[i*n+j+1][21] = [3,12,19]
				self.pred[i*n+j+1][28] = [5,10,26]
				self.pred[i*n+j+1].update(dict([j,2] for j in [3,4,5]))
				self.pred[i*n+j+1].update(dict([j,9] for j in [10,11,12]))
				self.pred[i*n+j+1].update(dict([j,17] for j in [18,19,20]))
				self.pred[i*n+j+1].update(dict([j,24] for j in [25,26,27]))
				self.pred[i*n+j+1].update({c: c-1 for c in list(set(self.C[i*n+j+1])-set(self.I1[i*n+j+1])-set(self.I2[i*n+j+1])-set(self.I3[i*n+j+1])-set(self.I4[i*n+j+1])-set(self.M[i*n+j+1]))})
				self.n_init[i*n+j+1] = np.zeros(len(self.C[i*n]))
				""" if random == False:
					# self.Demand[i*n+j+1].append([0.3*Max_Q*4]*self.T)
					self.beta[i*n] = np.array([[0.15,0.15,0.15,0.15],[0.7,0.7,0.7,0.7],[0.15,0.15,0.15,0.15]])
				if random == True:
					# self.Demand[i*n] = np.random.poisson(0.3*Max_Q*4, sample_size)
					for xi in sample_size:
						pick_prob = np.random.rand()
						if pick_prob < 1/3:
							self.beta[i*n][xi] = np.array([[0.1,0.1,0.1,0.1],[0.8,0.8,0.8,0.8],[0.1,0.1,0.1,0.1]])
						if pick_prob >= 1/3 and pick_prob < 2/3:
							self.beta[i*n][xi] = np.array([[0.1,0.1,0.1,0.1],[0.7,0.7,0.7,0.7],[0.2,0.2,0.2,0.2]])
						if pick_prob >= 2/3:
							self.beta[i*n][xi] = np.array([[0.2,0.2,0.2,0.2],[0.75,0.75,0.75,0.75],[0.05,0.05,0.05,0.05]]) """

			self.C[(i+1)*n-1] = list(range(32))
			self.O[(i+1)*n-1] = [9]
			self.Demand[(i+1)*n-1] = []
			self.D[(i+1)*n-1] = [8]
			self.BI[(i+1)*n-1] = [0,18,25]
			self.BO[(i+1)*n-1] = [17,24,31]
			self.I1[(i+1)*n-1] = [2,13]
			self.I2[(i+1)*n-1] = [3,4,14,15]
			self.I3[(i+1)*n-1] = [20,27]
			self.I4[(i+1)*n-1] = [21,22,28,29]
			self.V[(i+1)*n-1] = [1,12,19,26]
			self.M[(i+1)*n-1] = [5,16,23,30]
			# self.beta[(i+1)*n-1] = np.array([[0.15,0.15,0.15,0.15],[0.7,0.7,0.7,0.7],[0.15,0.15,0.15,0.15]])
			self.proc[(i+1)*n-1] = {}
			self.proc[(i+1)*n-1][2] = 23
			self.proc[(i+1)*n-1][3] = 5
			self.proc[(i+1)*n-1][4] = 30
			self.proc[(i+1)*n-1][13] = 30
			self.proc[(i+1)*n-1][14] = 16
			self.proc[(i+1)*n-1][15] = 23
			self.proc[(i+1)*n-1][20] = 16
			self.proc[(i+1)*n-1][21] = 23
			self.proc[(i+1)*n-1][22] = 5
			self.proc[(i+1)*n-1][27] = 5
			self.proc[(i+1)*n-1][28] = 30
			self.proc[(i+1)*n-1][29] = 16
			self.proc[(i+1)*n-1].update({c: c+1 for c in list(set(self.C[(i+1)*n-1])-set(self.I1[(i+1)*n-1])-set(self.I2[(i+1)*n-1])-set(self.I3[(i+1)*n-1])-set(self.I4[(i+1)*n-1]))})
			self.pred[(i+1)*n-1] = {}
			self.pred[(i+1)*n-1][5] = [3,22,27]
			self.pred[(i+1)*n-1][16] = [14,20,29]
			self.pred[(i+1)*n-1][23] = [2,15,21]
			self.pred[(i+1)*n-1][30] = [4,13,28]
			self.pred[(i+1)*n-1].update(dict([j,1] for j in [2,3,4]))
			self.pred[(i+1)*n-1].update(dict([j,12] for j in [13,14,15]))
			self.pred[(i+1)*n-1].update(dict([j,19] for j in [20,21,22]))
			self.pred[(i+1)*n-1].update(dict([j,26] for j in [27,28,29]))
			self.pred[(i+1)*n-1].update({c: c-1 for c in list(set(self.C[(i+1)*n-1])-set(self.I1[(i+1)*n-1])-set(self.I2[(i+1)*n-1])-set(self.I3[(i+1)*n-1])-set(self.I4[(i+1)*n-1])-set(self.M[(i+1)*n-1]))})
			self.n_init[(i+1)*n-1] = np.zeros(len(self.C[i*n]))
			if random == False:
				self.Demand[(i+1)*n-1].append([0.2*Max_Q]*self.T)
			if random == True:
				self.Demand[(i+1)*n-1].append([None]*sample_size)
				for xi in range(sample_size):
					self.Demand[(i+1)*n-1][-1][xi] = np.random.poisson(0.2*Max_Q, self.T)
			
		for i in range(n):	
			self.O[i].append(self.BI[i][-1])
			self.D[i].append(self.BO[i][-2])
			self.BI[i].remove(self.BI[i][-1])
			self.BO[i].remove(self.BO[i][-2])
			if random == False:
				self.Demand[i].append([0.1*Max_Q]*self.T)
			if random == True:
				self.Demand[i].append([None]*sample_size)
				for xi in range(sample_size):
					self.Demand[i][-1][xi] = np.random.poisson(0.1*Max_Q, self.T)
			# self.Demand[i].append(0.1)
			self.O[(m-1)*n+i].append(self.BI[(m-1)*n+i][-2])
			self.D[(m-1)*n+i].append(self.BO[(m-1)*n+i][-1])
			self.BI[(m-1)*n+i].remove(self.BI[(m-1)*n+i][-2])
			self.BO[(m-1)*n+i].remove(self.BO[(m-1)*n+i][-1])
			if random == False:
				self.Demand[(m-1)*n+i].append([0.1*Max_Q]*self.T)
			if random == True:
				self.Demand[(m-1)*n+i].append([None]*sample_size)
				for xi in range(sample_size):
					self.Demand[(m-1)*n+i][-1][xi] = np.random.poisson(0.1*Max_Q, self.T)
			# self.Demand[(m-1)*n+i].append(0.1)

		for i in range(self.N):
			self.Jam_N[i] = np.ones(len(self.C[i]))*Max_Jam_N
			self.Q[i] = np.ones(len(self.C[i]))*Max_Q
			for c in self.I1[i] + self.I2[i] + self.I3[i] + self.I4[i]:
				self.Jam_N[i][c] = Max_Jam_N/2
				self.Q[i][c] = Max_Q/2

		# set turning ratio
		""" f random == True:
			ratio = np.zeros((3,4,sample_size))
			for i in range(sample_size):
				ratio[0,:,i] = np.repeat(np.random.uniform(0,0.2,1),4)
				ratio[1,:,i] = np.repeat(np.random.uniform(0.6,0.8,1),4)
				ratio[2,:,i] = 1 - ratio[0,:,i] - ratio[1,:,i]
			for i in range(self.N):
				self.beta[i] = ratio """
		
		if random == False:
			for i in range(self.N):
				self.beta[i] = np.zeros((3,4))
				self.beta[i] = np.array([[0.15,0.15,0.15,0.15],[0.7,0.7,0.7,0.7],[0.15,0.15,0.15,0.15]])
		if random == True:
			for i in range(self.N):
				self.beta[i] = [None]*sample_size
				for xi in range(sample_size):
					self.beta[i][xi] = [None]*self.T
					for t in range(self.T):
						pick_prob = np.random.rand()
						if pick_prob < 1/3:
							self.beta[i][xi][t] = np.array([[0.1,0.1,0.1,0.1],[0.8,0.8,0.8,0.8],[0.1,0.1,0.1,0.1]])
						if pick_prob >= 1/3 and pick_prob < 2/3:
							self.beta[i][xi][t] = np.array([[0.1,0.1,0.1,0.1],[0.7,0.7,0.7,0.7],[0.2,0.2,0.2,0.2]])
						if pick_prob >= 2/3:
							self.beta[i][xi][t] = np.array([[0.2,0.2,0.2,0.2],[0.75,0.75,0.75,0.75],[0.05,0.05,0.05,0.05]])


