import numpy as np
import random as rand
from scipy import linalg
from scipy.sparse.linalg import eigs, LinearOperator
from scipy.sparse import random, csr_matrix
from scipy.optimize import brute
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

seed = 250
nTeachRuns = 1000
nTestRuns = 100
nToSkip = 100

rand.seed(seed) # set the seed
rvs = stats.uniform(-1,2).rvs # uniform distribution between [-1,1]

# Tunable parameters
p = 1.5 # spectral radius
alpha = 4.0 # regularization rate
scale_in = 1.0
scale_fb = 1.0
scale_b = 0.3

#dfm = pd.DataFrame(columns=['p','alpha', 'scale_in','scale_fb','scale_b','train_MSE'])
#dfm = pd.read_csv(results.csv)
## Initialize Reservoir
def ComputeRes(x, args):
	p,alpha, scale_in,scale_fb,scale_b = x
	seed, nTeachRuns, nTestRuns, nToSkip, rvs = args

	W = random(100,100,density = 0.1, random_state = seed, data_rvs = rvs).A # uniformly distributed
	vals, vecs = np.absolute(eigs(W))
	l = max(vals) # eigenvalues
	W /= l # divide W by it's largest eigenvalue
	W *= p # scale W by its spectral radius

	W_in = random(100,1,density=0.1, random_state=seed, data_rvs = rvs).A
	W_in *= scale_in

	W_fb = random(100,1,density=0.1, random_state=seed, data_rvs = rvs).A
	W_fb *= scale_fb

	b = np.array(rvs(100)).reshape(100,1)
	b *= scale_b

	# Create teacher signal With 10 points per curve
	sin_freq = np.linspace(15.0,10.0,nTeachRuns+nTestRuns)
	teacher = np.array([np.sin(2*np.pi*n/sin_freq[n]) for n in np.arange(0,nTeachRuns)])
	#plt.plot(teacher)
	#plt.show()

	input_data = np.linspace(1.0,0.0000000000000000000000000000000000000001,nTeachRuns+nTestRuns)

	## Teacher Forcing Stage
	X = np.zeros((100,nTeachRuns))
	X[:,0]=1.0

	for i in np.arange(2, nTeachRuns):
		W_dot_Xn = np.dot(W,X[:,i-1].reshape(100,1))
		Wfb_dot_Yn = np.dot(W_fb,teacher[i-1])
		Win_dot_Un = np.dot(W_in, input_data[i-1])
		X[:,i] = np.tanh(W_dot_Xn+Wfb_dot_Yn+Win_dot_Un+b).reshape(100)

	plt.plot(X[1,800:900])
	plt.plot(X[2,800:900])
	plt.plot(X[3,800:900])
	plt.plot(X[4,800:900])
	plt.title("Reservoir Dynamics")
	plt.show()
	#plt.savefig("plots/p{}alpha{}scale_in{}scale_fb{}scale_b{}Resevoir.png".format(p,alpha,scale_in,scale_fb,scale_b))
	#plt.close()

	X = X[:,nToSkip:nTeachRuns]
	Y_dot_Xt = np.dot(teacher[nToSkip:nTeachRuns].reshape(1,nTeachRuns-nToSkip),X.T)
	X_dot_Xt_w_regularization = np.dot(X,X.T)+alpha*np.identity(100)

	# Use Ridge Regression to calculate output Weights
	W_out = np.dot(Y_dot_Xt, linalg.inv(X_dot_Xt_w_regularization))
	training_results = np.dot(W_out,X).reshape(nTeachRuns-nToSkip)

	training_MSE = np.sqrt(np.sum((training_results-teacher[nToSkip:nTeachRuns])**2))/(nTeachRuns-nToSkip)

	plt.plot(training_results,teacher[nToSkip:nTeachRuns])
	plt.xlabel('Training Results after 100 time steps')
	plt.ylabel('Teacher Sine Wave after 100 time steps')
	plt.title('Training MSE: '+str(training_MSE))
	plt.show()
	#plt.savefig("newplots/p{}alpha{}scale_in{}scale_fb{}scale_b{}TrainingResults.png".format(p,alpha,scale_in,scale_fb,scale_b))
	#plt.close()

	## Pattern Generation Phase
	X_train = np.zeros((100,nTestRuns))
	X_train[:,0] = X[:,nTeachRuns-nToSkip-1]
	Y_train = np.zeros(nTestRuns)
	Y_train[0] = np.dot(W_out,X_train[:,0].reshape(100,1))

	for i in np.arange(1,nTestRuns):
		W_dot_Xn = np.dot(W,X_train[:,i-1].reshape(100,1))
		Wfb_dot_Yn = np.dot(W_fb,Y_train[i-1])
		Win_dot_Un = np.dot(W_in, input_data[nTeachRuns+i-1])
		X_train[:,i] = np.tanh(W_dot_Xn+Wfb_dot_Yn+Win_dot_Un+b).reshape(100)
		Y_train[i] = np.dot(W_out,X_train[:,i].reshape(100,1))

	#expected_output = np.sin(np.linspace(0, np.pi*nTestRuns/10, nTestRuns))
	expected_output = np.array([np.sin(2*np.pi*n/sin_freq[n]) for n in np.arange(nTeachRuns,nTeachRuns+nTestRuns)])
	testing_MSE = np.sqrt(np.sum((expected_output-Y_train)**2))/nTestRuns

	plt.plot(Y_train, label = 'Generated Pattern')
	plt.plot(expected_output, label = 'Sine Wave')
	plt.title("Testing MSE: "+str(testing_MSE))
	plt.legend()
	plt.show()
	#plt.savefig("newplots/p{}alpha{}scale_in{}scale_fb{}scale_b{}GeneratedPattern.png".format(p,alpha,scale_in,scale_fb,scale_b))
	#plt.close()
	#dfm = dfm.append(pd.Series(data = {'p':p,'alpha':alpha, 'scale_in':scale_in,'scale_fb':scale_fb,
	#'scale_b':scale_b,'train_MSE':training_MSE,'test_MSE':testing_MSE}),ignore_index=True)
	return(training_MSE)

#results = brute(ComputeRes,((1,10),(0.01,3),(1,10),(1,10),(0,1)))
#print(results.x0)
train = 0
#test = 0
#for p in np.linspace(0.5,2.0,20):
#for i in np.arange(0,20):
#	training_MSE= ComputeRes([p,alpha, scale_in,scale_fb,scale_b], [seed, nTeachRuns, nTestRuns, nToSkip, rvs])
#	train+=training_MSE
ComputeRes([p,alpha, scale_in,scale_fb,scale_b], [seed, nTeachRuns, nTestRuns, nToSkip, rvs])
#dfm.to_csv("results.csv")

#print(dfm)