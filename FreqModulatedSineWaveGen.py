import numpy as np
import random as rand
from scipy import linalg
from scipy.sparse.linalg import eigs, LinearOperator
from scipy.sparse import random
from scipy.optimize import brute
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

seed = 275 # 275, 276, 300
nTeachRuns = 1000
nTestRuns = 200
nToSkip = 200

rand.seed(seed) # set the seed
np.random.seed(seed)
rvs = stats.uniform(-1,2).rvs # uniform distribution between [-1,1]

# Tunable parameters
res_size = 100
scale_res = 1.0
alpha = 3.0 # regularization rate
scale_in = 1.2
scale_fb = 2.0
scale_b = 0.5
scale_noise = 0.01

#dfm = pd.DataFrame(columns=['p','alpha', 'scale_in','scale_fb','scale_b','train_MSE'])
#dfm = pd.read_csv(results.csv)
## Initialize Reservoir
def ComputeRes(x, args):
	f, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2,3)

	p,alpha, scale_in,scale_fb,scale_b = x
	seed, nTeachRuns, nTestRuns, nToSkip, rvs = args

	W = random(res_size,res_size,density = 0.1, data_rvs = rvs).A # uniformly distributed
	vals, vecs = np.absolute(eigs(W))
	l = max(vals) # eigenvalues
	W /= l # divide W by it's largest eigenvalue
	W *= scale_res # scale W

	W_in = random(res_size,1,density=0.1, data_rvs = rvs).A
	W_in *= scale_in

	W_fb = random(res_size,1,density=0.1, data_rvs = rvs).A
	W_fb *= scale_fb

	b = np.array(rvs(res_size)).reshape(res_size,1)
	b *= scale_b

	# Create teacher signal With 10 points per curve
	sin_freq = np.linspace(15.0,10.0,nTeachRuns)
	teacher = np.array([np.sin(2*np.pi*n/sin_freq[n]) for n in np.arange(0,nTeachRuns)])
	input_data = np.linspace(1.0,0.0000000000000000000000000000000000000001,nTeachRuns)

	## Teacher Forcing Stage
	X = np.zeros((res_size,nTeachRuns))
	X[:,0]=1.0

	for i in np.arange(1, nTeachRuns):
		W_dot_Xn = np.dot(W,X[:,i-1].reshape(res_size,1))
		Wfb_dot_Yn = np.dot(W_fb,teacher[i-1])
		Win_dot_Un = np.dot(W_in, input_data[i])
		noise = np.random.normal(0,scale_noise,size=res_size)
		X[:,i] = np.tanh(W_dot_Xn+Wfb_dot_Yn+Win_dot_Un+b+noise.reshape(res_size,1)).reshape(res_size)

	ax4.plot(np.arange(600,699),X[1,600:699])
	ax4.plot(np.arange(600,699),X[2,600:699])
	ax4.plot(np.arange(600,699),X[3,600:699])
	ax4.plot(np.arange(600,699),X[4,600:699])
	ax4.set_title("Subset Reservoir Dynamics 1-4")

	ax1.plot(X[1,:])
	ax1.plot(X[2,:])
	ax1.plot(X[3,:])
	ax1.plot(X[4,:])
	ax1.set_title("Full Reservoir Dynamics 1-4")


	X = X[:,nToSkip:nTeachRuns]
	Y_dot_Xt = np.dot(teacher[nToSkip:nTeachRuns].reshape(1,nTeachRuns-nToSkip),X.T)
	X_dot_Xt_w_regularization = np.dot(X,X.T)+alpha*np.identity(res_size)

	# Use Ridge Regression to calculate output Weights
	W_out = np.dot(Y_dot_Xt, linalg.inv(X_dot_Xt_w_regularization))

	training_results = np.dot(W_out,X).reshape(nTeachRuns-nToSkip)
	training_MSE = np.sum((training_results-teacher[nToSkip:nTeachRuns])**2)/(nTeachRuns-nToSkip)

	ax2.plot(np.arange(nToSkip,nTeachRuns),training_results,label ='Training Results')
	ax2.plot(np.arange(nToSkip,nTeachRuns),teacher[nToSkip:nTeachRuns],label = 'Teacher Sine Wave')
	ax2.legend()
	ax2.set_title('Full Training MSE: '+str(training_MSE))

	ax5.plot(np.arange(nToSkip+300,nToSkip+450),training_results[300:450],label ='Training Results')
	ax5.plot(np.arange(nToSkip+300,nToSkip+450),teacher[nToSkip+300:nToSkip+450], '--',label = 'Teacher Sine Wave')
	ax5.legend()
	ax5.set_title('Subset Training MSE: '+str(training_MSE))
	#plt.savefig("newplots/p{}alpha{}scale_in{}scale_fb{}scale_b{}TrainingResults.png".format(p,alpha,scale_in,scale_fb,scale_b))
	#plt.close()

	## Pattern Generation Phase

	sin_freq = np.linspace(10.0,15.0,nTestRuns)
	teacher = np.array([np.sin(2*np.pi*n/sin_freq[n]) for n in np.arange(0,nTestRuns)])
	input_data = np.linspace(1.0,0.0000000000000000000000000000000000000001,nTestRuns)

	X_train = np.zeros((res_size,nTestRuns))
	#X_train[:,0] = X[:,nTeachRuns-nToSkip-1]
	X_train[:,0]=1.0
	Y_train = np.zeros(nTestRuns)
	#Y_train[1] = np.dot(W_out,X_train[:,0].reshape(res_size,1))
	#Y_train[0] = teacher[nTeachRuns]


	for i in np.arange(1,nTestRuns):
		W_dot_Xn = np.dot(W,X_train[:,i-1].reshape(res_size,1))
		Win_dot_Un = np.dot(W_in, input_data[i])
		if i < nTestRuns/4 :
			Wfb_dot_Yn = np.dot(W_fb,teacher[i-1])
			X_train[:,i] = np.tanh(W_dot_Xn+Wfb_dot_Yn+Win_dot_Un+b).reshape(res_size)
			Y_train[i] = teacher[i]
		else:
			Wfb_dot_Yn = np.dot(W_fb,Y_train[i-1])
			X_train[:,i] = np.tanh(W_dot_Xn+Wfb_dot_Yn+Win_dot_Un+b).reshape(res_size)
			Y_train[i] = np.dot(W_out,X_train[:,i].reshape(res_size,1))

	expected_output = np.array([np.sin(2*np.pi*n/sin_freq[n]) for n in np.arange(0,nTestRuns)])
	testing_MSE = np.sum((expected_output-Y_train)**2)/nTestRuns

	ax3.plot(Y_train, label = 'Generated Pattern')
	ax3.plot(expected_output, label = 'Sine Wave')
	ax3.set_title("Full Testing MSE: "+str(testing_MSE))
	ax3.axvline(nTestRuns/4,ymin = 0.0,ymax = 1.0,color='k')
	ax3.legend()

	ax6.plot(np.arange(25,100),Y_train[25:100], label = 'Generated Pattern')
	ax6.plot(np.arange(25,100),expected_output[25:100], '--', label = 'Sine Wave')
	ax6.set_title("Subset Testing MSE: "+str(testing_MSE))
	ax6.axvline(nTestRuns/4,ymin = 0.0,ymax = 1.0,color='k')
	ax6.legend()

	plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.4)
	
	mng = plt.get_current_fig_manager()
	mng.window.showMaximized()
	plt.show()
	return(training_MSE)

#results = brute(ComputeRes,((1,10),(0.01,3),(1,10),(1,10),(0,1)))
#print(results.x0)
train = 0
#test = 0
#for p in np.linspace(0.5,2.0,20):
#for i in np.arange(0,20):
#	training_MSE= ComputeRes([p,alpha, scale_in,scale_fb,scale_b], [seed, nTeachRuns, nTestRuns, nToSkip, rvs])
#	train+=training_MSE
ComputeRes([scale_res,alpha, scale_in,scale_fb,scale_b], [seed, nTeachRuns, nTestRuns, nToSkip, rvs])
#dfm.to_csv("results.csv")

#print(dfm)