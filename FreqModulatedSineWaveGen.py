import numpy as np
import random as rand
from scipy import linalg
from scipy.sparse.linalg import eigs, LinearOperator
from scipy.sparse import random, csr_matrix
import scipy.stats as stats
import matplotlib.pyplot as plt

seed = 250
nTeachRuns = 500
nTestRuns = 100

rand.seed(seed) # set the seed

# Tunable parameters
p = 1.0  # spectral radius
beta = 4.0 # regularization rate


## Initialize Reservoir

rvs = stats.uniform(-1,2).rvs # uniform distribution between [-1,1]

W = random(100,100,density = 0.1, random_state = seed, data_rvs = rvs).A # uniformly distributed
vals, vecs = np.absolute(eigs(W))
l = max(vals) # eigenvalues
W /= l # divide W by it's largest eigenvalue
W *= p # scale W by its spectral radius

W_in = random(100,1,density=0.1, random_state=seed, data_rvs = rvs).A
W_in *= 1.2

b = np.array(rvs(100)).reshape(100,1)
b*=0.1

# Create teacher signal With 10 points per curve
teacher = np.array([np.sin(2*np.pi*n/10) for n in np.arange(0,nTeachRuns)])


M = np.ndarray(shape = (nTeachRuns,100))

## Teacher Forcing Stage
X = np.zeros((100,nTeachRuns))
X[0]=1.0

for i in np.arange(2, nTeachRuns):
	W_dot_Xn = np.dot(W,X[:,i-1].reshape(100,1))
	Win_dot_Yn = np.dot(W_in,teacher[i-1])
	X[:,i] = np.tanh(W_dot_Xn+Win_dot_Yn+b).reshape(100)

plt.plot(X[1,:])
plt.plot(X[2,:])
plt.plot(X[3,:])
plt.plot(X[4,:])
plt.title("Reservoir Dynamics")
plt.show()

Y_dot_Xt = np.dot(teacher.reshape(1,nTeachRuns),X.T)
X_dot_Xt_w_regularization = np.dot(X,X.T)+beta*np.identity(100)

# Use Ridge Regression to calculate output Weights
W_out = np.dot(Y_dot_Xt, linalg.inv(X_dot_Xt_w_regularization))
training_results = np.dot(W_out,X).reshape(nTeachRuns)
training_MSE = np.sqrt(np.sum((training_results[100:nTeachRuns]-teacher[100:nTeachRuns])**2))/(nTeachRuns-100)

plt.plot(training_results[100:nTeachRuns],teacher[100:nTeachRuns])
plt.xlabel('Training Results after 100 time steps')
plt.ylabel('Teacher Sine Wave after 100 time steps')
plt.title('Training MSE: '+str(training_MSE))
plt.show()

## Pattern Generation Phase
X_train = np.zeros((100,nTestRuns))
X_train[:,0] = X[:,nTeachRuns-1]
Y_train = np.zeros(nTestRuns)
Y_train[0] = np.dot(W_out,X_train[:,0].reshape(100,1))

for i in np.arange(1,nTestRuns):
	W_dot_Xn = np.dot(W,X[:,i-1].reshape(100,1))
	Win_dot_Yn = np.dot(W_in,Y_train[i-1])
	X_train[:,i] = np.tanh(W_dot_Xn+Win_dot_Yn+b).reshape(100)
	Y_train[i] = np.dot(W_out,X_train[:,i].reshape(100,1))

#expected_output = np.sin(np.linspace(0, np.pi*nTestRuns/10, nTestRuns))
expected_output = np.array([np.sin(2*np.pi*n/10) for n in np.arange(0,nTestRuns)])
testing_MSE = np.sqrt(np.sum((expected_output-Y_train)**2))/nTestRuns

plt.plot(Y_train, label = 'Generated Pattern')
plt.plot(expected_output, label = 'Sine Wave')
plt.title("Testing MSE: "+str(testing_MSE))
plt.legend()
plt.show()