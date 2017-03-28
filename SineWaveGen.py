import numpy as np
import random as rand
import scipy.linalg as linalg
from scipy.sparse.linalg import eigs, LinearOperator
from scipy.sparse import random, csr_matrix
import scipy.stats as stats
import matplotlib.pyplot as plt

seed = 250
nTeachRuns = 600
nTestRuns = 150

rand.seed(seed) # set the seed

# Tunable parameters
p = 1.0  # spectral radius
beta = 0.01 # regularization rate


## Initialize Reservoir

rvs = stats.uniform(-3,6).rvs # uniform distribution between [-1,1]

W = random(100,100,density = 0.1, random_state = seed, data_rvs = rvs).A # uniformly distributed
vals, vecs = np.absolute(eigs(W))
l = max(vals) # eigenvalues
W /= l # divide W by it's largest eigenvalue
W *= p # scale W by its spectral radius

W_in = random(100,1,density=0.1, random_state=seed, data_rvs = rvs).A

# Create teacher signal With 10 points per curve
teacher = np.sin(np.linspace(0., np.pi*nTeachRuns/10, nTeachRuns))

M = np.ndarray(shape = (nTeachRuns,100))

## Teacher Forcing Stage
X = np.zeros((100,nTeachRuns))
X[0]=1.0

for i in np.arange(2, nTeachRuns):
	W_dot_Xn = np.dot(W,X[:,i-1].reshape(100,1))
	Win_dot_Yn = np.dot(W_in,teacher[i-1])
	X[:,i] = np.tanh(W_dot_Xn+Win_dot_Yn).reshape(100)

#plt.plot(X[50,:])
#plt.show()

Y_dot_Xt = np.dot(teacher.reshape(1,nTeachRuns),X.T)
X_dot_Xt_w_regularization = np.dot(X,X.T)+beta*np.identity(100)

# Use Ridge Regression to calculate output Weights
W_out = np.dot(Y_dot_Xt, linalg.inv(X_dot_Xt_w_regularization))
training_results = np.dot(W_out,X).reshape(nTeachRuns)
training_error = np.sum((training_results[100:nTeachRuns]-teacher[100:nTeachRuns])**2)
print("Training Error: ",training_error)

plt.plot(training_results[100:nTeachRuns],teacher[100:nTeachRuns])
plt.xlabel('Training Results after 100 time steps')
plt.ylabel('Teacher Sine Wave after 100 time steps')
plt.show()

## Pattern Generation Phase
X_train = np.zeros((100,nTestRuns))
X_train[:,0] = X[:,nTeachRuns-1]
Y_train = np.zeros(nTestRuns)
Y_train[0] = np.dot(W_out,X_train[:,0].reshape(100,1))
print(Y_train[0])

for i in np.arange(1,nTestRuns):
	W_dot_Xn = np.dot(W,X[:,i-1].reshape(100,1))
	Win_dot_Yn = np.dot(W_in,Y_train[i-1])
	X_train[:,i] = np.tanh(W_dot_Xn+Win_dot_Yn).reshape(100)
	Y_train[i] = np.dot(W_out,X_train[:,i].reshape(100,1))

plt.plot(Y_train, label = 'Generated Pattern')
plt.plot(np.sin(np.linspace(0, np.pi*nTestRuns/10, nTestRuns)), label = 'Sine Wave')
plt.legend()
plt.show()