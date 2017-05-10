import numpy as np
import pandas as pd
import random as rand
from scipy import linalg
from scipy.sparse.linalg import eigs, LinearOperator
from scipy.sparse import random
from scipy.optimize import brute
import scipy.stats as stats
from os import listdir,getcwd
from functools import partial

zhang_dir = getcwd()+'\\ZhangData'
data = np.zeros(132, dtype=[('neuron_id','a13'),
								('raster_data',np.float32,(419,1000)),
								('raster_info','a20',4),
								('stimulus_id','a15',419),
								('stimulus_position','a15',419),
								('combined_id_position','a15',419)])

for i,data_dir in enumerate(listdir(zhang_dir)):
	raster_data, raster_info, raster_labels = [zhang_dir+'\\'+data_dir+'\\'+my_dir for my_dir in listdir(zhang_dir+'\\'+data_dir)]
	neuron_id = data_dir[0:(len(data_dir)-12)]
	data_arr = pd.read_csv(raster_data,header=None).values
	labels_arr = pd.read_csv(raster_labels,header=0).values
	if data_arr.shape == (420,1000):
		data_arr = data_arr[0:(len(data_arr)-1)]
		labels_arr = labels_arr[0:(len(labels_arr)-1)]
	data[i] = (neuron_id, data_arr, pd.read_csv(raster_info).values, labels_arr[:,0],labels_arr[:,1],labels_arr[:,2])

# Getting only the neurons measured together during run number 1018
index_1018=data['raster_info'][:,3]=='1018'
data_1018 = np.zeros(11, dtype=[('neuron_id','a13'),
								('raster_data',np.float32,(419,1000)),
								('raster_info','a20',4),
								('stimulus_id','a15',419),
								('stimulus_position','a15',419),
								('combined_id_position','a15',419)])
data_1018[np.arange(11)] = data[index_1018]

def getSpecificStimulus(sid, neuron_index = None, d=data):
	if neuron_index==None:
		to_return = [x['raster_data'][x[0]==sid] for x in d[['stimulus_id','raster_data']][:][:]]
	else:
		to_return = [x['raster_data'][x[0]==sid] for x in d[['stimulus_id','raster_data']][neuron_index][:]]
	return(to_return)

def addExponentialDecayToNeuronRead(neuron_read, rate_of_exp_decrease):
    activation = 0.0
    for i,x in enumerate(neuron_read):
        if x==1.0:
            activation+=1.0
        elif ((x==0.0) and (activation>0.0)):
            activation *= rate_of_exp_decrease
        if activation<0.:
            activation=0.
        neuron_read[i]=activation


def addExponentialDecay(speed,d=data):
    addExpDecay = partial(addExponentialDecayToNeuronRead, rate_of_exp_decrease = speed)
    for neuron in d:
        np.apply_along_axis(addExpDecay, axis = 1, arr = neuron['raster_data'])

def createBriefTeacherSignal(d=data):
	ind_dict = dict(zip(['car', 'couch', 'face', 'flower', 'guitar', 'hand', 'kiwi'],np.arange(7)))
	teacher_signal = np.zeros(len(d), dtype = (np.float32,(7,419,1000)))
	increase = [1.0/(1.0+np.exp(-0.7*x)) for x in np.linspace(-10,10,100)]
	decrease = [1.0/(1.0+np.exp(0.7*x)) for x in np.linspace(-10,10,100)]
	for i, neuron in enumerate(d):
		for j in np.arange(419):
			stimulus_index = ind_dict[neuron['stimulus_id'][j]]
			stimulus_shown = int(neuron['raster_info'][0])
			teacher_signal[stimulus_index][i][stimulus_shown:stimulus_shown+100] = increase
			teacher_signal[stimulus_index][i][stimulus_shown+100:stimulus_shown+200] = decrease

	return (teacher_signal)

def createExtendedTeacherSignal(d=data):
	ind_dict = dict(zip(['car', 'couch', 'face', 'flower', 'guitar', 'hand', 'kiwi'],np.arange(7)))
	teacher_signal = np.zeros(len(d), dtype = (np.float32,(7,419,1000)))
	increase = [1.0/(1.0+np.exp(-0.7*x)) for x in np.linspace(-10,10,100)]
	for i, neuron in enumerate(d):
		for j in np.arange(419):
			stimulus_index = ind_dict[neuron['stimulus_id'][j]]
			stimulus_shown = int(neuron['raster_info'][0])
			teacher_signal[stimulus_index][i][stimulus_shown:stimulus_shown+100] = increase
			teacher_signal[stimulus_index][i][stimulus_shown+100:1000] = 1.0

	return (teacher_signal)

data_exp_decay_1018 = data_1018.copy()
addExponentialDecay(0.995,data_exp_decay_1018)

seed = 275 # 275, 276, 300
nMax = 1000
nTeachRuns = 300
nTestRuns = 1000
nToSkip = 200
inputSize = 12

rand.seed(seed) # set the seed
np.random.seed(seed)
rvs = stats.uniform(-1,2).rvs # uniform distribution between [-1,1]

# Tunable parameters
res_size = 100
scale_res = 1.0
alpha = 3.0 # regularization rate
scale_in = 1.2
scale_b = 0.5
scale_noise = 0.01

W = random(res_size,res_size,density = 0.1, data_rvs = rvs).A # uniformly distributed
vals, vecs = np.absolute(eigs(W))
l = max(vals) # eigenvalues
W /= l # divide W by it's largest eigenvalue
W *= scale_res # scale W

W_in = random(res_size,inputSize,density=0.1, data_rvs = rvs).A
W_in *= scale_in

b = np.array(rvs(res_size,inputSize))
b *= scale_b

# Create output signal 
teacher = createExtendedTeacherSignal(data_1018)
teacher = teacher[0] # Because each of these 11 neurons saw the same stimuli, only need one

## State Harvesting
X = np.zeros((res_size,nMax*nTeachRuns))
X[:,0]=0.0
data_exp_decay_1018['raster_data'] = np.concatenate((np.ones((1,419,1000)),data_exp_decay_1018['raster_data']),axis=0)

for run_ind in np.arange(nTeachRuns):
	for i in np.arange(1, nMax):
		W_dot_Xn = np.dot(W,X[:,run_ind+i-1].reshape(res_size,1))
		Win_dot_Un = np.dot(W_in, data_exp_decay_1018['raster_data'][:,run_ind,i].reshape(inputSize,1))
		noise = np.random.normal(0,scale_noise,size=res_size)
		X[:,run_ind+i] = np.tanh(W_dot_Xn+Wfb_dot_Yn+Win_dot_Un+b+noise.reshape(res_size,1)).reshape(res_size)

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

X = X[:,nToSkip:(nMax*nTeachRuns)]
Y_dot_Xt = np.dot(teacher[:,:,nToSkip:nTeachRuns].reshape(7,nMax*nTeachRuns-nToSkip),X.T)
X_dot_Xt_w_regularization = np.dot(X,X.T)+alpha*np.identity(res_size)

# Use Ridge Regression to calculate output Weights
W_out = np.dot(Y_dot_Xt, linalg.inv(X_dot_Xt_w_regularization))
training_results = np.dot(W_out,X)
training_MSE = np.sum((training_results-teacher[:,:,nToSkip:nTeachRuns].reshape(7,nMax*nTeachRuns-nToSkip))**2)/(nMax*nTeachRuns-nToSkip)

ax2.plot(np.arange(nToSkip,nMax),training_results,label ='Training Results')
ax2.plot(np.arange(nToSkip,nMax),teacher[nToSkip:nMax],label = 'Teacher Signal')
ax2.legend()
ax2.set_title('Full Training MSE: '+str(training_MSE))


ax5.plot(np.arange(nToSkip+300,nToSkip+450),training_results[300:450],label ='Training Results')
ax5.plot(np.arange(nToSkip+300,nToSkip+450),teacher[nToSkip+300:nToSkip+450], '--',label = 'Teacher Signal')
ax5.legend()
ax5.set_title('Subset Training MSE: '+str(training_MSE))


## Testing Phase
X_train = np.zeros((res_size,nTestRuns))
#X_train[:,0] = X[:,nMax-nToSkip-1]
X_train[:,0]=1.0
Y_train = np.zeros(nTestRuns)
#Y_train[1] = np.dot(W_out,X_train[:,0].reshape(res_size,1))
#Y_train[0] = teacher[nMax]
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