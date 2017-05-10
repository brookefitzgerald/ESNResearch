import numpy as np
import pandas as pd
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

def createTeacherSignal(d=data):
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

data_exp_decay_1018 = data_1018.copy()
addExponentialDecay(0.995,data_exp_decay_1018)

