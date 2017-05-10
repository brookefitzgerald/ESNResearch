import numpy as np
import pandas as pd
from os import listdir,getcwd

zhang_dir = getcwd()+'\\ZhangData'
data = np.zeros((132,4), dtype=[('neuron_id','a13'),
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

def getSpecificStimulus(sid, neuron_index = None, d=data):
	if neuron_index==None:
		to_return = [x['raster_data'][x[0]==sid] for x in d[['stimulus_id','raster_data']][:][:,0]]
	else:
		to_return = [x['raster_data'][x[0]==sid] for x in d[['stimulus_id','raster_data']][neuron_index][:,0]]
	return(to_return)

def addExponentialDecayToNeuronRead(rate_of_exp_decrease, neuron_read):
    activation = 0.0
    to_return = np.zeros(len(neuron_read))
    for i,x in enumerate(neuron_read):
        if x==1.0:
            activation=1.0
        elif ((x==0.0) and (activation>0.0)):
            print(rate_of_exp_decrease*np.exp(activation))
            activation -= rate_of_exp_decrease*np.exp(activation)
        if activation<0.:
            activation=0.
        to_return[i]=activation

    return(to_return)


def addExponentialDecay(speed,d=data):
	raster_data =  d['raster_data'][:]
	data = np.apply(addExponentialDecayToNeuronRead, axis = 0)

