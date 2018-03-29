#!/usr/bin/python3.5

################
# Loading data #
################

import numpy as np
import scipy.io
import math
import sklearn
from sklearn.neural_network import MLPClassifier

param = scipy.io.whosmat('dont_commit_this/3d_sensor_epochs.mat')
#print(param) #prints info about mat file we are reading in

mat_contents = scipy.io.loadmat('dont_commit_this/3d_sensor_epochs.mat')

array_std = np.array(mat_contents['epoch_std'])
array_trg = np.array(mat_contents['epoch_trg'])

#print(array_std.shape)
#print(array_trg.shape)

###################################
# defining training + testing set #
###################################

htrials = 147
sensor = 5

trainsize = math.floor(htrials*2*(9/10))

std_dat = array_std[:,sensor,:]
trg_dat = array_trg[:,sensor,:]
sens_dat = np.transpose(np.concatenate((std_dat,trg_dat),axis=1))
#print(sens_dat.shape)

std_class = np.full((153,1),0)
trg_class = np.full((147,1),1)
sens_class = np.concatenate((std_class,trg_class),axis=0)
#print(sens_class.shape)

#############
# shufflin' #
#############

loops=20
thenumber=0

for a in range(0,loops):

	key = np.random.permutation(sens_dat[:,1].size)
	#print(key)

	sens_dat = sens_dat[key,:]
	sens_class = sens_class[key,:]
	#print(sens_dat.shape)
	#print(sens_class.shape)

	##################
	# defining model #
	##################

	clf = MLPClassifier(solver='lbfgs', alpha=1e-02,
		hidden_layer_sizes=(500, 300), random_state=1)
	clf.fit(sens_dat[:trainsize,:], sens_class[:trainsize])
	MLPClassifier(activation='relu', alpha=1e-02, batch_size='auto',
	       beta_1=0.9, beta_2=0.999, early_stopping=False,
	       epsilon=1e-08, hidden_layer_sizes=(500, 300), learning_rate='constant',
	       learning_rate_init=0.001, max_iter=200, momentum=0.9,
	       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=False,
	       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=True,
	       warm_start=False)

	######################
	# making predictions #
	######################

	true_class = sens_class[trainsize:]
	predict_class = clf.predict(sens_dat[trainsize:])
	accuracy = sklearn.metrics.accuracy_score(true_class,predict_class)
	thenumber += accuracy

	####################
	# persisting model #
	####################

	#from sklearn.externals import joblib
	#joblib.dump(clf, 'big_data_model.pkl')
	#clf = joblib.load('big_data_model.pkl')
print(thenumber/loops)