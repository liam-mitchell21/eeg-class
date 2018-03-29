#!/usr/bin/python3.5

################
# Loading data #
################

import numpy as np
import scipy.io
import math
import sklearn
from sklearn import svm

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
sensor = 10

trainsize = math.floor(htrials*2*(7.5/10))

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

key = np.random.permutation(sens_dat[:,1].size)
#print(key)

sens_dat = sens_dat[key,:]
sens_class = sens_class[key,:]
#print(sens_dat.shape)
#print(sens_class.shape)

##################
# defining model #
##################

clf = svm.SVC(gamma=0.001, C=100.0)
clf.fit(sens_dat[:trainsize,:], sens_class[:trainsize])
svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
 decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
 max_iter=-1, probability=False, random_state=None, shrinking=True,
 tol=0.001, verbose=False)

######################
# making predictions #
######################

true_class = sens_class[trainsize:]
predict_class = clf.predict(sens_dat[trainsize:])
accuracy = sklearn.metrics.accuracy_score(true_class,predict_class)
print(accuracy)

####################
# persisting model #
####################

#from sklearn.externals import joblib
#joblib.dump(clf, 'big_data_model.pkl')
#clf = joblib.load('big_data_model.pkl')
