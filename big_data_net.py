#!/usr/bin/python3.5

################
# Loading data #
################

import numpy
import scipy.io
import math
import sklearn
from sklearn import svm

param = scipy.io.whosmat('dont_commit_this/2d_sensor_epochs.mat')
#print(param) #prints info about mat file we are reading in

mat_contents = scipy.io.loadmat('dont_commit_this/2d_sensor_epochs.mat')

array_std = numpy.array(mat_contents['epoch2d_std'])
array_trg = numpy.array(mat_contents['epoch2d_trg'])

std_dat = array_std[:,:]
trg_dat = array_trg[:,:]

sens_dat = numpy.concatenate((std_dat,trg_dat),axis=0)

htrials = 20
trainsize = math.floor(htrials*129*(9/10))

std_class = numpy.full((htrials*129,1),0)
trg_class = numpy.full((htrials*129,1),1)
sens_class = numpy.concatenate((std_class,trg_class),axis=0)

#############
# shufflin' #
#############

rand_key = numpy.random.permutation(htrials*2*129)
numpy.subtract(rand_key,1)
#print(rand_key)
sens_dat = sens_dat[rand_key,:]
sens_class = sens_class[rand_key,:]
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
