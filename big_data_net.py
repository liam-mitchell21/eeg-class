#!/usr/bin/python3.5

################
# Loading data #
################

import numpy
import scipy.io

param = scipy.io.whosmat('dont_commit_this/2d_sensor_epochs.mat')
#print(param) #prints info about mat file we are reading in

mat_contents = scipy.io.loadmat('dont_commit_this/2d_sensor_epochs.mat')

array_std = numpy.array(mat_contents['epoch2d_std'])
array_trg = numpy.array(mat_contents['epoch2d_trg'])

std_dat = array_std[:,:]
trg_dat = array_trg[:,:]

sens_dat = numpy.concatenate((std_dat,trg_dat),axis=0)

std_class = numpy.full((153*129,1),0)
trg_class = numpy.full((147*129,1),1)
sens_class = numpy.concatenate((std_class,trg_class),axis=0)

#############
# shufflin' #
#############

rand_key = numpy.random.permutation(300*129)
numpy.subtract(rand_key,1)
#print(rand_key)
sens_dat = sens_dat[rand_key,:]
sens_class = sens_class[rand_key,:]
print(sens_dat.shape)
print(sens_class.shape)

##################
# defining model #
##################

from sklearn import svm

clf = svm.SVC(gamma=0.001, C=100.0)
clf.fit(sens_dat[:-300,:], sens_class[:-300])
svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
 decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
 max_iter=-1, probability=False, random_state=None, shrinking=True,
 tol=0.001, verbose=False)

######################
# making predictions #
######################

print(numpy.transpose(sens_class[-300:,:]))
print(clf.predict(sens_dat[-300:,:]))

####################
# persisting model #
####################

#from sklearn.externals import joblib
#joblib.dump(clf, 'big_data_model.pkl')
#clf = joblib.load('big_data_model.pkl')
