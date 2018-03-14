
import numpy
import scipy.io

param = scipy.io.whosmat('dont_commit_this/3d_sensor_epochs.mat')
print(param) #prints info about mat file we are reading in

mat_contents = scipy.io.loadmat('dont_commit_this/3d_sensor_epochs.mat')

array_std = numpy.array(mat_contents['epoch_std'])
array_trg = numpy.array(mat_contents['epoch_trg'])

