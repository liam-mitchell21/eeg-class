

import numpy
import scipy.io

param = scipy.io.whosmat('data_allSensors.mat')
print(param) #prints info about mat file we are reading in

mat_contents = scipy.io.loadmat('data_allSensors.mat')
matrices = numpy.array(mat_contents.values())
	#converts from a dict_values object into a numpy matrix
print(type(matrices))
	#checks that we successfully converted obj types

print(dir(mat_contents))