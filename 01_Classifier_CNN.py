# Hand writing 0s and 1s classifier CNN

#### Setup environment
from pylab import *
import matplotlib.pyplot as plt
import numpy as np

# add caffe to python path
import sys
caffe_root = '../../../'
sys.path.insert(0, caffe_root + 'python')
import caffe

#### Creating the net 
from caffe import layers as L, params as P

def zero1_net(source, batch_size, num_classes, train = True, learn_all = False):

	n = caffe.NetSpec()

	n.data, n.label = L.ImageData(source = source, batch_size = batch_size, ntop = 2)

	#n.conv1 = L.Convolution(n.data, kernel_size = 3, num_output = 10, weight_filler = dict(type = 'xavier'))

	n.score = L.InnerProduct(n.data, num_output = 10, weight_filler = dict(type = 'xavier'))
	n.loss = L.SoftmaxWithLoss(n.score, n.label)

	return n.to_proto()



dataset_source = '01_images/im_reference.txt'
batch_size = 10
num_classes = 2

untrained_zero1_net = zero1_net(train = False, 
											source = dataset_source,
											batch_size = batch_size, 
											num_classes = num_classes)

with open('zero1_net_train.prototxt', 'w') as f:
	f.write(str(untrained_zero1_net))


caffe.set_mode_cpu()



print
print
print "=================== 01_Classifier_CNN ==================="