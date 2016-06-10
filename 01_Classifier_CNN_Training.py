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

def zero1_net(source, batch_size):

	n = caffe.NetSpec()

	n.data, n.label = L.ImageData(source = source, batch_size = batch_size, ntop = 2)

	n.conv1 = L.Convolution(n.data, kernel_size = 3, num_output = 10, weight_filler = dict(type = 'xavier'))
	n.pool1 = L.Pooling(n.conv1, kernel_size = 2, stride = 2, pool = P.Pooling.MAX)
	n.fc1 = L.InnerProduct(n.pool1, num_output = 10, weight_filler = dict(type='xavier'))
	# n.relu1 = L.ReLU(n.fc1, in_place = True)
	# n.prob = L.Softmax(n.fc1)
	n.fc2 = L.InnerProduct(n.fc1, num_output = 2, weight_filler = dict(type = 'xavier'))
	n.prob = L.SoftmaxWithLoss(n.fc2, n.label)

	return n.to_proto()

def create_net(source, batch_size):
	untrained_zero1_net = zero1_net(source = dataset_source, batch_size = batch_size)

	with open('zero1_net_train.prototxt', 'w') as f:
		f.write(str(untrained_zero1_net))


# setup the CNN
dataset_source = '01_images/im_reference.txt'
batch_size = 10
create_net(dataset_source, batch_size)


# setup the solver
caffe.set_mode_cpu()

solver = None
solver = caffe.SGDSolver('zero1_net_solver.prototxt')

# get a view of the model
# check feature dimensions - (batch size, feature dim, spatial dim)
print [(k, v.data.shape) for k, v in solver.net.blobs.items()]
# just the weight sizes
print [(k, v[0].data.shape) for k, v, in solver.net.params.items()]
print

############ for testing
# tile the first 8 items
if False:
	plt.figure(1)
	solver.net.forward() # put a batch of data into the net
	plt.imshow(solver.net.blobs['data'].data[:8, 0].transpose(1,0,2).reshape(16, 8*16), 
			cmap = 'gray'); axis('off') # show what that data was
	print 'Train labels: ', solver.net.blobs['label'].data[:8]

	print (solver.net.params['score'][0].data)

#### Stepping the solver
# show filters before training
	plt.figure(2)
	plt.subplot(2,1,1)
	plt.imshow(solver.net.params['score'][0].data[2].reshape(16,48),
			cmap = 'gray'); axis('off')
	#take 1 step of minibatch SGD
	solver.step(1)
	# check our filters after 1 step- first layer 4x5 grid of 5x5 filters
	plt.subplot(2,1,2)
	plt.imshow(solver.net.params['score'][0].data[2].reshape(16,48),
			cmap = 'gray'); axis('off')
	plt.show()
print


# train the network
solver.solve() # the fast way, but we cant save stuff
if False:
	niter = 30
	test_interval = 10
	train_loss = np.zeros(niter)
	test_acc = np.zeros(int(np.ceil(niter / test_interval)))
	print "Start Training..."
	for it in range(niter):
		solver.step(1) # run 1 batch

		train_loss[it] = solver.net.blobs['loss'].data

##### plot some output

	_, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(np.arange(niter), train_loss)
	ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
	ax1.set_xlabel('iteration')
	ax1.set_ylabel('train loss')
	ax2.set_ylabel('test accuracy')
	ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
	plt.show()




print
print
print "=================== 01_Classifier_CNN ==================="