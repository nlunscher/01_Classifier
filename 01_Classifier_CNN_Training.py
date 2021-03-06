# Hand writing 0s and 1s classifier CNN

#### Setup environment
from pylab import *
import matplotlib.pyplot as plt # http://www.pyimagesearch.com/2015/08/24/resolved-matplotlib-figures-not-showing-up-or-displaying/
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
	n.prob = L.InnerProduct(n.fc1, num_output = 2, weight_filler = dict(type = 'xavier'))
	# n.loss = L.Softmax(n.prob)
	n.loss = L.SoftmaxWithLoss(n.prob, n.label)

	return n.to_proto()

def create_net(source, batch_size, folder, net_type = "train"):
	untrained_zero1_net = zero1_net(source = source, batch_size = batch_size)

	with open(folder + 'zero1_net_' + net_type + '.prototxt', 'w') as f:
		f.write(str(untrained_zero1_net))

# output model location
model_folder = 'Model_Files/'
model_output = model_folder + 'zero1_net_train.caffemodel'

# setup the CNN
dataset_source_test = '01_images/01_images_test/im_reference_test.txt'
dataset_source_train = '01_images/01_images_train/im_reference_train.txt'
batch_size_test = 8
batch_size_train = 20
create_net(dataset_source_test, batch_size_test, model_folder, 'test')
create_net(dataset_source_train, batch_size_train, model_folder, 'train')

# setup the solver
caffe.set_mode_cpu()

solver = None
solver = caffe.SGDSolver(model_folder + 'zero1_net_solver.prototxt')

# get a view of the model
# check feature dimensions - (batch size, feature dim, spatial dim)
print [(k, v.data.shape) for k, v in solver.net.blobs.items()]
# just the weight sizes
print [(k, v[0].data.shape) for k, v, in solver.net.params.items()]
print

# train the network
fast = False
if fast:
	solver.solve() # the fast way, but we cant save stuff
else:
	niter = 50
	test_interval = 5
	test_iter = 4
	train_loss = np.zeros(niter)
	test_acc = np.zeros(int(np.ceil(niter / test_interval)))
	print "Start Training..."
	for it in range(niter):
		solver.step(1) # run 1 batch

		train_loss[it] = solver.net.blobs['loss'].data

		if it % test_interval == 0:
			print "Iteration ", it, " Testing..."
			correct = 0
			for test_it in range(test_iter):
				solver.test_nets[0].forward()
				correct += sum(solver.test_nets[0].blobs['prob'].data.argmax(1)
								== solver.test_nets[0].blobs['label'].data)
				print "Classification: ", solver.test_nets[0].blobs['prob'].data.argmax(1), " Real Labels: ", solver.test_nets[0].blobs['label'].data
				# print "Data: ", solver.test_nets[0].blobs['prob'].data

			test_acc[it // test_interval] = 1.0 * correct / (test_iter * batch_size_test)
			print test_acc[it // test_interval]

	print "Saving model..."
	solver.net.save(model_output)
	# solver.test_nets[0].save('zero1_net_test.caffemodel')


##### plot some output

	_, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(np.arange(niter), train_loss)
	ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
	ax1.set_xlabel('Iteration')
	ax1.set_ylabel('Train loss')
	ax2.set_ylabel('Test accuracy')
	ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
	print "Accuracy: ", test_acc
	plt.show()




print
print
print "=================== 01_Classifier_CNN ==================="