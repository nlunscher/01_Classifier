# Using the Classifier

#### Setup environment
from pylab import *
import matplotlib.pyplot as plt
import numpy as np

# add caffe to python path
import sys
caffe_root = '../../../'
sys.path.insert(0, caffe_root + 'python')
import caffe

# Setup model
caffe.set_mode_cpu()

model_def = 'zero1_net_train.prototxt'
model_weights = 'zero1_net_snap_iter_100.caffemodel'

net = caffe.Net(model_def,		#defines the structure of the mdoel
				model_weights,	# contains the trained weights
				caffe.TEST)		# use test mode (e.g. dont perform dropout) - leave out nodes during testinf to help prevent overfitting

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))


# set the size of the input
# net.blobs['data'].reshape(	1,		# batch size
# 							3,		# 3-channel (RGB) images
# 							16,16)# image size of 227x227


# setup the data you want to use
# image_file = '01_images/1_0image.jpg'
image_file = '01_images/27_1image.jpg'
image = caffe.io.load_image(image_file)

# classify the image
net.blobs['data'].data[...] = transformer.preprocess('data', image) # copy image to memory allocated for the net
output = net.forward() # the actual classification
output_prob = output # the output probability vector for the first image in the batch

print net.blobs
# output_other = net.blobs['prob'].data
output_other = net.blobs['fc2'].data

print 'Predicted class is: ', output_prob
print 'Other Output: ', (output_other)


net.blobs['data'].data[...] = transformer.preprocess('data', image)
net.forward()
print net.blobs['fc2'].data





print
print
print "=================== 01_Classifier_CNN_Using ==================="