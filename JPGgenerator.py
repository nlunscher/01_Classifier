# Create image jpgs from the formatted file


import numpy as np
import Image

test_im_str = '0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 0.0000 0.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 0.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 0.0000 0.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1 0 0 0 0 0 0 0 0 0 '

# read in the image Str dataset
def read_dataset(dataset_file_name):
	im_strs = []
	with open(dataset_file_name) as f:
		im_strs = f.readlines()

	return im_strs

# determine image label and remove label from data
def determine_label(imStr):
	im_split = imStr.split(' ')

	im_labels = np.zeros(10)
	str_labels = im_split[-11:-1]
	for i in range(10):
		im_labels[i] = float(str_labels[i])

	# print im_labels

	return np.argmax(im_labels)

# read the bits and reshape it to the image
def format_image(imStr):
	im_dim = 16

	im_split = (imStr.split(' '))[:-11]
	im_split_float = np.zeros(len(im_split))

	for i in range(len(im_split)):
		im_split_float[i] = float(im_split[i])

	im_reshape = im_split_float.reshape(im_dim, im_dim) * 255
	return im_reshape

# create and save the image from a shape of floats
def create_jpg(im_float, fileName):
	im_out = Image.fromarray(im_float.astype('uint8')).convert('L')
	im_out.save(fileName)


# write to a file (for as a refernce for Caffe to read the images)
def writeTextLine(file_name, line_text):
	with open(file_name, 'a') as f:
		f.write(line_text + '\n')



dataset_file_name = 'SemeionHandwrittenDataset.txt'
image_folder = '01_images/'
im_ref_file_name = 'im_reference.txt'

im_strs = read_dataset(dataset_file_name)

im_num = 0
im_totals = np.zeros(10)
for im_str in im_strs:
	im_label = determine_label(im_str)
	# print im_label

	im_totals[im_label] += 1

	if im_label in [0, 1]:
		im_num += 1

		im_float = format_image(im_str)

		file_name = str(im_num) + '_' + str(im_label) + 'image.jpg'
		create_jpg(im_float, image_folder + file_name)

		writeTextLine(image_folder + im_ref_file_name, file_name + ' ' + str(im_label))

print im_totals


print
print
print "================ JPGs Generator done ================"