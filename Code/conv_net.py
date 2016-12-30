import tensorflow as tf
import tflearn as tfl

# Importing and categorizing Cifar - 10 datasets 
(X, Y), (X_test, Y_test) = tfl.datasets.cifar10.load_data()
X, Y = tfl.data_utils.shuffle(X, Y)
Y = tfl.data_utils.to_categorical(Y, 10)
Y_test = tfl.data_utils.to_categorical(Y_test, 10) 

# Image Processing using zero center and standard normalization
img_pre_processing = tfl.data_preprocessing.ImagePreprocessing()
img_pre_processing.add_featurewise_zero_center()
img_pre_processing.add_featurewise_stdnorm()

# Image Augmentation using Random flipping and Random rotation
img_augmentation = tfl.data_augmentation.ImageAugmentation()
img_augmentation.add_random_flip_leftright()
img_augmentation.add_random_rotation(max_angle=27.)

# Creating data model for the lenet model
lenet_max = tfl.layers.core.input_data(shape=[None, 32, 32, 3], data_preprocessing=img_pre_processing, data_augmentation=img_augmentation)

# 2D Convolution
lenet_max = tfl.layers.conv.conv_2d(lenet_max, 32, 3, activation='relu')

# 2D Max-pooling
lenet_max = tfl.layers.conv.max_pool_2d(lenet_max, 2)

# 2D Convolution
lenet_max = tfl.layers.conv.conv_2d(lenet_max, 64, 3, activation='relu')

# 2D Convolution
lenet_max = tfl.layers.conv.conv_2d(lenet_max, 64, 3, activation='relu')

# 2D Max-pooling
lenet_max = tfl.layers.conv.max_pool_2d(lenet_max, 2)

# Fully connected layer
lenet_max = tfl.layers.core.fully_connected(lenet_max, 512, activation='relu')

# Dropout layer
lenet_max = tfl.layers.core.dropout(lenet_max, 0.5)

# Fully connected layer
lenet_max = tfl.layers.core.fully_connected(lenet_max, 10, activation='softmax')

# Constructing an estimator using regression
lenet_max = tfl.layers.estimator.regression(lenet_max, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

# Constructing the Deep Neural Network using LeNet
model = tfl.DNN(lenet_max)

# Fitting the model with the validation set
model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test), show_metric=True, batch_size=96)
