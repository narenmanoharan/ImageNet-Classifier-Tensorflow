import tensorflow as tf
import tflearn as tfl

# Importing the Flower datasets 
X, Y = tfl.datasets.oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

# Image Processing using zero center and standard normalization
img_pre_processing = tfl.data_preprocessing.ImagePreprocessing()
img_pre_processing.add_featurewise_zero_center()
img_pre_processing.add_featurewise_stdnorm()

# Image Augmentation using Random flipping and Random rotation
img_augmentation = tfl.data_augmentation.ImageAugmentation()
img_augmentation.add_random_flip_leftright()
img_augmentation.add_random_rotation(max_angle=27.)

# Creating data model for the alexnet model
alex_net_max = tfl.layers.core.input_data(shape=[None, 227, 227, 3])

# 2D Convolution
alex_net_max = tfl.layers.conv.conv_2d(alex_net_max, 96, 11, strides=4, activation='relu')

# 2D Max-pooling
alex_net_max = tfl.layers.conv.max_pool_2d(alex_net_max, 3, strides=2)

# Normalization to decrease overfitting
alex_net_max = tfl.layers.normalization.local_response_normalization(alex_net_max)

# 2D Convolution
alex_net_max = tfl.layers.conv.conv_2d(alex_net_max, 256, 5, activation='relu')

# 2D Max-pooling
alex_net_max = tfl.layers.conv.max_pool_2d(alex_net_max, 3, strides=2)

# Normalization to decrease overfitting
alex_net_max = tfl.layers.normalization.local_response_normalization(alex_net_max)

# 2D Convolution
alex_net_max = tfl.layers.conv.conv_2d(alex_net_max, 384, 3, activation='relu')

# 2D Convolution
alex_net_max = tfl.layers.conv.conv_2d(alex_net_max, 384, 3, activation='relu')

# 2D Convolution
alex_net_max = tfl.layers.conv.conv_2d(alex_net_max, 256, 3, activation='relu')

# 2D Max-pooling
alex_net_max = tfl.layers.conv.max_pool_2d(alex_net_max, 3, strides=2)

# Normalization to decrease overfitting
alex_net_max = tfl.layers.normalization.local_response_normalization(alex_net_max)

# Fully connected layer
alex_net_max = tfl.layers.core.fully_connected(alex_net_max, 4096, activation='tanh')

# Dropout layer
alex_net_max = tfl.layers.core.dropout(alex_net_max, 0.5)

# Fully connected layer
alex_net_max = tfl.layers.core.fully_connected(alex_net_max, 4096, activation='tanh')

# Dropout layer
alex_net_max = tfl.layers.core.dropout(alex_net_max, 0.5)

# Fully connected layer
alex_net_max = tfl.layers.core.fully_connected(alex_net_max, 17, activation='softmax')

# Constructing an estimator using regression
alex_net_max = tfl.layers.estimator.regression(alex_net_max, optimizer='momentum', loss='categorical_crossentropy', learning_rate=0.001)

# Constructing the Deep Neural Network using created model
model = tfl.DNN(alex_net_max, checkpoint_path='alexnet_max', max_checkpoints=1, tensorboard_verbose=2)

# Fitting the model with the validation set
model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True, show_metric=True, batch_size=64, run_id='alex_flower_max')

