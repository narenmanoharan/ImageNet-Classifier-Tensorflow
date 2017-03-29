## ImageNet Classifier with TensorFlow

[![Build Status](https://travis-ci.org/narenkmanoharan/Set-Game-Implementation.svg?branch=master)](https://travis-ci.org/narenkmanoharan/Set-Game-Implementation)

*Technical Specifications*

* Python 2.7 
* TensorFlow 0.10.0
* TFLearn 0.2.1
* CUDA 8
* CuDnn v5

Dataset Links

@ CIFAR 10 dataset

https://www.cs.toronto.edu/~kriz/cifar.html

@ 102 Category Flower Dataset

http://www.robots.ox.ac.uk/~vgg/data/flowers/102/

### Initiate EC2 instance on AWS with the following specification

g2.8x - 80GB SSD - 32 vCPU

### Installing initial dependencies

```sudo apt-get update
sudo apt-get -y dist-upgrade
sudo apt-get install python
sudo apt install python3-pip
sudo apt-get install -y libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev gcc g++ gfortran build-essential git wget linux-image-generic libopenblas-dev python-dev liblapack-dev libblas-dev build-essential cmake git unzip pkg-config linux-image-generic linux-image-extra-virtual linux-source linux-headers-generic 
```

### Installing compression libraries

```
sudo apt-get install zlib1g-dev python-imaging
```

### Exporting TensorFlow 0.10.0 for Python 2.7

```
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl
```

### Installing TensorFlow

```
sudo pip install --upgrade $TF_BINARY_URL
```

### Installing and upgrading Python and Pip

```
sudo apt-get install python
sudo apt install python-pip
sudo pip install --upgrade pip
```

### Downloading and installing 

```
wget -qO- https://github.com/tflearn/tflearn/tarball/0.2.1 | tar xvz
cd tflearn-tflearn-a55c1fd/
sudo python setup.py install
```

### Installing scipy stack dependencies

```
sudo pip install pillow numpy scipy h5py
```

### Install CUDA

https://developer.nvidia.com/cuda-downloads

### Installing CuDNN

https://developer.nvidia.com/cudnn

### SCP the code into the remote server and run the code using python 2.7

```
python alex_net.py
```







