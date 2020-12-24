# Finger Count ResNet
This is a neural network that is based on the [ResNet algorithm](https://arxiv.org/abs/1512.03385) by He et al.

The ResNet algorithm allows for training very deep neural networks without the problem of exploding/vanishing gradients. This is avoided by using "skip connections" which make learning the identity function between layers very simple. In theory, this means more layers should only improve the performance of the network.

## Install
Note: Tensorflow doesn't currently work with Python 3.9

To setup the environment you will first need to install the dependencies. It's recommended that you create a virtual environment first.
```
python -m venv venv-name
```
Then run
```
pip install -r requirements.txt
```

## Running
Open main.py to set these variables:
```
train
model_path
epochs
test_img
```

## Analyzing results
Running the following command will allow you to visualize train and validation accuracy using Tensorboard
```
tensorboard --logdir=logs
```
You will need to navigate to the address that is provided as output and toggle the "train" and "validation" runs listed on the left hand side.
