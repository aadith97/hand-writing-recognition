from keras.models import model_from_json
# Simple CNN model for the CIFAR-10 Dataset
import numpy
import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import hw_imgextract
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from scipy import misc
# load json and create model
json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model1.h5")
print("Loaded model from disk")


C = misc.imread('test0.jpg')
C = np.array([C])

ans=loaded_model.predict(C)
print(ans)
a=numpy.argmax(ans)
print(a)