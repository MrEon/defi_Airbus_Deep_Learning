from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
import numpy as np
from keras import backend as K
import keras

def loadMNISTAsVector(subset=10000) :
    nb_classes=10
    (X_train_img, y_train_real), (X_test_img, y_test_real) = mnist.load_data()
    X_train_vect = X_train_img[:subset].reshape(subset, 784)
    X_test_vect = X_test_img.reshape(10000, 784)
    X_train_vect = X_train_vect.astype("float32")
    X_test_vect = X_test_vect.astype("float32")
    X_train_vect /= 255
    X_test_vect /= 255
    y_train_cat = np_utils.to_categorical(y_train_real[:subset], nb_classes)
    y_test_cat = np_utils.to_categorical(y_test_real, nb_classes)
    return (X_train_vect, y_train_cat), (X_test_vect, y_test_cat)
    
def loadMNISTAsMaxtrix(subset=10000) :
    nb_classes=10
    img_rows, img_cols = 28, 28
    (X_train_img, y_train_real), (X_test_img, y_test_real) = mnist.load_data()
    X_train_mat = X_train_img[:subset].reshape(X_train_img[:subset].shape[0], img_rows, img_cols, 1)
    X_test_mat = X_test_img.reshape(X_test_img.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    X_train_mat = X_train_mat.astype('float32')
    X_test_mat = X_test_mat.astype('float32')
    X_train_mat /= 255
    X_test_mat /= 255
    y_train_cat = np_utils.to_categorical(y_train_real[:subset], nb_classes)
    y_test_cat = np_utils.to_categorical(y_test_real, nb_classes)
    return (X_train_mat, y_train_cat), (X_test_mat, y_test_cat)
    
def plot_10_by_10_images(images):
    """ Plot 100 MNIST images in a 10 by 10 table. Note that we crop
    the images so that they appear reasonably close together.  The
    image is post-processed to give the appearance of being continued."""
    fig = plt.figure()
    #image = np.concatenate(images, axis=1)
    for x in range(10):
        for y in range(10):
            ax = fig.add_subplot(10, 10, 10*y+x+1)
            plt.imshow(images[10*y+x+1].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()

def plot_mnist_digit(image):
    """ Plot a single MNIST image."""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(image.reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()    
    
(X_train_mat, y_train_cat), (X_test_mat, y_test_cat) =loadMNISTAsMaxtrix()
(X_train_vect, y_train_cat), (X_test_vect, y_test_cat) =loadMNISTAsVector()

plot_10_by_10_images(X_train_vect)

plot_mnist_digit(X_train_vect[200])

print(X_train_vect.shape)
print(X_train_mat.shape)

def getSimplePerceptron(nb_classes=10):
    model = Sequential()
    model.add(Dense(28*28, input_shape=(784,),activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(28*8, activation='elu'))
    model.add(Dense(28*4, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],  optimizer=sgd)
    return model

model=getSimplePerceptron()
model.summary()

(X_train_vect, y_train_cat), (X_test_vect, y_test_cat) =loadMNISTAsVector()
model=getSimplePerceptron()
batch_size = 64
epochs=20
model.fit(X_train_vect, y_train_cat, batch_size=batch_size, epochs=epochs,  verbose=1, validation_data=(X_test_vect, y_test_cat))
