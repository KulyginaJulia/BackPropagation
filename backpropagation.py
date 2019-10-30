from keras.datasets import mnist
from keras import optimizers
from keras.utils import np_utils
import numpy as np
from datetime import datetime as dt
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Dense # the two types of neural network layer we will be using

def predict(X, Y, synapse_0, synapse_1, b1, b2):
    W_layer_1 = synapse_0.dot(X.T) + b1  # WX1
    layer_1 = reLU(W_layer_1)  # X1

    W_layer_2 = synapse_1.dot(layer_1) + b2
    Y_predict = softmax(W_layer_2)
    Y_predict = Y_predict.T

    cross = -np.sum(Y * np.log(Y_predict)) / Y.shape[0]

    Y_predict = np.argmax(Y_predict, axis=1)
    Y = np.argmax(Y, axis=1)
    accuracy = (Y == Y_predict).mean()
    return accuracy, cross

def reLu_derivative(X):
    return 1. * (X > 0)

def reLU(x):
    return np.maximum(x, 0)

def softmax(x):
    expX = np.exp(x)
    sum = expX.sum(axis=0, keepdims=True)
    return expX / sum

def crossEntropy(x, y):
    return -np.sum(y*np.log(x.T)) / x.shape[0]

def keras_(hidden_dim, num_epochs, batch_size, learning_rate):
    height, width, depth = 28, 28, 1
    num_classes = 10  # there are 10 classes (1 per digit)
   # hidden_dim = 512
   # num_epochs = 20
   # batch_size = 128
   # learning_rate = 0.1

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    num_train = 60000
    num_test = 10000
    X_train = X_train.reshape(num_train, height * width)

    X_test = X_test.reshape(num_test, height * width)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    Y_train = np_utils.to_categorical(Y_train, num_classes)  # One-hot encode the labels
    Y_test = np_utils.to_categorical(Y_test, num_classes)  # One-hot encode the labels
    time_start = dt.now()

    inp = Input(shape=(height * width,))  # Our input is a 1D vector of size 784
    hidden_1 = Dense(hidden_dim, activation='relu')(inp)  # First hidden ReLU layer
    out = Dense(num_classes, activation='softmax')(hidden_1)  # Output softmax layer

    model = Model(input=inp, output=out)  # To define a model, just specify its input and output layers
    sgd = optimizers.SGD(lr=learning_rate, momentum=0.0, nesterov=False)

    model.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
                  optimizer=sgd,  # using the SGD optimiser
                  metrics=['accuracy'])  # reporting the accuracy
    model.fit(X_train, Y_train,  # Train the model using the training set...
              batch_size=batch_size, nb_epoch=num_epochs,
              verbose=1, validation_split=0.1)  # ...holding out 10% of the data for validation
    delta_time = dt.now() - time_start
    print("Time for training = ", delta_time)
    model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!


def main(hidden_dim, training_epoch, batch_size, learning_rate):
    height, width = 28, 28
    num_classes = 10  # there are 10 classes (1 per digit)

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    num_train = 60000
    num_test = 10000
    X_train = X_train.reshape(num_train, height * width)

    X_test = X_test.reshape(num_test, height * width)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    Y_train = np_utils.to_categorical(Y_train, num_classes)  # One-hot encode the labels
    Y_test = np_utils.to_categorical(Y_test, num_classes)  # One-hot encode the labels

  #  hidden_dim = 512
  #  training_epoch = 20
  #  batch_size = 128
   # learning_rate = 0.1

    time_start = dt.now()

    synapse_0 = np.random.uniform(-0.2, 0.2, (hidden_dim, X_train.shape[1]))
    synapse_1 = np.random.uniform(-0.5, 0.5, (num_classes, hidden_dim))

    b1 = np.zeros((hidden_dim, 1))
    b2 = np.zeros((num_classes, 1))

    total_batch = int(num_train / batch_size)

    random_state = np.random.get_state()
    np.random.shuffle(X_train)
    np.random.set_state(random_state)
    np.random.shuffle(Y_train)
    # fit
    for epoch in range(training_epoch):
        for i in range(0, total_batch):
            start = i * batch_size
            end = (i + 1) * batch_size

            batch_x = X_train[start:end]  # layer_0
            batch_y = Y_train[start:end]

            W_layer_1 = synapse_0.dot(batch_x.T) + b1 #WX1
            layer_1 = reLU(W_layer_1) # X1

            W_layer_2 = synapse_1.dot(layer_1) + b2
            layer_2 = softmax(W_layer_2) #X2

            # derivative
            delta_2 = batch_y.T - layer_2
            dW2 = delta_2.dot(layer_1.T) / batch_size
            db2 = np.sum(delta_2, axis=1, keepdims=True) / batch_size

            delta_1 = synapse_1.T.dot(delta_2) * reLu_derivative(W_layer_1)
            dW1 = delta_1.dot(batch_x) / batch_size
            db1 = np.sum(delta_1, axis=1, keepdims=True) / batch_size

            synapse_0 += learning_rate*dW1
            synapse_1 += learning_rate*dW2
            b1 += learning_rate*db1
            b2 += learning_rate*db2

        if (epoch) % 10 == 0:
            accuracy, crossentropy = predict(X_train, Y_train, synapse_0, synapse_1, b1, b2)
            print("Epoch: " + str(epoch) + " accuracy: " + str(accuracy) + " loss: " + str(crossentropy))

    accuracy, crossentropy = predict(X_train, Y_train, synapse_0, synapse_1, b1, b2)
    print("Epoch: " + str(training_epoch) + " accuracy: " + str(accuracy) + " loss: " + str(crossentropy))

    delta_time = dt.now() - time_start
    print("Time for training = ", delta_time)
    test, cross = predict(X_test, Y_test, synapse_0, synapse_1, b1, b2)
    print("Accuracy for test = ", test)
    print("Loss for test = ", cross)

if __name__ == '__main__':
    hidden_dim = 300
    training_epochs = 20
    batch_size = 128
    learning_rate = 0.1
    main(hidden_dim, training_epochs, batch_size, learning_rate)

    #keras_(hidden_dim, num_epochs, batch_size, learning_rate)
