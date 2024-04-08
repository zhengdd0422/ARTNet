from keras.layers import Dense, Flatten,  Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential


def cnn_2(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=9, input_shape=input_shape, name='conv1'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=5, name='maxpool1'))
    model.add(Dropout(0.5, name='d1'))
    model.add(Conv1D(filters=128, kernel_size=7, name='conv2'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=5, name='maxpool2'))
    model.add(Dropout(0.5, name='d2'))
    model.add(Flatten(name='flatten1'))
    model.add(Dense(128, activation='relu', name='fc1'))
    model.add(Dense(1, activation='sigmoid', name='score'))
    return model


def adprtnet(input_shape):
    """construct the deviation network-based detection model"""
    model = cnn_2(input_shape)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
