import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical


def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'data'], dict[b'labels']


X_train, Y_train = [], []
for i in range(1, 6):  
    X_batch, Y_batch = load_cifar10_batch(f'data_batch_{i}')
    X_train.append(X_batch)
    Y_train += Y_batch

X_train = np.concatenate(X_train, axis=0)
Y_train = np.array(Y_train)


X_test, Y_test = load_cifar10_batch('test_batch')


X_train = X_train.reshape((len(X_train), 3, 32, 32)).transpose(0,2,3,1) / 255.0
X_test = X_test.reshape((len(X_test), 3, 32, 32)).transpose(0,2,3,1) / 255.0


Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  
])


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_data=(X_test, Y_test))


loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

