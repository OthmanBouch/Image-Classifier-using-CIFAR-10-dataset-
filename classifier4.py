import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
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


learning_rates = [1e-2, 1e-3, 1e-4]
batch_sizes = [32, 64, 128]
performance = {}

for lr in learning_rates:
    for batch_size in batch_sizes:
       
        model = Sequential([
            Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])

       
        optimizer = Adam(learning_rate=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        
        history = model.fit(X_train, Y_train, epochs=10, batch_size=batch_size, validation_split=0.2, verbose=0)
        
        
        loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
        performance[(lr, batch_size)] = (loss, accuracy)
        print(f"Learning rate: {lr}, Batch size: {batch_size}, Loss: {loss}, Accuracy: {accuracy}")


fig, axs = plt.subplots(len(learning_rates), 2, figsize=(12, len(learning_rates) * 4))

for i, lr in enumerate(learning_rates):
    losses = [performance[(lr, bs)][0] for bs in batch_sizes]
    accuracies = [performance[(lr, bs)][1] for bs in batch_sizes]
    
    axs[i, 0].plot(batch_sizes, losses, marker='o')
    axs[i, 0].set_title(f'Loss vs. Batch Size at learning rate {lr}')
    axs[i, 0].set_xlabel('Batch Size')
    axs[i, 0].set_ylabel('Loss')
    
    axs[i, 1].plot(batch_sizes, accuracies, marker='o')
    axs[i, 1].set_title(f'Accuracy vs. Batch Size at learning rate {lr}')
    axs[i, 1].set_xlabel('Batch Size')
    axs[i, 1].set_ylabel('Accuracy')

plt.tight_layout()
plt.show()
