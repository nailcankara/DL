from tqdm import tqdm
from keras.datasets import mnist
from keras.layers import Dense, Reshape, Conv2D, Flatten, Input, UpSampling2D, BatchNormalization, LeakyReLU
from keras.models import Sequential , Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
K.clear_session()

input_size = 100  
optimizer = Adam(lr=0.0002)

generator = Sequential()
generator.add(Dense(7*7*256, input_dim=input_size))
generator.add(BatchNormalization())
generator.add(LeakyReLU())
generator.add(Reshape((7, 7, 256)))

generator.add(UpSampling2D())
generator.add(Conv2D(128, kernel_size=5, strides=1, padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU())

generator.add(UpSampling2D())
generator.add(Conv2D(64, kernel_size=5, strides=1, padding='same' ))
generator.add(BatchNormalization())
generator.add(LeakyReLU())

generator.add(Conv2D(1, kernel_size=5, strides=1, padding='same', use_bias=False , activation='tanh'))
#generator.summary()

discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=(28, 28, 1)))
discriminator.add(LeakyReLU())
#discriminator.add(layers.Dropout(0.3))

discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
discriminator.add(LeakyReLU())
#discriminator.add(layers.Dropout(0.3))

discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
#discriminator.summary()

discriminator.trainable = False
x = Input((input_size,))
out_generator = generator(x)
out_discriminator = discriminator(out_generator)
gan = Model(inputs=(x,), outputs=(out_discriminator))
gan.compile(loss="binary_crossentropy", optimizer=optimizer)
#gan.summary()

def generateRandomData(sizey, sizex):
    return np.random.normal(0, 1, (sizey, sizex))

def showResults(gen):
    noise = generateRandomData(4, input_size)
    images = gen.predict(noise)
    plt.figure(figsize=(2, 2))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        im = np.reshape(images[i], (1, -1))
        im = (np.reshape(im, [28, 28]) + 1) * 255
        im = np.clip(im, 0, 255)
        im = np.uint8(im)
        plt.imshow(im, cmap='gray')
    plt.show()


(xtrain, _), (xtest, _) = mnist.load_data()
xtrain = np.reshape(xtrain, [-1, 28, 28, 1])
xtest = np.reshape(xtest, [-1, 28, 28, 1])
xtrain = (xtrain.astype(np.float32) - 127.5) / 127.5

epochs = 1000
batch_size = 100

for e in range(epochs):
    for i in tqdm(range(int(xtrain.shape[0]/batch_size))):
        xreal = xtrain[(i) * batch_size:(i + 1) * batch_size]
        noise = generateRandomData(batch_size, input_size)
        xfake = generator.predict_on_batch(noise)
        discriminator.trainable = True
        discriminator.train_on_batch(xreal, np.array([[0.9]] * batch_size))
        discriminator.train_on_batch(xfake, np.array([[0.]] * batch_size))
        discriminator.trainable = False
        gan.train_on_batch(noise, np.array([[1.]] * batch_size))

    print(e,")")
    showResults(generator)
