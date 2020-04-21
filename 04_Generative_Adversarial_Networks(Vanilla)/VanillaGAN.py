from tqdm import tqdm
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras import Sequential,Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
K.clear_session()

(xtrain, _), (xtest,_)  = mnist.load_data()

input_size = 100 
optimizer = Adam(lr=0.0002)
generator = Sequential()
generator.add(Dense(256,activation='relu',input_dim=input_size))
generator.add(Dense(512,activation='relu'))
generator.add(Dense(784,activation='tanh'))

discriminator = Sequential()
discriminator.add(Dense(512,activation='relu',input_dim=784))
discriminator.add(Dense(256,activation='relu'))
discriminator.add(Dense(1,activation='sigmoid'))
discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)

discriminator.trainable = False
x = Input((input_size,))
out_generator = generator(x)
out_discriminator = discriminator(out_generator)
gan = Model(inputs=(x,), outputs=(out_discriminator))
gan.compile(loss="binary_crossentropy", optimizer=optimizer)

def generateRandomData(sizey, sizex):
    return np.random.normal(0,1,(sizey,sizex))

def showResults(gen):
    noise = generateRandomData(32,input_size)
    images = gen.predict(noise)
    plt.figure(figsize=(4,8))
    for i in range(32):
        plt.subplot(4,8,i+1)
        im = np.reshape(images[i],(1,-1))
        im = (np.reshape(im,[28,28])+1) *255
        im = np.clip(im,0,255)
        im = np.uint8(im)
        plt.imshow(im, cmap='gray')
    plt.show()   

epochs = 100
batch_size = 100
xtrain = np.reshape(xtrain,[xtrain.shape[0],-1])
xtest = np.reshape(xtest,[xtest.shape[0],-1])
xtrain = (xtrain.astype(np.float32) - 127.5)/127.5
plt.imshow(np.reshape(xtrain[0],[28,28] ),cmap="gray")
plt.show()

for e in range(epochs):
    for i in tqdm(range(int(xtrain.shape[0]/batch_size))):
        xreal = xtrain[(i) * batch_size:(i+1)* batch_size]
        noise = generateRandomData(batch_size, input_size)
        xfake = generator.predict_on_batch(noise)
        discriminator.trainable = True
        discriminator.train_on_batch(xreal, np.array([[0.9]]*batch_size))
        discriminator.train_on_batch(xfake, np.array([[0.]]*batch_size))
        discriminator.trainable = False
        gan.train_on_batch(noise, np.array([[1.]]*batch_size))
    
    showResults(generator)
