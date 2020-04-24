!wget https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/P16-AutoEncoders.zip

!unzip P16-AutoEncoders.zip

cd /content/AutoEncoders

!unzip ml-100k.zip

!unzip ml-1m.zip

import numpy as np
import pandas as pd

training_set = pd.read_csv('/content/AutoEncoders/ml-100k/u1.base' , delimiter= '\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('/content/AutoEncoders/ml-100k/u1.test', delimiter= '\t')
test_set = np.array(test_set, dtype='int')

nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

def convert(data):
  new_data = []
  for id_users in range(1 , nb_users+1):
    id_movies = data[:,1][data[:,0] == id_users]
    id_ratings = data[:,2][data[:,0] == id_users]
    ratings = np.zeros(nb_movies)
    ratings[id_movies-1] = id_ratings  
    new_data.append(ratings)
  return new_data

training_set = np.array(convert(training_set))
test_set = np.array(convert(test_set))

from keras.models import Model
from keras.layers import Dense , Input 
import keras.backend as K
K.clear_session()

input = Input(shape=(nb_movies,))
encoded1 = Dense(50,activation='sigmoid')(input)
encoded2 = Dense(25,activation='sigmoid')(encoded1)
decoded1 = Dense(50,activation='sigmoid')(encoded2)
output = Dense(nb_movies)(encoded1)

autoencoder = Model(input,output)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

epochs = 250
batch_size = 1
import time
for epoch in range(epochs):
  train_loss=0
  s=0
  total_mean = 0
  for i in range(int(training_set.shape[0]/batch_size)):
    batch_input = training_set[(i) * batch_size:(i + 1) * batch_size]
    target = np.copy(batch_input)
    if np.sum(training_set[i]) > 0:
      predicted = autoencoder.predict_on_batch(training_set[(i) * batch_size:(i + 1) * batch_size]) 
      K.stop_gradient(target)
      predicted[target == 0] = 0
      autoencoder.train_on_batch(predicted,target)

      train_loss += np.sum(((predicted - target) * (predicted - target) / 2))
      s=s+1
      total_mean += len(target[target > 0])
  mean_corrector = total_mean / s
  train_loss = train_loss / (mean_corrector + 1e-10)
  
  print('epoch: ' +str(epoch) + ' loss: '+str(train_loss/s))
  print("----------------------------")
