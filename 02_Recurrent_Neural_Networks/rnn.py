!wget https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/P16-Recurrent-Neural-Networks.zip

!unzip P16-Recurrent-Neural-Networks.zip

cd Recurrent_Neural_Networks

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []

for i in range(60, len(training_set)):
  X_train.append(training_set_scaled[i-60:i , 0:1])
  y_train.append(training_set_scaled[i , 0])

X_train,y_train = np.array(X_train) , np.array(y_train)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()
regressor.add(LSTM(units = 50 , return_sequences = True , input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50 , return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50 , return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))

regressor.compile(optimizer= 'adam' , loss= 'mean_squared_error' , metrics=['mse'])

regressor.fit(X_train,y_train , epochs=100 , batch_size=32)

dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:,1:2].values

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']) , axis = 0)
dataset_total = dataset_total.values.reshape(-1,1)
inputs = dataset_total[len(dataset_total)-len(dataset_test) - 60 :]
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
  X_test.append(inputs[i-60:i , 0:1])

X_test = np.array(X_test)

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = 'red' ,  label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue' ,  label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
