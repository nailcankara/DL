import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1:].values
#%%
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
#%%
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [1])], remainder='passthrough' )
onehotencoder = OneHotEncoder(categories='auto')
X = ct.fit_transform(X)
X = X[:,1:]
#%%
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y, test_size = 0.2 , random_state=0)
#%%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
#%%
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
classifier = Sequential()
classifier.add(Dense(units=6 ,  kernel_initializer='uniform' , activation= 'relu' , input_dim=11))
classifier.add(Dropout(p=0.1)) #close %10 node
classifier.add(Dense(units=6 ,  kernel_initializer='uniform' , activation= 'relu'))
classifier.add(Dropout(p=0.1))
classifier.add(Dense(units=1 ,  kernel_initializer='uniform' , activation= 'sigmoid'))
classifier.compile(optimizer='adam' , loss= 'binary_crossentropy' , metrics=['accuracy'])  #categorical_crossentropy for multiple categories
classifier.fit(X_train , y_train , batch_size=10 , epochs=100)
#%%
y_pred = classifier.predict(X_test)
y_pred = y_pred > 0.5
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
