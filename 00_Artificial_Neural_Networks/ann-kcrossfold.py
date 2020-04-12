from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
  classifier = Sequential()
  classifier.add(Dense(units=6 ,  kernel_initializer='uniform' , activation= 'relu' , input_dim=11))
  classifier.add(Dense(units=6 ,  kernel_initializer='uniform' , activation= 'relu'))
  classifier.add(Dense(units=1 ,  kernel_initializer='uniform' , activation= 'sigmoid'))
  classifier.compile(optimizer='adam' , loss= 'binary_crossentropy' , metrics=['accuracy'])
  return classifier

classifier = KerasClassifier(build_fn = build_classifier , batch_size = 10 , epochs = 100)
accuracies = cross_val_score(estimator = classifier , X = X_train , y = y_train , cv=10)
mean = accuracies.mean()
variance = accuracies.std()