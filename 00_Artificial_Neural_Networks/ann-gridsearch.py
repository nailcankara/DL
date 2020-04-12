from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
  classifier = Sequential()
  classifier.add(Dense(units=6 ,  kernel_initializer='uniform' , activation= 'relu' , input_dim=11))
  classifier.add(Dense(units=6 ,  kernel_initializer='uniform' , activation= 'relu'))
  classifier.add(Dense(units=1 ,  kernel_initializer='uniform' , activation= 'sigmoid'))
  classifier.compile(optimizer=optimizer , loss= 'binary_crossentropy' , metrics=['accuracy'])
  return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[25,32],
              'epochs':[100,500],
              'optimizer':['adam','rmsprop']}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv=10)

grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_parameters
best_accuracy = grid_search.best_accuracy