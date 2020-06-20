
# Ce script montre un exemple pratique de l'overfit
# Le premier ANN entraîné a une structure complexe avec beaucoup de neurones
# Du coup il arrive à obtenir une très bonne précision sur le jeu d'entraînement (environ 96%)
# Mais il n'arrive pas à généraliser sur le jeu de test (précision à 81%)
# C'est un cas classique d'overfitting (surentraînement)
# 
# Le deuxième ANN utilise Dropout pour réduire ce problème.
# La précision d'entraînement est plus faible (90%) et 
# la précision de test a augmenté (à 85%)
# On a toujours un problème de surentraînement, mais bien mondre.

# Import libraries
import pandas as pd

# Import data
dataset = pd.read_csv('data/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

# Encode categorical data and scale continuous data
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
preprocess = make_column_transformer(
        (OneHotEncoder(), ['Geography', 'Gender']),
        (StandardScaler(), ['CreditScore', 'Age', 'Tenure', 'Balance',
                            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
                            'EstimatedSalary']))
X = preprocess.fit_transform(X)

# Split in train/test
y = y.values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# ANN - Overfitting
classifier = Sequential()
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))
classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
score_train = classifier.evaluate(X_train, y_train) # 96.6%
score_test = classifier.evaluate(X_test, y_test) # 80.7%

# ANN - Dropout

classifier = Sequential()
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
score_train = classifier.evaluate(X_train, y_train) # 90.6%
score_test = classifier.evaluate(X_test, y_test) # 85.7%













