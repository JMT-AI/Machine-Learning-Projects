# Artificial Neural Network 

"Implementation of an ANN to predict Churn - - - Jean-Martial Tagro"

# --------------------------
# PART 1 : DATA PREPARATION | 
# --------------------------
 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('data/Churn_Modelling.csv')

# creation des array numpy pour les features et la target
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data 
# Encoding the Independent Variable

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# One Hot Encoding ==> dummy variable 
# -----------------------------------
# getting dummy variables in a frame format
geography_dummies = pd.get_dummies(dataset['Geography'])
# remove one demmy column to avoid colinear issue 
geography_dummies = geography_dummies.drop('France', axis=1)

# convert in a array numpy
geography_dummies = geography_dummies.values

# remove categorical features and insert dummies columns in X numpy
X_dum = np.delete(X, 1, 1)
X_dum = np.insert(X_dum, 1, geography_dummies[:,0], 1)
X_dum = np.insert(X_dum, 2, geography_dummies[:,1], 1)

# -----------------------------------

# Feature Scaling - Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dum)

# train / test 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)


# -------------------------------
# PART 2 : CONSTRUCTION DE L'ANN |
# ------------------------------

# importation
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialisation du réseau de neuronne
classifier = Sequential()

# Ajouter la couche d'entree et la couche cachée
classifier.add(Dense(
    units=6, 
    activation='relu', # fonction d'activation Redresseur
    kernel_initializer='uniform', 
    input_dim=11)) # 1st layer

# to avoid overfitting ..
classifier.add(Dropout(rate=0.1)) # 10% of chance'desactivation for each 6 neuronnes

# Ajouter une 2e couche cachée 
classifier.add(Dense(
    units=6, 
    activation='relu', 
    kernel_initializer='uniform'))

classifier.add(Dropout(rate=0.1))

# Ajouter la couche de sortie
classifier.add(Dense(
    units=1, 
    activation='sigmoid', # to get probability 
    kernel_initializer='uniform'))

# compiler le reseau de neuronnes
# 'adam' : gradient stochastique
classifier.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy'])


# Entrainer le reseau de neuronnes 
classifier.fit(X_train, y_train, batch_size=10, epochs=100)


# ----------------------
# PART 3 : PREDICTIONS |
# ----------------------


# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Convert proba to binary 0-1 with a threshold = 50%
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Autre test prediction d'un client 
"""Pays : France
Score de crédit : 600
Genre : Masculin
Âge : 40 ans
Durée depuis entrée dans la banque : 3 ans
Balance : 60000 €
Nombre de produits : 2
Carte de crédit ? Oui
Membre actif ? : Oui
Salaire estimé : 50000 €"""

data_client = np.array([[600, 0, 0, 1, 40, 3, 60000, 2, 1, 1, 50000]])
data_client = scaler.fit_transform(data_client)

y_pred_client1 = classifier.predict(data_client)
y_pred_client1 = (y_pred_client1 > 0.5) 


# --------------------------------
# PART 4 : CROSS VALIDATION      |
# --------------------------------
# -------for bias-variance tradeoff

from keras.wrappers.scikit_learn import KerasClassifier # Bridge Keras <-> Sklearn 
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, activation='relu', 
                         kernel_initializer='uniform', input_dim=11)) # 1st layer
    classifier.add(Dense(units=6, activation='relu', 
                         kernel_initializer='uniform'))
    classifier.add(Dense(units=1, activation='sigmoid', # to get probability 
                         kernel_initializer='uniform'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', 
                       metrics=['accuracy'])
    

    return classifier 

# k-folfs cross validation
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
precisions = cross_val_score(
                            estimator = classifier, 
                            X = X_train,
                            y = y_train,
                            cv = 10,
                            n_jobs = -1)

score_moyenne = precisions.mean()
score_ecart_type = precisions.std()



# ---------------------------------------
# PART 5 : HYPER-PARAMETRES OPTIMIZATION |
# --------------------------------------

from keras.wrappers.scikit_learn import KerasClassifier # Bridge Keras <-> Sklearn 
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, activation='relu', 
                         kernel_initializer='uniform', input_dim=11)) # 1st layer
    classifier.add(Dense(units=6, activation='relu', 
                         kernel_initializer='uniform'))
    classifier.add(Dense(units=1, activation='sigmoid', # to get probability 
                         kernel_initializer='uniform'))
    
    classifier.compile(optimizer = optimizer, 
                       loss='binary_crossentropy', 
                       metrics=['accuracy'])

    return classifier 

# k-folfs cross validation
classifier = KerasClassifier(build_fn=build_classifier)

params = {"batch_size": [25, 32],
          "epochs": [100, 500],
          "optimizer": ['adam', 'rmsprop']}

classifier_grid = GridSearchCV(classifier,
                               params,
                               scoring = 'accuracy',
                               cv = 5,
                               n_jobs = -1)

classifier_grid.fit(X_train, y_train)

best_hyperparameters = classifier_grid.best_params_
best_precision = classifier_grid.best_score_

