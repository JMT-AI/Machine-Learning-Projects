#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 18:18:52 2020  @author: jean-martial
"""

#                           ************************
#                          *  DEEP LEARNING HYBRIDE  *
#                           ************************

# -------------------------------------------------------------------------
# NON SUPERVISÉ - Carte auto-adaptative pour la detection de fraude
# -------------------------------------------------------------------------

""" Il s'agit ...
"""
 
# importations et preparation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

dataset = pd.read_csv("Credit_Card_Applications.csv")
#dataset.head()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)


# Entrainement
from minisom import MiniSom

# pour creer la carte, on choisit ici 10x10=100 valeurs (~suffisant pour 690 observations ...)
som = MiniSom(x = 10, y = 10, input_len = 15) # 15 dimensions

# initialiser aleatoirement les poids
som.random_weights_init(X_scaled)
som.train_random(X_scaled, num_iteration = 100)

# Visualisation des résultats
from pylab import plot, colorbar, pcolor, show, bone
bone() # initialisation graphe
#pcolor(som.distance_map().T)
colorbar(pcolor(som.distance_map().T)) # add a color bar

markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X_scaled):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = "None",
         markersize = 10,
         markeredgewidth = 2)
    
show()

# Detection des fraudeurs
mappings = som.win_map(X_scaled)
fraudeurs = np.concatenate((mappings[(3,4)], mappings[(4,8)],
                            mappings[(5,8)]), axis=0)
fraudeurs = scaler.inverse_transform(fraudeurs) # use '%.8g' format in table display


# -------------------------------------------------------------------------
# NON SUPERVISÉ --> SUPERVISÉ : Réseau de neurones artificiel
# -------------------------------------------------------------------------

# Creation de la matrice de variable
customers = dataset.iloc[:, 1:].values

# Creation du vecteur de target
is_fraud = np.zeros(len(dataset))
for i in range(dataset.shape[0]):
    if dataset.iloc[i, 0] in fraudeurs:
        is_fraud[i] = 1
        
        
# Feature Scaling - Standardization
sc = StandardScaler()
customers = sc.fit_transform(customers)


# --> CONSTRUCTION DE L'ANN |

# importation
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialisation du réseau de neuronne
classifier = Sequential()

# Ajouter la couche d'entree et la couche cachée
classifier.add(Dense(
    units=2, 
    activation='relu', # fonction d'activation Redresseur
    kernel_initializer='uniform', 
    input_dim=15)) # 1st layer


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
classifier.fit(customers, is_fraud, batch_size=1, epochs=2)


# Predicting the Test set results
y_pred = classifier.predict(customers) # normally do on test data, not train

y_pred = np.concatenate((dataset.iloc[:, 0:1], y_pred), axis=1)

# to order
y_pred = y_pred[y_pred[:, 1].argsort()]
