#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 18:18:52 2020

@author: jean-martial
"""

# -------------------------------------------------------------------------
# Self Adapting Map - Carte auto-adaptative pour la detection de fraude
# -------------------------------------------------------------------------

""" Il s'agit d'analyser les clients d'une banque (depart sur un échantillon de 
690 clients) qui demande à avoir une carte de crédit et de détecter de potentiels
fraudeurs sur la base des informations remplies dans le formulaire de demande. 
"""

# importations et preparation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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
fraudeurs = np.concatenate((mappings[(7,6)], mappings[(8,7)]), axis=0)
fraudeurs = scaler.inverse_transform(fraudeurs) # use '%.8g' format in table display






