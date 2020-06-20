#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:10:54 2020

@author: jean-martial
"""
# -------------------------------------------------------------------------
# Recurrent Neural Networks 
# -------------------------------------------------------------------------

# importations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

# --------------------------
# PART 1 : DATA PREPARATION | 
# --------------------------

dataset = pd.read_csv('Google_Stock_Price_Train.csv')
#dataset.head()

X = dataset.iloc[:,1].values
X = X.reshape(-1, 1) # shape (n,) to (n,1)


# features scaling
sc = MinMaxScaler(feature_range = (0, 1))

X_scaled = sc.fit_transform(X)

# creation de la structure avec 60 timesteps et 1 sortie

X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(X_scaled[(i-60):i, 0])
    y_train.append(X_scaled[i, 0])
    
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# ----------------------------
# PART 2 : CONSTRUTION DU RNN | 
# ----------------------------

# Initalisation
regressor = Sequential()

# Couche LSTM + Dropout
regressor.add(LSTM(units = 50, return_sequences = True,
                         input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# 2e Couche LSTM + Dropout
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# 3e Couche LSTM + Dropout
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# 4e Couche LSTM + Dropout
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2))

# Couche de sortie
regressor.add(Dense(units = 1))
              
# Compilation - 'adam' good enough for several cases, or RMS prop good for RNN, see Keras
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Entrainement
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32,
              use_multiprocessing = True)



# --------------------------------------
# PART 3 : PREDICTIONS ET VISUALISATION | 
# --------------------------------------

# Données de 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
#dataset_test.head()

real_stock_price = dataset_test[['Open']].values


# Preedictions de 2017

#dataset      # train 
#dataset_test # test
dataset_total = pd.concat((dataset['Open'], dataset_test['Open']), axis = 0)

inputs = dataset_total[dataset_total.shape[0] - dataset_test.shape[0] - 60 : ].values

# Standardisation of inputs : Attention : use same normaliser for all (transform 
inputs = sc.transform(inputs.reshape(-1,1))

X_test = []
for i in range(60, 80):
    X_test.append(inputs[(i-60):i, 0])
    
X_test = np.array(X_test)

# Reshaping, add 1 more dimension
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualisation
plt.plot(real_stock_price, color='red', label="Prix réels de l'action Google")
plt.plot(predicted_stock_price, color='blue', label="Prix prédits de l'action Google")
plt.title("Action Google")
plt.xlabel("Jours de Janvier 2017")
plt.ylabel("Prix de l'action")
plt.legend()
plt.show()



