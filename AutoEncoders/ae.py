# Auto-Encoder - Architecture Diabolo
# -----------------------------------Jean-Martial Tagro

# ------------------------------------------------------------------------
# SYSTEME DE RECOMMANDATION DE FILM                                       |
# Prédiction de la note de film par un Auto-Encodeur Empilé               |                        |
# ------------------------------------------------------------------------
 
# Librairies
import pandas as pd
import numpy as np
import torch # pythorch library
import torch.nn as nn
import torch.nn.parallel as parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


# --------------------------
# PART 1 : DATA PREPARATION | 
# --------------------------

# Importation données
movies = pd.read_csv('ml-1m/movies.dat', sep='::', 
                     header=None,
                     engine='python',
                     encoding='latin-1')

users = pd.read_csv('ml-1m/users.dat', sep='::', 
                     header=None,
                     engine='python',
                     encoding='latin-1')

ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', 
                     header=None,
                     engine='python',
                     encoding='latin-1')

# Train and Test Split - 1st with 100k notes
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t', header=None)
training_set = np.array(training_set, dtype='int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t', header=None)
test_set = np.array(test_set, dtype='int')


# Construction de la Matrice pour le système de recommandation
# ------------------------------------------------------------

# Obtention du nombre de users et du nombre de films
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))



def convert(data):
    """ Conversion des donneés en matrice
    Pour chaque user, on affiche en 'features' les notes de 
    tous les films vu ou non
    """
    new_data = []
    
    for id_user in range(1, nb_users + 1):
        
        id_movies_seen  = data[data[:,0] == id_user, 1] #filtrage
        ratings_given   = data[data[:,0] == id_user, 2] #filtrage
        
        # creation de la liste de tous les movies avec notes du user
        user_ratings                     = np.zeros(nb_movies)
        user_ratings[id_movies_seen - 1] = ratings_given   
        
        new_data.append(list(user_ratings))
        
    return new_data

matrix_train = convert(training_set)
matrix_test = convert(test_set)

    
# Conversion des data en type 'tensor' pour pytorch
#--------------------------------------------------
matrix_train = torch.FloatTensor(matrix_train) 
matrix_test = torch.FloatTensor(matrix_test) 



# --------------------------
# PART 2 : RECOMMANDATIONS  |  Recommendation multiclasses (Scores)
# --------------------------

# CONSTRUCTION DE L"ARCHITECTURE DE L'AUTO-ENCODEUR (Stacked Auto-Encoder)

class SAE(nn.Module):
    
    def __init__(self):
        
        super(SAE, self).__init__()
        # Ajout, empilement des couches
        self.fc1 = nn.Linear(nb_movies, 20) # entree : nb_movies noeuds, 1er caché : 20 noeuds
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        
        # Definition activation function
        self.activation = nn.Sigmoid()
        
        
    # Fonction to encode/decode information et transferer
    def forward(self, x):
        
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x) # specificity : none activation on last layer (-> entry layer)
        
        return x

# instanciation
sae = SAE()

criterion = nn.MSELoss # cost function to get RMSE
optimizer = optim.RMSprop(sae.parameters(),     # for gradient algorithm
                          lr = 0.1,             # learning rate
                          weight_decay = 0.5)   # for ~ adjust lr ?


# Entraînement du SAE

nb_epoch = 200          # to test..where there's convergence

for epoch in range(1, nb_epoch + 1):
    
    train_loss = 0
    s = 0.    # counter to normalise after error train_loss
    
    for id_user in range(nb_users):
        
        inputs = Variable(matrix_train[id_user]).unsqueeze(0)
        target = inputs.clone()
        
        if (torch.sum(target.data > 0) > 0) :
            
            outputs = sae(inputs) # SAE() has one function so we can pass directly data x
            
            target.require_grad = False  # not apply gradient algo on target, only on input
            
            outputs[target == 0] = 0     # keep to 0 film where user didn't note

            # cost calculation
            loss = criterion(outputs, inputs)
            
            
            # Error factor corrector : .. users noted not the same number of movies
            # a user with error = 1 on ALL movies is less dangerous than 
            # a user with error = 1 on HALF movies
            mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 10**-10) # 10**-10 to avoid /0

            # GRADIENT : to specify direction for adjust weights 
            loss.backward()
            
            train_loss += np.sqrt(loss.data[0] * mean_corrector)

            s += 1.
            
            optimizer.step() # intensity of weight update

    print('Epoch : {} / Loss : {}'.format(epoch, train_loss / s))


# Test du SAE
    
test_loss = 0
s = 0.
for id_user in range(nb_users):
    inputs = Variable(matrix_train[id_user]).unsqueeze(0)
    target = Variable(matrix_test[id_user]).unsqueeze(0)
    if (torch.sum(target.data > 0) > 0) :
        outputs = sae(inputs) 
        target.require_grad = False  # not apply gradient algo on target, only on input
        outputs[target == 0] = 0     # keep to 0 film where user didn't note
        loss = criterion(outputs, inputs)
        mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 10**-10) # 10**-10 to avoid /0

        test_loss += np.sqrt(loss.data[0] * mean_corrector)
        s += 1.

print('Epoch : {} / Loss : {}'.format(epoch, test_loss / s))



