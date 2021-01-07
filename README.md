<h2><a href=''> Étude d'un modèle de prédiction de nombre de vélos loués à chaque heure </a></h2>
Le projet consiste à prévoir le nombre de vélos loués à chaque heure dans des bornes libres-services de la ville (système type Vélib'). La variable cible est ici la variable count.
Voici un descriptif de l'ensemble des variables :

- datetime - date et heure du relevé
- season - 1 = printemps , 2 = été, 3 = automne, 4 = hiver
- holiday – indique si le jour est un jour de vacances scolaires
- workingday - indique si le jour est travaillé (ni week-end ni vacances)
- weather - 1: Dégagé à nuageux, 2 : Brouillard, 3 : Légère pluie ou neige, 4 : Fortes averses ou neiges
- temp – température en degrés Celsius
- atemp – température ressentie en degrés Celsius
- humidity – taux d’humidité
- windspeed – vitesse du vent
- casual - nombre de locations d’usagers non abonnés
- registered – nombre de locations d’usagers abonnés
- count – nombre total de locations de vélos

<h2><a href=''> Prédiction de loyer d'appartements des arrondissements de Paris </a></h2>
Le but de ce travail est de trouver le meilleur model pour prédire le loyer d'un appartement, connaissant sa surface et son arrondissement dans Paris. On considèrera dans un premier temps l'evaluation dans le cas de la baseline (la régression linéaire avec une seule feature : la surface) puis nous améliorerons les performances en considérant des features bidimentionnels : la surface et l'arrondissement.
<a href ='https://github.com/JMT-AI/Portfolio/blob/master/1.%20Prediction%20Loyer%20-%20MNIST/Prediction%20de%20loyer_Livrable.ipynb'> En savoir plus </a>


<h2><a href=''>Le MNIST </a></h2>
Il s'agit d'un dataset très célèbre, appelé MNIST. Il est constitué d'un ensemble de 70000 images 28x28 pixels en noir et blanc annotées du chiffre correspondant (entre 0 et 9). L'objectif de ce jeu de données était de permettre à un ordinateur d'apprendre à reconnaître des nombres manuscrits automatiquement (pour lire des chèques par exemple). Ce dataset utilise des données réelles qui ont déjà été pré-traitées pour être plus facilement utilisables par un algorithme.
OBJECTIF : Entraîner un modèle qui sera capable de reconnaître les chiffres manuscrits.

   <h3><a href='https://github.com/JMT-AI/Portfolio/blob/master/1.%20Prediction%20Loyer%20-%20MNIST/K-NN%20sur%20le%20jeu%20de%20données%20MNIST.ipynb'>2.1. Étude du K-NN sur le jeu de données MNIST</a></h3>

   <h3><a href='https://github.com/JMT-AI/Portfolio/blob/master/6.%20Exploration%20via%20des%20algorithmes%20non%20supervisés/Clustering_MNIST_livrable.ipynb'>2.2. Clustering des images du MNIST</a></h3>


<h2><a href=''> Ré-implémentation de GridSearchCV de la librairie scikit-learn pour la prédiction de la qualité du vin </a></h2>
Il s'agit d'une ré-implémentation de la fonction de validation croisée de la libraire scikit-learn (la fonction GridSearchCV), dans l’objectif d’effectuer la classification d'un dataset sur la qualité du vin.
Attentes : L’algorithme devra permettre d’optimiser l’accuracy du modèle. La fonction prendra en entrée le tableau des hyperparamètres à tester ainsi que le nombre de folds. On utilisera des folds exacts (non randomisés) afin de pouvoir comparer les résultats.
Il s'agira ensuite de comparer les performances de mon implémentation par rapport à l’implémentation scikit-learn.
<a href ='https://github.com/JMT-AI/Portfolio/blob/master/2.%20Evaluations%20KNN/Ré-implémentation%20de%20GridSearchCV_Livrable.ipynb'> En savoir plus </a>


<h2><a href=''> Kaggle Competition : Leaf Classification - Classement automatique des feuilles d’arbres </a></h2>
Le but de cette étude lancée par Kaggle est de construire le meilleur classifieur multi-classes pour catégoriser (espèces) automatiquement les feuilles, compte tenu de leurs caractéristiques : 3 vecteurs de dimension 64 (marge, forme & texture).
Dataset source and description : <a href='https://www.kaggle.com/c/leaf-classification/data'> Kaggle</a>
<a href ="https://github.com/JMT-AI/Portfolio/blob/master/3.%20Entrainement%20de%20modèle%20prédictif%20linéaire/Classement%20feuilles%20d'arbres_Livrable.ipynb"> En savoir plus </a>


<h2><a href=''> SVM à noyau pour la prédiction de la qualité de vin </a></h2>
On va entrainer un algorithme svm.SVC à classifier nos vins (bonne qualité ou pas terrible) en fonction des caractéristiques physico-chimiques.<br>
Data source : <a href='https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality'>Archive UCI</a> 
<a href ='https://github.com/JMT-AI/Portfolio/blob/master/4.%20Etudes%20modèles%20supervisés%20non%20linéaires/Kernel%20SVM%20-%20wine%20quality.ipynb'> En savoir plus </a>


<h2><a href=''> Régression Ridge à noyau pour la prédiction de la qualité de vin </a></h2>
On va maintenant entrainer un algorithme de régression ridge à noyau. Nous allons utiliser les données concernant les caractéristiques physico-chimiques de vins blancs portugais disponibles sur <a href='https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality'>l'archive UCI</a>. Il s'agit ici de prédire le score (entre 3 et 9) donné par des experts aux différents vins.
<a href ='https://github.com/JMT-AI/Portfolio/blob/master/4.%20Etudes%20modèles%20supervisés%20non%20linéaires/Kernel%20ridge%20regression%20-%20wine%20quality.ipynb'> En savoir plus </a>


<h2><a href=''> Human Activity Recognition Using Smartphones Data - Mesure de la puissance des forêts aléatoires </a></h2>
From Kaggle : <a href="https://github.com/JMT-AI/Portfolio/blob/master/5.%20Méthodes%20ensemblistes/Classement%20feuilles%20d'arbre_Random%20forest.ipynb">Human Activity Recognition Using Smartphones Data Set</a>

Dans ce projet Kaggle, on va appliquer l’algorithme des forêts aléatoires sur un dataset réel : le <a href=''>Human Activity Recognition Using Smartphones Data Set</a>. Il permet de reconnaître l’activité physique à partir de données du smartphone. Ce jeu de donnée à peu d'echantillons mais possède de nombreuses variables (plus de 500 variables). Cette étude consistera dans un 1er temps en une modélisation 'brute' (sans features engineering) puis dans un second temps avec features engineering. Nous essayerons de trouver un compromis performance / temps de calcul.
<a href ='https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones'> En savoir plus </a>



### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
