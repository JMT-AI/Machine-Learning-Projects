<h4> <strong> 
   <i> Bonjour, je suis Jean-Martial Tagro, data scientist. Cette page Web donne un aperçu de mes principaux projets de science des données traitant divers problèmes d'optimisation. Un lien github associé à chaque projet vous permettra d'en savoir plus.🙂</i>
   </strong> </h6>


<h2><a href=''> 1. Prédiction de loyer d'appartements des arrondissements de Paris </a></h2>
Le but de ce travail est de trouver le meilleur model pour prédire le loyer d'un appartement, connaissant sa surface et son arrondissement dans Paris. On considèrera dans un premier temps l'evaluation dans le cas de la baseline (la régression linéaire avec une seule feature : la surface) puis nous améliorerons les performances en considérant des features bidimentionnels : la surface et l'arrondissement.
<a href =''> En savoir plus </a>


<h2><a href=''>2. Le MNIST </a></h2>
Il s'agit d'un dataset très célèbre, appelé MNIST. Il est constitué d'un ensemble de 70000 images 28x28 pixels en noir et blanc annotées du chiffre correspondant (entre 0 et 9). L'objectif de ce jeu de données était de permettre à un ordinateur d'apprendre à reconnaître des nombres manuscrits automatiquement (pour lire des chèques par exemple). Ce dataset utilise des données réelles qui ont déjà été pré-traitées pour être plus facilement utilisables par un algorithme.
OBJECTIF : Entraîner un modèle qui sera capable de reconnaître les chiffres manuscrits.

   <h3><a href=''>2.1. Étude du K-NN sur le jeu de données MNIST</a></h3>

   <h3><a href=''>2.2. Clustering des images du MNIST</a></h3>


<h2><a href=''> 3. Ré-implémentation de GridSearchCV de la librairie scikit-learn pour la prédiction de la qualité du vin </a></h2>
Il s'agit d'une ré-implémentation de la fonction de validation croisée de la libraire scikit-learn (la fonction GridSearchCV), dans l’objectif d’effectuer la classification d'un dataset sur la qualité du vin.
Attentes : L’algorithme devra permettre d’optimiser l’accuracy du modèle. La fonction prendra en entrée le tableau des hyperparamètres à tester ainsi que le nombre de folds. On utilisera des folds exacts (non randomisés) afin de pouvoir comparer les résultats.
Il s'agira ensuite de comparer les performances de mon implémentation par rapport à l’implémentation scikit-learn. Pour cela, le professeur, Chloé-Agathe Azencott (Chargée de recherche au CBIO de MINES ParisTech & Institut Curie, enseignante à CentraleSupélec - Machine learning & bioinformatique), conseille dans un premier temps de ne pas randomiser la sélection des sets, mais de faire une sélection exacte afin de pouvoir comparer des résultats qui doivent être identiques entre mon implémentation et celle de scikit-learn.
<a href =''> En savoir plus </a>


<h2><a href=''> 4. Kaggle Competition : Leaf Classification - Classement automatique des feuilles d’arbres </a></h2>
Le but de cette étude lancée par Kaggle est de construire le meilleur classifieur multi-classes pour catégoriser (espèces) automatiquement les feuilles, compte tenu de leurs caractéristiques : 3 vecteurs de dimension 64 (marge, forme & texture).
Dataset source and description : <a href='https://www.kaggle.com/c/leaf-classification/data'> Kaggle</a>
<a href =''> En savoir plus </a>


<h2><a href=''> 5. SVM à noyau pour la prédiction de la qualité de vin </a></h2>
On va entrainer un algorithme svm.SVC à classifier nos vins (bonne qualité ou pas terrible) en fonction des caractéristiques physico-chimiques.<br>
Data source : <a href='https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality'>Archive UCI</a> 
<a href =''> En savoir plus </a>


<h2><a href=''> 6. Régression Ridge à noyau pour la prédiction de la qualité de vin </a></h2>
On va maintenant entrainer un algorithme de régression ridge à noyau. Nous allons utiliser les données concernant les caractéristiques physico-chimiques de vins blancs portugais disponibles sur <a href='https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality'>l'archive UCI</a>. Il s'agit ici de prédire le score (entre 3 et 9) donné par des experts aux différents vins.
<a href =''> En savoir plus </a>


<h2><a href=''> 7. Human Activity Recognition Using Smartphones Data - Mesure de la puissance des forêts aléatoires </a></h2>
From Kaggle : <a href='https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones'>Human Activity Recognition Using Smartphones Data Set</a>

Dans ce projet Kaggle, on va appliquer l’algorithme des forêts aléatoires sur un dataset réel : le <a href='https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones'>Human Activity Recognition Using Smartphones Data Set</a>. Il permet de reconnaître l’activité physique à partir de données du smartphone. Ce jeu de donnée à peu d'echantillons mais possède de nombreuses variables (plus de 500 variables). Cette étude consistera dans un 1er temps en une modélisation 'brute' (sans features engineering) puis dans un second temps avec features engineering. Nous essayerons de trouver un compromis performance / temps de calcul.
<a href =''> En savoir plus </a>


<h2><a href=''> 8. Implementation du VGG-16 (CNN) et mise en oeuvre du Transfert Learning pour la Computer Vision </a></h2>
L'objet de notre étude est le célèbre VGG-16, une version du réseau de neurones convolutif très connu appelé VGG-Net. Nous allons d'abord l'implémenter de A à Z pour découvrir <a href='https://keras.io/api/'> Keras </a>, puis nous allons voir comment classifier des images de manière efficace. Pour cela, nous allons exploiter le réseau VGG-16 pré-entraîné fourni par Keras, et mettre en oeuvre le Transfer Learning.
<a href =''> En savoir plus </a>


<h2><a href=''> 9. Etude de Churn via un ANN (Réseaux de Neurones Artificiels) </a></h2>
Dans cette étude, il s'agit de trouver la meilleure architecture de réseau de neurones artificiels (garantissant bien sûr un temps de calcul raisonnable sur un PC moyen) pour prédire le churn d'une banque fictive.
<a href =''> En savoir plus </a>


<h2><a href=''> 10. Classification binaire d'animaux domestiques via un CNN (Réseaux de Neurones à Convolution) </a></h2>
On va dans ce sujet mettre en place un système capable de reconnaître la photo d'un chat et d'un chien avec un taux d'erreur de prédiction le plus faible possible. Nous utiliserons une fonction de Keras pour générer beaucoup plus d'images pour bien entrainer notre modèle.
<a href =''> En savoir plus </a>


<h2><a href=''> 11. Prédiction (Tentative de modélisation de tendance) du prix de l'action Google via un RNN (Réseaux de Neurones Récurrents). </a></h2>
Bien que le prix d'une action soit une métrique difficile à prédire, nous allons aller plus loin en exploitant la puissance des réseaux de neurones récurrents sur les données du prix de l'action Google entre 2012 et 2016 pour faire une prédiction sur 2017. L'objectif ici n'est pas de prédire la valeur de l'action Google sur 2017 mais de prédire sa <strong> tendance </strong> sur 2017. 
<a href =''> En savoir plus </a>


<h2><a href=''> 12. DEEP LEARNING HYBRIDE : Implémentation d'une Carte Auto Adaptive pour la détection de fraude dans une enquête bancaire </a></h2>
Dans cette étude, il s'agit d'une banque qui a reçu d'une partie de ses clients des demandes de cartes de crédit. Pour soumettre leur demande, les clients doivent remplir un formulaire précisant la raison de la demande et différentes informations. La banque, qui a su intégrer le machine learning dans sa stratégie, veut pondérer des décisions (donner ou non la carte de crédit au client après le verdict d'experts sur l'intention du client c'est-à-dire s'il est un fraudeur ou non) avec les informations suggérées par ses données. 
Pour découvrir ces 'informations cachées' nous mettrons en oeuvre premièrement un algorithme de deep learning non-supervisé : la SOM (Self Organizing Map) ou carte auto-adaptative pour découvrir les clients les plus susceptibles de frauder puis deuxièmement nous passerons du non-supervisé au supervisé pour faire des prédictions sur des données de test.
<a href =''> En savoir plus </a>


<h2><a href=''> 13. Mise en place d'un système de recommandation grâce à une Machines de Boltzmann </a></h2>
Ce travail concerne l'implémentation d'un système de recommandation de films via une machine de Boltzmann. L'objet est d'entraîner une machine de Boltzmann sur un jeu de données (voir <a href = 'https://grouplens.org/datasets/movielens/'> MovieLens </a>) très lourd (25 millions de films notés par les utilisateurs). On va dans un 1er temps se focaliser sur une portion de ce dataset (1 millions de films) pour construire notre machine de Boltzmann en vue de recommander aux clients des films de manière pertinente.
<a href =''> En savoir plus </a>   


<h2><a href=''> 14. Mise en place d'un auto encodeur empilé * - Système de scoring de films pour la recommandation </a></h2>
<i>* Les auto-encodeurs sont une technique de Deep Learning assez récente.</i><br>
L'objet de ce projet est la mise en place d'un système de scoring de films via un auto encodeur empilé entrainé sur le dataset publié sur <a href = 'https://grouplens.org/datasets/movielens/'> MovieLens </a>. Le système est mis en oeuvre avec <a href='https://pytorch.org/resources/'>PyTorch</a>.
<a href =''> En savoir plus </a>


<h2><a href=''> 15. Analyse de sentiments (NLP) via 3 approches : Sklearn simple, Pytorch simple, Pytorch-LSTM TORCH (en cours...) </a></h2>
Il s'agit de l'analyse d'un dataset de sentiments composé de quelques millions d'avis clients Amazon (texte d'entrée) et d'étoiles (étiquettes de sortie).
Ce dataset constitue de vraies données commerciales à une échelle raisonnable mais peut être appris en un temps relativement court sur un ordinateur portable modeste. Dans le dataset, label 1 : sentiment positif ; label 2 : sentiment négatif.<br>
Source : voir <a href='https://www.kaggle.com/bittlingmayer/amazonreviews?select=test.ft.txt.bz2'>Kaggle</a>
<br>Evidement, les 3 approches étudiées sont indépendantes.
<a href =''> En savoir plus </a>

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
