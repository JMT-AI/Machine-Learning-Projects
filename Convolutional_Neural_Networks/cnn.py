# Convolutional Neural Network

# - Learn to recognize dog and cat - Jean-Martial Tagro

# --------------------------
# PART 1 : DATA PREPARATION | 
# --------------------------
# Here in CNN we just need a 'correct' structure of data folder of images, which is 
# ging to be recognized and used by Keras (except we have to scale data too)


# importations des modules
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# -----------------------------
# ETAPE 1 : INITIALISER le CNN |
# -----------------------------

classifier = Sequential()

# -----------------------------
# ETAPE 2 : CONVOLUTION        |
# -----------------------------

classifier.add(Conv2D(
    filters = 32,
    kernel_size = 3, # W =3px, H = 3px
    strides = 1,
    input_shape = (64, 64, 3),
    activation = 'relu'))

# -----------------------------
# ETAPE 3 : POOLING            |
# -----------------------------
classifier.add(MaxPooling2D(pool_size = (2,2)))


# --------------------- AMELIORATIONS
# To better perform, we can Deep the CNN by adding here :
#    - a Conv Layer
 #   - a Pool Layer
# ---------------------


# -----------------------------
# ETAPE 4 : FLATTENING         |
# -----------------------------
classifier.add(Flatten())

# ------------------------------
# ETAPE 5 : Full connected layer|
# ------------------------------
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# ------------------------------
# ETAPE 6 : COMPILATION         |
# ------------------------------
classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy', # categorical_crossentropy for multi-class
                   metrics = ['accuracy'])

# ------------------------------
# ETAPE 7 : ENTRAINEMENT DU CNN |
# ------------------------------
# including scaling, data(image) augmented and training with cross val by Keras

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255, # Scaling du train [0,1]
        shear_range=0.2, # transvection
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255) # Scaling du testset

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size = (64, 64),
        batch_size = 32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch = 250, # 8000/32 - 8000 images train
        epochs = 25,
        validation_data = test_set,
        validation_steps = 63,
        use_multiprocessing=True) # IMPORTANT if False, kernel died !!


# accuracy: 0.8794 - val_accuracy: 0.7675


#------------------------------------------------------------------------
# SINGLE PREDICTION - mon chien
#------------------------------------------------------------------------

import numpy as np
from keras.preprocessing import image

# load image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_5.jpg',
        target_size = (64, 64))

# convert color image in right shape and numpy
test_image = image.img_to_array(test_image)

# add 1 dimension at 1st position for the 'group', need for the 'predict' function
test_image = np.expand_dims(test_image, axis=0)

result_vector = classifier.predict(test_image)

# training_set.class_indices

if result_vector[0,0] == 1:
    prediction = 'chien'
else:
    prediction = 'chat'


# --------------------- AMELIORATIONS 
# To better perform, we can :
#    rise images size px x px
#    rise number of epochs
#    Using Dropout to desactivate some neurons, avoid overfiting
# ...
# ---------------------

