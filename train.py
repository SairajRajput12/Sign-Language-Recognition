from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense , Dropout
import os

print("library imported succesfully")

sz = 128

classifier = Sequential() 

# First convolution network and pooling


# First convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Second convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation='relu'))# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# # extra
# classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# # input_shape is going to be the pooled feature maps from the previous convolution layer
# classifier.add(MaxPooling2D(pool_size=(2, 2)))


# Flattening the layersclassifier.add(Flatten())


# Adding a fully connected layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=3, activation='softmax')) # softmax for more than 2
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import os

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

sz=128


test_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory('Dataset/Train',
                                                 target_size=(sz, sz),
                                                 batch_size=27,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('Dataset/Test',
                                            target_size=(sz , sz),
                                            batch_size=27,
                                            color_mode='grayscale',
                                            class_mode='categorical')

classifier.fit(
        training_set,
        epochs=40,
        verbose=1, 
        validation_data=test_set,
)# No of images in test set