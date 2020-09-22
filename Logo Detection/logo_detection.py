import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Flatten
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from glob import glob
from keras.models import Model

image_size = [224,224]

vgg = VGG16(input_shape = image_size + [3], weights = 'imagenet', include_top = False)

for layer in vgg.layers:
    layer.trainable = False
    
folder = glob('datasets/training_set/*')
print(len(folder))

x = Flatten()(vgg.output)

prediction = Dense(len(folder), activation = 'softmax')(x)

model = Model(inputs = vgg.input, outputs = prediction)

model.summary()

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                   metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('datasets/training_set',
                                                 target_size = (224,224),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('datasets/test_set',
                                            target_size = (224,224),
                                            batch_size = 32,
                                            class_mode = 'binary')

history = model.fit_generator(training_set,
                         steps_per_epoch = (30),
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = (10))




