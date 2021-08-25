import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import cv2
import csv

lines = []
images = []
light_colors = []

source = r'/home/workspace/CarND-Capstone/traffic_light_imgs/'

# Import csv file with traffic light images and labels
with open('traffic_lights.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(csvfile) # skip first line with headers
    for line in reader:
        lines.append(line)

# Importing images and labels from csv
for line in lines:
    path = source + line[0] + '.jpg'
    color_label = line[1]

    if color_label == 'red':
        color_id = 0
    if color_label == 'yellow':
        color_id = 1
    if color_label == 'green':
        color_id = 2

    # Add images for traffic lights to training dataset
    light_img = cv2.imread(path)
    light_img = cv2.resize(light_img, (48,108), interpolation = cv2.INTER_AREA)
    images.append(light_img)
    light_colors.append(color_id)
    
    # Flip images to expand dataset
    images.append(np.fliplr(light_img))
    light_colors.append(color_id)
    
X_train = np.array(images)
print("X train shape:", X_train.shape)
y_train = np.array(light_colors)

# Switch to one hot encoding
y_encoded = to_categorical(y_train)
print("Y train shape:", y_encoded.shape)

# Build training model
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.applications.vgg16 import VGG16

model = Sequential()

# Add layers to pretrained model
model.add(VGG16(include_top = False, input_shape = (108,48,3)))
model.add(Flatten())
model.add(Dense(3, activation="softmax"))

# Use categorical cross entropy for classification
model.compile(loss='categorical_crossentropy', optimizer='adam')
history_object = model.fit(X_train, y_encoded, validation_split = 0.2, shuffle = True, epochs=10)

# Print the keys contained in the history object
print(history_object.history.keys())

model.save('lights_model.h5')
model.summary()

# Evaluate accuracy of the keras model
accuracy = model.evaluate(X_train, y_encoded)
print('Accuracy: %.2f' % (accuracy*100))