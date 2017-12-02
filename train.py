import csv
import cv2
import numpy as np


# dataset files
dataset_dir = "./driving_data"
dataset_csv_file = dataset_dir + "/driving_log.csv"

# read ground truth
lines = []
with open(dataset_csv_file) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


# read training data
images = []
steering_measurements = []
for line in lines:
    img_path = line[0]
    img = cv2.imread(img_path)
    images.append(img)
    steering_angle = float(line[3])
    steering_measurements.append(steering_angle)

# create training data
X_train = np.array(images)
Y_train = np.array(steering_measurements)

# build a simple regression network for testing data
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=X_train[0].shape))
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save("./model_lenet.h5")
