import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

# dataset files
dataset_dir = "./driving_data"
dataset_csv_file = dataset_dir + "/driving_log.csv"
additional_dataset_dir = "./additional_driving_data"
additional_dataset_csv_file = dataset_dir + "/driving_log.csv"

# read ground truth
lines = []
with open(dataset_csv_file) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

with open(additional_dataset_csv_file) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


# read training data
images = []
steering_measurements = []
for line in lines:
    img_center_path = line[0]
    img_left_path = line[1]
    img_right_path = line[2]
    img_center = cv2.imread(img_center_path)
    img_left = cv2.imread(img_left_path)
    img_right = cv2.imread(img_right_path)
    steering_angle_center = float(line[3])
    correction = 0.2
    steering_angle_left = steering_angle_center + correction
    steering_angle_right = steering_angle_center - correction
    images.append(img_center)
    steering_measurements.append(steering_angle_center)
    images.append(img_left)
    steering_measurements.append(steering_angle_left)
    images.append(img_right)
    steering_measurements.append(steering_angle_right)
    # augment training data
    img_center_flipped = np.fliplr(img_center)
    steering_angle_center_flipped = -steering_angle_center
    img_left_flipped = np.fliplr(img_left)
    steering_angle_left_flipped = -steering_angle_left
    img_right_flipped = np.fliplr(img_right)
    steering_angle_right_flipped = -steering_angle_right
    images.append(img_center_flipped)
    steering_measurements.append(steering_angle_center_flipped)
    images.append(img_left_flipped)
    steering_measurements.append(steering_angle_left_flipped)
    images.append(img_right_flipped)
    steering_measurements.append(steering_angle_right_flipped)


# create training data
X_train = np.array(images)
Y_train = np.array(steering_measurements)

# build a simple regression network for testing data
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=X_train[0].shape))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Dropout(0.2))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
history_object = model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=20)

model.save("./model_nvidia_augmented_data_cropped_multicam_additional_data_10.h5")

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
