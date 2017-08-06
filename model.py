import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D,MaxPooling2D, Dropout, Activation
from keras.layers.convolutional import Convolution2D
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#load CSV
lines = []
with open('./data/driving_log.csv') as csvfile:
    # Skip first header line
    next(csvfile, None)
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#print(len(lines))
#create train and validation set
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

#generator function which yields batches of augmented images and measurements
def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction1 = 0.2
    correction2 = 0.2
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            #read images from center, left and right camera
            #and calculate left and right measurements
            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    # rename the path
                    name = batch_sample[i]
                    # data coming from different directories
                    if name[1] =='I' or name[1] =='M':
                        current_path = './data/IMG/'+batch_sample[i].split('/')[-1] 
                    else:
                        current_path = './data/IMG/'+batch_sample[i].split('\\')[-1] 
#                        print (name[1], current_path)
                    image = mpimg.imread(current_path)
                    measurement = float(batch_sample[3])  
                    if i == 1:
                        measurement = measurement + correction1
                    elif i == 2:
                        measurement = measurement - correction2
                    images.append(image)
                    measurements.append(measurement)

                    # save example images
                    image_save = cv2.imread(current_path)
                    if i==1:
                        cv2.imwrite("image1.png",image_save)
                    elif i==2:
                        cv2.imwrite("image2.png",image_save)
                    else:
                        cv2.imwrite("image0.png",image_save)


            #augment the data by flipping the images and taking the opposite sign of the measurements
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)
                
            # save example images
            image_save = cv2.imread(current_path)
            cv2.imwrite("image3.png",image_save)
            cv2.imwrite("image4.png",cv2.flip(image_save,1))

            #create numpy arrays
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# create the generator objects used by model.fit_generator()
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
print (len(train_samples))

#build the model
model = Sequential()

#normalization using a lambda layer
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

#chose area of interest by cropping images
model.add(Cropping2D(cropping=((70,25),(0,0))))

#use NVIDIA architecture (4 convolution layers and 3 fully connected layers)
model.add(Convolution2D(24,5,5, activation="relu"))
model.add(MaxPooling2D())

model.add(Convolution2D(36,5,5, activation="relu"))
model.add(MaxPooling2D())

model.add(Convolution2D(48,5,5, activation="relu"))
model.add(MaxPooling2D())

model.add(Convolution2D(64,3,3, activation="relu"))

model.add(Flatten())

model.add(Dense(150))
model.add(Dense(50))
model.add(Dense(10))

# Apply dropout of 25%
model.add(Dropout(0.25))

model.add(Dense(1)) # The output is only the steering angle.

#compile, train and save the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= 
            len(train_samples), validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=12, verbose=1)

model.save('model.h5')
