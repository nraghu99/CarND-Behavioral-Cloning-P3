import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.regularizers import l2

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

## Resize the image to what nVidia Neural net archtecture expects i.e. 200,66,3
def resize(image):
    """
    Returns an image resized to match the input size of the network.
    :param image: Image represented as a numpy array.
    """
    return cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)


## Crop the 70 pixels i.e. sky , trees above horizon,
## and the dashoard extension i.e. 25 pixels in the bottom
def crop_image(image):
    """
    Returns an image cropped 70 pixels from top and 25 pixels from bottom.
    :param image: Image represented as a numpy array.
    """
    return image[70:-25,:]


## Adjust brightness randomly, this is more for generalization
def random_brightness(image):
    """
    Returns an image with a random degree of brightness.
    :param image: Image represented as a numpy array.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = .25 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * brightness
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image
## Image pre process
def process_image(image):
    """
    Returns an image after applying several preprocessing functions.
    :param image: Image represented as a numpy array.
    """
    image = random_brightness(image)
    image = crop_image(image)
    image = resize(image)
    return image

# Get the input data, I using udacity data only
def getCsvData(filepath, headerline):
    """
     Reads a csv file and retruns 2 lists i.e. images and steering angles
     input parameter is the path to the csv file
     """
    imagepaths, anglemeasures = [], []
    correction = 0.275
    with open(filepath + '/driving_log.csv') as csvfile:       
        reader =  csv.reader(csvfile)
        if headerline:
            next(reader,None)
        for line in reader:
            angle = float(line[3])
            center_img = filepath + '/IMG/' + line[0].split('/')[-1]
            left_img = filepath + '/IMG/' + line[1].split('/')[-1]
            right_img = filepath + '/IMG/' + line[2].split('/')[-1]

            ## We enhance the sample size by adding images from left and right camera
            ## as mostly we are driving straight, adding right and left camera, will
            ## make training data more uniformyl distributed as these images are turn data
            ## with the correction factor of 0.275 as the steering angle
            imagepaths.append(center_img)
            imagepaths.append(left_img)
            imagepaths.append(right_img)
            anglemeasures.append(angle)
            anglemeasures.append(angle + correction)
            anglemeasures.append(angle - correction)
        return imagepaths, anglemeasures




## The generator code
def generator(X_train,y_train, batch_size=64):
    images = []
    angles = []
    while True: # loop forever
        num_samples = len(X_train)
        skewedCount = 0
        for i in range(batch_size):
            sample_index =  np.random.randint(0,num_samples)
            angle = y_train[sample_index]

            ## As the track is mostly stright, we do not want the sample data to be skewed towards
            ## < 0.1 turning angle, so we will make the data uniform , if we have already
            ## added 40% of low turning angle, we will not add them any more fort this batch
            if abs(angle) < 0.1:
                skewedCount = skewedCount + 1
            if skewedCount > (batch_size * 0.4):
                while abs(y_train[sample_index]) < 0.1:
                    sample_index =  np.random.randint(0,num_samples)
            image = cv2.imread(X_train[sample_index])
            ## cv2 reads in BGR format, change to YUV as nviDia paper suggets
            ## in drive.py , the images comes in RGB format ans we change it to
            ## YUV as well
 
            image = process_image(image)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
            angle = y_train[sample_index]

            ## Flip the image for more uniform distribution base on coin flip
            if np.random.randint(0,2) == 1:
                 image = cv2.flip(image,1)
                 angle = -1.0 * angle
            images.append(image)
            angles.append(angle)
        yield np.array(images), np.array(angles)


## NVidia Architecture with dropouts added for generalization

def nvidiaNet(input_shape, dropout= 0.25):
    model = Sequential()
    ## Normaliza the data
    model.add(Lambda( lambda w: w / 255.0 - 0.5, input_shape=input_shape))
    model.add(Convolution2D(24,5,5, border_mode= 'valid',subsample=(2,2),activation='elu',W_regularizer=l2(0.001)))
    model.add(Dropout(0.1))
    model.add(Convolution2D(36,5,5, border_mode= 'valid',subsample=(2,2),activation='elu',W_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(48,5,5, border_mode= 'valid',subsample=(2,2),activation='elu',W_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64,3,3,border_mode= 'valid',subsample=(1,1),activation='elu',W_regularizer=l2(0.001)))
    model.add(Dropout(dropout))
    model.add(Convolution2D(64,3,3,border_mode= 'valid',subsample=(1,1),activation='elu',W_regularizer=l2(0.001)))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(100,activation='elu',W_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(50,activation='elu',W_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation='elu',W_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='linear'))
    return model



if __name__=="__main__":
    own_filepath = '/Users/nraghu/Documents/data'
    udacity_filepath =  './data'
    headerline = 'false'

    filepath_g  =  udacity_filepath
    headerline = 'true'
    X_train, y_train = getCsvData(filepath_g, 'true')                                             
    X_train, y_train = shuffle(X_train, y_train, random_state=14)
    X_train,X_validation,y_train,y_validation = train_test_split(X_train, y_train, test_size = 0.2,random_state=14)              
    model = nvidiaNet((66,200,3), dropout= 0.25)
    model.compile(loss='mse',optimizer='adam')
    model.summary()
    model.fit_generator(generator(X_train,y_train), samples_per_epoch=19264,nb_epoch=15,validation_data=generator(X_validation, y_validation), nb_val_samples=1024)
    print('Saving model weights and configuration file.')
    model.save('model_udacity_nvidia_current3.h5')

    

 
 



    
