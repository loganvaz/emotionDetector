#note: validation accuracy is similar to unaltered, increase model complexity (also, train for a while overnight)
#data from https://www.kaggle.com/astraszab/facial-expression-dataset-image-folders-fer2013


#so I can load the datasets
import os
import random
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
# example of progressively loading images from file
from keras.preprocessing.image import ImageDataGenerator

checkpoint_filepath= path = os.getcwd() + "/model2_efficient_keras/currentWeights.h5"

# create generator
datagen = ImageDataGenerator()
# prepare an iterators for each dataset
dataSize = (48,48)
"""
trainBasic = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.05,
        height_shift_range=0.05,
        brightness_range = [0.7, 1.3],
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0)
"""
trainBasic = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=8,
        zoom_range=0.15,
        shear_range = 0.25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range = [0.65, 1.35],
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0)


train_generator = trainBasic.flow_from_directory(
    directory="data/train/keras_regular",
    target_size=dataSize,
    color_mode="rgb",
    batch_size=64,
    class_mode="categorical",
    shuffle=True
)
#trainBasic = datagen.flow_from_directory("data/train/keras_regular")#image_size =dataSize )

"""
trainAug = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=4,
        zoom_range=0.02,
        width_shift_range=0.00205,
        height_shift_range=0.00025,
        fill_mode="nearest",
        validation_split=0)
trainAugment = trainAug.flow_from_directory(
    directory="data/train/noisy",
    target_size=dataSize,
    color_mode="rgb",
    batch_size=64,
    class_mode="categorical",
    shuffle=True
)
"""
#trainAugment = datagen.flow_from_directory("data/train/noisy")# image_size =dataSize)
#validation_it = datagen.flow_from_directory("data/val", image_size =dataSize)
#so I can prosess images
from keras.preprocessing.image import load_img, img_to_array, array_to_img
loadModel = True
import numpy as np
#for getting validation data

#definition of model
import pandas as pd
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Activation, Reshape, Lambda, Input, Concatenate
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, ZeroPadding2D
from keras.callbacks import ModelCheckpoint#updating model as go
from keras.initializers import glorot_uniform


from keras import backend as K #probably won't need this or lambda
from keras.optimizers import Adam#generally the best optimizer

from random import randint
import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, ZeroPadding2D
from keras.callbacks import ModelCheckpoint#updating model as go
from keras.initializers import glorot_uniform
from keras.optimizers import Adam
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import regularizers

justTestModel = True

#final model
#GOT test_accuracy OF 68.93% and test F1 of .6878
#stored in provenModel folder (do not alter below code or won't be able to load those weights)

def initializeModel(inputShape, outputDimension):
    #caution against MaxPooling for small images
    #if still overfitting, chagne Conv2Ds to 8, 16, 32
    X_input = Input(shape=inputShape)
    print(X_input.shape)
    #X = ZeroPadding2D((1,1))(X_input)#similar accuracy with (2,2) zero padding
    X = X_input
    print("drop out, should maintain above shape")
    
    print(X.shape)
    #added
    X = Conv2D(64, kernel_size=8,strides=1,padding="same")(X)
    X = BatchNormalization(axis=3,name="zero_batch_norm")(X)
    X = Activation("relu")(X)
    print(X.shape)
    X = Conv2D(128,kernel_size=5,strides=2,padding="valid")(X)
    print(X.shape)
    X = BatchNormalization(axis=3,name="first_batch_normalization")(X)
    print(X.shape)
    X = Activation("relu")(X)
    print(X.shape)
    #X = MaxPooling2D(pool_size=(2,2))(X)
    print("mid")
    print(X.shape)
    #X = Dropout(0.025)(X)
    X = Conv2D(256, kernel_size = 4, strides=2, padding="valid")(X)
    X = BatchNormalization(axis=3, name="second_batch_norm")(X)
    X = Activation("relu")(X)
    #X = Dropout(0.05)(X)
    #X = MaxPooling2D(pool_size=(2,2))(X)
    print(X.shape)
    X = Conv2D(300, kernel_size=3, strides=2, padding="valid")(X)
    X = BatchNormalization(axis=3,name="third_batch_norm")(X)
    X = Activation("relu")(X)
    X = Dropout(0.25)(X)
    print(X.shape)
    X = ZeroPadding2D((1,1))(X)
    print("before last")
    print(X.shape)
    #X = Dropout(0.095)(X)
    X = Conv2D(400, kernel_size=3, strides=1, padding="valid")(X)
    X = BatchNormalization(axis=3,name="fourth_batch_norm")(X)
    X = Activation("relu")(X)
    X = Dropout(0.35)(X)#okay, this one is necesart because we're going to feed the data in

    #have eliminated this 
    X_prev = X
    flat_prev = Flatten(name="flatten_1")(X_prev)
    X_prev = Dense(128, activation="relu",kernel_regularizer = regularizers.l1(.000015))(flat_prev)
    X_prev = Dropout(0.3)(X_prev)
    hold = X_prev
    X_prev = Dense(128, activation="relu",kernel_regularizer = regularizers.l1(.000015))(hold)
    X_prev = Dropout(0.275)(X_prev)
    X_prev = Dense(128,kernel_regularizer = regularizers.l1(.000015))(X_prev)
    X_prev = Dropout(0.3)(X_prev)
    X_prev = Activation("relu")(X_prev+hold)

    
    
    
    

    X = Conv2D(512, kernel_size=4, strides=1, padding="valid")(X)
    X = BatchNormalization(axis=3,name="fifth_batch_norm")(X)
    X = Activation("relu")(X)
    

    flat = Flatten(name="flatten_2")(X)
    flat = Dropout(0.485)(flat)

    #I added three dropouts below (btw dense) to try to reduce overfitting, see how that works
    flat= Dense(256, activation="relu",kernel_regularizer = regularizers.l2(.0000185))(flat)#tried changing l1 to l2
    flat = Dropout(0.485)(flat)
    X = Dense(128, activation="relu", kernel_regularizer = regularizers.l2(.00002))(flat)
    X = Dropout(0.5)(X)
    X = Dense(256,kernel_regularizer = regularizers.l2(.0000185) )(X)#no activation function, took away regularizer so it could scale below activation freely
    X = Dropout(0.485)(X)
    X = Activation("relu")(X+flat)
    

    #joining prev with current
    
##    X = Concatenate()([X_prev, X])
##    X = Dense(128, activation="relu", kernel_regularizer = regularizers.l1(.000015))(X)
##    X_old = Dropout(0.25)(X)
##    X = X_old
##    X = Dense(64, kernel_regularizer = regularizers.l1(.0000001), activation="relu")(X)
##    X = Dropout(0.15)(X)
##    X_old = X
##    X = Dense(16, activation="relu")(X)
##    X = Concatenate()([X, X_old])
    
    
    #added = Extra dense (128 layer) and doubled all convolution sizes
    #can try adding one more intermediete layer here if complexity still too low
    
    
    
    
    X = Dense(outputDimension, activation ="softmax",kernel_initializer = glorot_uniform(seed=0))(X)
    model = Model(inputs = X_input, outputs = X, name="myCNN")
    return model



#old model
"""
def initializeModel(inputShape, outputDimension):
    #caution against MaxPooling for small images
    #if still overfitting, chagne Conv2Ds to 8, 16, 32
    X_input = Input(shape=inputShape)
    print(X_input.shape)
    #X = ZeroPadding2D((1,1))(X_input)#similar accuracy with (2,2) zero padding
    X = X_input
    print("drop out, should maintain above shape")
    
    print(X.shape)
    #added
    X = Conv2D(64, kernel_size=8,strides=1,padding="same")(X)
    X = BatchNormalization(axis=3,name="zero_batch_norm")(X)
    X = Activation("relu")(X)
    print(X.shape)
    X = Conv2D(128,kernel_size=5,strides=2,padding="valid")(X)
    print(X.shape)
    X = BatchNormalization(axis=3,name="first_batch_normalization")(X)
    print(X.shape)
    X = Activation("relu")(X)
    print(X.shape)
    #X = MaxPooling2D(pool_size=(2,2))(X)
    print("mid")
    print(X.shape)
    #X = Dropout(0.025)(X)
    X = Conv2D(256, kernel_size = 4, strides=2, padding="valid")(X)
    X = BatchNormalization(axis=3, name="second_batch_norm")(X)
    X = Activation("relu")(X)
    #X = Dropout(0.05)(X)
    #X = MaxPooling2D(pool_size=(2,2))(X)
    print(X.shape)
    X = Conv2D(300, kernel_size=3, strides=2, padding="valid")(X)
    X = BatchNormalization(axis=3,name="third_batch_norm")(X)
    X = Activation("relu")(X)
    X = Dropout(0.25)(X)
    print(X.shape)
    X = ZeroPadding2D((1,1))(X)
    print("before last")
    print(X.shape)
    #X = Dropout(0.095)(X)
    X = Conv2D(400, kernel_size=3, strides=1, padding="valid")(X)
    X = BatchNormalization(axis=3,name="fourth_batch_norm")(X)
    X = Activation("relu")(X)
    X = Dropout(0.35)(X)#okay, this one is necesart because we're going to feed the data in

    #have eliminated this 
    X_prev = X
    flat_prev = Flatten(name="flatten_1")(X_prev)
    X_prev = Dense(128, activation="relu",kernel_regularizer = regularizers.l1(.000015))(flat_prev)
    X_prev = Dropout(0.3)(X_prev)
    hold = X_prev
    X_prev = Dense(128, activation="relu",kernel_regularizer = regularizers.l1(.000015))(hold)
    X_prev = Dropout(0.275)(X_prev)
    X_prev = Dense(128,kernel_regularizer = regularizers.l1(.000015))(X_prev)
    X_prev = Dropout(0.3)(X_prev)
    X_prev = Activation("relu")(X_prev+hold)

    
    
    
    

    X = Conv2D(512, kernel_size=4, strides=1, padding="valid")(X)
    X = BatchNormalization(axis=3,name="fifth_batch_norm")(X)
    X = Activation("relu")(X)
    

    flat = Flatten(name="flatten_2")(X)
    flat = Dropout(0.485)(flat)

    #I added three dropouts below (btw dense) to try to reduce overfitting, see how that works
    flat= Dense(256, activation="relu",kernel_regularizer = regularizers.l2(.0000185))(flat)#tried changing l1 to l2
    flat = Dropout(0.485)(flat)
    X = Dense(128, activation="relu", kernel_regularizer = regularizers.l2(.00002))(flat)
    X = Dropout(0.5)(X)
    X = Dense(256,kernel_regularizer = regularizers.l2(.0000185) )(X)#no activation function, took away regularizer so it could scale below activation freely
    X = Dropout(0.485)(X)
    X = Activation("relu")(X+flat)
    

    #joining prev with current
    
##    X = Concatenate()([X_prev, X])
##    X = Dense(128, activation="relu", kernel_regularizer = regularizers.l1(.000015))(X)
##    X_old = Dropout(0.25)(X)
##    X = X_old
##    X = Dense(64, kernel_regularizer = regularizers.l1(.0000001), activation="relu")(X)
##    X = Dropout(0.15)(X)
##    X_old = X
##    X = Dense(16, activation="relu")(X)
##    X = Concatenate()([X, X_old])
    
    
    #added = Extra dense (128 layer) and doubled all convolution sizes
    #can try adding one more intermediete layer here if complexity still too low
    
    
    
    
    X = Dense(outputDimension, activation ="softmax",kernel_initializer = glorot_uniform(seed=0))(X)
    model = Model(inputs = X_input, outputs = X, name="myCNN")
    return model


"""

def getAllData(limitData = False):
    print("getAllData()")
    if (limitData):
        limitData += 1
    holdOriginal = limitData
  
    data = list()
    for y in range(7):
        limitData = holdOriginal
        print("starting emotion # " + str(y))
        os.chdir(str(y))
        filenames = os.listdir()
        if (limitData):
            random.shuffle(filenames)
        for file in filenames:
           
            if (limitData):
                limitData = limitData -1
                if (limitData<=0):
                    print("exitting early")
                    break
            image = load_img(os.getcwd()+"/"+file)
            x = img_to_array(image).astype(np.uint8)/255
            data.append((x,y))
        os.chdir("..")
    print("shuffling data")
    random.shuffle(data)
    X = list()
    y = list()
    for d in data:
        X.append(d[0])
        y.append(d[1])
    print("before one hot")
    print(y[0])
    y = to_categorical(array(y))
    print("after one hot")
    print(y[0])
    X = np.array(X)
    y = np.array(y)
    print("shapes from getAllData()")
    print(X.shape)
    print(y.shape)
    return X,y

os.chdir("data")
os.chdir("val")
X_val, y_val = getAllData()
X_val = np.array(X_val)
y_val = np.array(y_val)

os.chdir("..")
os.chdir("train")
unaltered_X, unaltered_Y = getAllData()
unaltered_X = np.array(unaltered_X)
unaltered_Y = np.array(unaltered_Y)

os.chdir("..")
os.chdir("..")


#defining callback and metrics
from keras.callbacks import Callback,ModelCheckpoint
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val





if (justTestModel):
    first_image  = load_img(os.getcwd()+"/data/train/0/0.png")
    data_first_image = img_to_array(first_image)
    data_first_image = data_first_image.astype(np.uint8)
    model = initializeModel(data_first_image.shape, 7)
    model.load_weights(path)
    os.chdir("data")
    os.chdir("val")
    X, y  = getAllData()
    model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics = ["accuracy", get_f1])
    output = model.evaluate(X,y,verbose=0, batch_size=128)
    print("Validation Set Loss/Accuracy/F1")
    print(".........."*3)
    print(output)
    os.chdir("..")
    os.chdir("test")
    X, y  = getAllData()
    model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics = ["accuracy", get_f1])
    output = model.evaluate(X,y,verbose=0, batch_size=128)
    print("Test Set Loss/Accuracy/F1")
    print(".........."*3)
    print(output)
    exit(0)
    

    












first_image  = load_img(os.getcwd()+"/data/train/0/0.png")
data_first_image = img_to_array(first_image)
data_first_image = data_first_image.astype(np.uint8)
model = initializeModel(data_first_image.shape, 7)
print("directory")
print(os.getcwd())
if (loadModel):
    try:
        model.load_weights(path)
        print("weightes successfully loaded")
    except:
        print("model size different from what it was saved as")
print(model.summary())
opt = Adam(learning_rate=0.001,beta_1=.9, beta_2=.999)
resultsVal = list()
resultsRegPhoto = list()
resultsAltered = list()
x_alt = list()
common_x = list()
val_x = list()
cost_reg = list()
cost_val = list()
unaltered_x = list()
unalteredAccuracy = list()
unalteredCost = list()
x_at = 0
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.ion()
numRounds= 100
plt.axis([0, numRounds,0.3,2.5])
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics = ["accuracy", get_f1])
for i in range(numRounds):
    #common_x.append(i)
    #val_x.append(i-.5)
    #val_x.append(i)
    #resultsVal.append(model.evaluate(X_val, y_val, verbose=0, batch_size=128)[1])
    
    print("decimal done: " + str(i/numRounds))
    history = model.fit(train_generator, steps_per_epoch=128, epochs=8, verbose =2, callbacks = [model_checkpoint_callback], validation_data =(X_val, y_val) )#increase batch size to 64
    valData = history.history["val_accuracy"]
    unalteredCost_val = model.evaluate(unaltered_X, unaltered_Y, verbose=0, batch_size=128)
    unalteredAccuracy.append(unalteredCost_val[1])
    unalteredCost.append(unalteredCost_val[0])
    unaltered_x.append(i+1)
    
    for ii in range(len(valData)-1,len(valData)):
        common_x.append(i+ii/len(valData))
        val_x.append(i+ii/len(valData))
        resultsVal.append(valData[ii])
        resultsRegPhoto.append(history.history["accuracy"][ii])
        cost_reg.append(history.history["loss"][ii])
        cost_val.append(history.history["val_loss"][ii])
    if (i%2==1):#can't have 0 as need at least 2 pts for last 2
        listOfLoop = [(resultsVal, "r--", val_x, "validationAccuracy", "red"),(resultsRegPhoto, "g--", common_x,"smallAlterAccuracy", "green")]
        listOfLoop.append((cost_reg, "g--", common_x, "regular cost", "green"))
        listOfLoop.append((cost_val, "r--", val_x, "validation cost", "red"))
        listOfLoop.append((unalteredAccuracy, "y--", unaltered_x, "unaltered accuracy", "yellow"))
        listOfLoop.append((unalteredCost, "o--", unaltered_x, "unaltered cost", "orange"))
        patches = list()
        for y, color, x, label,c2 in listOfLoop:#,(resultsAltered, "b--", x_alt,"majorAlterAccuracy", "blue")]:
            if (label=="validation cost" or label =="unaltered cost" or label=="regular cost"):
                pass#y = [i/2 for i in y]#so everything fits nicely on the graph
            try:
                z = np.polyfit(x,y,1)
            except:
                x = x[:-1]
                z = np.polyfit(x,y,1)
            p = np.poly1d(z)
            plt.plot(x,y,color=c2)
            plt.draw()
            plt.pause(0.000000001)
            print(c2 +" is " + label)
            patch = mpatches.Patch(color=c2, label=label)
            patches.append(patch)
        plt.legend(handles=patches)
        plt.pause(0.00000001)
        
        plt.show()
        


    #resultsVal.append(history.history["val_accuracy"][-1])
    #resultsRegPhoto.append(history.history["accuracy"][-1])
    #adding these try/except b/c one of the images is corrupted (can fix by deleting and re-running program (think stopping early corrupted something, don't know where though)
    #in addition to helping stop overfitting, below should increase F1 score (perhaps at cost of accuracy unfortunately)
    """
    try:
        history = model.fit(trainAugment, steps_per_epoch=128, epochs=2, verbose =2, callbacks = [model_checkpoint_callback], validation_data =(X_val, y_val) )
        resultsAltered.append(history.history["accuracy"][-1])
        resultsVal.append(history.history["val_accuracy"][-1])
        x_alt.append(i)
        val_x.append(i+.5)
    except:
        try:
            model.fit(trainAugment, steps_per_epoch=64, epochs=2, verbose =2, callbacks = [model_checkpoint_callback], validation_data =(X_val, y_val) )
        except:
            try:
                model.fit(trainAugment, steps_per_epoch=32, epochs=1, verbose =2, callbacks = [model_checkpoint_callback], validation_data =(X_val, y_val) )
            except:
                pass


    """
import matplotlib.pyplot as plt
import seaborn as sns

for y, color, x, label,c2 in [(resultsVal, "r--", val_x, "validationAccuracy", "red"),(resultsRegPhoto, "g--", common_x,"smallAlterAccuracy", "green")]:#,(resultsAltered, "b--", x_alt,"majorAlterAccuracy", "blue")]:
    #y = [i[1] for i in y]
    try:
        z = np.polyfit(x,y,1)
    except:
        x = x[:-1]
        z = np.polyfit(x,y,1)
    p = np.poly1d(z)
    #plt.plot(x,p(x), color, label=label)
    plt.plot(x,y,color=c2)
    print(label + " line of best fit")
    print( "y=%.6fx+(%.6f)"%(z[0],z[1]))

plt.show()
loss = model.evaluate(X_val, y_val,verbose=0)
print(loss)
