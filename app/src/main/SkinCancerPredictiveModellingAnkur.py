#!/usr/bin/env python
# coding: utf-8

# In[459]:


import glob
from PIL import Image 
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import matplotlib
import os
import csv

from keras import backend as K
K.set_image_data_format('channels_first')

from numpy import*
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D,  MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
import theano

from keras.models import model_from_json
from keras.models import load_model

from sklearn.utils import shuffle

from keras import callbacks 
from tensorflow.keras.callbacks import TensorBoard
import time

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split
from keras import callbacks 
from tensorflow.keras.callbacks import TensorBoard
import time
from sklearn.model_selection import KFold


# In[460]:



path_string = '/home/phi/FYPs/Ankur/ISIC_2019_Training_Input/'
MelPath = '/home/phi/FYPs/Ankur/Data/Melanoma/'
NeviPath = '/home/phi/FYPs/Ankur/Data/Nevi/'
Datalist = '/home/phi/FYPs/Ankur/Data/'
Datadirlist = os.listdir(Datalist)
Metadata = pd.read_csv('/home/phi/FYPs/Ankur/ISIC_2019_Training_GroundTruth.csv')


# In[461]:


hist = 0
TestA = 0
TestL = 0 
m = 22
n = 8375
#m = 3522
#n = 11875
imgcol = 128
imgrow = 128
ImgData = []
numchannels = 1


MelanomaData = Metadata.filter(["image" ,"MEL"]) # Extract the Melanoma Column and the ID of images
MelanomaData = MelanomaData[MelanomaData.MEL == 1] # Sort column with positive images
MelanomaData = MelanomaData[:-m] # Reduce the amount of images to 4500, by subtracting 22 images

NeviData = Metadata.filter(["image" ,"NV"]) # Extract the Nevus Column and the ID of the images
NeviData = NeviData[NeviData.NV == 1] # Sort Column with positive images
NeviData = NeviData[:-n] # Reduce the amount of images to 4500, by subracting 8375 images 

mel_name_list  = MelanomaData['image'].tolist() # Add all melanoma images to the varable 
nv_name_list = NeviData['image'].tolist() # Add all Nevus Images to the varible 


# In[557]:


print(len(NeviData))
print(len(MelanomaData))


# In[ ]:


# for image in mel_name_list:
#         im = cv2.imread(path_string + image +'.jpg')
#         new = cv2.resize(im,(imgcol, imgrow))
#         new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
#         ImgData.append(new)
# for ima in nv_name_list:
#         imn = cv2.imread(path_string + ima +'.jpg')
#         newn = cv2.resize(im,(imgcol, imgrow))
#         newn = cv2.cvtColor(newn, cv2.COLOR_BGR2GRAY)
#         ImgData.append(newn)
    


# In[463]:


for image in mel_name_list:
    #Reads in the images and converts the images into grayscale
    im = cv2.imread(path_string + image +'.jpg',(cv2.IMREAD_GRAYSCALE))
    #resizes the images into 128x128
    new = cv2.resize(im,(imgcol, imgrow))
    ImgData.append(new)


# In[464]:


for ima in nv_name_list:
    #Reads in the images and converts the images into grayscale
    im = cv2.imread(path_string + ima +'.jpg',(cv2.IMREAD_GRAYSCALE))
    #resizes the images into 128x128
    newn = cv2.resize(im,(imgcol, imgrow))
    ImgData.append(newn)


# In[646]:


plt.imshow(ImgData[1], cmap = 'gray')


# In[736]:


#Convert images into Numpy arrays 
Img_data = np.array(ImgData)
#Convert images into float 32
Img_data = Img_data.astype('float32')
#Normalise all of the images
Img_data /= 255                         
print(Img_data.shape) 


# In[737]:


# Add the channel of the image to the sphape 
if numchannels == 1: # depending on the channel used 
    if K.image_data_format() == 'channels_first': # is always true, unless imports change
        Img_data = np.expand_dims(Img_data, axis=1) # single channel input, 
        print(Img_data.shape)
    else:
        Img_data = np.expand_dims(Img_data, axis = 4)
        print (Img_data.shape)
else: # only if channel not 1 (RGB)
    if K.image_data_format() == 'channels_first':
        Img_data=np.rollaxis(Img_data,3,1)
        print (Img_data.shape)
    else:
        img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
        print (img_data_scaled.shape)


# In[738]:


numclasses = 2 # Number of types of skin conditions used 
print(Img_data.shape[0]) # print the amount of images 
numsamples = Img_data.shape[0] 

Labels = np.ones((numsamples,),dtype = 'int64') # the numsamples are converted into NumPy and added to Labels
Labels[0:4500] = 0 # 0 - 4500 are melanoma images
Labels[4500:9000] = 1 # 4500 - 9000 are Nevus Images
names = ['Melanoma' , 'Nevi' ] # Only two classes 


Y = np_utils.to_categorical(Labels, numclasses) # the amount of vectors and labels corresponding 
x,y = shuffle(Img_data,Y, random_state=2) # Shuffle all images 
print(Img_data.shape)# Print the whole shape of the input

 


# In[739]:


#X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state = 2)

print(x_train.shape)
print(y_test.shape)

print(Img_data.shape[0])
input_shape=Img_data[0].shape
print(input_shape)


# In[734]:





# In[740]:


filename='empty.csv' # Declare the excel file 
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)
early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min') # Monitor Val loss
filepath="Best-weights-my_model-{epoch:04d}-{loss:.4f}-{acc:.4f}.hdf5" # Best weights after each epoch is added
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min') # Record the best epoch
callbacks_list = [csv_log,early_stopping,checkpoint] # added all methods into the arraylist


# In[650]:


#Underfitting
model = Sequential()
model.add(Convolution2D(128, (3, 3), activation='relu', input_shape= input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(250, activation='relu'))
model.add(Dense(10, activation='softmax'))


# In[563]:


#Overfitting
model = Sequential()
model.add(Convolution2D(32, kernel_size = (3,3),padding = 'same' ,input_shape= input_shape))
model.add(Activation('linear'))
model.add(Convolution2D(32, kernel_size = (3,3)))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same' ))
model.add(Dropout(0.5))

model.add(Convolution2D(64, kernel_size = (3,3)))
model.add(Activation('linear'))
model.add(Convolution2D(64, kernel_size = (3,3)))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same' ))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(10))
model.add(Dropout(0.5))

model.add(Activation('linear'))
model.add(Dense(numclasses))
model.add(Activation('softmax'))


# In[ ]:


#Overfitting 
model.add(Convolution2D(32, (3,3),padding = 'same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
#model.add(Convolution2D(64, 3, 3))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


# In[615]:


#Not Applicable 
model = Sequential()
    
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(input_shape)))
model.add(MaxPooling2D((2, 2)))
   
    
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
    
model.add(Convolution2D(64, (3, 3), activation='relu'))
 

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))


# In[741]:


model = Sequential()
model.add(Convolution2D(32, kernel_size = (3,3),padding = 'same' ,input_shape= input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, kernel_size = (3,3)))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same' ))
model.add(Dropout(0.5))

model.add(Convolution2D(64, kernel_size = (3,3)))
model.add(Activation('relu'))
model.add(Convolution2D(64, kernel_size = (3,3)))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same' ))
model.add(Dropout(0.5))

model.add(Convolution2D(128, kernel_size = (3,3)))
model.add(Activation('relu'))
model.add(Convolution2D(128, kernel_size = (3,3), padding = 'same' ))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(10))
model.add(Dropout(0.5))

model.add(Activation('relu'))
model.add(Dense(numclasses))
model.add(Activation('softmax'))


# In[744]:



model.compile(loss='categorical_crossentropy', 
              optimizer='RMSprop',
              metrics=['accuracy'])


# In[ ]:



model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])
hist = model.fit(X_train, y_train, batch_size=16, nb_epoch=25, verbose=1, validation_data=(X_test, y_test), callbacks = callbacks_list)


# In[652]:


def get_score (x_train, x_test, y_train, y_test):
    hist = model.fit(x_train, y_train, validation_data=(X_test, y_test), 
                     epochs = 10, callbacks = callbacks_list)# Callback records the outcomes 
    return x_test, y_test,hist 


# In[746]:


kfold = KFold(n_splits=10, shuffle=True) # Specify the amount of folds required 
for train_index, test_index in folds.split(x,y):# Split the data in to 10 folds
    x_train, x_test, y_train, y_test = x[train_index], x[test_index], \ 
                                       y[train_index], y[test_index]  
    a = (get_score(x_train, x_test, y_train, y_test)) # run the model 
    Score.append(a) # Save weights into array 


# In[806]:


scores = model.evaluate(x_test, y_test, verbose=0)


# In[749]:



with open("empty.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row[0:21]) 


# In[750]:


col_list = ["epoch", "acc", "loss", "val_acc", "val_loss"]
evalmodel = pd.read_csv("empty.csv", usecols=col_list)


# In[812]:


TrainLoss =  evalmodel['loss']
ValLoss = evalmodel['val_loss']
TrainAcc = evalmodel['acc']
ValAcc = evalmodel['val_acc']
Range = range(10)


# In[809]:


TrainLoss = hist.history['loss']
ValLoss = hist.history['val_loss']
TrainAcc = hist.history['acc']
ValAcc = hist.history['val_acc']
Range = range(10)


# In[813]:


print(ValLoss)
print(TrainLoss)
print(TrainAcc)
print(ValAcc)


# In[814]:


plt.figure(1,figsize=(10,6))
plt.plot(Range, TrainLoss, color='red', marker='o')
plt.plot(Range, ValLoss, color='blue', marker='o')
plt.ylim() 
plt.xlabel('Numb Of Folds')
plt.ylabel('TLoss and VLoss ')
plt.title('Train Loss Vs Value Loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])



plt.figure(5,figsize=(10,6))
plt.plot(Range, TrainAcc, color='red', marker='o')
plt.plot(Range, ValAcc, color='blue', marker='o')
plt.ylim(0.4)
plt.xlabel('Numb Of Folds')
plt.ylabel('TAcc and VAcc')
plt.title('Train Acc Vs Value Acc')
plt.grid(True)
plt.legend(['train','val'],loc = 3)
plt.style.use(['classic'])


# In[808]:


# goal = model.evaluate(x_test, y_test,  verbose = 0)
print(' The test Loss:', scores[0])
print(' The Test Accuracy:', scores[1])


# In[756]:


Test_image = x_test[0:1]


# In[757]:


print(Test_image.shape)
print(model.predict(Test_image))
print(model.predict_classes(Test_image))
print(y_test[0:1])


# In[821]:


im2 = 'ISIC_0033742'
testImage = cv2.imread(path_string + im2 + '.jpg')
testImage = cv2.resize(testImage,(128, 128))
testImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
testImage = np.array(testImage)
testImage = testImage.astype('float32')
testImage /= 255
print(testImage.shape)


# In[822]:


if numchannels == 1:
    if K.image_data_format() == 'channels_first':
        testImage = np.expand_dims(testImage, axis=0)
        testImage = np.expand_dims(testImage, axis=0)
        print(testImage.shape)
    else:
        testImage = np.expand_dims(testImage, axis = 3)
        testImage = np.expand_dims(testImage, axis=0)
        print (testImage.shape)
    
else: 
    if K.image_data_format() == 'channels_first':
        testImage=np.rollaxis(testImage.shape,2,0)
        testImage = np.expand_dims(testImage, axis=0)
        print (testImage.shape)
    else:
        testImage = np.expand_dims(testImage, axis = 0)
        print(testImage.shape)


# In[823]:


print((model.predict(testImage)))
print(model.predict_classes(testImage))


# In[795]:


testImageMel = 0


# In[838]:


im2 = 'ISIC_0033248'
testImageMel = cv2.imread(path_string + im2 + '.jpg')
testImageMel = cv2.resize(testImageMel,(128, 128))
testImageMel = cv2.cvtColor(testImageMel, cv2.COLOR_BGR2GRAY)
testImageMel = np.array(testImageMel)
testImageMel = testImageMel.astype('float32')
testImageMel /= 255
print(testImageMel.shape)


# In[839]:


if numchannels == 1:
    if K.image_data_format() == 'channels_first':
        testImageMel = np.expand_dims(testImageMel, axis=0)
        testImageMel = np.expand_dims(testImageMel, axis=0)
        print(testImageMel.shape)
    else:
        testImage = np.expand_dims(testImageMel, axis = 3)
        testImage = np.expand_dims(testImageMel, axis=0)
        print (testImageMel.shape)
else: 
    if K.image_data_format() == 'channels_first':
        testImage=np.rollaxis(testImage.shape,2,0)
        testImage = np.expand_dims(testImage, axis=0)
        print (testImage.shape)
    else:
        testImage = np.expand_dims(testImage, axis = 0)
        print(testImage.shape)


# In[840]:



print((model.predict(testImageMel)))
print(model.predict_classes(testImageMel))


# In[761]:


Y_pred = model.predict(x_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)


# In[725]:


from sklearn.metrics import classification_report,confusion_matrix
import itertools


# In[ ]:





# In[763]:


target_names = ['class 0(Mel)', 'class 1(Nevi)']

print(classification_report(np.argmax(y_test, axis=1), y_pred,target_names=target_names))

print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))


# In[765]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[766]:


cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
plt.show()


# In[ ]:


MODEL_NAME = 'Ankur'

def export_model(saver, model, input_node_names,output_node_names) :
    
    tf.train.write_graph(K.get_session().graph_def, 'new',                          MODEL_NAME + '_graph.pbtxt') # Specify the models saved name and file type  

    saver.save(K.get_session(), 'new/' + MODEL_NAME + '.chkp') # 'Saver' function saves the actual model

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None,                               False, 'out/' + MODEL_NAME + '.chkp', output_node_names,                               "save/restore_all", "save/Const:0",                               'out/frozen_' + MODEL_NAME + '.pb', True, "") # freeze the trained weigths 

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

        output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_names],
            tf.float32.as_datatype_enum) #Re-open the trained weights and convert to float 32

        with tf.gfile.FastGFile('new/opt_' + MODEL_NAME + '.pb', "wb") as f:
            f.write(output_graph_def.SerializeToString()) #Re open the model and save to seralized string

            print("graph saved!")

