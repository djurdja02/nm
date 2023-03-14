import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
main_path='./archive (1)'
img_size=(64,64)
batch_size=64
from keras.utils import image_dataset_from_directory
Xtrain=image_dataset_from_directory(main_path,subset='training',validation_split=0.2,image_size=img_size,batch_size=batch_size,seed=123)
Xval=image_dataset_from_directory(main_path,subset='validation',validation_split=0.2,image_size=img_size,batch_size=batch_size,seed=123)
classes=Xtrain.class_names

from keras import layers
from keras import Sequential
data_augmentation=Sequential([layers.RandomFlip("horizontal",input_shape=(img_size[0],img_size[1],3)),layers.RandomRotation(0.25),layers.RandomZoom(0.1),])

from keras import Sequential
from keras import layers
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
num_classes=len(classes)
model=Sequential([data_augmentation,
                  layers.Rescaling(1./255,input_shape=(64,64,3)),
                  layers.Conv2D(16,3,padding='same',activation='relu'),layers.MaxPooling2D(),
                  layers.Conv2D(32,3,padding='same',activation='relu'),layers.MaxPooling2D(),
                  layers.Conv2D(64,3,padding='same',activation='relu'),layers.MaxPooling2D(),
                  layers.Dropout(0.2),layers.Flatten(),layers.Dense(128,activation='relu'),
                  layers.Dense(num_classes,activation='softmax')])
model.summary()
model.compile(Adam(learning_rate=0.001),loss=SparseCategoricalCrossentropy(),metrics='accuracy')
from keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss',mode='min',patience=10,verbose=1)
history=model.fit(Xtrain,epochs=30,validation_data=Xval,verbose=1,callbacks=es)
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.show()
model.save('./cuvanje/sacuvanodropout.model')
from keras import models

model=models.load_model('./cuvanje/sacuvanodropout.model')
model.compile(Adam(learning_rate=0.001),loss=SparseCategoricalCrossentropy(),metrics='accuracy')


predV=np.array([])
labelsV=np.array([])
for img,lab in Xval:
    labelsV=np.append(labelsV,lab)
    predV=np.append(predV,np.argmax(model.predict(img,verbose=0),axis=1))
predT=np.array([])
labelsT=np.array([])
for img,lab in Xtrain:
    labelsT=np.append(labelsT,lab)
    predT=np.append(predT,np.argmax(model.predict(img,verbose=0),axis=1))

from sklearn.metrics import confusion_matrix, accuracy_score,ConfusionMatrixDisplay
cmV=confusion_matrix(labelsV, predV,normalize='true')
accV=accuracy_score(labelsV, predV)
cmDis=ConfusionMatrixDisplay(confusion_matrix=cmV,display_labels=classes)
cmDis.plot()
plt.show()
print("Tacnost na skupu validacije je" +str(100*accV)+'%')


cmT=confusion_matrix(labelsT, predT,normalize='true')
accT=accuracy_score(labelsT, predT)
cmDisT=ConfusionMatrixDisplay(confusion_matrix=cmT,display_labels=classes)
cmDisT.plot()
plt.show()
print("Tacnost na skupu trening je" +str(100*accT)+'%')