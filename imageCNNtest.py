#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pip
pip.main(["install","opencv-python"])
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sm
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf


# In[3]:


import cv2


# In[4]:


import os


# In[7]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop


# In[22]:


img = image.load_img("/Users/mahelwimaladasa/Desktop/BaseData/Test/01 Very Good/Image00001.jpeg")


# In[9]:


plt.imshow(img)


# In[24]:


cv2.imread("/Users/mahelwimaladasa/Desktop/BaseData/Test/01 Very Good/Image00001.jpeg").shape


# In[13]:


train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)


# In[26]:


train_dataset = train.flow_from_directory('/Users/mahelwimaladasa/Desktop/BaseData/Test/',
                                         target_size = (200,200),
                                         batch_size = 3,
                                         class_mode = 'binary')
validation_dataset = train.flow_from_directory('/Users/mahelwimaladasa/Desktop/BaseData/Validation/',
                                         target_size = (200,200),
                                         batch_size = 3,
                                         class_mode = 'binary')


# In[27]:


train_dataset.class_indices


# In[28]:


train_dataset.classes


# In[29]:


model = tf.keras.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation = 'relu',input_shape = (200,200,3)),
                            tf.keras.layers.MaxPool2D(2,2),
                            #
                            tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                            tf.keras.layers.MaxPool2D(2,2),
                            #
                            tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
                            tf.keras.layers.MaxPool2D(2,2),
                            ##
                            tf.keras.layers.Flatten(),
                            ##
                            tf.keras.layers.Dense(512,activation= 'relu'),
                            ##
                            tf.keras.layers.Dense(1,activation= 'sigmoid'),
                            ])


# In[30]:


model.compile(loss = 'binary_crossentropy',
             optimizer = RMSprop(lr=0.001),
             metrics = ['accuracy'])


# In[32]:


model_fit = model.fit(train_dataset,
                     steps_per_epoch = 3,
                     epochs = 10,
                     validation_data = validation_dataset)


# In[42]:


validation_dataset.class_indices


# In[47]:


dir_path = '/Users/mahelwimaladasa/Desktop/BaseData/Test/01 Very Good'

for i in os.listdir(dir_path):
    img = image.load_img(dir_path+'//'+i, target_size=(200,200))
    plt.imshow(img)
    plt.show()
    
    X = image.img_to_array(img)
    X = np.expand_dims(X,axis = 0)
    images = np.vstack([X])
    
    val = model.predict(images)
    if val == 0:
        print("Perfect Advertisement")
    else:
        print("Recreate this Advertisement")


# In[ ]:




