#!/usr/bin/env python
# coding: utf-8

# In[1]:


#
# if you do not have tensorflow,
# run:
#  $ pip3 install tensorflow
#
# if you do not have pip3, run:
#  $ sudo apt install python3-pip
#
import tensorflow as tf


# In[2]:


mnist = tf.keras.datasets.mnist


# In[3]:


(x_train, y_train),(x_test, y_test) = mnist.load_data()


# In[4]:


x_train, x_test = x_train / 255.0, x_test / 255.0


# In[5]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


# In[6]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[7]:


model.fit(x_train, y_train, epochs=5)


# In[8]:


model.evaluate(x_test, y_test)


# In[ ]:





# In[ ]:




