#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')


# In[2]:


pip install keras


# In[3]:


from numpy import loadtxt
import numpy as np
import pandas as pd
from keras import models
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt


# In[4]:


dataframe=pd.read_csv('pima-indians-diabetes.csv',delimiter=',')
dataframe.head()


# In[5]:


X=dataframe.iloc[:,:8]
y=dataframe.iloc[:,8]
dataframe.shape


# In[6]:


features_train,features_test,target_train,target_test=train_test_split(X,y,test_size=0.33,random_state=0)


# In[7]:


network=models.Sequential()
network.add(Dense(units=8,activation="relu",input_shape=(features_train.shape[1],)))


# In[8]:


network.add(Dense(units=8,activation="relu"))


# In[9]:


network.add(Dense(units=1,activation="sigmoid"))


# In[10]:


network.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[11]:


history=network.fit(features_train,target_train,epochs=20,verbose=1,batch_size=100,validation_data=(features_test,target_test))


# In[12]:


training_loss=history.history["loss"]
test_loss=history.history["val_loss"]
epoch_count=range(1,len(training_loss)+1)
plt.plot(epoch_count,training_loss,"r--")
plt.plot(epoch_count,test_loss,"b-")
plt.legend(["Training Loss","Test Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# In[13]:


_,accuracy=network.evaluate(features_train,target_train)
print('Accuracy: %.2f'%(accuracy*100))


# In[14]:


predicted_target=network.predict(features_test)
_,accuracy=network.evaluate(features_test,target_test)
print('Accuracy: %.2f'%(accuracy*100))


# In[15]:


for i in range(10):
    print(predicted_target[i])


# In[16]:


training_accuracy=history.history["accuracy"]
test_accuracy=history.history["val_accuracy"]
plt.plot(epoch_count,training_accuracy,"r--")
plt.plot(epoch_count,test_accuracy,"b-")
plt.legend(["Training Accuracy","Test Accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy Score")
plt.show()


# In[ ]:




