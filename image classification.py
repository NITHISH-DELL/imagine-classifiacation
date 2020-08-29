
# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[3]:


data=pd.read_csv("mnist_train.csv")


# In[6]:


a=data.iloc[3,1:].values


# In[7]:


a=a.reshape(28,28).astype("uint8")


# In[8]:


plt.imshow(a)


# In[9]:


df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]


# In[10]:


x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.3,random_state=4)


# In[12]:


rf=RandomForestClassifier(n_estimators=100)


# In[13]:


rf.fit(x_train,y_train)


# In[14]:


pred=rf.predict(x_test)


# In[15]:


s=y_test.values


# In[16]:


count=0
for i in range(len(pred)):
    if pred[i]==s[i]:
        count+=1
print(count)


# In[ ]:





# In[ ]:




