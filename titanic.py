
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('desktop/ML/train.csv')
test = pd.read_csv('desktop/ML/test.csv')


# In[3]:


train.head()


# In[4]:


train.shape


# In[5]:


test.shape


# In[6]:


train.describe()


# In[7]:


train.info()


# In[8]:


test.info()


# In[9]:


train.isnull().sum()


# In[10]:


test.isnull().sum()


# In[11]:


def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[12]:


bar_chart('Sex')


# In[13]:


bar_chart('Pclass')


# In[14]:


bar_chart('SibSp')


# In[15]:


bar_chart('Parch')


# In[16]:


bar_chart('Embarked')


# In[17]:


age = train['Age'].median()
print (age)


# In[18]:


def fills_na(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[19]:


train['Age'] = train[['Age', 'Pclass']].apply(fills_na, axis =1)


# In[20]:


train.drop('Cabin', axis = 1, inplace = True)


# In[21]:


train.head()


# In[22]:


sex = pd.get_dummies(train['Sex'], drop_first = True)
embarked = pd.get_dummies(train['Embarked'], drop_first = True)


# In[23]:


train = pd.concat([train, sex, embarked], axis = 1)


# In[24]:


train.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId'], axis = 1, inplace = True)


# In[25]:


X = train.drop('Survived', axis = 1)
y = train['Survived']


# In[26]:


train.head()


# In[27]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


# In[28]:


logreg = LogisticRegression()


# In[29]:


logreg.fit(X,y)


# In[30]:


test['Age'] = test[['Age', 'Pclass']].apply(fills_na, axis=1)
test.drop('Cabin', axis = 1, inplace = True)
sex1 = pd.get_dummies(test['Sex'], drop_first = True)
embarked1 = pd.get_dummies(test['Embarked'], drop_first = True)
test.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Embarked'],axis = 1, inplace = True)
test = pd.concat([test, sex1, embarked1], axis =1)
test=test.dropna()


# In[31]:


pred = logreg.predict(test)


# In[32]:


pred


# In[33]:


plt.hist(pred)


# In[34]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
logreg = LogisticRegression()
print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())

