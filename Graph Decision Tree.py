
# coding: utf-8

# In[1]:


# Python Notebook - Va's Draft

import numpy as np
import pandas as pd
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

from sklearn import tree


# In[2]:


data = pd.read_csv('datapv1.csv')


# In[3]:


data['extend'] = [0 if pd.isnull(x) == True else 1 for x in data['charge_time']]


# In[4]:


data['shop_level'] = data['shop_level'].fillna(1)
data['free_app_installed'] = data['free_app_installed'].fillna(0)
data['paid_app_installed'] = data['paid_app_installed'].fillna(0)


# In[5]:


data.head()


# In[6]:


data2 = data.drop(['shop_id', 'upgrade_time', 'charge_time', 'amount'], axis=1)
data2.head()


# In[7]:


target = data2['extend']
data3 = data2.drop('extend', axis=1)


# In[8]:


data3 = pd.get_dummies(data3, columns=['package_type'])


# In[9]:


data3.head()


# In[10]:


for x in data3['package']:
    if x == 'Starter':
        data3['package_encoded'] = 0
    elif x == 'Ninja':
        data3['package_encoded'] = 1
    else:
        data3['package_encoded'] = 2


# In[11]:


data3 = data3.drop(['package'], axis=1)


# In[12]:


data3.isnull().any()


# In[13]:


classifier = tree.DecisionTreeClassifier(max_depth=5)
classifier = classifier.fit(data3, target)


# In[14]:


import graphviz 
dot_data = tree.export_graphviz(classifier, out_file='graph2', feature_names=data3.columns, class_names=['not extend','extend'],
                               filled=True, rounded=True) 


# # Further explore

# In[17]:


extend_by_shop_level = pysqldf("""
SELECT
    shop_level,
    COUNT(shop_id) AS shops,
    SUM(extend) AS extend,
    AVG(extend) AS extend_pct
FROM data
WHERE upgrade_time >= '2018-02-01'
GROUP BY 1
""")


# In[18]:


extend_by_shop_level

