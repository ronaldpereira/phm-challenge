#!/usr/bin/env python
# coding: utf-8

# # PHM Challenge 2014

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib


# In[3]:


RANDOM_SEED = 1212


# In[4]:


train = pd.read_csv("../data/train_features.csv")


# In[5]:


train.drop(["Time_failure", "Time_diff"], axis=1, inplace=True)


# In[7]:


train = pd.get_dummies(train, columns=["Asset", "Reason", "Part"])


# In[9]:


x_train, x_test, y_train, y_test = train_test_split(
    train.drop("Failure", axis=1),
    train["Failure"],
    stratify=train["Failure"],
    test_size=0.333,
    random_state=RANDOM_SEED,
)


# In[10]:


model = RandomForestRegressor(
    n_estimators=2000, n_jobs=-1, verbose=1, random_state=RANDOM_SEED
)


# In[ ]:


model.fit(x_train, y_train)


# In[ ]:


probs = model.predict_proba(x_test)


# In[ ]:


print(probs)


# In[ ]:


sc = model.score(x_test, y_test)


# In[ ]:


joblib.dump(model, "random_forest_regressor_failure.joblib")

