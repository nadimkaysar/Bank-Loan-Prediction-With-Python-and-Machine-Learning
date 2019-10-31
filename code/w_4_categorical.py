#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('../Processed Data/df_processed_v2.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[7]:


describe = df.describe().T.reset_index()
# corr_with_target = df.corrwith(df.loan_status).sort_values(ascending = False).abs().to_frame('correlation_with_target').reset_index().head(20)
unique_values = df.nunique().to_frame('unique_values').reset_index()
# corr_unique = pd.merge(corr_with_target, unique_values, on = 'index', how = 'inner')
df_stat = pd.merge(describe, unique_values, on = 'index', how = 'inner')


# In[8]:


df_stat.shape


# In[9]:


df_stat.sort_values(by = 'unique_values', ascending=False)


# In[10]:


df.disbursement_method.unique()


# In[11]:


target_encoded_features = list(df_stat[(df_stat.unique_values>2) & (df_stat.unique_values<= 200)]['index'])


# In[12]:


target_encoded_features


# In[19]:


import category_encoders as ce


# In[20]:


target_encoder = ce.TargetEncoder(cols = target_encoded_features)


# In[21]:


target_encoder.fit(df, y = df.loan_status)


# In[22]:


encoded_df = target_encoder.transform(df)


# In[23]:


encoded_df.head()


# In[24]:


ordinal_encoder = ce.OrdinalEncoder(cols = ['term'])
ordinal_encoder.fit(encoded_df)
encoded_df = ordinal_encoder.transform(encoded_df)


# In[25]:


encoded_df.shape


# In[26]:


encoded_df.head()


# In[27]:


encoded_df.to_csv('../Processed Data/df_processed_categorical_v3.csv', index = False)


# In[ ]:




