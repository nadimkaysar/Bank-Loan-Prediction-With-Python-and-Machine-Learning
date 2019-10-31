#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('../Processed Data/df_processed_categorical_v3.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# ### Correlation with target

# In[6]:


corr_with_target = df.corrwith(df.loan_status, method = 'spearman').sort_values(ascending = False).abs().to_frame('correlation_with_target').reset_index()


# In[7]:


corr_with_target


# In[8]:


best_features = corr_with_target[(corr_with_target.correlation_with_target >= 0.08) & 
                                 (corr_with_target.correlation_with_target < 0.5)]['index']


# In[9]:


print(len(best_features))
list(best_features)


# ### Automated feature engineering

# In[10]:


for icol in best_features:
    for jcol in best_features:
        if icol != jcol:
            df['multiplied_'+icol+'_'+jcol] = df[icol] * df[jcol]
            df['minus_'+icol+'_'+jcol] = df[icol] - df[jcol]


# In[11]:


for icol in best_features:
    for jcol in best_features:
        for kcol in best_features:
            df['multiplied_'+icol+'_'+jcol + '_'+kcol] = df[icol] * df[jcol] * df[kcol]


# In[12]:


df.shape


# In[13]:


aa = df.corrwith(df.loan_status, method = 'spearman').sort_values(ascending = False).abs().to_frame('correlation_with_target').reset_index()


# In[14]:


aa.head(20)


# In[ ]:





# In[15]:


df.to_csv('../Processed Data/df_processed_v4.csv', index = False)


# In[ ]:




