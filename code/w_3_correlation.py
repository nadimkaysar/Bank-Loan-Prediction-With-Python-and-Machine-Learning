#!/usr/bin/env python
# coding: utf-8

# In[15]:


import warnings
warnings.filterwarnings("ignore")


# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle, class_weight
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[17]:


import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# In[18]:


init_notebook_mode(connected=True)
cf.go_offline()


# ### Read data

# In[19]:


df_selected = pd.read_csv('../Processed Data/df_selected.csv')


# In[20]:


df_selected.shape


# In[21]:


df_selected.describe().T


# In[22]:


def plot_feature(df, col_name, isContinuous):
    """
    Visualize a variable with and without faceting on the loan status.
    - col_name is the variable name in the dataframe
    - full_name is the full variable name
    - continuous is True if the variable is continuous, False otherwise
    """
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,3), dpi=90)
    
    # Plot without loan status
    if isContinuous:
        sns.distplot(df.loc[df[col_name].notnull(), col_name], kde=False, ax=ax1)
    else:
        sns.countplot(df[col_name], order=sorted(df[col_name].unique()), color='#5975A4', saturation=1, ax=ax1)
    ax1.set_xlabel(col_name)
    ax1.set_ylabel('Count')
    ax1.set_title(col_name)
    plt.xticks(rotation = 90)

    # Plot with loan status
    if isContinuous:
        sns.boxplot(y=col_name, x='loan_status', data=df, ax=ax2)
        ax2.set_ylabel('')
        ax2.set_title(col_name + ' by Loan Status')
    else:
        data = df.groupby(col_name)['loan_status'].value_counts(normalize=True).to_frame('proportion').reset_index()        
        sns.barplot(x = col_name, y = 'proportion', hue= "loan_status", data = data, saturation=1, ax=ax2)
        ax2.set_ylabel('Loan fraction')
        ax2.set_title('Loan status')
        plt.xticks(rotation = 90)
    ax2.set_xlabel(col_name)
    
    plt.tight_layout()


# ### Feature correlations

# In[23]:


corr = df_selected.corr(method = 'spearman')


# In[24]:


layout = cf.Layout(height=600,width=600)
corr.abs().iplot(kind = 'heatmap', layout=layout.to_plotly_json(), colorscale = 'RdBu')


# In[25]:


import scipy
import scipy.cluster.hierarchy as sch

X = df_selected.corr(method = 'spearman').values
d = sch.distance.pdist(X)   # vector of ('55' choose 2) pairwise distances
L = sch.linkage(d, method='complete')
ind = sch.fcluster(L, 0.25*d.max(), 'distance')
columns = [df_selected.columns.tolist()[i] for i in list((np.argsort(ind)))]
df_selected_new = df_selected.reindex_axis(columns, axis=1)


# In[26]:


clustered_corr = df_selected_new.corr(method = 'spearman')


# In[27]:


clustered_corr.abs().iplot(kind = 'heatmap', layout=layout.to_plotly_json(), colorscale='RdBu')


# ### Find highly correlated features

# In[28]:


new_corr = corr.abs()
new_corr.loc[:,:] = np.tril(new_corr, k=-1) # below main lower triangle of an array
new_corr = new_corr.stack().to_frame('correlation').reset_index().sort_values(by='correlation', ascending=False)


# In[29]:


new_corr[new_corr.correlation > 0.4]


# In[30]:


high_correlated_feat = ['funded_amnt','funded_amnt_inv', 'fico_range_high', 'grade', 
                        'credit_history', 'installment']


# In[31]:


df_selected.drop(high_correlated_feat, axis=1, inplace=True)


# In[32]:


df_selected.shape


# ### Correlation with target variable

# In[18]:


# df_selected.nunique().to_frame().reset_index()


# In[33]:


corr_with_target = df_selected.corrwith(df_selected.loan_status).sort_values(ascending = False).abs().to_frame('correlation_with_target').reset_index().head(20)
unique_values = df_selected.nunique().to_frame('unique_values').reset_index()
corr_with_unique = pd.merge(corr_with_target, unique_values, on = 'index', how = 'inner')


# In[34]:


corr_with_unique


# ### Vizualizations

# In[35]:


plot_feature(df_selected, 'sub_grade', False)


# In[36]:


plot_feature(df_selected, 'int_rate', True)


# In[37]:


plot_feature(df_selected, 'dti', True)


# In[38]:


plot_feature(df_selected, 'revol_util', True)


# In[39]:


plot_feature(df_selected, 'issue_month', False)


# ### Observe the selected features

# In[40]:


df_selected.shape


# In[41]:


df_selected.head().T


# In[42]:


df_selected.to_csv('../Processed Data/df_processed_v2.csv', index = False)


# In[ ]:




