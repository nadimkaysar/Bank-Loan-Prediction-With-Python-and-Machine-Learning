#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# ### Read data

# In[9]:


df1 = pd.read_csv('../Data/LoanStats_securev1_2017Q1.csv', skiprows=[0])
df2 = pd.read_csv('../Data/LoanStats_securev1_2017Q2.csv', skiprows=[0])
df3 = pd.read_csv('../Data/LoanStats_securev1_2017Q3.csv', skiprows=[0])
df4 = pd.read_csv('../Data/LoanStats3c_securev1_2014.csv', skiprows=[0])
df5 = pd.read_csv('../Data/LoanStats3d_securev1_2015.csv', skiprows=[0])


# ### Check if all the datasets have same column

# In[10]:


columns = np.dstack((list(df1.columns), list(df2.columns), list(df3.columns), list(df4.columns), list(df5.columns))) 


# In[11]:


coldf = pd.DataFrame(columns[0])


# In[2]:


# coldf.head()


# In[12]:


df = pd.concat([df1, df2, df3, df4, df5])


# ### Get familiar with data

# In[13]:


df.shape


# In[15]:


print(list(df.columns))


# In[16]:


df.head(5)


# In[17]:


df.dtypes.sort_values().to_frame('feature_type').groupby(by = 'feature_type').size().to_frame('count').reset_index()


# ### Select data with loan_status either Fully Paid or Charged Off 

# In[18]:


df.loan_status.value_counts()


# In[19]:


df = df.loc[(df['loan_status'].isin(['Fully Paid', 'Charged Off']))]


# In[20]:


df.shape


# ### Feature selections and clean

# ### Find the missing columns and their types

# In[21]:


df_dtypes = pd.merge(df.isnull().sum(axis = 0).sort_values().to_frame('missing_value').reset_index(),
         df.dtypes.to_frame('feature_type').reset_index(),
         on = 'index',
         how = 'inner')


# In[22]:


df_dtypes.sort_values(['missing_value', 'feature_type'])


# #### 1. Check columns have more than $400000$ missing values ($\approx90\%$)

# In[23]:


missing_df = df.isnull().sum(axis = 0).sort_values().to_frame('missing_value').reset_index()


# In[24]:


miss_4000 = list(missing_df[missing_df.missing_value >= 400000]['index'])


# In[25]:


print(len(miss_4000))


# In[26]:


print(sorted(miss_4000))


# In[27]:


df.drop(miss_4000, axis = 1, inplace = True)


# #### 2. Remove constant features

# In[28]:


def find_constant_features(dataFrame):
    const_features = []
    for column in list(dataFrame.columns):
        if dataFrame[column].unique().size < 2:
            const_features.append(column)
    return const_features


# In[29]:


const_features = find_constant_features(df)


# In[30]:


const_features


# In[31]:


df.hardship_flag.value_counts()


# In[32]:


df.drop(const_features, axis = 1, inplace = True)


# #### 3. Remove Duplicate rows

# In[33]:


df.shape


# In[34]:


df.drop_duplicates(inplace= True)


# In[35]:


df.shape


# #### 4. Remove duplicate columns

# In[36]:


def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break
    return dups


# In[37]:


duplicate_cols = duplicate_columns(df)


# In[38]:


duplicate_cols


# In[39]:


df.shape


# #### 5. Remove/process features manually

# In[40]:


features_to_be_removed = []


# In[41]:



def plot_feature(col_name, isContinuous):
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
        sns.boxplot(x=col_name, y='loan_status', data=df, ax=ax2)
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


# ### 0-10 features

# In[42]:


df.iloc[0:5, 0: 10]


# In[43]:


len(df.loan_amnt.value_counts())


# In[44]:


plot_feature('loan_amnt', True)


# It looks like all loans are not unique. Certain amount appear several times. It may be the reason, company has some range or certain amount to lend

# ##### Term feature

# In[45]:


df.term = df.term.str.replace('months', '').astype(np.int)


# In[46]:


df.term.value_counts()


# In[47]:


plot_feature('term', False)


# ##### interest rate

# In[48]:


df.int_rate = df.int_rate.str.replace('%', '').astype(np.float32)


# In[49]:


len(df.int_rate.value_counts())


# In[52]:


plot_feature('int_rate', True)


# It looks like applicants who could not afford to pay back or were charged off had higher interest rate.

# ##### grade and subgrade

# In[53]:


df.grade.value_counts()


# In[54]:


df.sub_grade.value_counts()


# In[55]:


plot_feature('grade', False)


# In[56]:


plot_feature('sub_grade', False)


# It seems that grade and sub grade have same shape and relation with loan status. IN this case I would keep sub_grade, because it carries more information than the grade.

# ##### emp_title

# In[57]:


len(df.emp_title.value_counts())


# It looks like emp_title has lots of unique value, which may not be strongly associated with predicted loan amount

# In[58]:


features_to_be_removed.extend(['emp_title', 'id'])


# #### 11-20 features

# In[59]:


df.iloc[0:5, 6: 20]


# ##### emp_length

# In[60]:


df.emp_length.value_counts()


# In[61]:


df.emp_length.fillna(value=0,inplace=True)
df['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
df['emp_length'] = df['emp_length'].astype(int)


# In[62]:


plot_feature('emp_length', False)


# It looks like emp lenght is not good predictor to determine the loan status. Sicne number of loanees remain same with the employment length.

# ##### home_ownership

# In[63]:


df.home_ownership.value_counts()


# In[64]:


plot_feature('home_ownership', False)


# home_ownership is also not that much discreminatory

# ##### verification_status

# In[65]:


df.verification_status.value_counts()


# In[66]:


df.verification_status = df.verification_status.map(lambda x: 1 if x == 'Not Verified' else 0)


# In[67]:


plot_feature('verification_status', False)


# verification_status is somewhat discreminative in the sense that, among the loanes whose source was verified are more charged off which is a bit wired.

# #### issue_d

# In[70]:


df.issue_d.value_counts()


# In[71]:


df['issue_month'] = pd.Series(df.issue_d).str.replace(r'-\d+', '')


# In[72]:


plot_feature('issue_month', False)


# It looks like people who borrowed in December, are more charged off than those who borrowed in other months.

# In[73]:


df.issue_month = df.issue_month.astype("category", categories=np.unique(df.issue_month)).cat.codes


# In[74]:


df.issue_month.value_counts()


# In[75]:


df['issue_year'] = pd.Series(df.issue_d).str.replace(r'\w+-', '').astype(np.int) 


# In[76]:


df.issue_year.value_counts()


# #### loan status

# In[77]:


df.loan_status.value_counts()


# In[78]:


df.loan_status = df.loan_status.map(lambda x: 1 if x == 'Charged Off' else 0)


# #### url

# In[79]:


features_to_be_removed.append('url')


# #### purpose

# In[80]:


df.purpose.value_counts()


# In[81]:


plot_feature('purpose', False)


# It looks like, purpose can be a good discrimnatory. For exmaple people who had a purpose for renewable energy are more charged off while people borrwed loan for car or educational purpose are less charged off.

# #### title

# In[82]:


len(df.title.value_counts())


# In[83]:


features_to_be_removed.append('title')


# #### zip_code

# In[84]:


len(df.zip_code.value_counts())


# In[85]:


features_to_be_removed.append('zip_code')


# ##### addr_state

# In[86]:


df.addr_state.value_counts()


# In[87]:


plot_feature('addr_state', False)


# addr_state can be a good discreminatory feature.

# ##### dti

# In[78]:


# plot_feature('dti', True)


# ### 21 - 30 features

# In[88]:


df.iloc[0:5, 15: 30]


# ##### earliest_cr_line

# In[89]:


df['earliest_cr_year'] = df.earliest_cr_line.str.replace(r'\w+-', '').astype(np.int)


# In[90]:


df['credit_history'] = np.absolute(df['issue_year']- df['earliest_cr_year'])


# In[91]:


df.credit_history.value_counts()


# In[92]:


features_to_be_removed.extend(['issue_d', 'mths_since_last_delinq', 'mths_since_last_record', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record'])


# ### 31 - 40 features

# In[93]:


df.iloc[0:5, 25: 40]


# In[94]:


df.revol_util = df.revol_util.str.replace('%', '').astype(np.float32)


# In[95]:


df.initial_list_status.value_counts()


# In[96]:


df.initial_list_status = df.initial_list_status.map(lambda x: 1 if x== 'w' else 0)


# In[97]:


features_to_be_removed.extend(['total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee'])


# ### 41 - 50 features

# In[98]:


df.iloc[0:5, 35: 50]


# In[99]:


df.application_type.value_counts()


# In[100]:


df.application_type = df.application_type.map(lambda x: 0 if x == 'Individual' else 1)


# In[101]:


features_to_be_removed.extend(['recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low', 'collections_12_mths_ex_med', 'mths_since_last_major_derog'])


# ### 51 - 60 features

# In[102]:


df.iloc[0:5, 45: 60]


# In[103]:


features_to_be_removed.extend([ 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt'])


# ### 61 - 70 features

# In[104]:


df.iloc[0:5, 55: 70]


# In[105]:


features_to_be_removed.extend(['mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq', 'mths_since_recent_inq', 'mths_since_recent_revol_delinq', 'num_accts_ever_120_pd'])


# #### 71 - 80 features

# In[106]:


df.iloc[0:5, 65: 80]


# In[107]:


features_to_be_removed.extend(['num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m'])


# #### 81 - 90 features

# In[108]:


df.iloc[0:5, 75: 90]


# In[109]:


features_to_be_removed.extend(['num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit'])


# ### 91 to rest of the features

# In[110]:


df.iloc[0:5, 85:]


# In[111]:


df.disbursement_method.value_counts()


# In[112]:


df.disbursement_method = df.disbursement_method.map(lambda x: 0 if x == 'Cash' else 1)


# In[113]:


df.debt_settlement_flag.value_counts()


# In[114]:


df.debt_settlement_flag = df.debt_settlement_flag.map(lambda x: 0 if x == 'N' else 1)


# In[115]:


features_to_be_removed.extend(['debt_settlement_flag', 'total_il_high_credit_limit'])


# ### Removed _ features

# In[116]:


print(features_to_be_removed)


# In[117]:


len(set(features_to_be_removed))


# ### Drop selected features

# In[118]:


df_selected = df.drop(list(set(features_to_be_removed)), axis = 1)


# In[119]:


df_selected.shape


# In[120]:


df_dtypes = pd.merge(df_selected.isnull().sum(axis = 0).sort_values().to_frame('missing_value').reset_index(),
         df_selected.dtypes.to_frame('feature_type').reset_index(),
         on = 'index',
         how = 'inner')


# In[121]:


df_dtypes.sort_values(['missing_value', 'feature_type'])


# In[122]:


df_selected.dropna(inplace=True)


# In[123]:


df_selected.shape


# In[124]:


df_selected.drop('earliest_cr_line', axis = True, inplace=True)


# In[125]:


df_selected.purpose.value_counts()


# In[126]:


df_selected.purpose = df_selected.purpose.astype("category", categories=np.unique(df_selected.purpose)).cat.codes


# In[127]:


df_selected.purpose.value_counts()


# In[128]:


df_selected.home_ownership = df_selected.home_ownership.astype("category", categories = np.unique(df_selected.home_ownership)).cat.codes


# In[129]:


df_selected.home_ownership.value_counts()


# In[130]:


df_selected.grade = df_selected.grade.astype("category", categories = np.unique(df_selected.grade)).cat.codes


# In[131]:


df_selected.grade.value_counts()


# In[132]:


df_selected.sub_grade = df_selected.sub_grade.astype("category", categories = np.unique(df_selected.sub_grade)).cat.codes


# In[133]:


df_selected.sub_grade.value_counts()


# In[134]:


df_selected.addr_state = df_selected.addr_state.astype("category", categories = np.unique(df_selected.addr_state)).cat.codes


# In[135]:


df_selected.sub_grade.value_counts()


# In[137]:


df_selected.columns


# ### Save selected features

# In[139]:


df_selected.to_csv('../Data/df_selected.csv', index = False)


# In[ ]:




